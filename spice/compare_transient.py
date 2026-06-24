"""Compare Python reference CSV with LTSpice-exported transient traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


_ENGINEERING_SUFFIXES = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def canonical_column(name):
    name = name.strip().strip('"')
    lowered = name.lower()
    if lowered in {"time", "time(s)", "t"}:
        return "time"
    match = re.fullmatch(r"[vV]\((.+)\)", name)
    if match:
        return match.group(1).lower()
    return name.lower()


def parse_number(value):
    text = value.strip().strip('"')
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([A-Za-z]+)", text)
    if not match:
        raise ValueError(f"Could not parse numeric value: {value!r}")
    base = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix not in _ENGINEERING_SUFFIXES:
        raise ValueError(f"Unknown engineering suffix in value: {value!r}")
    return base * _ENGINEERING_SUFFIXES[suffix]


def sniff_delimiter(header):
    if "," in header:
        return ","
    if "\t" in header:
        return "\t"
    return None


def read_trace_table(path):
    path = Path(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].startswith("Title:"):
        return read_ltspice_ascii_raw(lines)
    delimiter = sniff_delimiter(lines[0])
    if delimiter is None:
        rows = [line.split() for line in lines]
    else:
        rows = list(csv.reader(lines, delimiter=delimiter))
    headers = [canonical_column(h) for h in rows[0]]
    columns = {header: [] for header in headers}
    for row in rows[1:]:
        if len(row) != len(headers):
            continue
        for header, value in zip(headers, row):
            columns[header].append(parse_number(value))
    return {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}


def read_ltspice_ascii_raw(lines):
    variables_idx = lines.index("Variables:")
    values_idx = lines.index("Values:")
    variable_names = []
    for line in lines[variables_idx + 1:values_idx]:
        parts = line.split()
        if len(parts) >= 3:
            variable_names.append(canonical_column(parts[1]))
    columns = {name: [] for name in variable_names}
    n_vars = len(variable_names)
    idx = values_idx + 1
    while idx < len(lines):
        first = lines[idx].split()
        if len(first) < 2:
            break
        values = [parse_number(first[1])]
        idx += 1
        for _ in range(n_vars - 1):
            values.append(parse_number(lines[idx]))
            idx += 1
        for name, value in zip(variable_names, values):
            columns[name].append(value)
    return {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}


def compare_traces(reference_csv, ltspice_export, nodes=None):
    reference = read_trace_table(reference_csv)
    ltspice = read_trace_table(ltspice_export)
    if "time" not in reference or "time" not in ltspice:
        raise ValueError("Both files must contain a time column.")
    if nodes is None:
        nodes = sorted(set(reference.keys()) & set(ltspice.keys()) - {"time"})
    else:
        nodes = [canonical_column(node) for node in nodes]
    if not nodes:
        raise ValueError("No common trace columns found.")

    ref_time = reference["time"]
    spice_time = ltspice["time"]
    results = {}
    for node in nodes:
        if node not in reference or node not in ltspice:
            continue
        spice_interp = np.interp(ref_time, spice_time, ltspice[node])
        diff = spice_interp - reference[node]
        results[node] = {
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
        }
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Python reference CSV.")
    parser.add_argument("--ltspice", required=True, help="LTSpice exported text/CSV traces.")
    parser.add_argument("--nodes", nargs="*", default=None, help="Optional node names to compare.")
    parser.add_argument("--json-out", default=None, help="Optional JSON summary output.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = compare_traces(args.reference, args.ltspice, args.nodes)
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
