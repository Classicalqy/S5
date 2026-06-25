"""Run logit-only LTSpice MNIST accuracy validation with resume support."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from .compare_transient import canonical_column, read_trace_table
from .export_full_model import FULL_MODEL_CIRCUIT_SEMANTICS, extract_full_model
from .export_netlist import _state_node, format_spice_value, load_flax_params, SUPPORTED_SSM_PARAMS
from .trace_utils import linear_nodes, zoh_pwl_source_line
from .validate_digital_alignment import load_mnist_samples, simulate_full_digital
from .validate_transient import strip_final_end


DEFAULT_LTSPICE = "/Applications/LTspice.app/Contents/SharedSupport/ltspice/LTspice/run_ltspice"
COMPARISON_SEMANTICS = "ltspice_continuous_cascade_vs_digital_stacked_recurrence"
LOGIT_MAX_ABS_FIELD = "ltspice_vs_digital_final_logit_max_abs"
LOGIT_RMSE_FIELD = "ltspice_vs_digital_final_logit_rmse"
OLD_LOGIT_MAX_ABS_FIELD = "final_logit_max_abs"
OLD_LOGIT_RMSE_FIELD = "final_logit_rmse"


def final_logits_from_digital(model, inputs, sample_rate):
    traces = simulate_full_digital(model, inputs, sample_rate)
    return np.asarray([traces[node][-1] for node in linear_nodes("LOGIT", model.decoder_bias.shape[0])])


def final_logits_from_raw(raw_path, logit_nodes, final_time):
    table = read_trace_table(raw_path)
    if "time" not in table:
        raise ValueError(f"{raw_path} does not contain a time column.")
    logits = []
    for node in logit_nodes:
        key = canonical_column(node)
        if key not in table:
            raise ValueError(f"{raw_path} is missing {node}.")
        logits.append(float(np.interp(final_time, table["time"], table[key])))
    return np.asarray(logits, dtype=np.float64)


def runner_path_for_ltspice(path):
    path = Path(path).resolve()
    try:
        rel = path.relative_to(Path.home())
        return "Y:\\" + "\\".join(rel.parts)
    except ValueError:
        return str(path)


def write_logit_only_deck(base_cir_path, out_path, inputs, model, sample_rate, max_step_divisor):
    body = strip_final_end(Path(base_cir_path).read_text())
    dt = 1.0 / float(sample_rate)
    duration = inputs.shape[0] * dt
    logit_nodes = linear_nodes("LOGIT", model.decoder_bias.shape[0])

    lines = [body, "", "* MNIST LTSpice logit-only accuracy stimulus"]
    lines.append(zoh_pwl_source_line("VSTIM_IN0", "IN0", dt, inputs[:, 0]))
    for layer_idx, layer in enumerate(model.ssm_layers):
        for block_idx in range(layer.n_blocks):
            for state_idx in range(2):
                lines.append(f".ic V({_state_node(layer_idx, block_idx, state_idx)})=0")
    lines.append(".options plotwinsize=0")
    lines.append(".save " + " ".join(f"V({node})" for node in logit_nodes))
    lines.append(
        f".tran 0 {format_spice_value(duration)} 0 {format_spice_value(dt / float(max_step_divisor))} uic"
    )
    lines.append(".end")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if LOGIT_MAX_ABS_FIELD not in row and OLD_LOGIT_MAX_ABS_FIELD in row:
                row[LOGIT_MAX_ABS_FIELD] = row[OLD_LOGIT_MAX_ABS_FIELD]
            if LOGIT_RMSE_FIELD not in row and OLD_LOGIT_RMSE_FIELD in row:
                row[LOGIT_RMSE_FIELD] = row[OLD_LOGIT_RMSE_FIELD]
            rows[int(row["sample"])] = row
        return rows


def write_rows(path, rows, n_classes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample",
        "label",
        "digital_pred",
        "ltspice_pred",
        "status",
        LOGIT_MAX_ABS_FIELD,
        LOGIT_RMSE_FIELD,
        "error",
        "deck",
        "raw",
    ]
    fields.extend(f"digital_logit{i}" for i in range(n_classes))
    fields.extend(f"ltspice_logit{i}" for i in range(n_classes))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_idx in sorted(rows):
            writer.writerow({field: rows[sample_idx].get(field, "") for field in fields})


def _as_int(value):
    return None if value in {None, ""} else int(value)


def build_summary(rows, num_samples):
    complete = [row for row in rows.values() if row.get("status") == "complete"]
    digital_rows = [row for row in rows.values() if row.get("digital_pred", "") != ""]
    digital_correct = [
        _as_int(row["digital_pred"]) == _as_int(row["label"])
        for row in digital_rows
    ]
    ltspice_correct = [
        _as_int(row["ltspice_pred"]) == _as_int(row["label"])
        for row in complete
    ]
    disagreements = [
        _as_int(row["ltspice_pred"]) != _as_int(row["digital_pred"])
        for row in complete
    ]
    max_abs = [float(row[LOGIT_MAX_ABS_FIELD]) for row in complete if row.get(LOGIT_MAX_ABS_FIELD, "") != ""]
    rmses = [float(row[LOGIT_RMSE_FIELD]) for row in complete if row.get(LOGIT_RMSE_FIELD, "") != ""]
    return {
        "num_samples": int(num_samples),
        "num_completed": len(complete),
        "num_pending": int(num_samples) - len(complete),
        "digital_accuracy": float(np.mean(digital_correct)) if digital_correct else None,
        "ltspice_continuous_cascade_accuracy": float(np.mean(ltspice_correct)) if ltspice_correct else None,
        "digital_vs_ltspice_continuous_cascade_disagreement_rate": (
            float(np.mean(disagreements)) if disagreements else None
        ),
        "ltspice_vs_digital_final_logit_max_abs": max(max_abs) if max_abs else None,
        "ltspice_vs_digital_final_logit_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "status": "complete" if len(complete) == num_samples else "pending",
    }


def write_summary(path, rows, num_samples, metadata):
    summary = build_summary(rows, num_samples)
    summary.update(metadata)
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_ltspice(ltspice_bin, deck_path):
    return subprocess.run(
        [ltspice_bin, "-b", "-ascii", runner_path_for_ltspice(deck_path)],
        check=False,
    )


def sample_paths(out_dir, sample_idx):
    sample_dir = Path(out_dir) / f"sample_{sample_idx:05d}"
    deck_path = sample_dir / f"sample_{sample_idx:05d}.cir"
    return deck_path, deck_path.with_suffix(".raw"), deck_path.with_suffix(".log")


def run_accuracy_sample(
    sample_idx,
    inputs,
    label,
    model,
    cir_path,
    sample_rate,
    out_dir,
    logit_nodes,
    ltspice_bin,
    max_step_divisor,
    delete_raw_after_read,
    delete_log_after_read,
    run_sim,
):
    deck_path, raw_path, log_path = sample_paths(out_dir, sample_idx)
    deck_path = write_logit_only_deck(cir_path, deck_path, inputs, model, sample_rate, max_step_divisor)
    digital_logits = final_logits_from_digital(model, inputs, sample_rate)
    row = {
        "sample": sample_idx,
        "label": int(label),
        "digital_pred": int(np.argmax(digital_logits)),
        "status": "pending",
        "deck": str(deck_path),
        "raw": str(raw_path),
    }
    row.update({f"digital_logit{i}": format_spice_value(value) for i, value in enumerate(digital_logits)})

    if run_sim and not raw_path.exists():
        result = run_ltspice(ltspice_bin, deck_path)
        if result.returncode != 0:
            row.update({"status": "ltspice_failed", "ltspice_pred": "", "error": f"returncode={result.returncode}"})
            return sample_idx, row

    if raw_path.exists():
        try:
            final_time = inputs.shape[0] / float(sample_rate)
            ltspice_logits = final_logits_from_raw(raw_path, logit_nodes, final_time)
        except Exception as exc:
            row.update({"status": "raw_read_failed", "ltspice_pred": "", "error": str(exc)})
            return sample_idx, row
        diff = ltspice_logits - digital_logits
        row.update(
            {
                "ltspice_pred": int(np.argmax(ltspice_logits)),
                "status": "complete",
                LOGIT_MAX_ABS_FIELD: format_spice_value(np.max(np.abs(diff))),
                LOGIT_RMSE_FIELD: format_spice_value(np.sqrt(np.mean(diff ** 2))),
            }
        )
        row.update({f"ltspice_logit{i}": format_spice_value(value) for i, value in enumerate(ltspice_logits)})
        if delete_raw_after_read:
            raw_path.unlink(missing_ok=True)
        if delete_log_after_read:
            log_path.unlink(missing_ok=True)
    return sample_idx, row


def validate_ltspice_accuracy(
    params_path,
    cir_path,
    ssm_param,
    sample_rate,
    out_dir,
    num_samples=10000,
    cache_dir="cache_dir",
    seed=0,
    batch_size=256,
    ltspice_bin=DEFAULT_LTSPICE,
    max_step_divisor=10,
    delete_raw_after_read=False,
    delete_log_after_read=False,
    run_sim=True,
    samples=None,
    labels=None,
):
    params = load_flax_params(params_path)
    model = extract_full_model(params, ssm_param, sample_rate)
    if samples is None:
        samples, labels = load_mnist_samples(num_samples, cache_dir, seed, batch_size)
    else:
        samples = np.asarray(samples, dtype=np.float64)[:num_samples]
        labels = np.asarray(labels, dtype=np.int64)[: samples.shape[0]]
    num_samples = int(samples.shape[0])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.csv"
    summary_path = out_dir / "summary.json"
    rows = read_rows(per_sample_path)
    logit_nodes = linear_nodes("LOGIT", model.decoder_bias.shape[0])
    metadata = {
        "params": str(params_path),
        "base_cir": str(cir_path),
        "sample_rate": float(sample_rate),
        "ssm_param": ssm_param,
        "max_step_divisor": int(max_step_divisor),
        "per_sample_csv": str(per_sample_path),
        "circuit_semantics": FULL_MODEL_CIRCUIT_SEMANTICS,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "comparison_note": (
            "Final-logit differences include the model-semantics difference between "
            "a continuous analog cascade and the sampled stacked SSM recurrence."
        ),
    }

    pending = [
        (sample_idx, inputs, labels[sample_idx])
        for sample_idx, inputs in enumerate(samples)
        if not (rows.get(sample_idx) and rows[sample_idx].get("status") == "complete")
    ]

    def finish(sample_idx, row):
        rows[sample_idx] = row
        write_rows(per_sample_path, rows, model.decoder_bias.shape[0])
        write_summary(summary_path, rows, num_samples, metadata)

    for sample_idx, inputs, label in pending:
        try:
            _, row = run_accuracy_sample(
                sample_idx,
                inputs,
                label,
                model,
                cir_path,
                sample_rate,
                out_dir,
                logit_nodes,
                ltspice_bin,
                max_step_divisor,
                delete_raw_after_read,
                delete_log_after_read,
                run_sim,
            )
        except Exception as exc:
            deck_path, raw_path, _ = sample_paths(out_dir, sample_idx)
            row = {
                "sample": sample_idx,
                "label": int(labels[sample_idx]),
                "status": "error",
                "error": str(exc),
                "deck": str(deck_path),
                "raw": str(raw_path),
            }
        finish(sample_idx, row)

    write_rows(per_sample_path, rows, model.decoder_bias.shape[0])
    return write_summary(summary_path, rows, num_samples, metadata)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--cir", required=True)
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--out-dir", default="out/ltspice_accuracy")
    parser.add_argument("--cache-dir", default="cache_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ltspice-bin", default=DEFAULT_LTSPICE)
    parser.add_argument("--max-step-divisor", type=int, default=10)
    parser.add_argument("--delete-raw-after-read", action="store_true")
    parser.add_argument("--delete-log-after-read", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Generate decks and digital logits without running LTSpice.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = validate_ltspice_accuracy(
        args.params,
        args.cir,
        args.ssm_param,
        args.sample_rate,
        args.out_dir,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        ltspice_bin=args.ltspice_bin,
        max_step_divisor=args.max_step_divisor,
        delete_raw_after_read=args.delete_raw_after_read,
        delete_log_after_read=args.delete_log_after_read,
        run_sim=not args.no_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
