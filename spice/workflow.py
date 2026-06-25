"""Unified LTSpice export and validation workflow for the restricted S5 MNIST model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .compare_transient import canonical_column, read_trace_table
from .export_full_model import export_full_model, extract_full_model
from .export_netlist import (
    _output_node,
    _state_node,
    export_netlist,
    load_flax_params,
    SUPPORTED_SSM_PARAMS,
)
from .metrics import rrmse, trace_metrics
from .plots import plot_logit_bar, plot_trace_overlay
from .trace_utils import linear_nodes
from .validate_digital_alignment import (
    generate_digital_alignment_artifacts,
    load_mnist_samples,
)
from .validate_ltspice_accuracy import DEFAULT_LTSPICE, run_ltspice, validate_ltspice_accuracy
from .validate_transient import generate_validation_artifacts


def _path(value):
    return str(Path(value))


def _ltspice_raw_path(deck_path):
    return Path(deck_path).with_suffix(".raw")


def _table_trace(table, times, nodes):
    trace = {"time": np.asarray(times, dtype=np.float64)}
    for node in nodes:
        key = canonical_column(node)
        if key not in table:
            raise ValueError(f"Missing node {node} in LTSpice trace.")
        trace[node] = np.interp(times, table["time"], table[key])
    return trace


def _reference_trace(csv_path, nodes):
    table = read_trace_table(csv_path)
    trace = {"time": table["time"]}
    for node in nodes:
        trace[node] = table[canonical_column(node)]
    return trace


def _maybe_run_ltspice(deck_path, ltspice_bin, run_ltspice_enabled):
    raw_path = _ltspice_raw_path(deck_path)
    if run_ltspice_enabled and not raw_path.exists():
        result = run_ltspice(ltspice_bin, deck_path)
        if result.returncode != 0:
            return raw_path, f"ltspice_failed:returncode={result.returncode}"
    return raw_path, "complete" if raw_path.exists() else "pending"


def _layer_nodes(layer_idx, layer_manifest, role):
    if role == "state":
        return [
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(layer_manifest["n_blocks"])
            for state_idx in range(2)
        ]
    if role == "output":
        return [_output_node(layer_idx, idx) for idx in range(layer_manifest["output_dim"])]
    raise ValueError("role must be state or output.")


def _model_layer_nodes(model, layer_idx, role):
    layer = model.ssm_layers[layer_idx]
    if role == "state":
        return [
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(layer.n_blocks)
            for state_idx in range(2)
        ]
    if role == "output":
        return [_output_node(layer_idx, idx) for idx in range(layer.output_dim)]
    raise ValueError("role must be state or output.")


def _node_rrmse_rows(sample_idx, layer_idx, reference, candidate, nodes, kind):
    rows = []
    for node in nodes:
        rows.append(
            {
                "sample": int(sample_idx),
                "layer": int(layer_idx),
                "kind": kind,
                "node": node,
                "rrmse": rrmse(reference[node], candidate[node]),
            }
        )
    return rows


def _write_alignment_rrmse_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample", "layer", "kind", "node", "rrmse"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def run_layer_sanity(
    params_path,
    layer_cir,
    layer_manifest,
    ssm_param,
    sample_rate,
    out_dir,
    ltspice_bin,
    run_ltspice_enabled,
    duration,
    points,
    stimulus,
    amplitude,
):
    out_dir = Path(out_dir)
    rows = []
    for layer_record in layer_manifest["layers"]:
        layer_idx = int(layer_record["index"])
        layer_dir = out_dir / f"layer_{layer_idx:02d}"
        metadata = generate_validation_artifacts(
            params_path=params_path,
            cir_path=layer_cir,
            ssm_param=ssm_param,
            sample_rate=sample_rate,
            out_dir=layer_dir,
            layer_index=layer_idx,
            duration=duration,
            points=points,
            stimulus=stimulus,
            amplitude=amplitude,
        )
        deck_path = Path(metadata["validation_cir"])
        reference_path = Path(metadata["reference_csv"])
        raw_path, status = _maybe_run_ltspice(deck_path, ltspice_bin, run_ltspice_enabled)

        row = {
            "layer": layer_idx,
            "layer_path": metadata["layer_path"],
            "status": status,
            "validation_cir": _path(deck_path),
            "reference_csv": _path(reference_path),
            "raw": _path(raw_path),
            "state_rrmse": None,
            "output_rrmse": None,
            "plot": None,
        }
        state_nodes = _layer_nodes(layer_idx, layer_record, "state")
        output_nodes = _layer_nodes(layer_idx, layer_record, "output")
        if raw_path.exists():
            try:
                reference = _reference_trace(reference_path, state_nodes + output_nodes)
                raw = read_trace_table(raw_path)
                ltspice = _table_trace(raw, reference["time"], state_nodes + output_nodes)
                row["state_rrmse"] = trace_metrics(reference, ltspice, state_nodes)["rrmse"]
                row["output_rrmse"] = trace_metrics(reference, ltspice, output_nodes)["rrmse"]
                plot_nodes = state_nodes[:2]
                if plot_nodes:
                    plot_path = layer_dir / f"layer_{layer_idx:02d}_state_overlay.png"
                    plot_trace_overlay(plot_path, reference["time"], reference, ltspice, plot_nodes)
                    row["plot"] = _path(plot_path)
            except Exception as exc:
                row["status"] = f"compare_failed:{exc}"
        rows.append(row)

    summary = {"layers": rows}
    summary["max_state_rrmse"] = max((r["state_rrmse"] for r in rows if r["state_rrmse"] is not None), default=None)
    summary["max_output_rrmse"] = max((r["output_rrmse"] for r in rows if r["output_rrmse"] is not None), default=None)
    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _plot_full_alignment_sample(alignment_dir, model, sample_idx=0):
    sample_dir = Path(alignment_dir) / f"sample_{sample_idx:04d}"
    digital_path = sample_dir / "digital_reference.csv"
    raw_path = sample_dir / f"sample_{sample_idx:04d}.raw"
    if not digital_path.exists() or not raw_path.exists():
        return {}, []

    state_nodes = []
    output_nodes = []
    for layer_idx, layer in enumerate(model.ssm_layers):
        state_nodes.extend(_model_layer_nodes(model, layer_idx, "state"))
        output_nodes.extend(_model_layer_nodes(model, layer_idx, "output"))
    logit_nodes = linear_nodes("LOGIT", model.decoder_bias.shape[0])
    digital = _reference_trace(digital_path, state_nodes + output_nodes + logit_nodes)
    raw = read_trace_table(raw_path)
    ltspice = _table_trace(raw, digital["time"], state_nodes + output_nodes + logit_nodes)

    logit_plot = sample_dir / "final_logits.png"
    plots = {}
    rrmse_rows = []
    for layer_idx, _layer in enumerate(model.ssm_layers):
        layer_state_nodes = _model_layer_nodes(model, layer_idx, "state")
        layer_output_nodes = _model_layer_nodes(model, layer_idx, "output")
        state_rows = _node_rrmse_rows(sample_idx, layer_idx, digital, ltspice, layer_state_nodes, "state")
        output_rows = _node_rrmse_rows(sample_idx, layer_idx, digital, ltspice, layer_output_nodes, "output")
        rrmse_rows.extend(state_rows)
        rrmse_rows.extend(output_rows)

        worst_nodes = [
            row["node"]
            for row in sorted(state_rows, key=lambda row: row["rrmse"], reverse=True)[:2]
        ]
        first_block_nodes = layer_state_nodes[:2]
        layer_plots = {}
        if worst_nodes:
            worst_path = sample_dir / f"layer_{layer_idx:02d}_worst_states.png"
            plot_trace_overlay(
                worst_path,
                digital["time"],
                digital,
                ltspice,
                worst_nodes,
                reference_label="digital",
                candidate_label="ltspice",
            )
            layer_plots["worst_state_plot"] = _path(worst_path)
        if first_block_nodes:
            first_path = sample_dir / f"layer_{layer_idx:02d}_first_block_states.png"
            plot_trace_overlay(
                first_path,
                digital["time"],
                digital,
                ltspice,
                first_block_nodes,
                reference_label="digital",
                candidate_label="ltspice",
            )
            layer_plots["first_block_state_plot"] = _path(first_path)
        plots[f"layer_{layer_idx:02d}"] = {
            "worst_state_nodes": worst_nodes,
            "first_block_state_nodes": first_block_nodes,
            **layer_plots,
        }

    if state_nodes:
        plot_trace_overlay(
            sample_dir / "all_first_block_states.png",
            digital["time"],
            digital,
            ltspice,
            [_model_layer_nodes(model, layer_idx, "state")[idx] for layer_idx in range(len(model.ssm_layers)) for idx in range(2)],
            reference_label="digital",
            candidate_label="ltspice",
        )
    plot_logit_bar(
        logit_plot,
        [digital[node][-1] for node in logit_nodes],
        [ltspice[node][-1] for node in logit_nodes],
        title=f"Sample {sample_idx} final logits",
    )
    plots["logit_plot"] = _path(logit_plot)
    return plots, rrmse_rows


def plot_full_alignment_samples(alignment_dir, model, num_samples):
    plots = {}
    rrmse_rows = []
    for sample_idx in range(int(num_samples)):
        sample_plots, sample_rows = _plot_full_alignment_sample(alignment_dir, model, sample_idx)
        if sample_plots:
            plots[f"sample_{sample_idx:04d}"] = sample_plots
        rrmse_rows.extend(sample_rows)

    metrics_path = Path(alignment_dir) / "per_layer_node_rrmse.csv"
    _write_alignment_rrmse_csv(metrics_path, rrmse_rows)
    summary_rows = []
    for sample_idx in range(int(num_samples)):
        for layer_idx in range(len(model.ssm_layers)):
            for kind in ("state", "output"):
                values = [
                    row["rrmse"]
                    for row in rrmse_rows
                    if row["sample"] == sample_idx and row["layer"] == layer_idx and row["kind"] == kind
                ]
                if values:
                    summary_rows.append(
                        {
                            "sample": sample_idx,
                            "layer": layer_idx,
                            "kind": kind,
                            "max_rrmse": float(np.max(values)),
                            "mean_rrmse": float(np.mean(values)),
                        }
                    )
    summary_path = Path(alignment_dir) / "per_layer_rrmse_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    return {
        "plots": plots,
        "per_node_rrmse_csv": _path(metrics_path),
        "per_layer_rrmse_json": _path(summary_path),
    }


def run_workflow(
    params_path,
    ssm_param,
    sample_rate,
    out_dir,
    cache_dir="cache_dir",
    seed=0,
    batch_size=64,
    full_samples=5,
    accuracy_samples=100,
    ltspice_bin=DEFAULT_LTSPICE,
    run_ltspice_enabled=True,
    delete_raw_after_read=False,
    delete_log_after_read=False,
    layer_duration=0.02,
    layer_points=401,
    layer_stimulus="sine",
    layer_amplitude=0.1,
    samples=None,
    labels=None,
):
    out_dir = Path(out_dir)
    netlist_dir = out_dir / "netlists"
    netlist_dir.mkdir(parents=True, exist_ok=True)
    layer_cir, layer_json = export_netlist(
        params_path,
        ssm_param,
        sample_rate,
        netlist_dir / "ssm_layers.cir",
    )
    full_cir, full_json = export_full_model(
        params_path,
        ssm_param,
        sample_rate,
        netlist_dir / "full_model.cir",
    )
    layer_manifest = json.loads(Path(layer_json).read_text())

    max_samples = max(int(full_samples), int(accuracy_samples), 1)
    if samples is None:
        samples, labels = load_mnist_samples(max_samples, cache_dir, seed, batch_size)
    else:
        samples = np.asarray(samples, dtype=np.float64)[:max_samples]
        labels = np.asarray(labels, dtype=np.int64)[: samples.shape[0]]

    layer_summary = run_layer_sanity(
        params_path=params_path,
        layer_cir=layer_cir,
        layer_manifest=layer_manifest,
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=out_dir / "layer_sanity",
        ltspice_bin=ltspice_bin,
        run_ltspice_enabled=run_ltspice_enabled,
        duration=layer_duration,
        points=layer_points,
        stimulus=layer_stimulus,
        amplitude=layer_amplitude,
    )

    alignment_dir = out_dir / "full_alignment"
    alignment_summary = generate_digital_alignment_artifacts(
        params_path=params_path,
        cir_path=full_cir,
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=alignment_dir,
        num_samples=full_samples,
        cache_dir=cache_dir,
        seed=seed,
        batch_size=batch_size,
        samples=samples[:full_samples],
        labels=labels[:full_samples],
    )
    if run_ltspice_enabled:
        for sample_idx in range(int(full_samples)):
            deck_path = alignment_dir / f"sample_{sample_idx:04d}" / f"sample_{sample_idx:04d}.cir"
            _maybe_run_ltspice(deck_path, ltspice_bin, True)
        alignment_summary = generate_digital_alignment_artifacts(
            params_path=params_path,
            cir_path=full_cir,
            ssm_param=ssm_param,
            sample_rate=sample_rate,
            out_dir=alignment_dir,
            num_samples=full_samples,
            cache_dir=cache_dir,
            seed=seed,
            batch_size=batch_size,
            samples=samples[:full_samples],
            labels=labels[:full_samples],
        )

    params = load_flax_params(params_path)
    model = extract_full_model(params, ssm_param, sample_rate)
    alignment_plots = plot_full_alignment_samples(alignment_dir, model, full_samples)

    accuracy_summary = validate_ltspice_accuracy(
        params_path=params_path,
        cir_path=full_cir,
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=out_dir / "accuracy",
        num_samples=accuracy_samples,
        cache_dir=cache_dir,
        seed=seed,
        batch_size=batch_size,
        ltspice_bin=ltspice_bin,
        delete_raw_after_read=delete_raw_after_read,
        delete_log_after_read=delete_log_after_read,
        run_sim=run_ltspice_enabled,
        samples=samples[:accuracy_samples],
        labels=labels[:accuracy_samples],
    )

    summary = {
        "params": _path(params_path),
        "ssm_param": ssm_param,
        "sample_rate": float(sample_rate),
        "netlists": {
            "ssm_layers_cir": _path(layer_cir),
            "ssm_layers_components": _path(layer_json),
            "full_model_cir": _path(full_cir),
            "full_model_components": _path(full_json),
        },
        "layer_sanity": layer_summary,
        "full_alignment": alignment_summary,
        "full_alignment_plots": alignment_plots,
        "accuracy": accuracy_summary,
    }
    summary_path = out_dir / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--out-dir", default="out/spice_workflow")
    parser.add_argument("--cache-dir", default="cache_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-samples", type=int, default=5)
    parser.add_argument("--accuracy-samples", type=int, default=100)
    parser.add_argument("--ltspice-bin", default=DEFAULT_LTSPICE)
    parser.add_argument("--no-run-ltspice", action="store_true")
    parser.add_argument("--delete-raw-after-read", action="store_true")
    parser.add_argument("--delete-log-after-read", action="store_true")
    parser.add_argument("--layer-duration", type=float, default=0.02)
    parser.add_argument("--layer-points", type=int, default=401)
    parser.add_argument("--layer-stimulus", choices=["sine", "step", "impulse"], default="sine")
    parser.add_argument("--layer-amplitude", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_workflow(
        params_path=args.params,
        ssm_param=args.ssm_param,
        sample_rate=args.sample_rate,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        full_samples=args.full_samples,
        accuracy_samples=args.accuracy_samples,
        ltspice_bin=args.ltspice_bin,
        run_ltspice_enabled=not args.no_run_ltspice,
        delete_raw_after_read=args.delete_raw_after_read,
        delete_log_after_read=args.delete_log_after_read,
        layer_duration=args.layer_duration,
        layer_points=args.layer_points,
        layer_stimulus=args.layer_stimulus,
        layer_amplitude=args.layer_amplitude,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
