"""Unified LTSpice export and validation workflow for the restricted S5 MNIST model."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from itertools import product
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
from .hardware_projection import HardwareProjectionConfig
from .metrics import rrmse, trace_metrics
from .plots import plot_logit_bar, plot_multi_logit_bar, plot_multi_trace_overlay, plot_trace_overlay
from .projected_params import save_projected_params
from .trace_utils import linear_nodes
from .validate_digital_alignment import (
    alignment_nodes,
    generate_digital_alignment_artifacts,
    load_mnist_samples,
    logits_from_trace,
    read_ltspice_trace,
)
from .validate_ltspice_accuracy import DEFAULT_LTSPICE, run_ltspice, validate_ltspice_accuracy
from .validate_transient import generate_validation_artifacts


def _path(value):
    return str(Path(value))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _parse_sweep_values(values, cast):
    if values is None:
        return []
    parsed = []
    for value in values:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        for part in text.split(","):
            part = part.strip()
            if part:
                parsed.append(cast(part))
    return parsed


def _case_name(quant_bits, variation_sigma):
    sigma = "{:.6g}".format(float(variation_sigma)).replace("-", "m").replace(".", "p")
    return f"q{int(quant_bits)}_v{sigma}"


def _projection_config(quant_bits, variation_sigma, g_min, g_max, c_min, c_max, variation_seed):
    return HardwareProjectionConfig(
        hardware_projection="conductance",
        projection_scope="block",
        g_min=g_min,
        g_max=g_max,
        c_min=c_min,
        c_max=c_max,
        quant_bits=int(quant_bits),
        quant_mode="linear",
        variation_sigma=float(variation_sigma),
        variation_seed=int(variation_seed),
    )


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


def _trace_file_ready(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def _maybe_run_ltspice(deck_path, ltspice_bin, run_ltspice_enabled):
    raw_path = Path(deck_path).with_suffix(".raw")
    if run_ltspice_enabled and not _trace_file_ready(raw_path):
        raw_path.unlink(missing_ok=True)
        result = run_ltspice(ltspice_bin, deck_path)
        if result.returncode != 0:
            return raw_path, f"ltspice_failed:returncode={result.returncode}"
    return raw_path, "complete" if _trace_file_ready(raw_path) else "pending"


def _export_model_netlists(params_path, ssm_param, sample_rate, out_dir, projection_config=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_cir, layer_json = export_netlist(
        params_path,
        ssm_param,
        sample_rate,
        out_dir / "ssm_layers.cir",
        json_out=out_dir / "ssm_layers_manifest.json",
        projection_config=projection_config,
        projection_report=out_dir / "projection_report.json" if projection_config else None,
    )
    full_cir, full_json = export_full_model(
        params_path,
        ssm_param,
        sample_rate,
        out_dir / "full_model.cir",
        json_out=out_dir / "full_model_manifest.json",
        projection_config=projection_config,
    )
    return {
        "params": _path(params_path),
        "ssm_layers_cir": _path(layer_cir),
        "ssm_layers_manifest": _path(layer_json),
        "full_model_cir": _path(full_cir),
        "full_model_manifest": _path(full_json),
        "projection_report": _path(out_dir / "projection_report.json") if projection_config else None,
    }


def _layer_nodes(layer_idx, layer_manifest, role):
    if role == "state":
        state_width = int(layer_manifest.get("state_width", 2))
        return [
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(layer_manifest["n_blocks"])
            for state_idx in range(state_width)
        ]
    if role == "output":
        return [_output_node(layer_idx, idx) for idx in range(layer_manifest["output_dim"])]
    raise ValueError("role must be state or output.")


def _model_layer_nodes(model, layer_idx, role):
    layer = model.ssm_layers[layer_idx]
    if role == "state":
        return layer.state_nodes(layer_idx)
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


def run_layer_sanity_pair(
    exact_params,
    exact_cir,
    exact_manifest,
    projected_params,
    projected_cir,
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
    for layer_record in exact_manifest["layers"]:
        layer_idx = int(layer_record["index"])
        layer_dir = out_dir / f"layer_{layer_idx:02d}"
        exact_meta = generate_validation_artifacts(
            exact_params,
            exact_cir,
            ssm_param,
            sample_rate,
            layer_dir / "exact",
            layer_index=layer_idx,
            duration=duration,
            points=points,
            stimulus=stimulus,
            amplitude=amplitude,
        )
        projected_meta = generate_validation_artifacts(
            projected_params,
            projected_cir,
            ssm_param,
            sample_rate,
            layer_dir / "projected",
            layer_index=layer_idx,
            duration=duration,
            points=points,
            stimulus=stimulus,
            amplitude=amplitude,
        )
        exact_raw, exact_status = _maybe_run_ltspice(exact_meta["validation_cir"], ltspice_bin, run_ltspice_enabled)
        projected_raw, projected_status = _maybe_run_ltspice(projected_meta["validation_cir"], ltspice_bin, run_ltspice_enabled)
        state_nodes = _layer_nodes(layer_idx, layer_record, "state")[:2]
        traces = {}
        exact_ref = _reference_trace(exact_meta["reference_csv"], state_nodes)
        projected_ref = _reference_trace(projected_meta["reference_csv"], state_nodes)
        traces["exact_continuous"] = exact_ref
        traces["projected_continuous"] = projected_ref
        if _trace_file_ready(exact_raw):
            traces["exact_ltspice"] = _table_trace(read_trace_table(exact_raw), exact_ref["time"], state_nodes)
        if _trace_file_ready(projected_raw):
            traces["projected_ltspice"] = _table_trace(read_trace_table(projected_raw), projected_ref["time"], state_nodes)
        plot_path = layer_dir / "state_compare.png"
        plot_multi_trace_overlay(plot_path, exact_ref["time"], traces, state_nodes)
        rows.append(
            {
                "layer": layer_idx,
                "nodes": state_nodes,
                "plot": _path(plot_path),
                "exact_status": exact_status,
                "projected_status": projected_status,
                "exact_validation_cir": exact_meta["validation_cir"],
                "projected_validation_cir": projected_meta["validation_cir"],
                "exact_raw": _path(exact_raw),
                "projected_raw": _path(projected_raw),
            }
        )
    summary = {"layers": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _plot_full_alignment_sample(alignment_dir, model, sample_idx=0):
    sample_dir = Path(alignment_dir) / f"sample_{sample_idx:04d}"
    digital_path = sample_dir / "digital_reference.csv"
    raw_path = sample_dir / f"sample_{sample_idx:04d}.raw"
    if not digital_path.exists() or not _trace_file_ready(raw_path):
        return {}, []

    state_nodes = []
    output_nodes = []
    for layer_idx, layer in enumerate(model.ssm_layers):
        state_nodes.extend(_model_layer_nodes(model, layer_idx, "state"))
        output_nodes.extend(_model_layer_nodes(model, layer_idx, "output"))
    logit_nodes = linear_nodes("LOGIT", model.decoder_bias.shape[0])
    digital = _reference_trace(digital_path, state_nodes + output_nodes + logit_nodes)
    try:
        raw = read_trace_table(raw_path)
        ltspice = _table_trace(raw, digital["time"], state_nodes + output_nodes + logit_nodes)
    except Exception:
        return {}, []

    logit_plot = sample_dir / "final_logits.png"
    plots = {}
    rrmse_rows = []
    first_block_nodes = []
    worst_nodes = []
    layer_summaries = {}
    for layer_idx, _layer in enumerate(model.ssm_layers):
        layer_state_nodes = _model_layer_nodes(model, layer_idx, "state")
        layer_output_nodes = _model_layer_nodes(model, layer_idx, "output")
        state_rows = _node_rrmse_rows(sample_idx, layer_idx, digital, ltspice, layer_state_nodes, "state")
        output_rows = _node_rrmse_rows(sample_idx, layer_idx, digital, ltspice, layer_output_nodes, "output")
        rrmse_rows.extend(state_rows)
        rrmse_rows.extend(output_rows)

        layer_worst_nodes = [
            row["node"]
            for row in sorted(state_rows, key=lambda row: row["rrmse"], reverse=True)[:1]
        ]
        layer_first_block_nodes = layer_state_nodes[: _layer.state_width]
        worst_nodes.extend(layer_worst_nodes)
        first_block_nodes.extend(layer_first_block_nodes)
        layer_summaries[f"layer_{layer_idx:02d}"] = {
            "worst_state_nodes": layer_worst_nodes,
            "first_block_state_nodes": layer_first_block_nodes,
        }

    if worst_nodes:
        worst_path = sample_dir / "worst_states.png"
        plot_trace_overlay(
            worst_path,
            digital["time"],
            digital,
            ltspice,
            worst_nodes,
            reference_label="digital",
            candidate_label="ltspice",
        )
        plots["worst_state_plot"] = _path(worst_path)
    if first_block_nodes:
        first_path = sample_dir / "first_block_states.png"
        plot_trace_overlay(
            first_path,
            digital["time"],
            digital,
            ltspice,
            first_block_nodes,
            reference_label="digital",
            candidate_label="ltspice",
        )
        plots["first_block_state_plot"] = _path(first_path)
    plots["worst_state_nodes"] = worst_nodes
    plots["first_block_state_nodes"] = first_block_nodes
    plots["layers"] = layer_summaries

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


def run_full_alignment_pair(
    exact_params,
    exact_cir,
    projected_params,
    projected_cir,
    ssm_param,
    sample_rate,
    out_dir,
    num_samples,
    cache_dir,
    seed,
    batch_size,
    samples,
    labels,
    ltspice_bin,
    run_ltspice_enabled,
):
    out_dir = Path(out_dir)
    exact_dir = out_dir / "exact"
    projected_dir = out_dir / "projected"
    exact_summary = generate_digital_alignment_artifacts(
        exact_params,
        exact_cir,
        ssm_param,
        sample_rate,
        exact_dir,
        num_samples=num_samples,
        cache_dir=cache_dir,
        seed=seed,
        batch_size=batch_size,
        samples=samples[:num_samples],
        labels=labels[:num_samples],
    )
    projected_summary = generate_digital_alignment_artifacts(
        projected_params,
        projected_cir,
        ssm_param,
        sample_rate,
        projected_dir,
        num_samples=num_samples,
        cache_dir=cache_dir,
        seed=seed,
        batch_size=batch_size,
        samples=samples[:num_samples],
        labels=labels[:num_samples],
    )
    if run_ltspice_enabled:
        for sample_idx in range(int(num_samples)):
            _maybe_run_ltspice(exact_dir / f"sample_{sample_idx:04d}" / f"sample_{sample_idx:04d}.cir", ltspice_bin, True)
            _maybe_run_ltspice(projected_dir / f"sample_{sample_idx:04d}" / f"sample_{sample_idx:04d}.cir", ltspice_bin, True)
        exact_summary = generate_digital_alignment_artifacts(
            exact_params,
            exact_cir,
            ssm_param,
            sample_rate,
            exact_dir,
            num_samples=num_samples,
            cache_dir=cache_dir,
            seed=seed,
            batch_size=batch_size,
            samples=samples[:num_samples],
            labels=labels[:num_samples],
        )
        projected_summary = generate_digital_alignment_artifacts(
            projected_params,
            projected_cir,
            ssm_param,
            sample_rate,
            projected_dir,
            num_samples=num_samples,
            cache_dir=cache_dir,
            seed=seed,
            batch_size=batch_size,
            samples=samples[:num_samples],
            labels=labels[:num_samples],
        )

    exact_model = extract_full_model(load_flax_params(exact_params), ssm_param, sample_rate)
    projected_model = extract_full_model(load_flax_params(projected_params), ssm_param, sample_rate)
    plots = {}
    rows = []
    state_nodes = []
    for layer_idx, layer in enumerate(exact_model.ssm_layers):
        state_nodes.extend(layer.state_nodes(layer_idx))
    logit_nodes = linear_nodes("LOGIT", exact_model.decoder_bias.shape[0])
    for sample_idx in range(int(num_samples)):
        sample_key = f"sample_{sample_idx:04d}"
        sample_dir = out_dir / sample_key
        exact_sample = exact_dir / sample_key
        projected_sample = projected_dir / sample_key
        exact_digital = _reference_trace(exact_sample / "digital_reference.csv", state_nodes + logit_nodes)
        projected_digital = _reference_trace(projected_sample / "digital_reference.csv", state_nodes + logit_nodes)
        trace_map = {
            "exact_digital": exact_digital,
            "projected_digital": projected_digital,
        }
        exact_raw = exact_sample / f"{sample_key}.raw"
        projected_raw = projected_sample / f"{sample_key}.raw"
        if _trace_file_ready(exact_raw):
            trace_map["exact_ltspice"] = read_ltspice_trace(exact_raw, exact_digital["time"], state_nodes + logit_nodes)
        if _trace_file_ready(projected_raw):
            trace_map["projected_ltspice"] = read_ltspice_trace(projected_raw, projected_digital["time"], state_nodes + logit_nodes)
        state_error = [
            (node, rrmse(exact_digital[node], projected_digital[node]))
            for node in state_nodes
        ]
        chosen_nodes = state_nodes[:2] + [node for node, _ in sorted(state_error, key=lambda item: item[1], reverse=True)[:2]]
        chosen_nodes = list(dict.fromkeys(chosen_nodes))
        states_path = sample_dir / "state_compare.png"
        plot_multi_trace_overlay(states_path, exact_digital["time"], trace_map, chosen_nodes)
        logits_path = sample_dir / "logits_compare.png"
        logits_by_label = {
            label: [trace[node][-1] for node in logit_nodes]
            for label, trace in trace_map.items()
        }
        plot_multi_logit_bar(logits_path, logits_by_label, title=f"Sample {sample_idx} final logits")
        rows.append(
            {
                "sample": sample_idx,
                "label": int(labels[sample_idx]),
                "exact_digital_pred": int(np.argmax([exact_digital[node][-1] for node in logit_nodes])),
                "projected_digital_pred": int(np.argmax([projected_digital[node][-1] for node in logit_nodes])),
                "state_nodes": chosen_nodes,
                "state_plot": _path(states_path),
                "logit_plot": _path(logits_path),
            }
        )
        plots[sample_key] = {"state_plot": _path(states_path), "logit_plot": _path(logits_path), "state_nodes": chosen_nodes}
    combined = {
        "exact": exact_summary,
        "projected": projected_summary,
        "rows": rows,
        "plots": plots,
        "exact_digital_accuracy": float(np.mean([r["exact_digital_pred"] == r["label"] for r in rows])) if rows else None,
        "projected_digital_accuracy": float(np.mean([r["projected_digital_pred"] == r["label"] for r in rows])) if rows else None,
        "digital_prediction_disagreement_rate": float(np.mean([r["exact_digital_pred"] != r["projected_digital_pred"] for r in rows])) if rows else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True))
    return combined


def run_accuracy_pair(
    exact_params,
    exact_cir,
    projected_params,
    projected_cir,
    ssm_param,
    sample_rate,
    out_dir,
    accuracy_samples,
    full_alignment_summary,
    full_samples,
    cache_dir,
    seed,
    batch_size,
    samples,
    labels,
    ltspice_bin,
    run_ltspice_enabled,
    delete_raw_after_read,
    delete_log_after_read,
):
    out_dir = Path(out_dir)
    if int(accuracy_samples) <= int(full_samples):
        summary = {
            "source": "full_alignment",
            "num_samples": int(accuracy_samples),
            "exact_digital_accuracy": full_alignment_summary.get("exact_digital_accuracy"),
            "projected_digital_accuracy": full_alignment_summary.get("projected_digital_accuracy"),
            "digital_prediction_disagreement_rate": full_alignment_summary.get("digital_prediction_disagreement_rate"),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    exact = validate_ltspice_accuracy(
        exact_params,
        exact_cir,
        ssm_param,
        sample_rate,
        out_dir / "exact",
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
    projected = validate_ltspice_accuracy(
        projected_params,
        projected_cir,
        ssm_param,
        sample_rate,
        out_dir / "projected",
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
    summary = {"source": "accuracy_runner", "exact": exact, "projected": projected}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


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
    hardware_projection=False,
    quant_bits=(0,),
    variation_sigma=(0.0,),
    variation_seed=0,
    g_min=1e-6,
    g_max=150e-6,
    c_min=1e-12,
    c_max=1e-6,
):
    quant_bits = tuple(quant_bits) if quant_bits else (0,)
    variation_sigma = tuple(variation_sigma) if variation_sigma else (0.0,)
    out_dir = Path(out_dir)
    netlist_dir = out_dir / "netlist"
    original_netlist_dir = netlist_dir / "original"
    original_netlist_dir.mkdir(parents=True, exist_ok=True)
    original_params = original_netlist_dir / "params.msgpack"
    shutil.copyfile(params_path, original_params)
    original_netlists = _export_model_netlists(
        params_path,
        ssm_param,
        sample_rate,
        original_netlist_dir,
    )
    layer_manifest = json.loads(Path(original_netlists["ssm_layers_manifest"]).read_text())

    max_samples = max(int(full_samples), int(accuracy_samples), 1)
    if samples is None:
        samples, labels = load_mnist_samples(max_samples, cache_dir, seed, batch_size)
    else:
        samples = np.asarray(samples, dtype=np.float64)[:max_samples]
        labels = np.asarray(labels, dtype=np.int64)[: samples.shape[0]]

    layer_summary = run_layer_sanity(
        params_path=params_path,
        layer_cir=original_netlists["ssm_layers_cir"],
        layer_manifest=layer_manifest,
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=out_dir / "layer_sanity" / "original",
        ltspice_bin=ltspice_bin,
        run_ltspice_enabled=run_ltspice_enabled,
        duration=layer_duration,
        points=layer_points,
        stimulus=layer_stimulus,
        amplitude=layer_amplitude,
    )

    original_alignment_dir = out_dir / "full_alignment" / "original"
    alignment_summary = generate_digital_alignment_artifacts(
        params_path=params_path,
        cir_path=original_netlists["full_model_cir"],
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=original_alignment_dir,
        num_samples=full_samples,
        cache_dir=cache_dir,
        seed=seed,
        batch_size=batch_size,
        samples=samples[:full_samples],
        labels=labels[:full_samples],
    )
    if run_ltspice_enabled:
        for sample_idx in range(int(full_samples)):
            deck_path = original_alignment_dir / f"sample_{sample_idx:04d}" / f"sample_{sample_idx:04d}.cir"
            _maybe_run_ltspice(deck_path, ltspice_bin, True)
        alignment_summary = generate_digital_alignment_artifacts(
            params_path=params_path,
            cir_path=original_netlists["full_model_cir"],
            ssm_param=ssm_param,
            sample_rate=sample_rate,
            out_dir=original_alignment_dir,
            num_samples=full_samples,
            cache_dir=cache_dir,
            seed=seed,
            batch_size=batch_size,
            samples=samples[:full_samples],
            labels=labels[:full_samples],
        )

    params = load_flax_params(params_path)
    model = extract_full_model(params, ssm_param, sample_rate)
    alignment_plots = plot_full_alignment_samples(original_alignment_dir, model, full_samples)

    accuracy_summary = validate_ltspice_accuracy(
        params_path=params_path,
        cir_path=original_netlists["full_model_cir"],
        ssm_param=ssm_param,
        sample_rate=sample_rate,
        out_dir=out_dir / "accuracy" / "original",
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

    projection_runs = []
    if hardware_projection:
        for bits, sigma in product(quant_bits, variation_sigma):
            case = _case_name(bits, sigma)
            case_netlist_dir = netlist_dir / case
            config = _projection_config(bits, sigma, g_min, g_max, c_min, c_max, variation_seed)
            projected_params, projection_report = save_projected_params(
                params_path,
                ssm_param,
                sample_rate,
                config,
                case_netlist_dir / "params.msgpack",
            )
            projected_netlists = _export_model_netlists(
                params_path,
                ssm_param,
                sample_rate,
                case_netlist_dir,
                projection_config=config,
            )
            layer_pair = run_layer_sanity_pair(
                exact_params=params_path,
                exact_cir=original_netlists["ssm_layers_cir"],
                exact_manifest=layer_manifest,
                projected_params=projected_params,
                projected_cir=projected_netlists["ssm_layers_cir"],
                ssm_param=ssm_param,
                sample_rate=sample_rate,
                out_dir=out_dir / "layer_sanity" / case,
                ltspice_bin=ltspice_bin,
                run_ltspice_enabled=run_ltspice_enabled,
                duration=layer_duration,
                points=layer_points,
                stimulus=layer_stimulus,
                amplitude=layer_amplitude,
            )
            full_pair = run_full_alignment_pair(
                exact_params=params_path,
                exact_cir=original_netlists["full_model_cir"],
                projected_params=projected_params,
                projected_cir=projected_netlists["full_model_cir"],
                ssm_param=ssm_param,
                sample_rate=sample_rate,
                out_dir=out_dir / "full_alignment" / case,
                num_samples=full_samples,
                cache_dir=cache_dir,
                seed=seed,
                batch_size=batch_size,
                samples=samples,
                labels=labels,
                ltspice_bin=ltspice_bin,
                run_ltspice_enabled=run_ltspice_enabled,
            )
            accuracy_pair = run_accuracy_pair(
                exact_params=params_path,
                exact_cir=original_netlists["full_model_cir"],
                projected_params=projected_params,
                projected_cir=projected_netlists["full_model_cir"],
                ssm_param=ssm_param,
                sample_rate=sample_rate,
                out_dir=out_dir / "accuracy" / case,
                accuracy_samples=accuracy_samples,
                full_alignment_summary=full_pair,
                full_samples=full_samples,
                cache_dir=cache_dir,
                seed=seed,
                batch_size=batch_size,
                samples=samples,
                labels=labels,
                ltspice_bin=ltspice_bin,
                run_ltspice_enabled=run_ltspice_enabled,
                delete_raw_after_read=delete_raw_after_read,
                delete_log_after_read=delete_log_after_read,
            )
            projection_runs.append(
                {
                    "case": case,
                    "quant_bits": int(bits),
                    "variation_sigma": float(sigma),
                    "netlists": projected_netlists,
                    "projection": projection_report,
                    "layer_sanity": layer_pair,
                    "full_alignment": full_pair,
                    "accuracy": accuracy_pair,
                }
            )

    summary = {
        "params": _path(params_path),
        "ssm_param": ssm_param,
        "sample_rate": float(sample_rate),
        "hardware_projection": bool(hardware_projection),
        "netlists": {"original": original_netlists},
        "layer_sanity": layer_summary,
        "full_alignment": alignment_summary,
        "full_alignment_plots": alignment_plots,
        "accuracy": accuracy_summary,
        "projection_runs": projection_runs,
    }
    for run in projection_runs:
        summary["netlists"][run["case"]] = run["netlists"]
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
    parser.add_argument("--hardware-projection", "--hardware_projection", default=False)
    parser.add_argument("--quant-bits", "--quant_bits", nargs="*", default=["0"])
    parser.add_argument("--variation-sigma", "--variation_sigma", nargs="*", default=["0.0"])
    parser.add_argument("--variation-seed", "--variation_seed", type=int, default=0)
    parser.add_argument("--g-min", "--g_min", type=float, default=1e-6)
    parser.add_argument("--g-max", "--g_max", type=float, default=150e-6)
    parser.add_argument("--c-min", "--c_min", type=float, default=1e-12)
    parser.add_argument("--c-max", "--c_max", type=float, default=1e-6)
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
        hardware_projection=_as_bool(args.hardware_projection),
        quant_bits=_parse_sweep_values(args.quant_bits, int),
        variation_sigma=_parse_sweep_values(args.variation_sigma, float),
        variation_seed=args.variation_seed,
        g_min=args.g_min,
        g_max=args.g_max,
        c_min=args.c_min,
        c_max=args.c_max,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
