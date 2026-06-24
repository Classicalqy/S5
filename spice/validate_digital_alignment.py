"""Compare MNIST digital recurrence, continuous reference, and LTSpice traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from s5.dataloading import Datasets
from s5.ssm_parameterizations import discretize_2x2_blocks

from .compare_transient import canonical_column, read_trace_table
from .export_full_model import extract_full_model
from .export_netlist import _output_node, _state_node, format_spice_value, load_flax_params, SUPPORTED_SSM_PARAMS
from .trace_utils import full_model_nodes, linear_nodes, write_trace_csv, zoh_pwl_source_line, zoh_sample_times
from .validate_transient import layer_input_matrix, layer_state_matrix, strip_final_end


def layer_discrete_matrices(layer):
    B_blocks = layer.B.reshape((layer.n_blocks, 2, layer.input_dim))
    A_bar, B_bar = discretize_2x2_blocks(
        layer.q * layer.alpha,
        layer.q * layer.omega,
        B_blocks,
        layer.delta,
        "zoh",
    )
    return np.asarray(A_bar, dtype=np.float64), np.asarray(B_bar, dtype=np.float64)


def simulate_layer_digital(layer, inputs):
    A_bar, B_bar = layer_discrete_matrices(layer)
    state = np.zeros((layer.n_blocks, 2), dtype=np.float64)
    states = np.zeros((inputs.shape[0], 2 * layer.n_blocks), dtype=np.float64)
    outputs = np.zeros((inputs.shape[0], layer.output_dim), dtype=np.float64)
    for idx, current in enumerate(inputs):
        state = np.einsum("kij,kj->ki", A_bar, state) + np.einsum("kih,h->ki", B_bar, current)
        flat = state.reshape(-1)
        states[idx] = flat
        outputs[idx] = layer.C @ flat
    return states, outputs


def simulate_full_digital(model, inputs, sample_rate):
    traces = {"time": zoh_sample_times(inputs.shape[0], sample_rate), "IN0": inputs[:, 0]}
    current = inputs @ model.encoder_kernel + model.encoder_bias
    for idx, node in enumerate(linear_nodes("ENC", current.shape[1])):
        traces[node] = current[:, idx]

    for layer_idx, layer in enumerate(model.ssm_layers):
        states, outputs = simulate_layer_digital(layer, current)
        for state_idx in range(states.shape[1]):
            traces[_state_node(layer_idx, state_idx // 2, state_idx % 2)] = states[:, state_idx]
        for out_idx, node in enumerate(linear_nodes(f"L{layer_idx}_out", layer.output_dim)):
            traces[node] = outputs[:, out_idx]
        current = np.maximum(outputs, 0.0)
        for out_idx, node in enumerate(linear_nodes(f"RELU{layer_idx}_", current.shape[1])):
            traces[node] = current[:, out_idx]

    logits = current @ model.decoder_kernel + model.decoder_bias
    for idx, node in enumerate(linear_nodes("LOGIT", logits.shape[1])):
        traces[node] = logits[:, idx]
    return traces


def simulate_full_continuous_zoh(model, inputs, sample_rate):
    sample_rate = float(sample_rate)
    dt = 1.0 / sample_rate
    times = zoh_sample_times(inputs.shape[0], sample_rate)
    state_sizes = [2 * layer.n_blocks for layer in model.ssm_layers]
    offsets = np.cumsum([0] + state_sizes)
    As = [layer_state_matrix(layer) for layer in model.ssm_layers]
    Bs = [layer_input_matrix(layer) for layer in model.ssm_layers]

    def split_state(state, idx):
        return state[offsets[idx]:offsets[idx + 1]]

    def rhs(t, state):
        input_idx = min(int(np.floor(max(float(t), 0.0) / dt)), inputs.shape[0] - 1)
        current = inputs[input_idx] @ model.encoder_kernel + model.encoder_bias
        derivs = []
        for idx, layer in enumerate(model.ssm_layers):
            x = split_state(state, idx)
            derivs.append(As[idx] @ x + Bs[idx] @ current)
            current = np.maximum(layer.C @ x, 0.0)
        return np.concatenate(derivs)

    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.zeros((sum(state_sizes),), dtype=np.float64),
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
        max_step=dt / 10.0,
    )
    if not solution.success:
        raise RuntimeError(f"Continuous reference integration failed: {solution.message}")
    states = solution.y.T

    traces = {"time": times, "IN0": inputs[:, 0]}
    encoder = inputs @ model.encoder_kernel + model.encoder_bias
    for idx, node in enumerate(linear_nodes("ENC", encoder.shape[1])):
        traces[node] = encoder[:, idx]
    for layer_idx, layer in enumerate(model.ssm_layers):
        layer_states = states[:, offsets[layer_idx]:offsets[layer_idx + 1]]
        for state_idx in range(layer_states.shape[1]):
            traces[_state_node(layer_idx, state_idx // 2, state_idx % 2)] = layer_states[:, state_idx]
        outputs = layer_states @ layer.C.T
        for out_idx, node in enumerate(linear_nodes(f"L{layer_idx}_out", layer.output_dim)):
            traces[node] = outputs[:, out_idx]
        current = np.maximum(outputs, 0.0)
        for out_idx, node in enumerate(linear_nodes(f"RELU{layer_idx}_", current.shape[1])):
            traces[node] = current[:, out_idx]
    logits = current @ model.decoder_kernel + model.decoder_bias
    for idx, node in enumerate(linear_nodes("LOGIT", logits.shape[1])):
        traces[node] = logits[:, idx]
    return traces


def alignment_nodes(model):
    nodes = []
    for layer_idx, layer in enumerate(model.ssm_layers):
        nodes.extend(_output_node(layer_idx, idx) for idx in range(layer.output_dim))
    nodes.extend(linear_nodes("LOGIT", model.decoder_bias.shape[0]))
    return nodes


def logits_from_trace(trace, model):
    return np.stack([trace[node] for node in linear_nodes("LOGIT", model.decoder_bias.shape[0])], axis=-1)


def trace_error(reference, candidate, nodes):
    diffs = []
    for node in nodes:
        diffs.append(np.asarray(candidate[node]) - np.asarray(reference[node]))
    diff = np.concatenate([d.reshape(-1) for d in diffs])
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def read_ltspice_trace(raw_path, times, nodes):
    table = read_trace_table(raw_path)
    if "time" not in table:
        raise ValueError(f"{raw_path} does not contain a time column.")
    traces = {"time": times}
    missing = []
    for node in nodes:
        key = canonical_column(node)
        if key not in table:
            missing.append(node)
            continue
        traces[node] = np.interp(times, table["time"], table[key])
    if missing:
        raise ValueError(f"{raw_path} is missing nodes: {', '.join(missing)}")
    return traces


def write_sample_deck(base_cir_path, out_path, inputs, model, sample_rate):
    body = strip_final_end(Path(base_cir_path).read_text())
    dt = 1.0 / float(sample_rate)
    duration = inputs.shape[0] * dt
    save_nodes = full_model_nodes(model)
    lines = [body, "", "* MNIST digital-alignment ZOH stimulus"]
    lines.append(zoh_pwl_source_line("VSTIM_IN0", "IN0", dt, inputs[:, 0]))
    for layer_idx, layer in enumerate(model.ssm_layers):
        for block_idx in range(layer.n_blocks):
            for state_idx in range(2):
                lines.append(f".ic V({_state_node(layer_idx, block_idx, state_idx)})=0")
    lines.append(".options plotwinsize=0")
    lines.append(".save " + " ".join(f"V({node})" for node in save_nodes))
    lines.append(f".tran 0 {format_spice_value(duration)} 0 {format_spice_value(dt / 10.0)} uic")
    lines.append(".end")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def load_mnist_samples(num_samples, cache_dir, seed, batch_size):
    _, _, testloader, _, _, _, _, _ = Datasets["mnist-classification"](
        cache_dir=cache_dir,
        seed=seed,
        bsz=max(1, min(batch_size, num_samples)),
    )
    xs, ys = [], []
    for batch in testloader:
        batch_x, batch_y = batch[:2]
        xs.append(batch_x.detach().cpu().numpy())
        ys.append(batch_y.detach().cpu().numpy())
        if sum(x.shape[0] for x in xs) >= num_samples:
            break
    inputs = np.concatenate(xs, axis=0)[:num_samples].astype(np.float64)
    labels = np.concatenate(ys, axis=0)[:num_samples].astype(np.int64)
    if inputs.shape[1:] != (784, 1):
        raise ValueError(f"Expected MNIST inputs with shape (N, 784, 1), got {inputs.shape}.")
    return inputs, labels


def _accuracy(preds, labels):
    return float(np.mean(np.asarray(preds) == np.asarray(labels))) if len(preds) else None


def _disagreement(a, b):
    return float(np.mean(np.asarray(a) != np.asarray(b))) if len(a) and len(b) else None


def write_per_sample_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "label",
        "digital_pred",
        "continuous_pred",
        "ltspice_pred",
        "continuous_final_logit_max_abs",
        "continuous_trace_max_abs",
        "ltspice_final_logit_max_abs",
        "ltspice_trace_max_abs",
        "ltspice_vs_continuous_final_logit_max_abs",
        "ltspice_vs_continuous_trace_max_abs",
        "ltspice_status",
        "deck",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def build_summary(rows):
    labels = [row["label"] for row in rows]
    digital = [row["digital_pred"] for row in rows]
    continuous = [row["continuous_pred"] for row in rows]
    lt_rows = [row for row in rows if row.get("ltspice_pred") is not None]
    ltspice = [row["ltspice_pred"] for row in lt_rows]
    lt_labels = [row["label"] for row in lt_rows]
    return {
        "num_samples": len(rows),
        "ltspice_samples": len(lt_rows),
        "ltspice_status": "complete" if len(lt_rows) == len(rows) else "pending",
        "digital_accuracy": _accuracy(digital, labels),
        "continuous_accuracy": _accuracy(continuous, labels),
        "ltspice_accuracy": _accuracy(ltspice, lt_labels),
        "digital_vs_continuous_disagreement_rate": _disagreement(digital, continuous),
        "digital_vs_ltspice_disagreement_rate": _disagreement(
            [row["digital_pred"] for row in lt_rows],
            ltspice,
        ),
        "continuous_final_logit_max_abs": max(row["continuous_final_logit_max_abs"] for row in rows),
        "continuous_trace_max_abs": max(row["continuous_trace_max_abs"] for row in rows),
        "ltspice_final_logit_max_abs": (
            max(row["ltspice_final_logit_max_abs"] for row in lt_rows) if lt_rows else None
        ),
        "ltspice_trace_max_abs": max(row["ltspice_trace_max_abs"] for row in lt_rows) if lt_rows else None,
        "ltspice_vs_continuous_final_logit_max_abs": (
            max(row["ltspice_vs_continuous_final_logit_max_abs"] for row in lt_rows) if lt_rows else None
        ),
        "ltspice_vs_continuous_trace_max_abs": (
            max(row["ltspice_vs_continuous_trace_max_abs"] for row in lt_rows) if lt_rows else None
        ),
    }


def generate_digital_alignment_artifacts(
    params_path,
    cir_path,
    ssm_param,
    sample_rate,
    out_dir,
    num_samples=16,
    cache_dir="cache_dir",
    seed=0,
    batch_size=64,
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = alignment_nodes(model)
    logit_nodes = linear_nodes("LOGIT", model.decoder_bias.shape[0])

    rows = []
    for sample_idx, (inputs, label) in enumerate(zip(samples, labels)):
        sample_dir = out_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        deck_path = write_sample_deck(cir_path, sample_dir / f"sample_{sample_idx:04d}.cir", inputs, model, sample_rate)
        digital = simulate_full_digital(model, inputs, sample_rate)
        continuous = simulate_full_continuous_zoh(model, inputs, sample_rate)
        write_trace_csv(sample_dir / "digital_reference.csv", digital)
        write_trace_csv(sample_dir / "continuous_reference.csv", continuous)

        digital_logits = logits_from_trace(digital, model)
        continuous_logits = logits_from_trace(continuous, model)
        continuous_final = np.abs(continuous_logits[-1] - digital_logits[-1])
        continuous_trace_error = trace_error(digital, continuous, nodes)
        row = {
            "sample": sample_idx,
            "label": int(label),
            "digital_pred": int(np.argmax(digital_logits[-1])),
            "continuous_pred": int(np.argmax(continuous_logits[-1])),
            "continuous_final_logit_max_abs": float(np.max(continuous_final)),
            "continuous_trace_max_abs": continuous_trace_error["max_abs"],
            "ltspice_pred": None,
            "ltspice_final_logit_max_abs": None,
            "ltspice_trace_max_abs": None,
            "ltspice_vs_continuous_final_logit_max_abs": None,
            "ltspice_vs_continuous_trace_max_abs": None,
            "ltspice_status": "pending",
            "deck": str(deck_path),
        }

        raw_path = deck_path.with_suffix(".raw")
        if raw_path.exists():
            ltspice = read_ltspice_trace(raw_path, digital["time"], nodes + logit_nodes)
            ltspice_logits = logits_from_trace(ltspice, model)
            ltspice_continuous_final = np.abs(ltspice_logits[-1] - continuous_logits[-1])
            row.update(
                {
                    "ltspice_pred": int(np.argmax(ltspice_logits[-1])),
                    "ltspice_final_logit_max_abs": float(np.max(np.abs(ltspice_logits[-1] - digital_logits[-1]))),
                    "ltspice_trace_max_abs": trace_error(digital, ltspice, nodes)["max_abs"],
                    "ltspice_vs_continuous_final_logit_max_abs": float(np.max(ltspice_continuous_final)),
                    "ltspice_vs_continuous_trace_max_abs": trace_error(continuous, ltspice, nodes)["max_abs"],
                    "ltspice_status": "complete",
                }
            )
        rows.append(row)

    per_sample_path = write_per_sample_csv(out_dir / "per_sample.csv", rows)
    summary = build_summary(rows)
    summary.update(
        {
            "params": str(params_path),
            "base_cir": str(cir_path),
            "sample_rate": float(sample_rate),
            "ssm_param": ssm_param,
            "per_sample_csv": str(per_sample_path),
            "alignment_nodes": nodes,
        }
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--cir", required=True)
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--out-dir", default="out/validation_digital")
    parser.add_argument("--cache-dir", default="cache_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = generate_digital_alignment_artifacts(
        args.params,
        args.cir,
        args.ssm_param,
        args.sample_rate,
        args.out_dir,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
