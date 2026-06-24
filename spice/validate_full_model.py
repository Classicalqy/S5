"""Generate full-model Python reference traces and LTSpice validation decks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .export_full_model import extract_full_model
from .export_netlist import (
    _output_node,
    _state_node,
    format_spice_value,
    load_flax_params,
    SUPPORTED_SSM_PARAMS,
)
from .validate_transient import (
    interpolate_inputs,
    layer_input_matrix,
    layer_state_matrix,
    make_stimulus,
    pwl_source_line,
    strip_final_end,
)


def _linear_nodes(prefix, count):
    return [f"{prefix}{idx}" for idx in range(count)]


def full_model_nodes(model):
    nodes = ["IN0"]
    nodes.extend(_linear_nodes("ENC", model.encoder_bias.shape[0]))
    for layer_idx, layer in enumerate(model.ssm_layers):
        nodes.extend(
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(layer.n_blocks)
            for state_idx in range(2)
        )
        nodes.extend(_output_node(layer_idx, idx) for idx in range(layer.output_dim))
        nodes.extend(_linear_nodes(f"RELU{layer_idx}_", layer.output_dim))
    nodes.extend(_linear_nodes("LOGIT", model.decoder_bias.shape[0]))
    return nodes


def simulate_full_reference(model, times, inputs):
    times = np.asarray(times, dtype=np.float64)
    state_sizes = [2 * layer.n_blocks for layer in model.ssm_layers]
    offsets = np.cumsum([0] + state_sizes)
    x0 = np.zeros((sum(state_sizes),), dtype=np.float64)
    As = [layer_state_matrix(layer) for layer in model.ssm_layers]
    Bs = [layer_input_matrix(layer) for layer in model.ssm_layers]

    def split_state(state, idx):
        return state[offsets[idx]:offsets[idx + 1]]

    def rhs(t, state):
        u = interpolate_inputs(times, inputs, t)
        current = u @ model.encoder_kernel + model.encoder_bias
        derivs = []
        for idx, layer in enumerate(model.ssm_layers):
            x = split_state(state, idx)
            derivs.append(As[idx] @ x + Bs[idx] @ current)
            current = np.maximum(layer.C @ x, 0.0)
        return np.concatenate(derivs)

    if times.shape[0] > 1:
        steps = np.diff(times)
        if np.any(steps <= 0):
            raise ValueError("times must be strictly increasing.")
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            x0,
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
            max_step=float(np.min(steps) / 10.0),
        )
        if not solution.success:
            raise RuntimeError(f"Full-model reference integration failed: {solution.message}")
        states = solution.y.T
    else:
        states = x0[None, :]

    traces = {"time": times, "IN0": inputs[:, 0]}
    encoder = inputs @ model.encoder_kernel + model.encoder_bias
    for idx, node in enumerate(_linear_nodes("ENC", encoder.shape[1])):
        traces[node] = encoder[:, idx]

    current = encoder
    for layer_idx, layer in enumerate(model.ssm_layers):
        layer_states = states[:, offsets[layer_idx]:offsets[layer_idx + 1]]
        for state_idx in range(layer_states.shape[1]):
            node = _state_node(layer_idx, state_idx // 2, state_idx % 2)
            traces[node] = layer_states[:, state_idx]
        y = layer_states @ layer.C.T
        for idx, node in enumerate(_linear_nodes(f"L{layer_idx}_out", layer.output_dim)):
            traces[node] = y[:, idx]
        current = np.maximum(y, 0.0)
        for idx, node in enumerate(_linear_nodes(f"RELU{layer_idx}_", current.shape[1])):
            traces[node] = current[:, idx]

    logits = current @ model.decoder_kernel + model.decoder_bias
    for idx, node in enumerate(_linear_nodes("LOGIT", logits.shape[1])):
        traces[node] = logits[:, idx]
    return traces


def write_reference_csv(path, traces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(traces.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_idx in range(len(traces["time"])):
            writer.writerow([format_spice_value(traces[key][row_idx]) for key in headers])
    return path


def write_validation_deck(base_cir_path, out_path, times, inputs, model):
    body = strip_final_end(Path(base_cir_path).read_text())
    duration = float(times[-1])
    output_step = duration / max(len(times) - 1, 1)
    max_step = output_step / 10.0
    save_nodes = full_model_nodes(model)

    lines = [body, "", "* Full-model transient validation stimulus"]
    lines.append(pwl_source_line("VSTIM_IN0", "IN0", times, inputs[:, 0]))
    for layer_idx, layer in enumerate(model.ssm_layers):
        for block_idx in range(layer.n_blocks):
            for state_idx in range(2):
                lines.append(f".ic V({_state_node(layer_idx, block_idx, state_idx)})=0")
    lines.append(".options plotwinsize=0")
    lines.append(".save " + " ".join(f"V({node})" for node in save_nodes))
    lines.append(f".tran 0 {format_spice_value(duration)} 0 {format_spice_value(max_step)} uic")
    lines.append(".end")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path, save_nodes


def generate_full_validation_artifacts(
    params_path,
    cir_path,
    ssm_param,
    sample_rate,
    out_dir,
    duration=0.02,
    points=401,
    stimulus="sine",
    amplitude=0.05,
):
    params = load_flax_params(params_path)
    model = extract_full_model(params, ssm_param, sample_rate)
    times = np.linspace(0.0, duration, points, dtype=np.float64)
    inputs = make_stimulus(times, 1, stimulus, amplitude)
    traces = simulate_full_reference(model, times, inputs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cir_path).stem
    reference_path = write_reference_csv(out_dir / f"{stem}_reference.csv", traces)
    deck_path, save_nodes = write_validation_deck(
        cir_path,
        out_dir / f"{stem}_validation.cir",
        times,
        inputs,
        model,
    )
    metadata = {
        "params": str(params_path),
        "base_cir": str(cir_path),
        "validation_cir": str(deck_path),
        "reference_csv": str(reference_path),
        "sample_rate": float(sample_rate),
        "ssm_param": ssm_param,
        "duration": float(duration),
        "points": int(points),
        "stimulus": stimulus,
        "amplitude": float(amplitude),
        "saved_nodes": save_nodes,
        "logit_nodes": _linear_nodes("LOGIT", model.decoder_bias.shape[0]),
        "ltspice_app": "/Applications/LTspice.app/Contents/SharedSupport/ltspice/LTspice/run_ltspice",
    }
    metadata_path = out_dir / f"{stem}_validation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--cir", required=True)
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--out-dir", default="out/validation_full")
    parser.add_argument("--duration", type=float, default=0.02)
    parser.add_argument("--points", type=int, default=401)
    parser.add_argument("--stimulus", choices=["sine", "step", "impulse"], default="sine")
    parser.add_argument("--amplitude", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = generate_full_validation_artifacts(
        args.params,
        args.cir,
        args.ssm_param,
        args.sample_rate,
        args.out_dir,
        args.duration,
        args.points,
        args.stimulus,
        args.amplitude,
    )
    print(f"Wrote full-model validation deck: {metadata['validation_cir']}")
    print(f"Wrote full-model Python reference: {metadata['reference_csv']}")
    print(f"Wrote full-model validation metadata: {Path(metadata['validation_cir']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
