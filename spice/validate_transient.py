"""Generate a Python reference trace and LTSpice transient validation deck."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .export_netlist import (
    _input_node,
    _output_node,
    _state_node,
    find_ssm_modules,
    format_spice_value,
    load_flax_params,
    module_to_layer,
    SUPPORTED_SSM_PARAMS,
)


def layer_state_matrix(layer):
    state_dim = layer.state_dim
    A = np.zeros((state_dim, state_dim), dtype=np.float64)
    for block_idx in range(layer.n_blocks):
        start = layer.state_width * block_idx
        stop = start + layer.state_width
        A[start:stop, start:stop] = layer.A_tr[block_idx]
    return A


def layer_input_matrix(layer):
    return layer.B_tr.reshape((layer.state_dim, layer.input_dim))


def make_stimulus(times, input_dim, kind="sine", amplitude=0.1):
    times = np.asarray(times, dtype=np.float64)
    inputs = np.zeros((times.shape[0], input_dim), dtype=np.float64)
    if kind == "step":
        for idx in range(input_dim):
            inputs[:, idx] = amplitude * (1.0 if idx % 2 == 0 else -1.0)
        inputs[0, :] = 0.0
        return inputs
    if kind == "impulse":
        if times.shape[0] > 1:
            inputs[1, :] = amplitude
        return inputs
    if kind != "sine":
        raise ValueError("stimulus kind must be one of: sine, step, impulse")

    duration = max(float(times[-1]), 1e-12)
    base_hz = 1.0 / duration
    for idx in range(input_dim):
        freq = base_hz * (idx + 1)
        phase = idx * np.pi / max(input_dim, 1)
        inputs[:, idx] = amplitude * np.sin(2.0 * np.pi * freq * times + phase)
    inputs[0, :] = 0.0
    return inputs


def interpolate_inputs(times, inputs, t):
    out = np.empty((inputs.shape[1],), dtype=np.float64)
    for idx in range(inputs.shape[1]):
        out[idx] = np.interp(t, times, inputs[:, idx])
    return out


def simulate_layer_reference(layer, times, inputs):
    times = np.asarray(times, dtype=np.float64)
    A = layer_state_matrix(layer)
    B = layer_input_matrix(layer)
    x = np.zeros((A.shape[0],), dtype=np.float64)
    states = np.zeros((times.shape[0], A.shape[0]), dtype=np.float64)

    def rhs(t, state):
        return A @ state + B @ interpolate_inputs(times, inputs, t)

    if times.shape[0] > 1:
        positive_steps = np.diff(times)
        if np.any(positive_steps <= 0):
            raise ValueError("times must be strictly increasing.")
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            x,
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
            max_step=float(np.min(positive_steps) / 10.0),
        )
        if not solution.success:
            raise RuntimeError(f"Python reference integration failed: {solution.message}")
        states = solution.y.T

    outputs = states @ layer.C.T
    return states, outputs


def write_reference_csv(path, times, inputs, states, outputs, layer_index, state_width=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["time"]
    headers.extend(_input_node(layer_index, idx) for idx in range(inputs.shape[1]))
    headers.extend(
        _state_node(layer_index, block_idx, state_idx)
        for block_idx in range(states.shape[1] // state_width)
        for state_idx in range(state_width)
    )
    headers.extend(_output_node(layer_index, idx) for idx in range(outputs.shape[1]))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_idx, time in enumerate(times):
            writer.writerow(
                [format_spice_value(time)]
                + [format_spice_value(v) for v in inputs[row_idx]]
                + [format_spice_value(v) for v in states[row_idx]]
                + [format_spice_value(v) for v in outputs[row_idx]]
            )
    return path


def pwl_source_line(name, node, times, values):
    pairs = []
    for time, value in zip(times, values):
        pairs.append(f"{format_spice_value(time)} {format_spice_value(value)}")
    return f"{name} {node} 0 PWL({' '.join(pairs)})"


def strip_final_end(netlist):
    lines = netlist.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().lower() == ".end":
        lines.pop()
    return "\n".join(lines)


def write_validation_deck(
    base_cir_path,
    out_path,
    times,
    inputs,
    layer,
    layer_index,
):
    base = Path(base_cir_path).read_text()
    body = strip_final_end(base)
    duration = float(times[-1])
    output_step = duration / max(len(times) - 1, 1)
    max_step = output_step / 10.0
    save_nodes = []
    save_nodes.extend(_input_node(layer_index, idx) for idx in range(layer.input_dim))
    save_nodes.extend(
        _state_node(layer_index, block_idx, state_idx)
        for block_idx in range(layer.n_blocks)
        for state_idx in range(layer.state_width)
    )
    save_nodes.extend(_output_node(layer_index, idx) for idx in range(layer.output_dim))

    lines = [body, "", "* Transient validation stimulus"]
    for input_idx in range(layer.input_dim):
        lines.append(
            pwl_source_line(
                f"VSTIM_L{layer_index}_IN{input_idx}",
                _input_node(layer_index, input_idx),
                times,
                inputs[:, input_idx],
            )
        )
    for block_idx in range(layer.n_blocks):
        for state_idx in range(layer.state_width):
            lines.append(f".ic V({_state_node(layer_index, block_idx, state_idx)})=0")
    lines.append(".options plotwinsize=0")
    lines.append(".save " + " ".join(f"V({node})" for node in save_nodes))
    lines.append(f".tran 0 {format_spice_value(duration)} 0 {format_spice_value(max_step)} uic")
    lines.append(".end")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path, save_nodes


def generate_validation_artifacts(
    params_path,
    cir_path,
    ssm_param,
    sample_rate,
    out_dir,
    layer_index=0,
    duration=0.02,
    points=401,
    stimulus="sine",
    amplitude=0.1,
):
    params = load_flax_params(params_path)
    modules = find_ssm_modules(params)
    if layer_index < 0 or layer_index >= len(modules):
        raise ValueError(f"layer_index must be in [0, {len(modules) - 1}]")
    layer = module_to_layer(modules[layer_index][0], modules[layer_index][1], ssm_param, sample_rate)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, duration, points, dtype=np.float64)
    inputs = make_stimulus(times, layer.input_dim, stimulus, amplitude)
    states, outputs = simulate_layer_reference(layer, times, inputs)

    stem = Path(cir_path).stem
    reference_path = write_reference_csv(
        out_dir / f"{stem}_layer{layer_index}_reference.csv",
        times,
        inputs,
        states,
        outputs,
        layer_index,
        state_width=layer.state_width,
    )
    deck_path, save_nodes = write_validation_deck(
        cir_path,
        out_dir / f"{stem}_layer{layer_index}_validation.cir",
        times,
        inputs,
        layer,
        layer_index,
    )
    metadata = {
        "params": str(params_path),
        "base_cir": str(cir_path),
        "validation_cir": str(deck_path),
        "reference_csv": str(reference_path),
        "ssm_param": ssm_param,
        "sample_rate": float(sample_rate),
        "layer_index": int(layer_index),
        "layer_path": modules[layer_index][0],
        "duration": float(duration),
        "points": int(points),
        "stimulus": stimulus,
        "amplitude": float(amplitude),
        "saved_nodes": save_nodes,
        "ltspice_app": "/Applications/LTspice.app/Contents/SharedSupport/ltspice/LTspice/run_ltspice",
    }
    metadata_path = out_dir / f"{stem}_layer{layer_index}_validation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="Flax msgpack parameter file.")
    parser.add_argument("--cir", required=True, help="Base exported LTSpice .cir file.")
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--out-dir", default="out/validation")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.02)
    parser.add_argument("--points", type=int, default=401)
    parser.add_argument("--stimulus", choices=["sine", "step", "impulse"], default="sine")
    parser.add_argument("--amplitude", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = generate_validation_artifacts(
        params_path=args.params,
        cir_path=args.cir,
        ssm_param=args.ssm_param,
        sample_rate=args.sample_rate,
        out_dir=args.out_dir,
        layer_index=args.layer_index,
        duration=args.duration,
        points=args.points,
        stimulus=args.stimulus,
        amplitude=args.amplitude,
    )
    print(f"Wrote validation deck: {metadata['validation_cir']}")
    print(f"Wrote Python reference: {metadata['reference_csv']}")
    print(f"Wrote validation metadata: {Path(metadata['reference_csv']).with_name(Path(metadata['reference_csv']).stem.replace('_reference', '_validation') + '.json')}")
    print("Run in LTSpice, then compare saved node traces against the reference CSV.")


if __name__ == "__main__":
    main()
