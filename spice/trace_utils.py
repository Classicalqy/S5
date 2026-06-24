"""Small helpers shared by SPICE trace validators."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .export_netlist import _output_node, _state_node, format_spice_value


def linear_nodes(prefix, count):
    return [f"{prefix}{idx}" for idx in range(count)]


def full_model_nodes(model):
    nodes = ["IN0"]
    nodes.extend(linear_nodes("ENC", model.encoder_bias.shape[0]))
    for layer_idx, layer in enumerate(model.ssm_layers):
        nodes.extend(
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(layer.n_blocks)
            for state_idx in range(2)
        )
        nodes.extend(_output_node(layer_idx, idx) for idx in range(layer.output_dim))
        nodes.extend(linear_nodes(f"RELU{layer_idx}_", layer.output_dim))
    nodes.extend(linear_nodes("LOGIT", model.decoder_bias.shape[0]))
    return nodes


def write_trace_csv(path, traces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(traces.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_idx in range(len(traces["time"])):
            writer.writerow([format_spice_value(traces[key][row_idx]) for key in headers])
    return path


def zoh_sample_times(length, sample_rate):
    return (np.arange(length, dtype=np.float64) + 1.0) / float(sample_rate)


def zoh_value_at(times, inputs, t):
    if len(times) < 2:
        return inputs[0]
    step = float(times[1] - times[0])
    idx = int(np.floor(max(float(t), 0.0) / step))
    return inputs[min(idx, inputs.shape[0] - 1)]


def zoh_pwl_source_line(name, node, step_seconds, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("ZOH PWL values must be a non-empty rank-1 array.")
    step = float(step_seconds)
    if step <= 0.0:
        raise ValueError("ZOH PWL step must be positive.")

    eps = step * 1e-6
    pairs = [(0.0, values[0])]
    for idx in range(1, values.shape[0]):
        boundary = idx * step
        pairs.append((max(boundary - eps, 0.0), values[idx - 1]))
        pairs.append((boundary, values[idx]))
    pairs.append((values.shape[0] * step, values[-1]))
    text = " ".join(f"{format_spice_value(t)} {format_spice_value(v)}" for t, v in pairs)
    return f"{name} {node} 0 PWL({text})"
