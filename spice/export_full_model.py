"""Export the restricted MNIST S5 model as one LTSpice netlist."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .export_netlist import (
    DEFAULT_FEEDBACK_RESISTANCE,
    NetlistBuilder,
    SUPPORTED_SSM_PARAMS,
    _block_name,
    _output_node,
    _state_node,
    add_model_header,
    emit_block_subckt,
    emit_output_stage,
    find_ssm_modules,
    format_spice_value,
    load_flax_params,
    module_to_layer,
)


@dataclass(frozen=True)
class FullModel:
    encoder_kernel: np.ndarray
    encoder_bias: np.ndarray
    ssm_layers: tuple
    decoder_kernel: np.ndarray
    decoder_bias: np.ndarray


def _get_path(tree, path):
    node = tree
    for part in path:
        if not hasattr(node, "items") or part not in node:
            raise ValueError("Missing parameter path: " + "/".join(path))
        node = node[part]
    return np.asarray(node, dtype=np.float64)


def _layer_number(path):
    match = re.search(r"layers_(\d+)/seq$", path)
    return int(match.group(1)) if match else 10**9


def _encoder_ssm_modules(params):
    return [
        (path, module)
        for path, module in find_ssm_modules(params)
        if re.fullmatch(r"encoder/layers_\d+/seq", path)
    ]


def extract_full_model(params, ssm_param, sample_rate):
    encoder_kernel = _get_path(params, ("encoder", "encoder", "kernel"))
    encoder_bias = _get_path(params, ("encoder", "encoder", "bias"))
    decoder_kernel = _get_path(params, ("decoder", "kernel"))
    decoder_bias = _get_path(params, ("decoder", "bias"))

    modules = sorted(_encoder_ssm_modules(params), key=lambda item: _layer_number(item[0]))
    if len(modules) != 2:
        raise ValueError(f"Full-model export expects exactly 2 SSM layers, found {len(modules)}.")
    ssm_layers = tuple(module_to_layer(path, module, ssm_param, sample_rate) for path, module in modules)

    if encoder_kernel.shape[1] != ssm_layers[0].input_dim:
        raise ValueError("Encoder output dimension must match first SSM input dimension.")
    for idx in range(1, len(ssm_layers)):
        if ssm_layers[idx - 1].output_dim != ssm_layers[idx].input_dim:
            raise ValueError(f"SSM layer {idx - 1} output dimension must match layer {idx} input dimension.")
    if decoder_kernel.shape[0] != ssm_layers[-1].output_dim:
        raise ValueError("Decoder input dimension must match final SSM output dimension.")
    if decoder_kernel.shape[1] != decoder_bias.shape[0]:
        raise ValueError("Decoder kernel output dimension must match decoder bias.")
    return FullModel(encoder_kernel, encoder_bias, ssm_layers, decoder_kernel, decoder_bias)


def _linear_nodes(prefix, count):
    return [f"{prefix}{idx}" for idx in range(count)]


def emit_linear_stage(builder, name, source_nodes, kernel, bias, output_prefix):
    """Emit y = source @ kernel + bias using inverting summers."""
    kernel = np.asarray(kernel, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if kernel.shape[0] != len(source_nodes):
        raise ValueError(f"{name}: kernel input dimension does not match source nodes.")
    if kernel.shape[1] != bias.shape[0]:
        raise ValueError(f"{name}: kernel output dimension does not match bias.")

    output_nodes = _linear_nodes(output_prefix, kernel.shape[1])
    components = []
    for out_idx, out_node in enumerate(output_nodes):
        sum_node = f"{out_node}_sum"
        builder.component(f"X_{name}_{out_idx}", f"X_{name}_{out_idx} 0 {sum_node} {out_node} ideal_opamp")
        builder.component(
            f"RF_{name}_{out_idx}",
            f"RF_{name}_{out_idx} {out_node} {sum_node} {format_spice_value(DEFAULT_FEEDBACK_RESISTANCE)}",
        )
        components.extend([f"X_{name}_{out_idx}", f"RF_{name}_{out_idx}"])

        if bias[out_idx] != 0.0:
            bias_node = f"{out_node}_bias"
            builder.component(
                f"V_{name}_BIAS_{out_idx}",
                f"V_{name}_BIAS_{out_idx} {bias_node} 0 {format_spice_value(-bias[out_idx])}",
            )
            builder.component(
                f"R_{name}_BIAS_{out_idx}",
                f"R_{name}_BIAS_{out_idx} {bias_node} {sum_node} {format_spice_value(DEFAULT_FEEDBACK_RESISTANCE)}",
            )
            components.extend([f"V_{name}_BIAS_{out_idx}", f"R_{name}_BIAS_{out_idx}"])

        for in_idx, source in enumerate(source_nodes):
            weight = kernel[in_idx, out_idx]
            if weight == 0.0:
                continue
            actual_source = source
            if weight > 0.0:
                actual_source = f"{source}_inv_{name}_{out_idx}"
                inv_name = f"XINV_{name}_{out_idx}_{in_idx}"
                builder.component(inv_name, f"{inv_name} {source} {actual_source} unity_inverter")
                components.append(inv_name)
            resistor = DEFAULT_FEEDBACK_RESISTANCE / abs(weight)
            r_name = f"R_{name}_{out_idx}_{in_idx}"
            builder.component(r_name, f"{r_name} {actual_source} {sum_node} {format_spice_value(resistor)}")
            components.append(r_name)
    builder.line()
    return output_nodes, components


def emit_relu_stage(builder, name, source_nodes, output_prefix):
    output_nodes = _linear_nodes(output_prefix, len(source_nodes))
    for idx, (source, out_node) in enumerate(zip(source_nodes, output_nodes)):
        builder.component(f"B_{name}_{idx}", f"B_{name}_{idx} {out_node} 0 V=max(V({source}),0)")
    builder.line()
    return output_nodes


def emit_ssm_stage(builder, layer, layer_idx, input_nodes):
    for block_idx in range(layer.n_blocks):
        emit_block_subckt(builder, layer, layer_idx, block_idx, 1e-6)
    for block_idx in range(layer.n_blocks):
        pins = input_nodes + [_state_node(layer_idx, block_idx, 0), _state_node(layer_idx, block_idx, 1)]
        builder.component(
            f"XFULL_L{layer_idx}_B{block_idx}",
            f"XFULL_L{layer_idx}_B{block_idx} {' '.join(pins)} {_block_name(layer_idx, block_idx)}",
        )
    emit_output_stage(builder, layer, layer_idx)
    return [_output_node(layer_idx, idx) for idx in range(layer.output_dim)]


def build_full_netlist(params, ssm_param, sample_rate):
    model = extract_full_model(params, ssm_param, sample_rate)
    builder = NetlistBuilder()
    add_model_header(builder, state_capacitance=1e-6, dense_included=True)
    builder.line("* Full restricted model: Dense encoder -> SSM/ReLU -> SSM/ReLU -> Dense decoder")

    input_nodes = ["IN0"]
    encoder_nodes, encoder_components = emit_linear_stage(
        builder, "ENC", input_nodes, model.encoder_kernel, model.encoder_bias, "ENC"
    )
    ssm0_nodes = emit_ssm_stage(builder, model.ssm_layers[0], 0, encoder_nodes)
    relu0_nodes = emit_relu_stage(builder, "RELU0", ssm0_nodes, "RELU0_")
    ssm1_nodes = emit_ssm_stage(builder, model.ssm_layers[1], 1, relu0_nodes)
    relu1_nodes = emit_relu_stage(builder, "RELU1", ssm1_nodes, "RELU1_")
    logit_nodes, decoder_components = emit_linear_stage(
        builder, "DEC", relu1_nodes, model.decoder_kernel, model.decoder_bias, "LOGIT"
    )
    builder.line(".end")

    manifest = {
        "sample_rate": float(sample_rate),
        "ssm_param": ssm_param,
        "input_nodes": input_nodes,
        "encoder_nodes": encoder_nodes,
        "ssm_output_nodes": [[_output_node(i, j) for j in range(layer.output_dim)] for i, layer in enumerate(model.ssm_layers)],
        "relu_nodes": [relu0_nodes, relu1_nodes],
        "logit_nodes": logit_nodes,
        "encoder_components": encoder_components,
        "decoder_components": decoder_components,
    }
    return builder.render(), manifest


def export_full_model(params_path, ssm_param, sample_rate, out_path, json_out=None):
    params = load_flax_params(params_path)
    netlist, manifest = build_full_netlist(params, ssm_param, sample_rate)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(netlist)
    json_path = Path(json_out) if json_out else out_path.with_name(f"{out_path.stem}_components.json")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_path, json_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cir_path, json_path = export_full_model(
        args.params,
        args.ssm_param,
        args.sample_rate,
        args.out,
        args.json_out,
    )
    print(f"Wrote full-model LTSpice netlist: {cir_path}")
    print(f"Wrote full-model component manifest: {json_path}")


if __name__ == "__main__":
    main()
