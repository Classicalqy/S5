"""Export hardware-friendly S5 SSM layers to an LTSpice netlist."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flax.serialization import msgpack_restore


POSITIVE_EPS = 1e-4
SUPPORTED_SSM_PARAMS = {"resonant_2x2", "energy_shaped_2x2"}
DEFAULT_FEEDBACK_RESISTANCE = 10_000.0
DEFAULT_DUMMY_MARGIN = 1.0


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def positive(raw, eps=POSITIVE_EPS):
    return softplus(raw) + eps


def sanitize_name(name):
    out = []
    for char in str(name):
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    sanitized = "".join(out).strip("_")
    return sanitized or "root"


def format_spice_value(value):
    if not np.isfinite(value):
        raise ValueError("SPICE component values must be finite.")
    return "{:.12g}".format(float(value))


def resistance_for_gain(gain, capacitance):
    gain = float(gain)
    if gain == 0.0:
        return None
    return 1.0 / (abs(gain) * capacitance)


def load_flax_params(path):
    with open(path, "rb") as handle:
        tree = msgpack_restore(handle.read())
    if hasattr(tree, "get") and "params" in tree:
        return tree["params"]
    return tree


def _is_mapping(value):
    return hasattr(value, "items")


def find_ssm_modules(params):
    modules = []

    def walk(prefix, tree):
        if not _is_mapping(tree):
            return
        keys = set(tree.keys())
        if {"B", "C", "raw_alpha", "omega", "log_step"}.issubset(keys):
            modules.append((prefix or "root", tree))
        for key, value in tree.items():
            if _is_mapping(value):
                next_prefix = f"{prefix}/{key}" if prefix else str(key)
                walk(next_prefix, value)

    walk("", params)
    return modules


@dataclass(frozen=True)
class SsmLayer:
    path: str
    B: np.ndarray
    C: np.ndarray
    alpha: np.ndarray
    omega: np.ndarray
    q: np.ndarray
    delta: np.ndarray
    A_tr: np.ndarray
    B_tr: np.ndarray

    @property
    def n_blocks(self):
        return int(self.A_tr.shape[0])

    @property
    def input_dim(self):
        return int(self.B_tr.shape[2])

    @property
    def output_dim(self):
        return int(self.C.shape[0])


def module_to_layer(path, module, ssm_param, sample_rate):
    B = np.asarray(module["B"], dtype=np.float64)
    C = np.asarray(module["C"], dtype=np.float64)
    raw_alpha = np.asarray(module["raw_alpha"], dtype=np.float64)
    omega = np.asarray(module["omega"], dtype=np.float64)
    log_step = np.asarray(module["log_step"], dtype=np.float64)

    if ssm_param not in SUPPORTED_SSM_PARAMS:
        raise ValueError(
            "Only resonant_2x2 and energy_shaped_2x2 are supported by this exporter."
        )
    if B.ndim != 2 or C.ndim != 2:
        raise ValueError(f"{path}: expected B and C to be rank-2 arrays.")
    if B.shape[0] % 2 != 0:
        raise ValueError(f"{path}: B first dimension must be even for 2x2 blocks.")

    n_blocks = B.shape[0] // 2
    if raw_alpha.shape != (n_blocks,):
        raise ValueError(f"{path}: raw_alpha shape must be ({n_blocks},).")
    if omega.shape != (n_blocks,):
        raise ValueError(f"{path}: omega shape must be ({n_blocks},).")
    if log_step.shape[0] != n_blocks:
        raise ValueError(f"{path}: log_step first dimension must be {n_blocks}.")
    if C.shape[1] != B.shape[0]:
        raise ValueError(f"{path}: C second dimension must match B first dimension.")

    alpha = positive(raw_alpha)
    if ssm_param == "energy_shaped_2x2":
        if "raw_q" not in module:
            raise ValueError(f"{path}: energy_shaped_2x2 requires raw_q.")
        q = positive(np.asarray(module["raw_q"], dtype=np.float64))
    else:
        q = np.ones((n_blocks,), dtype=np.float64)

    delta = np.exp(log_step[:, 0])
    decay = q * alpha
    frequency = q * omega
    A = np.stack(
        (
            np.stack((-decay, -frequency), axis=-1),
            np.stack((frequency, -decay), axis=-1),
        ),
        axis=-2,
    )
    A_tr = sample_rate * delta[:, None, None] * A
    B_blocks = B.reshape((n_blocks, 2, B.shape[1]))
    B_tr = sample_rate * delta[:, None, None] * B_blocks
    return SsmLayer(
        path=path,
        B=B,
        C=C,
        alpha=alpha,
        omega=omega,
        q=q,
        delta=delta,
        A_tr=A_tr,
        B_tr=B_tr,
    )


class NetlistBuilder:
    def __init__(self):
        self.lines = []
        self.component_names = set()

    def line(self, text=""):
        self.lines.append(text)

    def component(self, name, text):
        if name in self.component_names:
            raise ValueError(f"Duplicate SPICE component id: {name}")
        self.component_names.add(name)
        self.lines.append(text)

    def render(self):
        return "\n".join(self.lines) + "\n"


def _state_node(layer_idx, block_idx, state_idx):
    return f"L{layer_idx}_B{block_idx}_x{state_idx}"


def _input_node(layer_idx, input_idx):
    return f"L{layer_idx}_in{input_idx}"


def _output_node(layer_idx, output_idx):
    return f"L{layer_idx}_out{output_idx}"


def _block_name(layer_idx, block_idx):
    return f"s5_L{layer_idx}_B{block_idx}_2x2"


def _subckt_pin_list(layer, layer_idx, block_idx):
    inputs = [_input_node(layer_idx, i) for i in range(layer.input_dim)]
    states = [_state_node(layer_idx, block_idx, 0), _state_node(layer_idx, block_idx, 1)]
    return inputs + states


def _component_record(name, kind, nodes, value=None, role=None, **extra):
    record = {
        "name": name,
        "kind": kind,
        "nodes": list(nodes),
    }
    if value is not None:
        record["value"] = float(value)
    if role is not None:
        record["role"] = role
    record.update(extra)
    return record


def add_model_header(builder, state_capacitance, dense_included=False):
    builder.line("* LTSpice netlist generated from hardware-friendly S5 SSM layers")
    if dense_included:
        builder.line("* Dense encoder/decoder are included; normalization, residual, and non-ReLU activations are not included.")
    else:
        builder.line("* Dense encoder/decoder, normalization, residual, and activation layers are not included.")
    builder.line(".subckt ideal_opamp noninv inv out")
    builder.line("Eop out 0 noninv inv 1e6")
    builder.line(".ends ideal_opamp")
    builder.line()
    builder.line(".subckt unity_inverter in out")
    builder.line("Rin in n_inv 10k")
    builder.line("Rfb out n_inv 10k")
    builder.line("Xop 0 n_inv out ideal_opamp")
    builder.line(".ends unity_inverter")
    builder.line()
    builder.line(f".param CSTATE={format_spice_value(state_capacitance)}")
    builder.line()


def source_for_integrator_gain(layer_idx, block_idx, state_idx, gain, source_node):
    if gain < 0:
        return source_node, "direct"
    inv_node = f"{source_node}_inv_L{layer_idx}_B{block_idx}_S{state_idx}"
    return inv_node, "inverted"


def emit_block_subckt(builder, layer, layer_idx, block_idx, state_capacitance):
    block = _block_name(layer_idx, block_idx)
    pins = _subckt_pin_list(layer, layer_idx, block_idx)
    input_pins = pins[: layer.input_dim]
    x0, x1 = pins[layer.input_dim :]
    sum0 = "n_sum0"
    sum1 = "n_sum1"
    x0_inv = "x0_inv"
    components = []

    builder.line(f".subckt {block} {' '.join(pins)}")
    builder.component(f"XOP_L{layer_idx}_B{block_idx}_S0", f"XOP_L{layer_idx}_B{block_idx}_S0 0 {sum0} {x0} ideal_opamp")
    builder.component(f"XOP_L{layer_idx}_B{block_idx}_S1", f"XOP_L{layer_idx}_B{block_idx}_S1 0 {sum1} {x1} ideal_opamp")
    builder.component(f"C_L{layer_idx}_B{block_idx}_S0", f"C_L{layer_idx}_B{block_idx}_S0 {x0} {sum0} {{CSTATE}}")
    builder.component(f"C_L{layer_idx}_B{block_idx}_S1", f"C_L{layer_idx}_B{block_idx}_S1 {x1} {sum1} {{CSTATE}}")
    components.extend(
        [
            _component_record(f"XOP_L{layer_idx}_B{block_idx}_S0", "opamp", ["0", sum0, x0], role="state_integrator"),
            _component_record(f"XOP_L{layer_idx}_B{block_idx}_S1", "opamp", ["0", sum1, x1], role="state_integrator"),
            _component_record(f"C_L{layer_idx}_B{block_idx}_S0", "capacitor", [x0, sum0], state_capacitance, role="state_capacitor"),
            _component_record(f"C_L{layer_idx}_B{block_idx}_S1", "capacitor", [x1, sum1], state_capacitance, role="state_capacitor"),
        ]
    )

    builder.component(f"XINV_L{layer_idx}_B{block_idx}", f"XINV_L{layer_idx}_B{block_idx} {x0} {x0_inv} unity_inverter")
    components.append(_component_record(f"XINV_L{layer_idx}_B{block_idx}", "subckt", [x0, x0_inv], role="unity_inverter"))

    for state_idx, (state_node, sum_node) in enumerate(((x0, sum0), (x1, sum1))):
        gain = layer.A_tr[block_idx, state_idx, state_idx]
        resistor = resistance_for_gain(gain, state_capacitance)
        if resistor is not None:
            name = f"R_L{layer_idx}_B{block_idx}_alpha{state_idx}"
            builder.component(name, f"{name} {state_node} {sum_node} {format_spice_value(resistor)}")
            components.append(
                _component_record(
                    name,
                    "resistor",
                    [state_node, sum_node],
                    resistor,
                    role="state_decay",
                    coefficient=float(gain),
                )
            )

    cross_specs = [
        (0, 1, layer.A_tr[block_idx, 0, 1], x1, sum0),
        (1, 0, layer.A_tr[block_idx, 1, 0], x0, sum1),
    ]
    for target_state, source_state, gain, source_node, sum_node in cross_specs:
        resistor = resistance_for_gain(gain, state_capacitance)
        if resistor is None:
            continue
        actual_source = source_node
        polarity = "direct"
        if gain > 0 and source_node == x0:
            actual_source = x0_inv
            polarity = "inverted_by_block_inverter"
        elif gain > 0:
            actual_source, polarity = source_for_integrator_gain(
                layer_idx, block_idx, target_state, gain, source_node
            )
            inv_name = f"XINV_L{layer_idx}_B{block_idx}_cross{source_state}_{target_state}"
            builder.component(inv_name, f"{inv_name} {source_node} {actual_source} unity_inverter")
            components.append(_component_record(inv_name, "subckt", [source_node, actual_source], role="unity_inverter"))
        name = f"R_L{layer_idx}_B{block_idx}_omega{source_state}_to_{target_state}"
        builder.component(name, f"{name} {actual_source} {sum_node} {format_spice_value(resistor)}")
        components.append(
            _component_record(
                name,
                "resistor",
                [actual_source, sum_node],
                resistor,
                role="cross_coupling",
                coefficient=float(gain),
                polarity=polarity,
            )
        )

    for state_idx, sum_node in enumerate((sum0, sum1)):
        for input_idx, input_node in enumerate(input_pins):
            gain = layer.B_tr[block_idx, state_idx, input_idx]
            resistor = resistance_for_gain(gain, state_capacitance)
            if resistor is None:
                continue
            actual_source = input_node
            polarity = "direct"
            if gain > 0:
                actual_source = f"{input_node}_inv_B{block_idx}_S{state_idx}"
                polarity = "inverted"
                inv_name = f"XINV_L{layer_idx}_B{block_idx}_U{input_idx}_S{state_idx}"
                builder.component(inv_name, f"{inv_name} {input_node} {actual_source} unity_inverter")
                components.append(_component_record(inv_name, "subckt", [input_node, actual_source], role="unity_inverter"))
            name = f"R_L{layer_idx}_B{block_idx}_B{state_idx}_{input_idx}"
            builder.component(name, f"{name} {actual_source} {sum_node} {format_spice_value(resistor)}")
            components.append(
                _component_record(
                    name,
                    "resistor",
                    [actual_source, sum_node],
                    resistor,
                    role="input_weight",
                    coefficient=float(gain),
                    input_index=input_idx,
                    state_index=state_idx,
                    polarity=polarity,
                )
            )

    builder.line(f".ends {block}")
    builder.line()
    return components


def sign_aware_resistors(weights, sources, layer_idx, output_idx, prefix):
    neg = [(idx, abs(w), src) for idx, (w, src) in enumerate(zip(weights, sources)) if w < 0]
    pos = [(idx, w, src) for idx, (w, src) in enumerate(zip(weights, sources)) if w > 0]
    sum_neg = sum(w for _, w, _ in neg)
    sum_pos = sum(w for _, w, _ in pos)
    dummy_weight = 0.0
    denom = 1.0 + sum_neg - sum_pos
    if denom <= 0.0:
        dummy_weight = -denom + DEFAULT_DUMMY_MARGIN
        denom = DEFAULT_DUMMY_MARGIN

    resistors = []
    for idx, weight, source in neg:
        resistors.append(
            {
                "source_index": idx,
                "source": source,
                "sign": "negative",
                "coefficient": -float(weight),
                "resistance": DEFAULT_FEEDBACK_RESISTANCE / weight,
            }
        )
    if dummy_weight > 0.0:
        resistors.append(
            {
                "source_index": None,
                "source": "0",
                "sign": "negative_dummy_ground",
                "coefficient": -float(dummy_weight),
                "resistance": DEFAULT_FEEDBACK_RESISTANCE / dummy_weight,
            }
        )
    for idx, weight, source in pos:
        resistors.append(
            {
                "source_index": idx,
                "source": source,
                "sign": "positive",
                "coefficient": float(weight),
                "resistance": DEFAULT_FEEDBACK_RESISTANCE * denom / weight,
            }
        )
    return resistors, {
        "sum_negative_abs": float(sum_neg + dummy_weight),
        "sum_positive": float(sum_pos),
        "dummy_negative_weight": float(dummy_weight),
        "positive_denominator": float(denom),
    }


def emit_output_stage(builder, layer, layer_idx):
    components = []
    state_nodes = [
        _state_node(layer_idx, block_idx, state_idx)
        for block_idx in range(layer.n_blocks)
        for state_idx in range(2)
    ]
    output_records = []

    for output_idx in range(layer.output_dim):
        out_node = _output_node(layer_idx, output_idx)
        weights = layer.C[output_idx]
        nonzero = [(idx, float(w)) for idx, w in enumerate(weights) if float(w) != 0.0]
        sum_node = f"L{layer_idx}_out{output_idx}_sum"
        builder.component(
            f"XOUT_L{layer_idx}_O{output_idx}",
            f"XOUT_L{layer_idx}_O{output_idx} 0 {sum_node} {out_node} ideal_opamp",
        )
        builder.component(
            f"RFOUT_L{layer_idx}_O{output_idx}",
            f"RFOUT_L{layer_idx}_O{output_idx} {out_node} {sum_node} {format_spice_value(DEFAULT_FEEDBACK_RESISTANCE)}",
        )
        components.extend(
            [
                _component_record(f"XOUT_L{layer_idx}_O{output_idx}", "opamp", ["0", sum_node, out_node], role="output_adder"),
                _component_record(
                    f"RFOUT_L{layer_idx}_O{output_idx}",
                    "resistor",
                    [out_node, sum_node],
                    DEFAULT_FEEDBACK_RESISTANCE,
                    role="output_feedback",
                ),
            ]
        )

        nonzero_components = []
        for state_index, weight in nonzero:
            source = state_nodes[state_index]
            polarity = "direct"
            if weight > 0:
                inverted = f"{source}_inv_O{output_idx}"
                inv_name = f"XINV_L{layer_idx}_O{output_idx}_C{state_index}"
                builder.component(inv_name, f"{inv_name} {source} {inverted} unity_inverter")
                components.append(
                    _component_record(
                        inv_name,
                        "subckt",
                        [source, inverted],
                        role="unity_inverter",
                    )
                )
                source = inverted
                polarity = "inverted"
            resistance = DEFAULT_FEEDBACK_RESISTANCE / abs(weight)
            name = f"R_L{layer_idx}_O{output_idx}_C{state_index}"
            builder.component(name, f"{name} {source} {sum_node} {format_spice_value(resistance)}")
            components.append(
                _component_record(
                    name,
                    "resistor",
                    [source, sum_node],
                    resistance,
                    role="output_weight",
                    coefficient=float(weight),
                    source_index=state_index,
                    sign="positive" if weight > 0 else "negative",
                    polarity=polarity,
                )
            )
            nonzero_components.append(name)
        output_records.append(
            {
                "output_index": output_idx,
                "node": out_node,
                "nonzero_weights": len(nonzero),
                "adder_metadata": {
                    "feedback_resistance": float(DEFAULT_FEEDBACK_RESISTANCE),
                    "implementation": "inverting_summer_with_unity_inverters_for_positive_weights",
                    "components": nonzero_components,
                },
            }
        )
    builder.line()
    return components, output_records


def emit_top_level_instances(builder, layers):
    builder.line("* Top-level block instances")
    for layer_idx, layer in enumerate(layers):
        for block_idx in range(layer.n_blocks):
            block = _block_name(layer_idx, block_idx)
            pins = _subckt_pin_list(layer, layer_idx, block_idx)
            name = f"X_L{layer_idx}_B{block_idx}"
            builder.component(name, f"{name} {' '.join(pins)} {block}")
    builder.line()
    builder.line(".end")


def build_netlist(params, ssm_param, sample_rate, state_capacitance=1e-6):
    modules = find_ssm_modules(params)
    if not modules:
        raise ValueError("No hardware-friendly SSM modules found in parameter tree.")

    layers = [module_to_layer(path, module, ssm_param, sample_rate) for path, module in modules]
    builder = NetlistBuilder()
    add_model_header(builder, state_capacitance)

    manifest = {
        "ssm_param": ssm_param,
        "sample_rate": float(sample_rate),
        "state_capacitance": float(state_capacitance),
        "layers": [],
    }

    for layer_idx, layer in enumerate(layers):
        layer_record = {
            "index": layer_idx,
            "path": layer.path,
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "n_blocks": layer.n_blocks,
            "alpha": layer.alpha.tolist(),
            "omega": layer.omega.tolist(),
            "q": layer.q.tolist(),
            "delta": layer.delta.tolist(),
            "A_tr": layer.A_tr.tolist(),
            "B_tr": layer.B_tr.tolist(),
            "blocks": [],
            "outputs": [],
            "components": [],
        }
        builder.line(f"* Layer {layer_idx}: {layer.path}")
        for block_idx in range(layer.n_blocks):
            components = emit_block_subckt(builder, layer, layer_idx, block_idx, state_capacitance)
            layer_record["blocks"].append(
                {
                    "index": block_idx,
                    "subckt": _block_name(layer_idx, block_idx),
                    "state_nodes": [
                        _state_node(layer_idx, block_idx, 0),
                        _state_node(layer_idx, block_idx, 1),
                    ],
                    "components": components,
                }
            )
            layer_record["components"].extend(components)
        output_components, output_records = emit_output_stage(builder, layer, layer_idx)
        layer_record["components"].extend(output_components)
        layer_record["outputs"] = output_records
        manifest["layers"].append(layer_record)

    emit_top_level_instances(builder, layers)
    return builder.render(), manifest


def export_netlist(params_path, ssm_param, sample_rate, out_path, json_out=None, state_capacitance=1e-6):
    params = load_flax_params(params_path)
    netlist, manifest = build_netlist(params, ssm_param, sample_rate, state_capacitance)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(netlist)

    json_path = Path(json_out) if json_out else out_path.with_name(f"{out_path.stem}_components.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_path, json_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="Flax msgpack parameter file.")
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--state-capacitance", type=float, default=1e-6)
    parser.add_argument("--out", required=True, help="Output LTSpice .cir file.")
    parser.add_argument("--json-out", default=None, help="Optional component manifest path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cir_path, json_path = export_netlist(
        params_path=args.params,
        ssm_param=args.ssm_param,
        sample_rate=args.sample_rate,
        out_path=args.out,
        json_out=args.json_out,
        state_capacitance=args.state_capacitance,
    )
    print(f"Wrote LTSpice netlist: {cir_path}")
    print(f"Wrote component manifest: {json_path}")


if __name__ == "__main__":
    main()
