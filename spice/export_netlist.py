"""Export hardware-friendly S5 SSM layers to an LTSpice netlist."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flax.serialization import msgpack_restore

from .hardware_projection import HardwareProjectionConfig, PROJECTION_NONE, project_layers


POSITIVE_EPS = 1e-4
SSM_PARAM_REAL_DECAY = "real_decay"
SSM_PARAM_RESONANT_2X2 = "resonant_2x2"
SSM_PARAM_ENERGY_SHAPED_2X2 = "energy_shaped_2x2"
SUPPORTED_SSM_PARAMS = {
    SSM_PARAM_REAL_DECAY,
    SSM_PARAM_RESONANT_2X2,
    SSM_PARAM_ENERGY_SHAPED_2X2,
}
DEFAULT_FEEDBACK_RESISTANCE = 10_000.0


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def positive(raw, eps=POSITIVE_EPS):
    return softplus(raw) + eps


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
        if {"B", "C", "raw_alpha", "log_step"}.issubset(keys):
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
    ssm_param: str
    state_width: int
    B: np.ndarray
    C: np.ndarray
    alpha: np.ndarray
    omega: np.ndarray
    q: np.ndarray
    delta: np.ndarray
    A_tr: np.ndarray
    B_tr: np.ndarray
    capacitances: np.ndarray | None = None

    @property
    def n_blocks(self):
        return int(self.A_tr.shape[0])

    @property
    def input_dim(self):
        return int(self.B_tr.shape[2])

    @property
    def output_dim(self):
        return int(self.C.shape[0])

    @property
    def state_dim(self):
        return int(self.n_blocks * self.state_width)

    @property
    def block_suffix(self):
        return "real_decay" if self.state_width == 1 else "2x2"

    def state_nodes(self, layer_idx):
        return [
            _state_node(layer_idx, block_idx, state_idx)
            for block_idx in range(self.n_blocks)
            for state_idx in range(self.state_width)
        ]


def module_to_layer(path, module, ssm_param, sample_rate):
    B = np.asarray(module["B"], dtype=np.float64)
    C = np.asarray(module["C"], dtype=np.float64)
    raw_alpha = np.asarray(module["raw_alpha"], dtype=np.float64)
    log_step = np.asarray(module["log_step"], dtype=np.float64)

    if ssm_param not in SUPPORTED_SSM_PARAMS:
        raise ValueError(
            "Only real_decay, resonant_2x2, and energy_shaped_2x2 are supported by this exporter."
        )
    if B.ndim != 2 or C.ndim != 2:
        raise ValueError(f"{path}: expected B and C to be rank-2 arrays.")

    if ssm_param == SSM_PARAM_REAL_DECAY:
        state_width = 1
        n_blocks = B.shape[0]
    else:
        state_width = 2
        if B.shape[0] % 2 != 0:
            raise ValueError(f"{path}: B first dimension must be even for 2x2 blocks.")
        n_blocks = B.shape[0] // 2

    if raw_alpha.shape != (n_blocks,):
        raise ValueError(f"{path}: raw_alpha shape must be ({n_blocks},).")
    if log_step.shape[0] != n_blocks:
        raise ValueError(f"{path}: log_step first dimension must be {n_blocks}.")
    if C.shape[1] != B.shape[0]:
        raise ValueError(f"{path}: C second dimension must match B first dimension.")

    alpha = positive(raw_alpha)
    delta = np.exp(log_step[:, 0])
    if ssm_param == SSM_PARAM_REAL_DECAY:
        omega = np.zeros((n_blocks,), dtype=np.float64)
        q = np.ones((n_blocks,), dtype=np.float64)
        A_tr = sample_rate * delta[:, None, None] * (-alpha[:, None, None])
        B_tr = sample_rate * delta[:, None, None] * B[:, None, :]
    else:
        if "omega" not in module:
            raise ValueError(f"{path}: {ssm_param} requires omega.")
        omega = np.asarray(module["omega"], dtype=np.float64)
        if omega.shape != (n_blocks,):
            raise ValueError(f"{path}: omega shape must be ({n_blocks},).")
        if ssm_param == SSM_PARAM_ENERGY_SHAPED_2X2:
            if "raw_q" not in module:
                raise ValueError(f"{path}: energy_shaped_2x2 requires raw_q.")
            q = positive(np.asarray(module["raw_q"], dtype=np.float64))
        else:
            q = np.ones((n_blocks,), dtype=np.float64)

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

    if ssm_param == SSM_PARAM_ENERGY_SHAPED_2X2 and q.shape != (n_blocks,):
        raise ValueError(f"{path}: raw_q shape must be ({n_blocks},).")
    return SsmLayer(
        path=path,
        ssm_param=ssm_param,
        state_width=state_width,
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


def _block_name(layer_idx, block_idx, suffix="2x2"):
    return f"s5_L{layer_idx}_B{block_idx}_{suffix}"


def _layer_block_name(layer, layer_idx, block_idx):
    return _block_name(layer_idx, block_idx, layer.block_suffix)


def _subckt_pin_list(layer, layer_idx, block_idx):
    inputs = [_input_node(layer_idx, i) for i in range(layer.input_dim)]
    states = [
        _state_node(layer_idx, block_idx, state_idx)
        for state_idx in range(layer.state_width)
    ]
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


def add_model_header(builder, state_capacitance, dense_included=False, use_global_state_capacitance=True):
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
    if use_global_state_capacitance:
        builder.line(f".param CSTATE={format_spice_value(state_capacitance)}")
        builder.line()


def add_unity_inverter(builder, components, name, source_node, output_node):
    builder.component(name, f"{name} {source_node} {output_node} unity_inverter")
    components.append(_component_record(name, "subckt", [source_node, output_node], role="unity_inverter"))


def add_integrator_state(
    builder,
    components,
    layer_idx,
    block_idx,
    state_idx,
    state_node,
    sum_node,
    state_capacitance,
    use_global_state_capacitance=True,
):
    op_name = f"XOP_L{layer_idx}_B{block_idx}_S{state_idx}"
    cap_name = f"C_L{layer_idx}_B{block_idx}_S{state_idx}"
    cap_value = "{CSTATE}" if use_global_state_capacitance else format_spice_value(state_capacitance)
    builder.component(op_name, f"{op_name} 0 {sum_node} {state_node} ideal_opamp")
    builder.component(cap_name, f"{cap_name} {state_node} {sum_node} {cap_value}")
    components.extend(
        [
            _component_record(op_name, "opamp", ["0", sum_node, state_node], role="state_integrator"),
            _component_record(cap_name, "capacitor", [state_node, sum_node], state_capacitance, role="state_capacitor"),
        ]
    )


def add_gain_resistor(builder, components, name, source_node, sum_node, gain, state_capacitance, role, **extra):
    resistor = resistance_for_gain(gain, state_capacitance)
    if resistor is None:
        return False
    builder.component(name, f"{name} {source_node} {sum_node} {format_spice_value(resistor)}")
    components.append(
        _component_record(
            name,
            "resistor",
            [source_node, sum_node],
            resistor,
            role=role,
            coefficient=float(gain),
            **extra,
        )
    )
    return True


def _state_capacitance(layer, block_idx, state_idx, state_capacitance):
    if layer.capacitances is None:
        return float(state_capacitance)
    return float(layer.capacitances[block_idx, state_idx])


def emit_block_subckt(builder, layer, layer_idx, block_idx, state_capacitance, use_global_state_capacitance=True):
    block = _layer_block_name(layer, layer_idx, block_idx)
    pins = _subckt_pin_list(layer, layer_idx, block_idx)
    input_pins = pins[: layer.input_dim]
    state_pins = pins[layer.input_dim :]
    sum_nodes = [f"n_sum{idx}" for idx in range(layer.state_width)]
    components = []

    builder.line(f".subckt {block} {' '.join(pins)}")
    for state_idx, (state_node, sum_node) in enumerate(zip(state_pins, sum_nodes)):
        cap = _state_capacitance(layer, block_idx, state_idx, state_capacitance)
        add_integrator_state(
            builder,
            components,
            layer_idx,
            block_idx,
            state_idx,
            state_node,
            sum_node,
            cap,
            use_global_state_capacitance,
        )

    for state_idx, (state_node, sum_node) in enumerate(zip(state_pins, sum_nodes)):
        gain = layer.A_tr[block_idx, state_idx, state_idx]
        cap = _state_capacitance(layer, block_idx, state_idx, state_capacitance)
        add_gain_resistor(
            builder,
            components,
            f"R_L{layer_idx}_B{block_idx}_alpha{state_idx}",
            state_node,
            sum_node,
            gain,
            cap,
            "state_decay",
        )

    if layer.state_width == 2:
        x0, x1 = state_pins
        cross_specs = [
            (0, 1, layer.A_tr[block_idx, 0, 1], x1, sum_nodes[0]),
            (1, 0, layer.A_tr[block_idx, 1, 0], x0, sum_nodes[1]),
        ]
        inverted_sources = {}
        for target_state, source_state, gain, source_node, sum_node in cross_specs:
            cap = _state_capacitance(layer, block_idx, target_state, state_capacitance)
            if resistance_for_gain(gain, cap) is None:
                continue
            actual_source = source_node
            polarity = "direct"
            if gain > 0:
                if source_node not in inverted_sources:
                    inv_node = f"{source_node}_inv_L{layer_idx}_B{block_idx}"
                    inv_name = f"XINV_L{layer_idx}_B{block_idx}_cross{source_state}"
                    add_unity_inverter(builder, components, inv_name, source_node, inv_node)
                    inverted_sources[source_node] = inv_node
                actual_source = inverted_sources[source_node]
                polarity = "inverted"
            name = f"R_L{layer_idx}_B{block_idx}_omega{source_state}_to_{target_state}"
            add_gain_resistor(
                builder,
                components,
                name,
                actual_source,
                sum_node,
                gain,
                cap,
                "cross_coupling",
                source_state=source_state,
                target_state=target_state,
                polarity=polarity,
            )

    for state_idx, sum_node in enumerate(sum_nodes):
        cap = _state_capacitance(layer, block_idx, state_idx, state_capacitance)
        for input_idx, input_node in enumerate(input_pins):
            gain = layer.B_tr[block_idx, state_idx, input_idx]
            if resistance_for_gain(gain, cap) is None:
                continue
            actual_source = input_node
            polarity = "direct"
            if gain > 0:
                actual_source = f"{input_node}_inv_B{block_idx}_S{state_idx}"
                polarity = "inverted"
                inv_name = f"XINV_L{layer_idx}_B{block_idx}_U{input_idx}_S{state_idx}"
                add_unity_inverter(builder, components, inv_name, input_node, actual_source)
            name = f"R_L{layer_idx}_B{block_idx}_B{state_idx}_{input_idx}"
            add_gain_resistor(
                builder,
                components,
                name,
                actual_source,
                sum_node,
                gain,
                cap,
                "input_weight",
                input_index=input_idx,
                state_index=state_idx,
                polarity=polarity,
            )

    builder.line(f".ends {block}")
    builder.line()
    return components


def emit_output_stage(builder, layer, layer_idx):
    components = []
    state_nodes = layer.state_nodes(layer_idx)
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
            block = _layer_block_name(layer, layer_idx, block_idx)
            pins = _subckt_pin_list(layer, layer_idx, block_idx)
            name = f"X_L{layer_idx}_B{block_idx}"
            builder.component(name, f"{name} {' '.join(pins)} {block}")
    builder.line()
    builder.line(".end")


def _normalize_projection_config(projection_config=None, **kwargs):
    if projection_config is None:
        values = {key: value for key, value in kwargs.items() if value is not None}
        return HardwareProjectionConfig(**values)
    if isinstance(projection_config, HardwareProjectionConfig):
        return projection_config
    if hasattr(projection_config, "items"):
        values = dict(projection_config)
        values.update({key: value for key, value in kwargs.items() if value is not None})
        return HardwareProjectionConfig(**values)
    raise TypeError("projection_config must be None, a dict, or HardwareProjectionConfig.")


def _projection_enabled(projection_config):
    return projection_config.hardware_projection != PROJECTION_NONE


def build_netlist(params, ssm_param, sample_rate, state_capacitance=1e-6, projection_config=None):
    modules = find_ssm_modules(params)
    if not modules:
        raise ValueError("No hardware-friendly SSM modules found in parameter tree.")

    layers = [module_to_layer(path, module, ssm_param, sample_rate) for path, module in modules]
    projection_config = _normalize_projection_config(projection_config)
    projection_report = None
    if _projection_enabled(projection_config):
        layers, projection_report = project_layers(layers, projection_config)
    builder = NetlistBuilder()
    use_global_state_capacitance = not _projection_enabled(projection_config)
    add_model_header(builder, state_capacitance, use_global_state_capacitance=use_global_state_capacitance)

    manifest = {
        "ssm_param": ssm_param,
        "sample_rate": float(sample_rate),
        "state_capacitance": float(state_capacitance),
        "layers": [],
    }
    if projection_report is not None:
        manifest["projection"] = projection_report

    for layer_idx, layer in enumerate(layers):
        layer_record = {
            "index": layer_idx,
            "path": layer.path,
            "state_width": layer.state_width,
            "state_dim": layer.state_dim,
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
        if layer.capacitances is not None:
            layer_record["state_capacitances"] = np.asarray(layer.capacitances, dtype=np.float64).tolist()
        builder.line(f"* Layer {layer_idx}: {layer.path}")
        for block_idx in range(layer.n_blocks):
            components = emit_block_subckt(
                builder,
                layer,
                layer_idx,
                block_idx,
                state_capacitance,
                use_global_state_capacitance=use_global_state_capacitance,
            )
            layer_record["blocks"].append(
                {
                    "index": block_idx,
                    "subckt": _layer_block_name(layer, layer_idx, block_idx),
                    "state_nodes": [
                        _state_node(layer_idx, block_idx, state_idx)
                        for state_idx in range(layer.state_width)
                    ],
                    "components": components,
                }
            )
            if layer.capacitances is not None:
                layer_record["blocks"][-1]["state_capacitances"] = [
                    float(layer.capacitances[block_idx, state_idx])
                    for state_idx in range(layer.state_width)
                ]
            layer_record["components"].extend(components)
        output_components, output_records = emit_output_stage(builder, layer, layer_idx)
        layer_record["components"].extend(output_components)
        layer_record["outputs"] = output_records
        manifest["layers"].append(layer_record)

    emit_top_level_instances(builder, layers)
    return builder.render(), manifest


def export_netlist(
    params_path,
    ssm_param,
    sample_rate,
    out_path,
    json_out=None,
    state_capacitance=1e-6,
    projection_config=None,
    projection_report=None,
):
    params = load_flax_params(params_path)
    netlist, manifest = build_netlist(params, ssm_param, sample_rate, state_capacitance, projection_config=projection_config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(netlist)

    json_path = Path(json_out) if json_out else out_path.with_name(f"{out_path.stem}_components.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if projection_report is not None and "projection" in manifest:
        report_path = Path(projection_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest["projection"], indent=2, sort_keys=True))
    return out_path, json_path


def add_projection_args(parser):
    parser.add_argument("--hardware-projection", "--hardware_projection", default=PROJECTION_NONE, choices=["none", "conductance"])
    parser.add_argument("--projection-scope", "--projection_scope", default="block", choices=sorted({"global", "layer", "block", "row"}))
    parser.add_argument("--g-min", "--g_min", type=float, default=1e-6)
    parser.add_argument("--g-max", "--g_max", type=float, default=150e-6)
    parser.add_argument("--c-min", "--c_min", type=float, default=1e-12)
    parser.add_argument("--c-max", "--c_max", type=float, default=1e-6)
    parser.add_argument("--variation-sigma", "--variation_sigma", type=float, default=0.0)
    parser.add_argument("--variation-seed", "--variation_seed", type=int, default=0)
    parser.add_argument("--projection-report", "--projection_report", default=None)
    return parser


def projection_config_from_args(args):
    return HardwareProjectionConfig(
        hardware_projection=args.hardware_projection,
        projection_scope=args.projection_scope,
        g_min=args.g_min,
        g_max=args.g_max,
        c_min=args.c_min,
        c_max=args.c_max,
        variation_sigma=args.variation_sigma,
        variation_seed=args.variation_seed,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="Flax msgpack parameter file.")
    parser.add_argument("--ssm-param", required=True, choices=sorted(SUPPORTED_SSM_PARAMS))
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--state-capacitance", type=float, default=1e-6)
    parser.add_argument("--out", required=True, help="Output LTSpice .cir file.")
    parser.add_argument("--json-out", default=None, help="Optional component manifest path.")
    add_projection_args(parser)
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
        projection_config=projection_config_from_args(args),
        projection_report=args.projection_report,
    )
    print(f"Wrote LTSpice netlist: {cir_path}")
    print(f"Wrote component manifest: {json_path}")


if __name__ == "__main__":
    main()
