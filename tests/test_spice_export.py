import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_bytes

from s5.train import save_params_msgpack
from s5.ssm_parameterizations import discretize_2x2_blocks, discretize_real_decay, init_RealValuedSSM
import spice.workflow as spice_workflow
from spice.compare_transient import compare_traces
from spice.compare_transient import read_trace_table
from spice.export_full_model import (
    build_full_netlist,
    emit_linear_stage,
    extract_full_model,
)
from spice.export_netlist import (
    NetlistBuilder,
    build_netlist,
    find_ssm_modules,
    load_flax_params,
    module_to_layer,
    positive,
)
from spice.hardware_projection import (
    HardwareProjectionConfig,
    project_layers,
    quantize_conductance,
)
from spice.metrics import trace_metrics
from spice.plots import trace_line_style
from spice.trace_utils import zoh_pwl_source_line
from spice.validate_digital_alignment import (
    generate_digital_alignment_artifacts,
    layer_discrete_matrices,
    simulate_full_continuous_zoh,
    simulate_full_digital,
)
from spice.validate_ltspice_accuracy import build_margin_analysis, validate_ltspice_accuracy, write_logit_only_deck
from spice.validate_transient import generate_validation_artifacts, make_stimulus, simulate_layer_reference
from spice.workflow import run_workflow


def _single_ssm_params(ssm_param="resonant_2x2", H=2, P=4):
    cls = init_RealValuedSSM(
        H=H,
        P=P,
        ssm_param=ssm_param,
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
    )
    variables = cls().init(jax.random.PRNGKey(0), jnp.ones((8, H)))
    return variables["params"]


def _nested_params(module):
    return {
        "encoder": {
            "layers_0": {
                "seq": module,
            },
            "ignored_dense": {
                "kernel": np.ones((2, 2)),
            },
        }
    }


def _full_model_params(ssm_param="resonant_2x2"):
    return {
        "encoder": {
            "encoder": {
                "kernel": np.array([[0.5, -0.25]], dtype=np.float32),
                "bias": np.array([0.1, -0.2], dtype=np.float32),
            },
            "layers_0": {"seq": _single_ssm_params(ssm_param=ssm_param, H=2, P=4)},
            "layers_1": {"seq": _single_ssm_params(ssm_param=ssm_param, H=2, P=4)},
        },
        "decoder": {
            "kernel": np.array([[0.2, -0.3, 0.1], [-0.4, 0.5, -0.6]], dtype=np.float32),
            "bias": np.array([0.01, -0.02, 0.03], dtype=np.float32),
        },
    }


def test_find_ssm_modules_records_paths():
    params = _nested_params(_single_ssm_params())

    modules = find_ssm_modules(params)

    assert len(modules) == 1
    assert modules[0][0] == "encoder/layers_0/seq"


def test_module_to_layer_maps_continuous_time_resonant_values():
    module = {
        "B": np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float32),
        "C": np.array([[1.0, 0.0], [0.5, -0.25]], dtype=np.float32),
        "raw_alpha": np.array([0.0], dtype=np.float32),
        "omega": np.array([2.0], dtype=np.float32),
        "log_step": np.array([[math.log(0.25)]], dtype=np.float32),
    }

    layer = module_to_layer("seq", module, "resonant_2x2", sample_rate=100.0)

    alpha = positive(np.array([0.0]))[0]
    expected_scale = 25.0
    np.testing.assert_allclose(
        layer.A_tr[0],
        expected_scale * np.array([[-alpha, -2.0], [2.0, -alpha]]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        layer.B_tr[0],
        expected_scale * np.array([[1.0, -2.0], [0.0, 3.0]]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(layer.C, module["C"], rtol=1e-6)


def test_module_to_layer_maps_energy_shaped_q():
    module = {
        "B": np.ones((2, 1), dtype=np.float32),
        "C": np.ones((1, 2), dtype=np.float32),
        "raw_alpha": np.array([0.0], dtype=np.float32),
        "omega": np.array([1.0], dtype=np.float32),
        "raw_q": np.array([0.0], dtype=np.float32),
        "log_step": np.array([[0.0]], dtype=np.float32),
    }

    layer = module_to_layer("seq", module, "energy_shaped_2x2", sample_rate=10.0)

    q = positive(np.array([0.0]))[0]
    alpha = positive(np.array([0.0]))[0]
    np.testing.assert_allclose(
        layer.A_tr[0],
        10.0 * q * np.array([[-alpha, -1.0], [1.0, -alpha]]),
        rtol=1e-6,
    )


def test_module_to_layer_maps_real_decay_values():
    module = {
        "B": np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float32),
        "C": np.array([[1.0, 0.0], [0.5, -0.25]], dtype=np.float32),
        "raw_alpha": np.array([0.0, 1.0], dtype=np.float32),
        "log_step": np.array([[math.log(0.25)], [math.log(0.5)]], dtype=np.float32),
    }

    layer = module_to_layer("seq", module, "real_decay", sample_rate=100.0)

    alpha = positive(np.array([0.0, 1.0]))
    np.testing.assert_allclose(layer.A_tr[:, 0, 0], 100.0 * np.array([0.25, 0.5]) * -alpha, rtol=1e-6)
    np.testing.assert_allclose(layer.B_tr[:, 0, :], 100.0 * np.array([[0.25], [0.5]]) * module["B"], rtol=1e-6)
    assert layer.state_width == 1
    assert layer.state_dim == 2
    np.testing.assert_allclose(layer.C, module["C"], rtol=1e-6)


def test_netlist_skips_zero_weight_resistors_and_has_positive_values():
    module = {
        "B": np.array([[1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
        "C": np.array([[0.0, 1.5], [-0.5, 0.0]], dtype=np.float32),
        "raw_alpha": np.array([0.0], dtype=np.float32),
        "omega": np.array([1.0], dtype=np.float32),
        "log_step": np.array([[0.0]], dtype=np.float32),
    }
    netlist, manifest = build_netlist(
        _nested_params(module),
        ssm_param="resonant_2x2",
        sample_rate=1.0,
        state_capacitance=1e-6,
    )

    components = manifest["layers"][0]["components"]
    input_resistors = [c for c in components if c.get("role") == "input_weight"]
    output_resistors = [
        c
        for c in components
        if c.get("role") == "output_weight" and c.get("source_index") is not None
    ]

    assert len(input_resistors) == 2
    assert len(output_resistors) == 2
    assert all(c["value"] > 0 for c in components if c["kind"] in {"resistor", "capacitor"})
    assert "R_L0_B0_B0_1" not in netlist
    assert "R_L0_B0_B1_0" not in netlist


def test_real_decay_netlist_uses_single_state_blocks_without_cross_coupling():
    module = {
        "B": np.array([[1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
        "C": np.array([[0.0, 1.5], [-0.5, 0.0]], dtype=np.float32),
        "raw_alpha": np.array([0.0, 1.0], dtype=np.float32),
        "log_step": np.array([[0.0], [math.log(0.5)]], dtype=np.float32),
    }

    netlist, manifest = build_netlist(
        _nested_params(module),
        ssm_param="real_decay",
        sample_rate=1.0,
        state_capacitance=1e-6,
    )

    layer = manifest["layers"][0]
    assert ".subckt s5_L0_B0_real_decay" in netlist
    assert "omega" not in netlist
    assert layer["state_width"] == 1
    assert layer["blocks"][0]["state_nodes"] == ["L0_B0_x0"]
    assert not [c for c in layer["components"] if c.get("role") == "cross_coupling"]


def test_netlist_smoke_contains_expected_block_parts_and_unique_ids():
    params = _nested_params(_single_ssm_params())

    netlist, manifest = build_netlist(params, "resonant_2x2", sample_rate=16000.0)

    assert ".subckt s5_L0_B0_2x2" in netlist
    assert "ideal_opamp" in netlist
    assert "unity_inverter" in netlist
    assert "state_integrator" in {c["role"] for c in manifest["layers"][0]["components"] if "role" in c}
    component_ids = []
    for line in netlist.splitlines():
        if not line or line.startswith((".", "*", "+")):
            continue
        component_ids.append(line.split()[0])
    assert len(component_ids) == len(set(component_ids))


def test_state_compare_styles_digital_dashed_and_ltspice_solid():
    assert trace_line_style("exact_digital") == "--"
    assert trace_line_style("projected_digital") == "--"
    assert trace_line_style("exact_ltspice") == "-"
    assert trace_line_style("projected_ltspice") == "-"


def test_hardware_projection_none_preserves_netlist():
    params = _nested_params(_single_ssm_params())

    baseline, _ = build_netlist(params, "resonant_2x2", sample_rate=16000.0)
    projected, _ = build_netlist(
        params,
        "resonant_2x2",
        sample_rate=16000.0,
        projection_config=HardwareProjectionConfig(),
    )

    assert projected == baseline


def test_hardware_projection_wide_ranges_preserve_coefficients():
    params = _nested_params(_single_ssm_params())
    layer = module_to_layer("seq", find_ssm_modules(params)[0][1], "resonant_2x2", sample_rate=10.0)

    projected_layers, report = project_layers(
        [layer],
        HardwareProjectionConfig(
            hardware_projection="conductance",
            g_min=1e-12,
            g_max=1e12,
            c_min=1e-12,
            c_max=1e12,
            quant_bits=0,
        ),
    )

    np.testing.assert_allclose(projected_layers[0].A_tr, layer.A_tr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(projected_layers[0].B_tr, layer.B_tr, rtol=1e-12, atol=1e-12)
    assert report["aggregate"]["clip_fraction"] == 0.0


def test_quant_bits_reduce_distinct_conductance_levels():
    conductances = np.linspace(1e-6, 1e-4, 64)

    high = quantize_conductance(conductances, 1e-6, 1e-4, bits=6, mode="linear")
    low = quantize_conductance(conductances, 1e-6, 1e-4, bits=2, mode="linear")

    assert np.unique(low).size < np.unique(high).size


def test_hardware_projection_variation_seed_reproducibility():
    params = _nested_params(_single_ssm_params())
    layer = module_to_layer("seq", find_ssm_modules(params)[0][1], "resonant_2x2", sample_rate=10.0)
    cfg = HardwareProjectionConfig(
        hardware_projection="conductance",
        g_min=1e-12,
        g_max=1e12,
        c_min=1e-12,
        c_max=1e12,
        variation_sigma=0.05,
        variation_seed=7,
    )

    same_a, _ = project_layers([layer], cfg)
    same_b, _ = project_layers([layer], cfg)
    different, _ = project_layers([layer], HardwareProjectionConfig(**{**cfg.to_dict(), "variation_seed": 8}))

    np.testing.assert_allclose(same_a[0].A_tr, same_b[0].A_tr)
    assert not np.allclose(same_a[0].A_tr, different[0].A_tr)


def test_hardware_projection_state_rescale_updates_output_weights():
    module = {
        "B": np.array([[1e-4], [1e-4]], dtype=np.float32),
        "C": np.array([[2.0, -4.0]], dtype=np.float32),
        "raw_alpha": np.array([10.0], dtype=np.float32),
        "omega": np.array([1000.0], dtype=np.float32),
        "log_step": np.array([[0.0]], dtype=np.float32),
    }
    layer = module_to_layer("seq", module, "resonant_2x2", sample_rate=1.0)

    projected, report = project_layers(
        [layer],
        HardwareProjectionConfig(
            hardware_projection="conductance",
            g_min=1e-6,
            g_max=1e-4,
            c_min=1e-12,
            c_max=1e-6,
            quant_bits=0,
        ),
    )

    scale = report["state_rescale"]["blocks"][0]["scale"]
    assert scale > 1.0
    np.testing.assert_allclose(projected[0].C[:, :2], layer.C[:, :2] / scale, rtol=1e-12)


def test_projected_manifest_records_stats_and_capacitances():
    params = _nested_params(_single_ssm_params())

    netlist, manifest = build_netlist(
        params,
        "resonant_2x2",
        sample_rate=10.0,
        projection_config=HardwareProjectionConfig(
            hardware_projection="conductance",
            g_min=1e-6,
            g_max=1e-4,
            c_min=1e-9,
            c_max=1e-6,
            quant_bits=2,
            quant_mode="linear",
        ),
    )

    assert "projection" in manifest
    assert manifest["projection"]["aggregate"]["num_conductances"] > 0
    assert "state_capacitances" in manifest["layers"][0]
    assert "{CSTATE}" not in netlist
    assert "e-09" in netlist or "e-06" in netlist
    assert all(c["value"] > 0 for c in manifest["layers"][0]["components"] if c["kind"] in {"resistor", "capacitor"})


def test_load_flax_params_msgpack_file(tmp_path):
    params = _single_ssm_params()
    path = tmp_path / "params.msgpack"
    path.write_bytes(to_bytes({"params": params}))

    loaded = load_flax_params(path)

    modules = find_ssm_modules(loaded)
    assert len(modules) == 1


def test_train_save_params_msgpack_matches_exporter_format(tmp_path):
    params = _single_ssm_params()
    path = save_params_msgpack(params, tmp_path / "best_params.msgpack")

    loaded = load_flax_params(path)

    modules = find_ssm_modules(loaded)
    assert len(modules) == 1


def test_python_reference_has_expected_shapes():
    params = _nested_params(_single_ssm_params())
    layer = module_to_layer("seq", find_ssm_modules(params)[0][1], "resonant_2x2", sample_rate=10.0)
    times = np.linspace(0.0, 0.01, 11)
    inputs = np.ones((11, layer.input_dim)) * 0.1

    states, outputs = simulate_layer_reference(layer, times, inputs)

    assert states.shape == (11, layer.state_dim)
    assert outputs.shape == (11, layer.output_dim)
    np.testing.assert_allclose(states[0], 0.0)
    np.testing.assert_allclose(outputs[0], 0.0)


def test_sine_stimulus_starts_at_zero_for_ltspice_uic():
    times = np.linspace(0.0, 0.01, 11)

    inputs = make_stimulus(times, input_dim=4, kind="sine", amplitude=0.1)

    np.testing.assert_allclose(inputs[0], 0.0)


def test_zoh_pwl_source_holds_previous_sample():
    line = zoh_pwl_source_line("VIN", "IN0", 0.1, np.array([1.0, 2.0, -1.0]))

    assert line.startswith("VIN IN0 0 PWL(")
    assert "0 1" in line
    assert "0.0999999 1" in line
    assert "0.1 2" in line
    assert "0.1999999 2" in line
    assert "0.2 -1" in line


def test_validation_artifacts_are_generated(tmp_path):
    params = _nested_params(_single_ssm_params())
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")
    cir_path = tmp_path / "model.cir"
    netlist, _ = build_netlist(params, "resonant_2x2", sample_rate=10.0)
    cir_path.write_text(netlist)

    metadata = generate_validation_artifacts(
        params_path=params_path,
        cir_path=cir_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "validation",
        duration=0.01,
        points=11,
        amplitude=0.1,
    )

    validation_cir = tmp_path / "validation" / "model_layer0_validation.cir"
    reference_csv = tmp_path / "validation" / "model_layer0_reference.csv"
    assert validation_cir.exists()
    assert reference_csv.exists()
    assert ".tran" in validation_cir.read_text()
    assert "PWL(" in validation_cir.read_text()
    assert metadata["saved_nodes"]


def test_compare_traces_accepts_ltspice_style_columns(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text("time,L0_out0\n0,0\n1e-3,1\n2e-3,2\n")
    ltspice = tmp_path / "ltspice.txt"
    ltspice.write_text("time\tV(L0_out0)\n0\t0\n1e-3\t1.1\n2e-3\t1.9\n")

    results = compare_traces(reference, ltspice)

    assert set(results) == {"l0_out0"}
    np.testing.assert_allclose(results["l0_out0"]["max_abs"], 0.1)


def test_compare_traces_rejects_missing_requested_nodes(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text("time,L0_out0\n0,0\n1e-3,1\n")
    ltspice = tmp_path / "ltspice.txt"
    ltspice.write_text("time\tV(L0_out0)\n0\t0\n1e-3\t1\n")

    with pytest.raises(ValueError, match="missing"):
        compare_traces(reference, ltspice, nodes=["L0_out1"])


def test_compare_traces_can_focus_final_sample(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text("time,L0_out0\n0,0\n1e-3,1\n2e-3,2\n")
    ltspice = tmp_path / "ltspice.txt"
    ltspice.write_text("time\tV(L0_out0)\n0\t100\n1e-3\t1.1\n2e-3\t2.01\n")

    results = compare_traces(reference, ltspice, nodes=["L0_out0"], final_only=True)

    np.testing.assert_allclose(results["l0_out0"]["max_abs"], 0.01)


def test_read_trace_table_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.raw"
    empty.write_text("")

    with pytest.raises(ValueError, match="empty"):
        read_trace_table(empty)


def test_trace_metrics_include_rrmse():
    reference = {"x": np.array([1.0, 2.0, 3.0])}
    candidate = {"x": np.array([1.0, 2.0, 4.0])}

    metrics = trace_metrics(reference, candidate, ["x"])

    assert metrics["rmse"] > 0
    assert metrics["rrmse"] > 0
    np.testing.assert_allclose(metrics["max_abs"], 1.0)


def test_linear_stage_signs_and_bias_are_explicit():
    builder = NetlistBuilder()

    outputs, _ = emit_linear_stage(
        builder,
        "TEST",
        ["SRC0", "SRC1"],
        np.array([[2.0, -3.0], [-4.0, 0.0]]),
        np.array([0.5, -0.25]),
        "OUT",
    )
    netlist = builder.render()

    assert outputs == ["OUT0", "OUT1"]
    assert "XINV_TEST_0_0 SRC0 SRC0_inv_TEST_0 unity_inverter" in netlist
    assert "R_TEST_1_0 SRC0 OUT1_sum" in netlist
    assert "V_TEST_BIAS_0 OUT0_bias 0 -0.5" in netlist
    assert "V_TEST_BIAS_1 OUT1_bias 0 0.25" in netlist


def test_full_model_exporter_rejects_missing_decoder():
    params = _full_model_params()
    del params["decoder"]

    with pytest.raises(ValueError, match="decoder/kernel"):
        build_full_netlist(params, "resonant_2x2", sample_rate=10.0)


def test_full_model_exporter_only_accepts_encoder_seq_ssm_layers():
    params = _full_model_params()
    params["encoder"]["layers_0"]["not_seq"] = params["encoder"]["layers_0"].pop("seq")

    with pytest.raises(ValueError, match="exactly 2 SSM layers"):
        build_full_netlist(params, "resonant_2x2", sample_rate=10.0)


def test_full_model_manifest_records_continuous_cascade_semantics():
    netlist, manifest = build_full_netlist(_full_model_params(), "resonant_2x2", sample_rate=10.0)

    assert "continuous_cascade_without_inter_layer_sample_hold" in netlist
    assert manifest["circuit_semantics"] == "continuous_cascade_without_inter_layer_sample_hold"
    assert "activation_fn=relu" in manifest["assumptions"]


def test_digital_discrete_matrices_match_training_zoh_discretizer():
    model = extract_full_model(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    layer = model.ssm_layers[0]

    A_bar, B_bar = layer_discrete_matrices(layer)
    expected_A, expected_B = discretize_2x2_blocks(
        layer.q * layer.alpha,
        layer.q * layer.omega,
        layer.B.reshape((layer.n_blocks, 2, layer.input_dim)),
        layer.delta,
        "zoh",
    )

    np.testing.assert_allclose(A_bar, np.asarray(expected_A), rtol=1e-6)
    np.testing.assert_allclose(B_bar, np.asarray(expected_B), rtol=1e-6)


def test_real_decay_digital_discrete_matrices_match_training_zoh_discretizer():
    model = extract_full_model(_full_model_params("real_decay"), "real_decay", sample_rate=10.0)
    layer = model.ssm_layers[0]

    Lambda_bar, B_bar = layer_discrete_matrices(layer)
    expected_Lambda, expected_B = discretize_real_decay(
        layer.alpha,
        layer.B,
        layer.delta,
        "zoh",
    )

    np.testing.assert_allclose(Lambda_bar, np.asarray(expected_Lambda), rtol=1e-6)
    np.testing.assert_allclose(B_bar, np.asarray(expected_B), rtol=1e-6)


def test_full_model_digital_and_continuous_alignment_shapes():
    model = extract_full_model(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    inputs = np.linspace(0.0, 0.2, 5, dtype=np.float64)[:, None]

    digital = simulate_full_digital(model, inputs, sample_rate=10.0)
    continuous = simulate_full_continuous_zoh(model, inputs, sample_rate=10.0)

    assert digital["LOGIT0"].shape == (5,)
    assert continuous["LOGIT0"].shape == (5,)
    assert digital["L0_out0"].shape == (5,)
    np.testing.assert_allclose(digital["time"], np.arange(1, 6) / 10.0)


def test_real_decay_full_model_digital_and_continuous_alignment_shapes():
    model = extract_full_model(_full_model_params("real_decay"), "real_decay", sample_rate=10.0)
    inputs = np.linspace(0.0, 0.2, 5, dtype=np.float64)[:, None]

    digital = simulate_full_digital(model, inputs, sample_rate=10.0)
    continuous = simulate_full_continuous_zoh(model, inputs, sample_rate=10.0)

    assert digital["LOGIT0"].shape == (5,)
    assert continuous["LOGIT0"].shape == (5,)
    assert "L0_B0_x0" in digital
    assert "L0_B0_x1" not in digital
    np.testing.assert_allclose(digital["time"], np.arange(1, 6) / 10.0)


def test_digital_alignment_summary_is_pending_without_raw(tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")
    cir_path = tmp_path / "full.cir"
    netlist, _ = build_full_netlist(params, "resonant_2x2", sample_rate=10.0)
    cir_path.write_text(netlist)

    summary = generate_digital_alignment_artifacts(
        params_path=params_path,
        cir_path=cir_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "digital",
        num_samples=2,
        samples=np.zeros((2, 4, 1), dtype=np.float64),
        labels=np.array([0, 1]),
    )

    assert summary["ltspice_status"] == "pending"
    assert summary["ltspice_accuracy"] is None
    assert summary["digital_accuracy"] is not None
    assert (tmp_path / "digital" / "sample_0000" / "sample_0000.cir").exists()
    assert (tmp_path / "digital" / "per_sample.csv").exists()


def test_logit_only_accuracy_deck_saves_only_logits(tmp_path):
    model = extract_full_model(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    cir_path = tmp_path / "full.cir"
    netlist, _ = build_full_netlist(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    cir_path.write_text(netlist)

    deck_path = write_logit_only_deck(
        cir_path,
        tmp_path / "sample.cir",
        np.zeros((4, 1), dtype=np.float64),
        model,
        sample_rate=10.0,
        max_step_divisor=20,
    )
    deck = deck_path.read_text()

    save_line = next(line for line in deck.splitlines() if line.startswith(".save "))
    assert "V(LOGIT0)" in save_line
    assert "V(L0_B0_x0)" not in save_line
    assert "V(L0_out0)" not in save_line
    assert ".tran 0 0.4 0 0.005 uic" in deck


def test_ltspice_accuracy_no_run_writes_pending_summary(tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")
    cir_path = tmp_path / "full.cir"
    netlist, _ = build_full_netlist(params, "resonant_2x2", sample_rate=10.0)
    cir_path.write_text(netlist)

    summary = validate_ltspice_accuracy(
        params_path=params_path,
        cir_path=cir_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "accuracy",
        num_samples=2,
        samples=np.zeros((2, 4, 1), dtype=np.float64),
        labels=np.array([0, 1]),
        run_sim=False,
    )

    assert summary["status"] == "pending"
    assert summary["num_completed"] == 0
    assert summary["digital_accuracy"] is not None
    assert summary["comparison_semantics"] == "ltspice_continuous_cascade_vs_digital_stacked_recurrence"
    assert "ltspice_vs_digital_final_logit_max_abs" in summary
    assert "final_logit_max_abs" not in summary
    assert "margin_analysis" in summary
    assert "digital_margin" in (tmp_path / "accuracy" / "per_sample.csv").read_text()
    assert (tmp_path / "accuracy" / "sample_00000" / "sample_00000.cir").exists()
    assert (tmp_path / "accuracy" / "per_sample.csv").exists()


def test_margin_analysis_reports_buckets_and_low_margin_concentration():
    rows = {
        0: {
            "sample": 0,
            "label": 0,
            "digital_pred": 0,
            "ltspice_pred": 1,
            "status": "complete",
            "digital_logit0": "1.0",
            "digital_logit1": "0.9",
            "digital_logit2": "0.0",
            "ltspice_logit0": "0.8",
            "ltspice_logit1": "1.0",
            "ltspice_logit2": "0.0",
        },
        1: {
            "sample": 1,
            "label": 0,
            "digital_pred": 0,
            "ltspice_pred": 0,
            "status": "complete",
            "digital_logit0": "1.0",
            "digital_logit1": "0.0",
            "digital_logit2": "0.0",
            "ltspice_logit0": "1.0",
            "ltspice_logit1": "0.0",
            "ltspice_logit2": "0.0",
        },
        2: {
            "sample": 2,
            "label": 0,
            "digital_pred": 1,
            "ltspice_pred": 0,
            "status": "complete",
            "digital_logit0": "0.8",
            "digital_logit1": "1.0",
            "digital_logit2": "0.0",
            "ltspice_logit0": "1.0",
            "ltspice_logit1": "0.0",
            "ltspice_logit2": "0.0",
        },
        3: {
            "sample": 3,
            "label": 0,
            "digital_pred": 1,
            "ltspice_pred": 2,
            "status": "complete",
            "digital_logit0": "0.0",
            "digital_logit1": "2.0",
            "digital_logit2": "0.0",
            "ltspice_logit0": "0.0",
            "ltspice_logit1": "0.0",
            "ltspice_logit2": "2.0",
        },
    }

    analysis = build_margin_analysis(rows, n_classes=3)

    assert analysis["bucket_stats"]["correct_wrong"]["count"] == 1
    assert analysis["bucket_stats"]["correct_correct"]["count"] == 1
    assert analysis["bucket_stats"]["wrong_correct"]["count"] == 1
    assert analysis["bucket_stats"]["wrong_wrong"]["count"] == 1
    np.testing.assert_allclose(analysis["low_margin"]["disagreement_rate"], 1.0)
    np.testing.assert_allclose(analysis["low_margin"]["disagreement_low_margin_fraction"], 1.0 / 3.0)


def test_unified_workflow_no_run_generates_artifacts(tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")

    summary = run_workflow(
        params_path=params_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "workflow",
        full_samples=2,
        accuracy_samples=2,
        run_ltspice_enabled=False,
        samples=np.zeros((2, 4, 1), dtype=np.float64),
        labels=np.array([0, 1]),
    )

    assert (tmp_path / "workflow" / "netlist" / "original" / "ssm_layers.cir").exists()
    assert (tmp_path / "workflow" / "netlist" / "original" / "full_model.cir").exists()
    assert (tmp_path / "workflow" / "netlist" / "original" / "params.msgpack").exists()
    assert (tmp_path / "workflow" / "layer_sanity" / "original" / "summary.json").exists()
    assert (tmp_path / "workflow" / "full_alignment" / "original" / "summary.json").exists()
    assert (tmp_path / "workflow" / "accuracy" / "original" / "summary.json").exists()
    assert summary["accuracy"]["status"] == "pending"


def test_workflow_attempts_full_alignment_plots_for_each_sample(monkeypatch, tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")
    calls = []

    def fake_plot(alignment_dir, model, sample_idx):
        calls.append(sample_idx)
        return {"state_plot": f"sample_{sample_idx}.png"}, []

    monkeypatch.setattr(spice_workflow, "_plot_full_alignment_sample", fake_plot)

    summary = run_workflow(
        params_path=params_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "workflow_plots",
        full_samples=2,
        accuracy_samples=2,
        run_ltspice_enabled=False,
        samples=np.zeros((2, 4, 1), dtype=np.float64),
        labels=np.array([0, 1]),
    )

    assert calls == [0, 1]
    assert sorted(summary["full_alignment_plots"]["plots"]) == ["sample_0000", "sample_0001"]
    assert (tmp_path / "workflow_plots" / "full_alignment" / "original" / "per_layer_node_rrmse.csv").exists()


def test_workflow_treats_empty_raw_as_not_ready(monkeypatch, tmp_path):
    deck = tmp_path / "sample.cir"
    raw = tmp_path / "sample.raw"
    deck.write_text(".end\n")
    raw.write_text("")
    calls = []

    def fake_run_ltspice(_ltspice_bin, deck_path):
        calls.append(deck_path)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(spice_workflow, "run_ltspice", fake_run_ltspice)

    raw_path, status = spice_workflow._maybe_run_ltspice(deck, "ltspice", True)

    assert raw_path == raw
    assert status == "pending"
    assert calls == [deck]
    assert not raw.exists()


def test_workflow_hardware_projection_sweep_writes_outputs(tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")

    summary = run_workflow(
        params_path=params_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "hardware",
        hardware_projection=True,
        quant_bits=[2],
        variation_sigma=[0.0],
        g_min=1e-6,
        g_max=1e-4,
        c_min=1e-9,
        c_max=1e-6,
        run_ltspice_enabled=False,
        samples=np.zeros((2, 4, 1), dtype=np.float64),
        labels=np.array([0, 1]),
        full_samples=2,
        accuracy_samples=2,
    )

    case_dir = tmp_path / "hardware" / "netlist" / "q2_v0"
    assert (case_dir / "params.msgpack").exists()
    assert (case_dir / "ssm_layers.cir").exists()
    assert (case_dir / "full_model.cir").exists()
    assert (case_dir / "projection_report.json").exists()
    projected = load_flax_params(case_dir / "params.msgpack")
    original_modules = find_ssm_modules(params)
    projected_modules = find_ssm_modules(projected)
    assert [path for path, _ in projected_modules] == [path for path, _ in original_modules]
    for (_path, original), (_projected_path, mapped) in zip(original_modules, projected_modules):
        assert set(mapped) == set(original)
        for key in ("B", "C", "raw_alpha", "log_step", "omega"):
            if key in original:
                assert np.asarray(mapped[key]).shape == np.asarray(original[key]).shape
    assert (tmp_path / "hardware" / "layer_sanity" / "q2_v0" / "summary.json").exists()
    assert (tmp_path / "hardware" / "full_alignment" / "q2_v0" / "summary.json").exists()
    assert (tmp_path / "hardware" / "accuracy" / "q2_v0" / "summary.json").exists()
    assert summary["projection_runs"][0]["case"] == "q2_v0"
    assert summary["projection_runs"][0]["accuracy"]["source"] == "full_alignment"
