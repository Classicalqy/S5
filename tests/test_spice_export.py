import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_bytes

from s5.train import save_params_msgpack
from s5.ssm_parameterizations import discretize_2x2_blocks, init_RealValuedSSM
from spice.export_netlist import (
    build_netlist,
    find_ssm_modules,
    load_flax_params,
    module_to_layer,
    positive,
)
from spice.compare_transient import compare_traces
from spice.metrics import trace_metrics
from spice.export_full_model import (
    build_full_netlist,
    emit_linear_stage,
    extract_full_model,
)
from spice.export_netlist import NetlistBuilder
from spice.trace_utils import zoh_pwl_source_line
from spice.validate_digital_alignment import (
    generate_digital_alignment_artifacts,
    layer_discrete_matrices,
    simulate_full_continuous_zoh,
    simulate_full_digital,
)
from spice.validate_full_model import generate_full_validation_artifacts, simulate_full_reference
from spice.validate_ltspice_accuracy import validate_ltspice_accuracy, write_logit_only_deck
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


def _full_model_params():
    return {
        "encoder": {
            "encoder": {
                "kernel": np.array([[0.5, -0.25]], dtype=np.float32),
                "bias": np.array([0.1, -0.2], dtype=np.float32),
            },
            "layers_0": {"seq": _single_ssm_params(H=2, P=4)},
            "layers_1": {"seq": _single_ssm_params(H=2, P=4)},
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

    assert states.shape == (11, 2 * layer.n_blocks)
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


def test_full_model_reference_returns_logit_traces():
    model = extract_full_model(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    times = np.linspace(0.0, 0.01, 11)
    inputs = make_stimulus(times, input_dim=1, kind="sine", amplitude=0.1)

    traces = simulate_full_reference(model, times, inputs)

    assert traces["LOGIT0"].shape == (11,)
    assert traces["LOGIT2"].shape == (11,)
    np.testing.assert_allclose(traces["LOGIT0"][0], model.decoder_bias[0])


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


def test_full_model_digital_and_continuous_alignment_shapes():
    model = extract_full_model(_full_model_params(), "resonant_2x2", sample_rate=10.0)
    inputs = np.linspace(0.0, 0.2, 5, dtype=np.float64)[:, None]

    digital = simulate_full_digital(model, inputs, sample_rate=10.0)
    continuous = simulate_full_continuous_zoh(model, inputs, sample_rate=10.0)

    assert digital["LOGIT0"].shape == (5,)
    assert continuous["LOGIT0"].shape == (5,)
    assert digital["L0_out0"].shape == (5,)
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
    assert (tmp_path / "accuracy" / "sample_00000" / "sample_00000.cir").exists()
    assert (tmp_path / "accuracy" / "per_sample.csv").exists()


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

    assert (tmp_path / "workflow" / "netlists" / "ssm_layers.cir").exists()
    assert (tmp_path / "workflow" / "netlists" / "full_model.cir").exists()
    assert (tmp_path / "workflow" / "layer_sanity" / "summary.json").exists()
    assert (tmp_path / "workflow" / "full_alignment" / "summary.json").exists()
    assert (tmp_path / "workflow" / "accuracy" / "summary.json").exists()
    assert summary["accuracy"]["status"] == "pending"


def test_full_model_validation_artifacts_are_generated(tmp_path):
    params = _full_model_params()
    params_path = save_params_msgpack(params, tmp_path / "params.msgpack")
    cir_path = tmp_path / "full.cir"
    netlist, _ = build_full_netlist(params, "resonant_2x2", sample_rate=10.0)
    cir_path.write_text(netlist)

    metadata = generate_full_validation_artifacts(
        params_path=params_path,
        cir_path=cir_path,
        ssm_param="resonant_2x2",
        sample_rate=10.0,
        out_dir=tmp_path / "validation_full",
        duration=0.01,
        points=11,
    )

    deck = (tmp_path / "validation_full" / "full_validation.cir").read_text()
    assert ".tran" in deck and "uic" in deck
    assert "PWL(" in deck
    assert "B_RELU0_0" in deck
    assert "V(LOGIT0)" in deck
    assert metadata["logit_nodes"] == ["LOGIT0", "LOGIT1", "LOGIT2"]
