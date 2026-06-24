import math

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes

from s5.train import save_params_msgpack
from s5.ssm_parameterizations import init_RealValuedSSM
from spice.export_netlist import (
    build_netlist,
    find_ssm_modules,
    load_flax_params,
    module_to_layer,
    positive,
)
from spice.compare_transient import compare_traces
from spice.validate_transient import generate_validation_artifacts, simulate_layer_reference


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
