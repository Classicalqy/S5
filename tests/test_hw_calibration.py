import subprocess
import sys

import jax
import jax.numpy as np
import pytest
import optax
from flax import linen as nn
from flax.training import train_state

from s5.train import (
    _hw_calibration_enabled,
    _hw_calibrated_params_out,
    _hw_variation_aware_epoch_sigma,
    _hw_variation_aware_eval_seed,
    _hw_variation_aware_eval_samples,
    _hw_variation_aware_nominal_fraction,
    _hw_variation_aware_nominal_train_samples,
    _hw_variation_aware_params_out,
    _hw_variation_aware_score,
    _hw_variation_aware_select_samples,
    _hw_variation_aware_select_seed,
    _hw_variation_aware_select_sigma,
    _hw_variation_aware_sigma_schedule,
    _hw_train_noise_epoch_sigma,
    _hw_train_noise_enabled,
    _hw_train_noise_samples,
    _hw_train_noise_sigma_schedule,
    _hw_variation_aware_train_seed,
    _hw_variation_aware_train_samples,
)
from s5.train_helpers import (
    analog_calibration_param_labels,
    create_hw_calibration_optimizer,
    cvar_top_mean,
    decoder_only_param_labels,
    perturb_physical_params,
    physical_noise_cvar_train_step,
    stack_variation_offsets,
    train_step,
    variation_aware_train_step,
)


class Args:
    pass


class _TinyClassifier(nn.Module):
    @nn.compact
    def __call__(self, inputs, _integration_times):
        return nn.Dense(2, use_bias=False)(inputs)


def _tiny_state(learning_rate=0.1):
    model = _TinyClassifier()
    inputs = np.ones((2, 1))
    integration_times = np.ones((2, 1))
    params = model.init(jax.random.PRNGKey(0), inputs, integration_times)["params"]
    return model, train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.sgd(learning_rate),
    )


def test_decoder_only_param_labels_train_only_top_level_decoder():
    params = {
        "encoder": {"kernel": np.ones((2, 2))},
        "layers_0": {"seq": {"B": np.ones((2, 1)), "C": np.ones((1, 2))}},
        "decoder": {"kernel": np.ones((2, 3)), "bias": np.ones((3,))},
        "decoder_aux": {"kernel": np.ones((2, 3))},
    }

    labels = decoder_only_param_labels(params)

    assert labels["decoder"]["kernel"] == "decoder"
    assert labels["decoder"]["bias"] == "decoder"
    assert labels["encoder"]["kernel"] == "frozen"
    assert labels["layers_0"]["seq"]["B"] == "frozen"
    assert labels["layers_0"]["seq"]["C"] == "frozen"
    assert labels["decoder_aux"]["kernel"] == "frozen"


def test_analog_param_labels_train_hardware_realizable_params():
    params = {
        "encoder": {
            "encoder": {"kernel": np.ones((1, 2)), "bias": np.ones((2,))},
            "layers_0": {
                "seq": {
                    "B": np.ones((2, 2)),
                    "C": np.ones((2, 2)),
                    "raw_alpha": np.ones((1,)),
                    "omega": np.ones((1,)),
                    "raw_q": np.ones((1,)),
                    "log_step": np.ones((1, 1)),
                    "Lambda_re": np.ones((1,)),
                },
                "norm": {"scale": np.ones((2,))},
                "out2": {"kernel": np.ones((2, 2))},
            },
        },
        "decoder": {"kernel": np.ones((2, 3)), "bias": np.ones((3,))},
        "decoder_aux": {"kernel": np.ones((2, 3))},
    }

    labels = analog_calibration_param_labels(params)

    assert labels["encoder"]["encoder"]["kernel"] == "analog"
    assert labels["encoder"]["encoder"]["bias"] == "analog"
    assert labels["decoder"]["kernel"] == "analog"
    assert labels["decoder"]["bias"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["B"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["C"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["raw_alpha"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["omega"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["raw_q"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["log_step"] == "analog"
    assert labels["encoder"]["layers_0"]["seq"]["Lambda_re"] == "frozen"
    assert labels["encoder"]["layers_0"]["norm"]["scale"] == "frozen"
    assert labels["encoder"]["layers_0"]["out2"]["kernel"] == "frozen"
    assert labels["decoder_aux"]["kernel"] == "frozen"


def test_hw_calibration_optimizer_rejects_unknown_mode():
    params = {"decoder": {"kernel": np.ones((2, 3))}}

    with pytest.raises(ValueError, match="Unknown hardware calibration mode"):
        create_hw_calibration_optimizer(params, 1e-4, "unknown")


def test_variation_aware_single_nominal_sample_matches_train_step():
    model, state = _tiny_state()
    inputs = np.ones((2, 1))
    labels = np.array([0, 1])
    integration_times = np.ones((2, 1))
    rng = jax.random.PRNGKey(1)

    regular_state, regular_loss = train_step(
        state, rng, inputs, labels, integration_times, model, False
    )
    aware_state, aware_loss = variation_aware_train_step(
        state,
        [rng],
        inputs,
        labels,
        integration_times,
        model,
        False,
        stack_variation_offsets(state.params, [state.params]),
    )

    assert float(aware_loss) == pytest.approx(float(regular_loss))
    for expected, actual in zip(
        jax.tree_util.tree_leaves(regular_state.params),
        jax.tree_util.tree_leaves(aware_state.params),
    ):
        assert np.allclose(expected, actual)


def test_variation_aware_batched_nominal_samples_match_train_step():
    model, state = _tiny_state()
    inputs = np.ones((2, 1))
    labels = np.array([0, 1])
    integration_times = np.ones((2, 1))
    rng = jax.random.PRNGKey(3)

    regular_state, regular_loss = train_step(
        state, rng, inputs, labels, integration_times, model, False
    )
    aware_state, aware_loss = variation_aware_train_step(
        state,
        jax.random.split(rng, 2),
        inputs,
        labels,
        integration_times,
        model,
        False,
        stack_variation_offsets(state.params, [state.params, state.params]),
    )

    assert float(aware_loss) == pytest.approx(float(regular_loss))
    for expected, actual in zip(
        jax.tree_util.tree_leaves(regular_state.params),
        jax.tree_util.tree_leaves(aware_state.params),
    ):
        assert np.allclose(expected, actual)


def test_variation_aware_step_does_not_replace_master_params_with_chip_params():
    model, state = _tiny_state(learning_rate=0.0)
    inputs = np.ones((2, 1))
    labels = np.array([0, 1])
    integration_times = np.ones((2, 1))
    chip_params = jax.tree_util.tree_map(lambda value: value + 10.0, state.params)

    updated_state, _ = variation_aware_train_step(
        state,
        [jax.random.PRNGKey(2)],
        inputs,
        labels,
        integration_times,
        model,
        False,
        stack_variation_offsets(state.params, [chip_params]),
    )

    for original, updated, chip in zip(
        jax.tree_util.tree_leaves(state.params),
        jax.tree_util.tree_leaves(updated_state.params),
        jax.tree_util.tree_leaves(chip_params),
    ):
        assert np.allclose(original, updated)
        assert not np.allclose(updated, chip)


def test_physical_noise_cvar_zero_sigma_matches_train_step():
    model, state = _tiny_state()
    inputs = np.ones((2, 1))
    labels = np.array([0, 1])
    integration_times = np.ones((2, 1))
    rng = jax.random.PRNGKey(5)

    regular_state, regular_loss = train_step(
        state, rng, inputs, labels, integration_times, model, False
    )
    aware_state, aware_loss = physical_noise_cvar_train_step(
        state,
        jax.random.split(rng, 3),
        inputs,
        labels,
        integration_times,
        model,
        False,
        0.0,
        "resonant_2x2",
        0.5,
        0.5,
    )

    assert float(aware_loss) == pytest.approx(float(regular_loss))
    for expected, actual in zip(
        jax.tree_util.tree_leaves(regular_state.params),
        jax.tree_util.tree_leaves(aware_state.params),
    ):
        assert np.allclose(expected, actual)


def test_cvar_top_mean_uses_highest_fraction():
    losses = np.array([1.0, 4.0, 2.0, 8.0])

    assert float(cvar_top_mean(losses, 0.5)) == pytest.approx(6.0)
    assert float(cvar_top_mean(losses, 0.25)) == pytest.approx(8.0)
    assert float(cvar_top_mean(losses, 1.0)) == pytest.approx(3.75)


def test_perturb_physical_params_preserves_tree_shape_and_finiteness():
    params = {
        "encoder": {
            "layers_0": {
                "seq": {
                    "B": np.ones((2, 2)),
                    "C": np.ones((2, 2)),
                    "raw_alpha": np.zeros((1,)),
                    "omega": np.ones((1,)),
                    "raw_q": np.zeros((1,)),
                    "log_step": np.ones((1, 1)),
                },
            },
        },
        "decoder": {"kernel": np.ones((2, 3))},
    }

    perturbed = perturb_physical_params(
        params,
        jax.random.PRNGKey(7),
        0.05,
        "energy_shaped_2x2",
    )

    assert jax.tree_util.tree_structure(perturbed) == jax.tree_util.tree_structure(params)
    assert all(bool(np.all(np.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(perturbed))
    assert np.allclose(perturbed["decoder"]["kernel"], params["decoder"]["kernel"])
    assert not np.allclose(perturbed["encoder"]["layers_0"]["seq"]["B"], params["encoder"]["layers_0"]["seq"]["B"])


def test_hw_calibration_gate_requires_flag_and_epochs():
    args = Args()
    args.hw_calibrate_readout = False
    args.hw_calibrate_epochs = 10
    args.hw_variation_aware_epochs = 0
    assert not _hw_calibration_enabled(args)

    args.hw_calibrate_readout = True
    args.hw_calibrate_epochs = 0
    assert not _hw_calibration_enabled(args)

    args.hw_variation_aware_epochs = 1
    assert _hw_calibration_enabled(args)

    args.hw_variation_aware_epochs = 0
    args.hw_calibrate_epochs = 1
    assert _hw_calibration_enabled(args)


def test_hw_calibrated_params_out_defaults_to_projected_path():
    args = Args()
    args.params_out = "./checkpoints/model_params.msgpack"
    args.hw_calibrated_params_out = None
    assert str(_hw_calibrated_params_out(args)).endswith("model_params_projected.msgpack")

    args.hw_calibrated_params_out = "./checkpoints/calibrated.msgpack"
    assert str(_hw_calibrated_params_out(args)).endswith("calibrated.msgpack")


def test_hw_variation_aware_params_out_defaults_beside_params_out():
    args = Args()
    args.params_out = "./checkpoints/model_params.msgpack"
    args.hw_variation_aware_params_out = None
    assert str(_hw_variation_aware_params_out(args)).endswith("model_params_variation_aware.msgpack")

    args.hw_variation_aware_params_out = "./checkpoints/aware.msgpack"
    assert str(_hw_variation_aware_params_out(args)).endswith("aware.msgpack")


def test_hw_variation_aware_samples_and_seed_schedule():
    args = Args()
    assert _hw_variation_aware_train_samples(args) == 3
    assert _hw_variation_aware_eval_samples(args) == 3

    args.hw_variation_aware_train_samples = 0
    args.hw_variation_aware_eval_samples = -2
    assert _hw_variation_aware_train_samples(args) == 1
    assert _hw_variation_aware_eval_samples(args) == 1

    assert _hw_variation_aware_train_seed(7, epoch=2, sample_index=1, train_samples=3) == 14
    assert _hw_variation_aware_eval_seed(7, epoch=2, sample_index=1, eval_samples=3) == 10014


def test_hw_variation_aware_selection_uses_fixed_held_out_configuration():
    args = Args()
    args.hw_variation_aware_sigma = 0.03
    args.hw_variation_aware_sigma_schedule = "0.03,0.05,0.1"
    args.hw_variation_aware_eval_samples = 4
    args.hw_variation_aware_select_sigma = 0.05
    args.hw_variation_aware_select_samples = None

    assert _hw_variation_aware_select_sigma(args) == pytest.approx(0.05)
    assert _hw_variation_aware_select_samples(args) == 4
    assert _hw_variation_aware_select_seed(7, 0) == 20007
    assert _hw_variation_aware_select_seed(7, 3) == 20010

    args.hw_variation_aware_select_sigma = None
    args.hw_variation_aware_select_samples = 2
    assert _hw_variation_aware_select_sigma(args) == pytest.approx(0.1)
    assert _hw_variation_aware_select_samples(args) == 2


def test_hw_variation_aware_sigma_schedule_reuses_last_value():
    args = Args()
    args.hw_variation_aware_sigma = 0.05
    assert _hw_variation_aware_sigma_schedule(args) == [0.05]
    assert _hw_variation_aware_epoch_sigma(args, 3) == pytest.approx(0.05)

    args.hw_variation_aware_sigma_schedule = "0.02,0.05 0.1"
    assert _hw_variation_aware_sigma_schedule(args) == [0.02, 0.05, 0.1]
    assert _hw_variation_aware_epoch_sigma(args, 0) == pytest.approx(0.02)
    assert _hw_variation_aware_epoch_sigma(args, 1) == pytest.approx(0.05)
    assert _hw_variation_aware_epoch_sigma(args, 4) == pytest.approx(0.1)


def test_hw_train_noise_sigma_schedule_reuses_last_value():
    args = Args()
    args.hw_train_noise_sigma = 0.0
    args.hw_train_noise_sigma_schedule = None
    args.hw_train_noise_samples = 0
    assert _hw_train_noise_sigma_schedule(args) == [0.0]
    assert not _hw_train_noise_enabled(args)
    assert _hw_train_noise_samples(args) == 1

    args.hw_train_noise_sigma_schedule = "0,0.01,0.05"
    args.hw_train_noise_samples = 4
    assert _hw_train_noise_sigma_schedule(args) == [0.0, 0.01, 0.05]
    assert _hw_train_noise_epoch_sigma(args, 0) == pytest.approx(0.0)
    assert _hw_train_noise_epoch_sigma(args, 5) == pytest.approx(0.05)
    assert _hw_train_noise_enabled(args)
    assert _hw_train_noise_samples(args) == 4


def test_hw_variation_aware_nominal_mix_clamps_and_counts_samples():
    args = Args()
    assert _hw_variation_aware_nominal_fraction(args) == 0.0
    assert _hw_variation_aware_nominal_train_samples(6, 0.0) == 0
    assert _hw_variation_aware_nominal_train_samples(6, 0.2) == 1
    assert _hw_variation_aware_nominal_train_samples(6, 0.5) == 3
    assert _hw_variation_aware_nominal_train_samples(6, 2.0) == 6

    args.hw_variation_aware_nominal_fraction = -1.0
    assert _hw_variation_aware_nominal_fraction(args) == 0.0
    args.hw_variation_aware_nominal_fraction = 2.0
    assert _hw_variation_aware_nominal_fraction(args) == 1.0


def test_hw_variation_aware_selection_uses_mean_accuracy():
    assert _hw_variation_aware_score([0.7, 0.8, 0.9], "mean_acc") == pytest.approx(0.8)
    assert _hw_variation_aware_score([0.7, 0.8, 0.9], "mean_std") == pytest.approx(0.8 - 0.5 * 0.08164966)
    assert _hw_variation_aware_score([0.7, 0.8, 0.9], "mean_std_strong") == pytest.approx(0.8 - 0.08164966)
    assert _hw_variation_aware_score([0.7, 0.8, 0.9], "min_acc") == pytest.approx(0.7)
    assert _hw_variation_aware_score([0.7, 0.8, 0.9], "p10_acc") == pytest.approx(0.72)
    with pytest.raises(ValueError, match="Unknown variation-aware selection metric"):
        _hw_variation_aware_score([0.7], "unknown_metric")


def test_run_train_exposes_variation_aware_cli_flags():
    result = subprocess.run(
        [sys.executable, "run_train.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--hw_variation_aware_train_samples" in result.stdout
    assert "--hw_variation_aware_eval_samples" in result.stdout
    assert "--hw_variation_aware_sigma_schedule" in result.stdout
    assert "--hw_variation_aware_nominal_fraction" in result.stdout
    assert "--hw_variation_aware_select_metric" in result.stdout
    assert "--hw_variation_aware_select_sigma" in result.stdout
    assert "--hw_variation_aware_select_samples" in result.stdout
    assert "--hw_variation_aware_nominal_gate" in result.stdout
    assert "--hw_variation_aware_loss" in result.stdout
    assert "--hw_variation_aware_consistency_weight" in result.stdout
    assert "--hw_variation_aware_cvar_fraction" in result.stdout
    assert "--hw_train_noise_sigma" in result.stdout
    assert "--hw_train_noise_sigma_schedule" in result.stdout
    assert "--hw_train_noise_samples" in result.stdout
    assert "--hw_train_noise_consistency_weight" in result.stdout
    assert "--hw_train_noise_cvar_fraction" in result.stdout
    assert "mean_std" in result.stdout
    assert "p10_acc" in result.stdout
