import jax.numpy as np
import pytest

from s5.train import _hw_calibration_enabled, _hw_calibrated_params_out, _hw_variation_aware_params_out
from s5.train_helpers import (
    analog_calibration_param_labels,
    create_hw_calibration_optimizer,
    decoder_only_param_labels,
)


class Args:
    pass


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
