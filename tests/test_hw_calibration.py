import jax.numpy as np

from s5.train import _hw_calibration_enabled, _hw_calibrated_params_out
from s5.train_helpers import decoder_only_param_labels


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


def test_hw_calibration_gate_requires_flag_and_epochs():
    args = Args()
    args.hw_calibrate_readout = False
    args.hw_calibrate_epochs = 10
    assert not _hw_calibration_enabled(args)

    args.hw_calibrate_readout = True
    args.hw_calibrate_epochs = 0
    assert not _hw_calibration_enabled(args)

    args.hw_calibrate_epochs = 1
    assert _hw_calibration_enabled(args)


def test_hw_calibrated_params_out_defaults_to_projected_path():
    args = Args()
    args.params_out = "./checkpoints/model_params.msgpack"
    args.hw_calibrated_params_out = None
    assert str(_hw_calibrated_params_out(args)).endswith("model_params_projected.msgpack")

    args.hw_calibrated_params_out = "./checkpoints/calibrated.msgpack"
    assert str(_hw_calibrated_params_out(args)).endswith("calibrated.msgpack")
