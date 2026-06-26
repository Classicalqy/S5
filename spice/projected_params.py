"""Map projected continuous-time SSM coefficients back to Flax params."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from flax.serialization import to_bytes

from .export_netlist import (
    POSITIVE_EPS,
    SSM_PARAM_ENERGY_SHAPED_2X2,
    SSM_PARAM_REAL_DECAY,
    find_ssm_modules,
    load_flax_params,
    module_to_layer,
)
from .hardware_projection import project_layers


def inverse_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def raw_from_positive(value):
    value = np.maximum(np.asarray(value, dtype=np.float64) - POSITIVE_EPS, np.finfo(np.float64).tiny)
    return inverse_softplus(value)


def projected_params_tree(params, ssm_param, sample_rate, projection_config):
    params_out = deepcopy(params)
    modules = find_ssm_modules(params)
    layers = [module_to_layer(path, module, ssm_param, sample_rate) for path, module in modules]
    projected_layers, report = project_layers(layers, projection_config)
    for (path, _module), layer in zip(modules, projected_layers):
        target = _get_mapping(params_out, path)
        _write_layer_params(target, layer, ssm_param, sample_rate)
    return params_out, report


def save_projected_params(params_path, ssm_param, sample_rate, projection_config, out_path):
    params = load_flax_params(params_path)
    projected, report = projected_params_tree(params, ssm_param, sample_rate, projection_config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(to_bytes({"params": projected}))
    return out_path, report


def _get_mapping(params, path):
    node = params
    if path == "root":
        return node
    for part in path.split("/"):
        node = node[part]
    return node


def _write_layer_params(module, layer, ssm_param, sample_rate):
    delta = np.asarray(layer.delta, dtype=np.float64)
    scale = float(sample_rate) * delta
    if ssm_param == SSM_PARAM_REAL_DECAY:
        alpha = np.maximum(-layer.A_tr[:, 0, 0] / scale, POSITIVE_EPS)
        module["raw_alpha"] = raw_from_positive(alpha).astype(np.asarray(module["raw_alpha"]).dtype)
        module["B"] = (layer.B_tr[:, 0, :] / scale[:, None]).astype(np.asarray(module["B"]).dtype)
        return

    q = np.asarray(layer.q, dtype=np.float64)
    decay = np.maximum((-layer.A_tr[:, 0, 0] - layer.A_tr[:, 1, 1]) / (2.0 * scale), POSITIVE_EPS)
    frequency = (layer.A_tr[:, 1, 0] - layer.A_tr[:, 0, 1]) / (2.0 * scale)
    alpha = np.maximum(decay / q, POSITIVE_EPS)
    omega = frequency / q
    module["raw_alpha"] = raw_from_positive(alpha).astype(np.asarray(module["raw_alpha"]).dtype)
    module["omega"] = omega.astype(np.asarray(module["omega"]).dtype)
    if ssm_param == SSM_PARAM_ENERGY_SHAPED_2X2 and "raw_q" in module:
        module["raw_q"] = np.asarray(module["raw_q"])
    module["B"] = (layer.B_tr / scale[:, None, None]).reshape(layer.B.shape).astype(np.asarray(module["B"]).dtype)
