"""Project Flax SSM params into the hardware-aware coefficient space."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from flax.serialization import to_bytes

from .export_netlist import (
    DEFAULT_FEEDBACK_RESISTANCE,
    POSITIVE_EPS,
    SSM_PARAM_ENERGY_SHAPED_2X2,
    SSM_PARAM_REAL_DECAY,
    find_ssm_modules,
    load_flax_params,
    module_to_layer,
)
from .hardware_projection import PROJECTION_NONE, choose_dense_weight_scale, project_dense_layers, project_layers


def inverse_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def raw_from_positive(value):
    value = np.maximum(np.asarray(value, dtype=np.float64) - POSITIVE_EPS, np.finfo(np.float64).tiny)
    return inverse_softplus(value)


def project_params_tree(params, ssm_param, sample_rate, projection_config):
    params_out = deepcopy(params)
    dense_rescale_report = _apply_dense_boundary_rescale(params_out, projection_config)
    modules = find_ssm_modules(params_out)
    layers = [module_to_layer(path, module, ssm_param, sample_rate) for path, module in modules]
    projected_layers, report = project_layers(layers, projection_config)
    for (path, _module), layer in zip(modules, projected_layers):
        target = _get_mapping(params_out, path)
        _write_layer_params(target, layer, ssm_param, sample_rate)
    dense_specs = _dense_param_specs(params_out)
    if dense_specs:
        dense_layers = [(name, kernel) for name, _path, kernel in dense_specs]
        projected_dense, dense_report = project_dense_layers(
            dense_layers,
            projection_config,
            DEFAULT_FEEDBACK_RESISTANCE,
        )
        for name, path, kernel in dense_specs:
            target = _get_mapping(params_out, "/".join(path[:-1]))
            target[path[-1]] = projected_dense[name].astype(np.asarray(kernel).dtype)
        report["dense"] = dense_report
    if dense_rescale_report["enabled"]:
        report["dense_rescale"] = dense_rescale_report
    return params_out, report


def save_projected_params(params_path, ssm_param, sample_rate, projection_config, out_path):
    params = load_flax_params(params_path)
    projected, report = project_params_tree(params, ssm_param, sample_rate, projection_config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(to_bytes({"params": projected}))
    return out_path, report


def default_projected_params_path(params_out):
    path = Path(params_out)
    return path.with_name(f"{path.stem}_projected{path.suffix or '.msgpack'}")


def _get_mapping(params, path):
    node = params
    if path == "root":
        return node
    for part in path.split("/"):
        node = node[part]
    return node


def _dense_param_specs(params):
    specs = []
    for name, path in (
        ("encoder", ("encoder", "encoder", "kernel")),
        ("encoder_bias", ("encoder", "encoder", "bias")),
        ("decoder", ("decoder", "kernel")),
        ("decoder_bias", ("decoder", "bias")),
    ):
        value = _get_optional_path(params, path)
        if value is not None:
            specs.append((name, path, np.asarray(value)))
    return specs


def _get_optional_path(params, path):
    node = params
    for part in path:
        if not hasattr(node, "items") or part not in node:
            return None
        node = node[part]
    return node


def _apply_dense_boundary_rescale(params, projection_config):
    projection_config = projection_config.validate()
    if projection_config.hardware_projection == PROJECTION_NONE:
        return {
            "enabled": False,
            "feedback_resistance": float(DEFAULT_FEEDBACK_RESISTANCE),
            "num_boundaries": 0,
            "num_clipped_before": 0,
            "num_clipped_after": 0,
            "records": [],
        }

    modules = sorted(find_ssm_modules(params), key=lambda item: item[0])
    records = []
    encoder_kernel = _get_optional_path(params, ("encoder", "encoder", "kernel"))
    encoder_bias = _get_optional_path(params, ("encoder", "encoder", "bias"))
    decoder_kernel = _get_optional_path(params, ("decoder", "kernel"))

    if encoder_kernel is not None and modules:
        first_module = modules[0][1]
        first_b = first_module.get("B")
        if first_b is not None:
            encoder_kernel_rescaled = np.array(encoder_kernel, copy=True)
            encoder_bias_rescaled = np.array(encoder_bias, copy=True) if encoder_bias is not None else None
            first_b_rescaled = np.array(first_b, copy=True)
            records.extend(
                _scale_encoder_boundary(
                    encoder_kernel_rescaled,
                    encoder_bias_rescaled,
                    first_b_rescaled,
                    projection_config,
                )
            )
            _get_mapping(params, "encoder/encoder")["kernel"] = encoder_kernel_rescaled.astype(np.asarray(encoder_kernel).dtype)
            if encoder_bias is not None:
                _get_mapping(params, "encoder/encoder")["bias"] = encoder_bias_rescaled.astype(np.asarray(encoder_bias).dtype)
            first_module["B"] = first_b_rescaled.astype(np.asarray(first_b).dtype)

    if decoder_kernel is not None and modules:
        last_module = modules[-1][1]
        last_c = last_module.get("C")
        if last_c is not None:
            last_c_rescaled = np.array(last_c, copy=True)
            decoder_kernel_rescaled = np.array(decoder_kernel, copy=True)
            records.extend(
                _scale_decoder_boundary(
                    last_c_rescaled,
                    decoder_kernel_rescaled,
                    projection_config,
                )
            )
            last_module["C"] = last_c_rescaled.astype(np.asarray(last_c).dtype)
            _get_mapping(params, "decoder")["kernel"] = decoder_kernel_rescaled.astype(np.asarray(decoder_kernel).dtype)

    scales = [record["scale"] for record in records]
    return {
        "enabled": bool(records),
        "feedback_resistance": float(DEFAULT_FEEDBACK_RESISTANCE),
        "scale_min": float(np.min(scales)) if scales else None,
        "scale_median": float(np.median(scales)) if scales else None,
        "scale_max": float(np.max(scales)) if scales else None,
        "num_boundaries": int(len(records)),
        "num_clipped_before": int(sum(record["num_clipped_before"] for record in records)),
        "num_clipped_after": int(sum(record["num_clipped_after"] for record in records)),
        "records": records,
    }


def _scale_encoder_boundary(encoder_kernel, encoder_bias, first_b, projection_config):
    encoder = np.asarray(encoder_kernel)
    bias = np.asarray(encoder_bias) if encoder_bias is not None else None
    b = np.asarray(first_b)
    if encoder.ndim != 2 or b.ndim != 2 or encoder.shape[1] != b.shape[1]:
        return []
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != encoder.shape[1]):
        return []

    records = []
    for idx in range(encoder.shape[1]):
        weights = encoder[:, idx]
        if bias is not None:
            weights = np.concatenate([np.asarray(weights).reshape(-1), np.asarray([bias[idx]])])
        scale, record = choose_dense_weight_scale(
            weights,
            projection_config,
            DEFAULT_FEEDBACK_RESISTANCE,
        )
        encoder[:, idx] = encoder[:, idx] * scale
        if bias is not None:
            bias[idx] = bias[idx] * scale
        b[:, idx] = b[:, idx] / scale
        record.update({"boundary": "encoder_to_ssm0", "channel": int(idx)})
        records.append(record)
    return records


def _scale_decoder_boundary(last_c, decoder_kernel, projection_config):
    c = np.asarray(last_c)
    decoder = np.asarray(decoder_kernel)
    if c.ndim != 2 or decoder.ndim != 2 or c.shape[0] != decoder.shape[0]:
        return []

    records = []
    for idx in range(decoder.shape[0]):
        scale, record = choose_dense_weight_scale(
            decoder[idx, :],
            projection_config,
            DEFAULT_FEEDBACK_RESISTANCE,
        )
        decoder[idx, :] = decoder[idx, :] * scale
        c[idx, :] = c[idx, :] / scale
        record.update({"boundary": "ssm_last_to_decoder", "channel": int(idx)})
        records.append(record)
    return records


def _write_layer_params(module, layer, ssm_param, sample_rate):
    delta = np.asarray(layer.delta, dtype=np.float64)
    scale = float(sample_rate) * delta
    module["C"] = np.asarray(layer.C, dtype=np.asarray(module["C"]).dtype)
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
