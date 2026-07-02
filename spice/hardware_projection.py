"""Hardware-aware projection for continuous-time SSM coefficients."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


PROJECTION_NONE = "none"
PROJECTION_CONDUCTANCE = "conductance"
PROJECTION_SCOPES = {"global", "layer", "block", "row"}
DENSE_BOUNDARY_SCALE_MIN = 0.25
DENSE_BOUNDARY_SCALE_MAX = 4.0


@dataclass(frozen=True)
class HardwareProjectionConfig:
    hardware_projection: str = PROJECTION_NONE
    projection_scope: str = "block"
    g_min: float = 1e-6
    g_max: float = 150e-6
    c_min: float = 1e-12
    c_max: float = 1e-6
    variation_sigma: float = 0.0
    variation_seed: int = 0

    def validate(self):
        if self.hardware_projection not in {PROJECTION_NONE, PROJECTION_CONDUCTANCE}:
            raise ValueError("hardware_projection must be 'none' or 'conductance'.")
        if self.projection_scope not in PROJECTION_SCOPES:
            raise ValueError("projection_scope must be one of global, layer, block, row.")
        if self.g_min <= 0 or self.g_max <= 0 or self.g_min >= self.g_max:
            raise ValueError("Expected 0 < g_min < g_max.")
        if self.c_min <= 0 or self.c_max <= 0 or self.c_min > self.c_max:
            raise ValueError("Expected 0 < c_min <= c_max.")
        if self.variation_sigma < 0:
            raise ValueError("variation_sigma must be non-negative.")
        return self

    def to_dict(self):
        return {
            "hardware_projection": self.hardware_projection,
            "projection_scope": self.projection_scope,
            "g_min": float(self.g_min),
            "g_max": float(self.g_max),
            "c_min": float(self.c_min),
            "c_max": float(self.c_max),
            "variation_sigma": float(self.variation_sigma),
            "variation_seed": int(self.variation_seed),
        }


def choose_capacitance(weights, g_min, g_max, c_min, c_max):
    weights = np.abs(np.asarray(weights, dtype=np.float64))
    nonzero = weights[weights > 0.0]
    if nonzero.size == 0:
        return float(c_min)
    return _optimized_capacitance(nonzero, g_min, g_max, c_min, c_max)


def add_component_variation(values, sigma, rng):
    values = np.asarray(values, dtype=np.float64)
    if float(sigma) <= 0.0 or values.size == 0:
        return values.copy()
    varied = values * (1.0 + rng.normal(0.0, float(sigma), size=values.shape))
    return np.maximum(varied, np.finfo(np.float64).tiny)


def project_signed_weights_to_conductance(weights, c, g_min, g_max, variation_sigma=0.0, rng=None):
    """Project signed coefficients through G = C*|w| and w = sign(w)*G/C."""
    weights = np.asarray(weights, dtype=np.float64)
    projected = weights.copy()
    mask = weights != 0.0
    g_before = float(c) * np.abs(weights[mask])
    if g_before.size == 0:
        return projected, _group_stats(c, g_before, g_before, g_before, 0, 0)
    low = g_before < float(g_min)
    high = g_before > float(g_max)
    g_clipped = np.where(low, 0.0, np.minimum(g_before, float(g_max)))
    if rng is None:
        rng = np.random.default_rng(0)
    c_after = float(add_component_variation(np.asarray([float(c)]), variation_sigma, rng)[0])
    g_after = g_clipped.copy()
    nonzero_g = g_after > 0.0
    g_after[nonzero_g] = add_component_variation(g_after[nonzero_g], variation_sigma, rng)
    projected[mask] = np.sign(weights[mask]) * g_after / c_after
    stats = _group_stats(
        c_after,
        g_before,
        g_after,
        g_clipped,
        int(np.count_nonzero(g_before < float(g_min))),
        int(np.count_nonzero(g_before > float(g_max))),
    )
    return projected, stats


def project_dense_weights_to_conductance(weights, config, feedback_resistance, rng=None):
    """Project dense weights through G = |w|/Rf and w = sign(w)*G*Rf."""
    config = config.validate()
    weights = np.asarray(weights, dtype=np.float64)
    projected = weights.copy()
    mask = weights != 0.0
    g_before = np.abs(weights[mask]) / float(feedback_resistance)
    if g_before.size == 0:
        return projected, _dense_group_stats(feedback_resistance, g_before, g_before, g_before, 0, 0)

    low = g_before < float(config.g_min)
    high = g_before > float(config.g_max)
    g_clipped = np.where(low, 0.0, np.minimum(g_before, float(config.g_max)))
    if rng is None:
        rng = np.random.default_rng(0)
    g_after = g_clipped.copy()
    nonzero_g = g_after > 0.0
    g_after[nonzero_g] = add_component_variation(g_after[nonzero_g], config.variation_sigma, rng)
    projected[mask] = np.sign(weights[mask]) * g_after * float(feedback_resistance)
    stats = _dense_group_stats(
        feedback_resistance,
        g_before,
        g_after,
        g_clipped,
        int(np.count_nonzero(low)),
        int(np.count_nonzero(high)),
    )
    return projected, stats


def choose_dense_weight_scale(
    weights,
    config,
    feedback_resistance,
    scale_min=DENSE_BOUNDARY_SCALE_MIN,
    scale_max=DENSE_BOUNDARY_SCALE_MAX,
):
    """Choose a positive scale that moves dense weights toward the conductance window."""
    config = config.validate()
    weights = np.abs(np.asarray(weights, dtype=np.float64))
    weights = weights[weights > 0.0]
    if weights.size == 0:
        return 1.0, {
            "num_weights": 0,
            "scale": 1.0,
            "clip_fraction_before": 0.0,
            "clip_fraction_after": 0.0,
            "num_clipped_before": 0,
            "num_clipped_after": 0,
        }

    min_weight = float(config.g_min) * float(feedback_resistance)
    max_weight = float(config.g_max) * float(feedback_resistance)
    lower = min_weight / weights
    upper = max_weight / weights
    bounds = np.concatenate([lower, upper, np.asarray([1.0])])
    bounds = bounds[np.isfinite(bounds) & (bounds > 0.0)]
    bounds = np.unique(bounds)
    bounds.sort()
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    candidates = [1.0, scale_min, scale_max]
    candidates.extend(np.clip(bounds, scale_min, scale_max).tolist())
    if bounds.size > 1:
        candidates.extend(np.clip(np.sqrt(bounds[:-1] * bounds[1:]), scale_min, scale_max).tolist())

    def score(scale):
        scaled = weights * float(scale)
        low = scaled < min_weight
        high = scaled > max_weight
        clipped = np.clip(scaled, min_weight, max_weight)
        log_error = float(np.mean(np.abs(np.log(clipped / scaled))))
        return (
            int(np.count_nonzero(low) + np.count_nonzero(high)),
            int(np.count_nonzero(high)),
            int(np.count_nonzero(low)),
            log_error,
            abs(np.log(float(scale))),
        )

    before = score(1.0)
    best_scale = min((float(c) for c in candidates if np.isfinite(c) and c > 0.0), key=score)
    after = score(best_scale)
    return best_scale, {
        "num_weights": int(weights.size),
        "scale": float(best_scale),
        "clip_fraction_before": float(before[0] / weights.size),
        "clip_fraction_after": float(after[0] / weights.size),
        "num_clipped_before": int(before[0]),
        "num_clipped_after": int(after[0]),
    }


def project_dense_layers(dense_layers, config, feedback_resistance):
    """Project named dense kernels and return projected kernels plus a report."""
    config = config.validate()
    if config.hardware_projection == PROJECTION_NONE:
        return {name: np.asarray(kernel, dtype=np.float64).copy() for name, kernel in dense_layers}, {
            "enabled": False,
            "config": config.to_dict(),
        }

    rng = np.random.default_rng(int(config.variation_seed))
    projected = {}
    groups = []
    for name, kernel in dense_layers:
        mapped, stats = project_dense_weights_to_conductance(kernel, config, feedback_resistance, rng=rng)
        stats["name"] = name
        projected[name] = mapped
        groups.append(stats)

    return projected, {
        "enabled": True,
        "config": config.to_dict(),
        "feedback_resistance": float(feedback_resistance),
        "aggregate": _aggregate_dense_stats(config, groups, feedback_resistance),
        "groups": groups,
    }


def project_layers(layers, config):
    config = config.validate()
    if config.hardware_projection == PROJECTION_NONE:
        return tuple(layers), {"enabled": False, "config": config.to_dict()}

    rng = np.random.default_rng(int(config.variation_seed))
    projected_layers = [
        replace(
            layer,
            C=np.array(layer.C, dtype=np.float64, copy=True),
            A_tr=np.array(layer.A_tr, dtype=np.float64, copy=True),
            B_tr=np.array(layer.B_tr, dtype=np.float64, copy=True),
            capacitances=np.full((layer.n_blocks, layer.state_width), np.nan, dtype=np.float64),
        )
        for layer in layers
    ]
    rescale_report = _apply_state_rescale(projected_layers, config)
    group_stats = []

    for items in _group_items(projected_layers, config.projection_scope):
        entries = [entry for item in items for entry in item]
        weights = np.asarray([_read_entry(projected_layers, entry) for entry in entries], dtype=np.float64)
        c = choose_capacitance(weights, config.g_min, config.g_max, config.c_min, config.c_max)
        projected_items, stats = _project_items_to_conductance(
            projected_layers,
            items,
            c,
            config,
            rng,
        )
        for entry, value in projected_items:
            _write_entry(projected_layers, entry, value)
        projected_c = stats["capacitance"]
        for entry in entries:
            _write_capacitance(projected_layers, entry, projected_c)
        stats.update(_group_identity(entries))
        group_stats.append(stats)

    for layer in projected_layers:
        caps = layer.capacitances
        if np.isnan(caps).any():
            caps[np.isnan(caps)] = config.c_min

    report = {
        "enabled": True,
        "config": config.to_dict(),
        "state_rescale": rescale_report,
        "aggregate": _aggregate_stats(config, group_stats),
        "layers": _layer_stats(projected_layers, group_stats),
        "groups": group_stats,
    }
    return tuple(projected_layers), report


def _optimized_capacitance(weights, g_min, g_max, c_min, c_max):
    weights = np.abs(np.asarray(weights, dtype=np.float64))
    weights = weights[weights > 0.0]
    if weights.size == 0:
        return float(c_min)
    c_min = float(c_min)
    c_max = float(c_max)
    points = np.concatenate(
        (
            np.asarray([c_min, c_max], dtype=np.float64),
            float(g_min) / weights,
            float(g_max) / weights,
        )
    )
    points = np.unique(np.clip(points[np.isfinite(points)], c_min, c_max))
    candidates = [c_min, c_max]
    candidates.extend(points.tolist())
    if points.size > 1:
        candidates.extend(np.sqrt(points[:-1] * points[1:]).tolist())
    median_c = np.sqrt(float(g_min) * float(g_max)) / float(np.median(weights))
    median_c = float(np.clip(median_c, c_min, c_max))
    candidates.append(median_c)

    best_key = None
    best_c = median_c
    for c in candidates:
        c = float(np.clip(c, c_min, c_max))
        conductance = c * weights
        clipped = np.clip(conductance, float(g_min), float(g_max))
        clip_count = int(np.count_nonzero(conductance < float(g_min)) + np.count_nonzero(conductance > float(g_max)))
        log_error = float(np.mean(np.abs(np.log(clipped / conductance))))
        key = (clip_count, log_error, abs(np.log(c / median_c)) if median_c > 0.0 else 0.0)
        if best_key is None or key < best_key:
            best_key = key
            best_c = c
    return float(best_c)


def _apply_state_rescale(layers, config):
    records = []
    for layer_idx, layer in enumerate(layers):
        for block_idx in range(layer.n_blocks):
            block_items = _block_items(layer_idx, layer, block_idx)
            scale, before, after = _choose_state_scale(layers, block_items, config)
            if scale != 1.0:
                layer.B_tr[block_idx] *= scale
                start = block_idx * layer.state_width
                end = start + layer.state_width
                layer.C[:, start:end] /= scale
            records.append(
                {
                    "layer": int(layer_idx),
                    "block": int(block_idx),
                    "scale": float(scale),
                    "clip_fraction_before": float(before["clip_fraction"]),
                    "clip_fraction_after": float(after["clip_fraction"]),
                    "num_clipped_before": int(before["num_clipped"]),
                    "num_clipped_after": int(after["num_clipped"]),
                    "capacitance_after": float(after["capacitance"]),
                }
            )
    before_total = int(sum(record["num_clipped_before"] for record in records))
    after_total = int(sum(record["num_clipped_after"] for record in records))
    conductances = int(sum(_block_nonzero_count(layers[record["layer"]], record["block"]) for record in records))
    return {
        "enabled": True,
        "scope": "block",
        "num_blocks": int(len(records)),
        "num_conductances": conductances,
        "num_clipped_before": before_total,
        "num_clipped_after": after_total,
        "clip_fraction_before": float(before_total / conductances) if conductances else 0.0,
        "clip_fraction_after": float(after_total / conductances) if conductances else 0.0,
        "scale_min": _finite_summary([record["scale"] for record in records])["min"],
        "scale_median": _finite_summary([record["scale"] for record in records])["median"],
        "scale_max": _finite_summary([record["scale"] for record in records])["max"],
        "blocks": records,
    }


def _choose_state_scale(layers, block_items, config):
    base_values = _scaled_block_values(layers, block_items, 1.0)
    before = _clip_summary_for_values(base_values, config)
    log_grid = np.linspace(-6.0, 6.0, 601)
    best_scale = 1.0
    best = before
    best_key = (
        before["num_clipped"],
        before["log_error"],
        0.0,
    )
    for log_scale in log_grid:
        scale = float(np.exp(log_scale))
        values = _scaled_block_values(layers, block_items, scale)
        summary = _clip_summary_for_values(values, config)
        key = (
            summary["num_clipped"],
            summary["log_error"],
            abs(log_scale),
        )
        if key < best_key:
            best_key = key
            best_scale = scale
            best = summary
    return best_scale, before, best


def _scaled_block_values(layers, block_items, scale):
    values = []
    for item in block_items:
        item_values = np.asarray([_read_entry(layers, entry) for entry in item], dtype=np.float64)
        if item and item[0]["array"] == "B_tr":
            item_values = item_values * float(scale)
        values.extend(item_values.tolist())
    return np.asarray(values, dtype=np.float64)


def _clip_summary_for_values(values, config):
    values = np.abs(np.asarray(values, dtype=np.float64))
    values = values[values > 0.0]
    if values.size == 0:
        return {"num_clipped": 0, "clip_fraction": 0.0, "log_error": 0.0, "capacitance": float(config.c_min)}
    capacitance = choose_capacitance(values, config.g_min, config.g_max, config.c_min, config.c_max)
    conductance = float(capacitance) * values
    clipped = np.clip(conductance, float(config.g_min), float(config.g_max))
    num_clipped = int(np.count_nonzero(conductance < float(config.g_min)) + np.count_nonzero(conductance > float(config.g_max)))
    return {
        "num_clipped": num_clipped,
        "clip_fraction": float(num_clipped / values.size),
        "log_error": float(np.mean(np.abs(np.log(clipped / conductance)))),
        "capacitance": float(capacitance),
    }


def _block_nonzero_count(layer, block_idx):
    count = int(np.count_nonzero(layer.A_tr[block_idx]))
    count += int(np.count_nonzero(layer.B_tr[block_idx]))
    return count


def _project_items_to_conductance(layers, items, c, config, rng):
    projected_items = []
    g_before_values = []
    g_after_values = []
    g_clipped_values = []
    clipped_low = 0
    clipped_high = 0
    c_after = float(add_component_variation(np.asarray([float(c)]), config.variation_sigma, rng)[0])
    for item in items:
        values = np.asarray([_read_entry(layers, entry) for entry in item], dtype=np.float64)
        mask = values != 0.0
        if not np.any(mask):
            for entry, value in zip(item, values):
                projected_items.append((entry, float(value)))
            continue
        representative_abs = float(np.median(np.abs(values[mask])))
        g_before = float(c) * representative_abs
        if g_before < config.g_min:
            g_clipped = 0.0
            g_after = 0.0
        else:
            g_clipped = float(min(g_before, config.g_max))
            g_after = float(add_component_variation(np.asarray([g_clipped]), config.variation_sigma, rng)[0])
        for entry, value in zip(item, values):
            if value == 0.0:
                projected_items.append((entry, 0.0))
                continue
            projected_items.append((entry, float(np.sign(value) * g_after / c_after)))
            g_before_values.append(g_before)
            g_clipped_values.append(g_clipped)
            g_after_values.append(g_after)
            clipped_low += int(g_before < config.g_min)
            clipped_high += int(g_before > config.g_max)
    stats = _group_stats(
        c_after,
        np.asarray(g_before_values, dtype=np.float64),
        np.asarray(g_after_values, dtype=np.float64),
        np.asarray(g_clipped_values, dtype=np.float64),
        clipped_low,
        clipped_high,
    )
    return projected_items, stats


def _entry(layer_idx, array_name, block_idx, row_idx, col_idx):
    return {
        "layer": int(layer_idx),
        "array": array_name,
        "block": int(block_idx),
        "row": int(row_idx),
        "col": int(col_idx),
    }


def _group_items(layers, scope):
    if scope == "global":
        return [[item for layer_idx, layer in enumerate(layers) for item in _layer_items(layer_idx, layer)]]
    if scope == "layer":
        return [[item for item in _layer_items(layer_idx, layer)] for layer_idx, layer in enumerate(layers)]
    groups = []
    for layer_idx, layer in enumerate(layers):
        for block_idx in range(layer.n_blocks):
            if scope == "block":
                groups.append(_block_items(layer_idx, layer, block_idx))
            elif scope == "row":
                for row_idx in range(layer.state_width):
                    groups.append(_row_items(layer_idx, layer, block_idx, row_idx))
            else:
                raise ValueError("projection_scope must be one of global, layer, block, row.")
    return groups


def _layer_items(layer_idx, layer):
    items = []
    for block_idx in range(layer.n_blocks):
        items.extend(_block_items(layer_idx, layer, block_idx))
    return items


def _block_items(layer_idx, layer, block_idx):
    if layer.state_width == 2:
        items = [
            [
                _entry(layer_idx, "A_tr", block_idx, 0, 0),
                _entry(layer_idx, "A_tr", block_idx, 1, 1),
            ],
            [
                _entry(layer_idx, "A_tr", block_idx, 0, 1),
                _entry(layer_idx, "A_tr", block_idx, 1, 0),
            ],
        ]
    else:
        items = [[_entry(layer_idx, "A_tr", block_idx, 0, 0)]]
    for row_idx in range(layer.state_width):
        for col_idx in range(layer.input_dim):
            items.append([_entry(layer_idx, "B_tr", block_idx, row_idx, col_idx)])
    return items


def _row_items(layer_idx, layer, block_idx, row_idx):
    items = []
    for col_idx in range(layer.state_width):
        items.append([_entry(layer_idx, "A_tr", block_idx, row_idx, col_idx)])
    for col_idx in range(layer.input_dim):
        items.append([_entry(layer_idx, "B_tr", block_idx, row_idx, col_idx)])
    return items


def _read_entry(layers, entry):
    layer = layers[entry["layer"]]
    array = getattr(layer, entry["array"])
    return float(array[entry["block"], entry["row"], entry["col"]])


def _write_entry(layers, entry, value):
    layer = layers[entry["layer"]]
    array = getattr(layer, entry["array"])
    array[entry["block"], entry["row"], entry["col"]] = float(value)


def _write_capacitance(layers, entry, capacitance):
    layers[entry["layer"]].capacitances[entry["block"], entry["row"]] = float(capacitance)


def _group_identity(entries):
    layers = sorted({entry["layer"] for entry in entries})
    blocks = sorted({entry["block"] for entry in entries})
    rows = sorted({entry["row"] for entry in entries})
    return {
        "layers": layers,
        "blocks": blocks,
        "rows": rows,
        "arrays": sorted({entry["array"] for entry in entries}),
    }


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _group_stats(c, g_before, g_after, g_clipped, clipped_low, clipped_high):
    g_before = np.asarray(g_before, dtype=np.float64)
    g_after = np.asarray(g_after, dtype=np.float64)
    count = int(g_before.size)
    return {
        "capacitance": float(c),
        "num_conductances": count,
        "num_clipped_low": int(clipped_low),
        "num_clipped_high": int(clipped_high),
        "clip_fraction": float((clipped_low + clipped_high) / count) if count else 0.0,
        "g_before_min": _finite_summary(g_before)["min"],
        "g_before_max": _finite_summary(g_before)["max"],
        "g_before_median": _finite_summary(g_before)["median"],
        "g_after_min": _finite_summary(g_after)["min"],
        "g_after_max": _finite_summary(g_after)["max"],
        "g_after_median": _finite_summary(g_after)["median"],
        "g_clipped_min": _finite_summary(g_clipped)["min"],
        "g_clipped_max": _finite_summary(g_clipped)["max"],
        "g_clipped_median": _finite_summary(g_clipped)["median"],
    }


def _dense_group_stats(feedback_resistance, g_before, g_after, g_clipped, clipped_low, clipped_high):
    g_before = np.asarray(g_before, dtype=np.float64)
    g_after = np.asarray(g_after, dtype=np.float64)
    count = int(g_before.size)
    return {
        "feedback_resistance": float(feedback_resistance),
        "num_conductances": count,
        "num_clipped_low": int(clipped_low),
        "num_clipped_high": int(clipped_high),
        "clip_fraction": float((clipped_low + clipped_high) / count) if count else 0.0,
        "g_before_min": _finite_summary(g_before)["min"],
        "g_before_max": _finite_summary(g_before)["max"],
        "g_before_median": _finite_summary(g_before)["median"],
        "g_after_min": _finite_summary(g_after)["min"],
        "g_after_max": _finite_summary(g_after)["max"],
        "g_after_median": _finite_summary(g_after)["median"],
        "g_clipped_min": _finite_summary(g_clipped)["min"],
        "g_clipped_max": _finite_summary(g_clipped)["max"],
        "g_clipped_median": _finite_summary(g_clipped)["median"],
    }


def _aggregate_stats(config, groups):
    counts = np.asarray([group["num_conductances"] for group in groups], dtype=np.int64)
    total = int(np.sum(counts)) if counts.size else 0
    clipped_low = int(sum(group["num_clipped_low"] for group in groups))
    clipped_high = int(sum(group["num_clipped_high"] for group in groups))
    capacitances = [group["capacitance"] for group in groups]
    g_before = _expand_group_values(groups, "g_before", counts)
    g_after = _expand_group_values(groups, "g_after", counts)
    return {
        "g_min": float(config.g_min),
        "g_max": float(config.g_max),
        "c_min": float(config.c_min),
        "c_max": float(config.c_max),
        "variation_sigma": float(config.variation_sigma),
        "num_conductances": total,
        "num_clipped_low": clipped_low,
        "num_clipped_high": clipped_high,
        "clip_fraction": float((clipped_low + clipped_high) / total) if total else 0.0,
        "capacitance_min": _finite_summary(capacitances)["min"],
        "capacitance_max": _finite_summary(capacitances)["max"],
        "capacitance_median": _finite_summary(capacitances)["median"],
        "g_before_min": g_before["min"],
        "g_before_max": g_before["max"],
        "g_before_median": g_before["median"],
        "g_after_min": g_after["min"],
        "g_after_max": g_after["max"],
        "g_after_median": g_after["median"],
    }


def _aggregate_dense_stats(config, groups, feedback_resistance):
    counts = np.asarray([group["num_conductances"] for group in groups], dtype=np.int64)
    total = int(np.sum(counts)) if counts.size else 0
    clipped_low = int(sum(group["num_clipped_low"] for group in groups))
    clipped_high = int(sum(group["num_clipped_high"] for group in groups))
    g_before = _expand_group_values(groups, "g_before", counts)
    g_after = _expand_group_values(groups, "g_after", counts)
    return {
        "g_min": float(config.g_min),
        "g_max": float(config.g_max),
        "variation_sigma": float(config.variation_sigma),
        "feedback_resistance": float(feedback_resistance),
        "num_conductances": total,
        "num_clipped_low": clipped_low,
        "num_clipped_high": clipped_high,
        "clip_fraction": float((clipped_low + clipped_high) / total) if total else 0.0,
        "g_before_min": g_before["min"],
        "g_before_max": g_before["max"],
        "g_before_median": g_before["median"],
        "g_after_min": g_after["min"],
        "g_after_max": g_after["max"],
        "g_after_median": g_after["median"],
    }


def _expand_group_values(groups, prefix, counts):
    values = []
    for group, count in zip(groups, counts):
        median = group[f"{prefix}_median"]
        if median is not None and int(count) > 0:
            values.extend([median] * int(count))
        for key in (f"{prefix}_min", f"{prefix}_max"):
            if group[key] is not None:
                values.append(group[key])
    return _finite_summary(values)


def _layer_stats(layers, groups):
    records = []
    for layer_idx, layer in enumerate(layers):
        layer_groups = [group for group in groups if layer_idx in group.get("layers", [])]
        aggregate = _aggregate_stats_for_groups(layer_groups)
        records.append(
            {
                "index": int(layer_idx),
                "path": layer.path,
                "capacitances": np.asarray(layer.capacitances, dtype=np.float64).tolist(),
                "aggregate": aggregate,
                "groups": layer_groups,
            }
        )
    return records


def _aggregate_stats_for_groups(groups):
    counts = np.asarray([group["num_conductances"] for group in groups], dtype=np.int64)
    total = int(np.sum(counts)) if counts.size else 0
    clipped_low = int(sum(group["num_clipped_low"] for group in groups))
    clipped_high = int(sum(group["num_clipped_high"] for group in groups))
    capacitances = [group["capacitance"] for group in groups]
    g_before = _expand_group_values(groups, "g_before", counts)
    g_after = _expand_group_values(groups, "g_after", counts)
    return {
        "num_conductances": total,
        "num_clipped_low": clipped_low,
        "num_clipped_high": clipped_high,
        "clip_fraction": float((clipped_low + clipped_high) / total) if total else 0.0,
        "capacitance_min": _finite_summary(capacitances)["min"],
        "capacitance_max": _finite_summary(capacitances)["max"],
        "capacitance_median": _finite_summary(capacitances)["median"],
        "g_before_min": g_before["min"],
        "g_before_max": g_before["max"],
        "g_before_median": g_before["median"],
        "g_after_min": g_after["min"],
        "g_after_max": g_after["max"],
        "g_after_median": g_after["median"],
    }
