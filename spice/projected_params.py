"""Compatibility wrappers for hardware-aware Flax param projection."""

from __future__ import annotations

from .hardware_projector import (
    default_projected_params_path,
    inverse_softplus,
    project_params_tree,
    raw_from_positive,
    save_projected_params,
)


projected_params_tree = project_params_tree


__all__ = [
    "default_projected_params_path",
    "inverse_softplus",
    "project_params_tree",
    "projected_params_tree",
    "raw_from_positive",
    "save_projected_params",
]
