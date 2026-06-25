"""Numeric trace comparison helpers for SPICE validation workflows."""

from __future__ import annotations

import numpy as np


def rmse(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)))


def rrmse(reference, candidate, eps=1e-12):
    reference = np.asarray(reference, dtype=np.float64)
    denom = float(np.sqrt(np.mean(reference ** 2)))
    return rmse(reference, candidate) / max(denom, float(eps))


def trace_metrics(reference, candidate, nodes):
    diffs = []
    refs = []
    for node in nodes:
        ref = np.asarray(reference[node], dtype=np.float64)
        cand = np.asarray(candidate[node], dtype=np.float64)
        refs.append(ref.reshape(-1))
        diffs.append((cand - ref).reshape(-1))
    ref_all = np.concatenate(refs)
    diff_all = np.concatenate(diffs)
    error = rmse(ref_all, ref_all + diff_all)
    return {
        "max_abs": float(np.max(np.abs(diff_all))),
        "rmse": error,
        "rrmse": rrmse(ref_all, ref_all + diff_all),
    }
