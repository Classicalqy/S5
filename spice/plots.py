"""Plot helpers for SPICE validation reports."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_tmpdir = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", str(Path(_tmpdir) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trace_line_style(label):
    label = str(label).lower()
    if "digital" in label:
        return "--"
    if "ltspice" in label:
        return "-"
    return "-"


def plot_trace_overlay(path, times, reference, candidate, nodes, reference_label="reference", candidate_label="ltspice"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, len(nodes))
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count), sharex=True)
    if count == 1:
        axes = [axes]
    for ax, node in zip(axes, nodes):
        ax.plot(times, reference[node], label=reference_label, linewidth=1.8)
        ax.plot(times, candidate[node], "--", label=candidate_label, linewidth=1.4)
        ax.set_ylabel(node)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_logit_bar(path, digital_logits, ltspice_logits=None, title="Final logits"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digital_logits = np.asarray(digital_logits, dtype=np.float64)
    x = np.arange(digital_logits.shape[0])
    width = 0.35 if ltspice_logits is not None else 0.6
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - width / 2, digital_logits, width=width, label="digital")
    if ltspice_logits is not None:
        ax.bar(x + width / 2, np.asarray(ltspice_logits, dtype=np.float64), width=width, label="ltspice")
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("logit")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_multi_trace_overlay(path, times, traces, nodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, len(nodes))
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count), sharex=True)
    if count == 1:
        axes = [axes]
    for ax, node in zip(axes, nodes):
        for label, trace in traces.items():
            if node in trace:
                ax.plot(times, trace[node], trace_line_style(label), label=label, linewidth=1.4)
        ax.set_ylabel(node)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_multi_logit_bar(path, logits_by_label, title="Final logits"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(logits_by_label)
    logits = [np.asarray(logits_by_label[label], dtype=np.float64) for label in labels]
    if not logits:
        return None
    x = np.arange(logits[0].shape[0])
    width = min(0.8 / len(logits), 0.25)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    offsets = (np.arange(len(logits)) - (len(logits) - 1) / 2.0) * width
    for offset, label, values in zip(offsets, labels, logits):
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("logit")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
