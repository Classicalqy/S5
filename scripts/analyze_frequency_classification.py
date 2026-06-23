"""Parse frequency-classification S5 comparison logs and plot summaries."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 19,
        "figure.titlesize": 24,
    }
)


RUN_RE = re.compile(
    r"^Running dataset=(\S+) split_mode=(\S+) ssm_param=(\S+) seed=(\d+)",
    re.MULTILINE,
)

SHORT_RUN_RE = re.compile(r"^Running ssm_param=(\S+) seed=(\d+)", re.MULTILINE)

CURRENT_RE = re.compile(
    r"Train Loss:\s*([0-9.]+)\s*-- Val Loss:\s*([0-9.]+)\s*--Test Loss:\s*([0-9.]+)"
    r"\s*-- Val Accuracy:\s*([0-9.]+)\s*Test Accuracy:\s*([0-9.]+)"
)

BEST_VAL_RE = re.compile(
    r"Best Val Loss:\s*([0-9.]+)\s*-- Best Val Accuracy:\s*([0-9.]+)\s*at Epoch\s*(\d+)"
)

BEST_TEST_RE = re.compile(
    r"Best Test Loss:\s*([0-9.]+)\s*-- Best Test Accuracy:\s*([0-9.]+)\s*at Epoch\s*(\d+)"
)

MODEL_LABELS = {
    "original": "S5 origin",
    "original_no_D": "S5",
    "real_decay": "Real decay",
    "resonant_2x2": "Resonant 2x2 block",
    "energy_shaped_2x2": "Energy shaped",
}

PLOT_ORDER = ["S5", "Real decay", "Resonant 2x2 block", "Energy shaped"]
COLORS = {
    "S5": "#4C78A8",
    "Real decay": "#F58518",
    "Resonant 2x2 block": "#54A24B",
    "Energy shaped": "#B279A2",
}


def _last_float(pattern: str, block: str) -> float | None:
    matches = re.findall(pattern, block)
    return float(matches[-1]) if matches else None


def parse_log(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    matches = list(RUN_RE.finditer(text))
    short_header = False
    if not matches:
        matches = list(SHORT_RUN_RE.finditer(text))
        short_header = True
    rows: list[dict[str, object]] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]

        current = CURRENT_RE.findall(block)
        best_val = BEST_VAL_RE.findall(block)
        best_test = BEST_TEST_RE.findall(block)
        if not best_val or not best_test:
            raise ValueError(f"Missing best metrics in run starting at byte {start}")

        if short_header:
            ssm_param, seed = match.groups()
            dataset_match = re.search(r"Starting S5 Training on `([^`]+)`", block)
            dataset = dataset_match.group(1) if dataset_match else "unknown"
            split_mode = "unknown"
        else:
            dataset, split_mode, ssm_param, seed = match.groups()
        best_val_loss, best_val_acc, best_val_epoch = best_val[-1]
        best_test_loss, best_test_acc, best_test_epoch = best_test[-1]

        row = {
            "dataset": dataset,
            "split_mode": split_mode,
            "ssm_param": ssm_param,
            "model": MODEL_LABELS.get(ssm_param, ssm_param),
            "seed": int(seed),
            "num_layers": _last_float(r"num_layers:\s*(\d+)", block),
            "trainable_parameters": _last_float(r"Trainable Parameters:\s*(\d+)", block),
            "total_parameters": _last_float(r"total_parameters:\s*(\d+)", block),
            "best_val_loss": float(best_val_loss),
            "best_val_accuracy": float(best_val_acc),
            "best_val_epoch": int(best_val_epoch),
            "best_test_loss": float(best_test_loss),
            "best_test_accuracy": float(best_test_acc),
            "best_test_epoch": int(best_test_epoch),
        }

        if current:
            train_loss, val_loss, test_loss, val_acc, test_acc = current[-1]
            row.update(
                {
                    "final_train_loss": float(train_loss),
                    "final_val_loss": float(val_loss),
                    "final_test_loss": float(test_loss),
                    "final_val_accuracy": float(val_acc),
                    "final_test_accuracy": float(test_acc),
                    "final_epoch": len(current),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        runs.groupby(["dataset", "num_layers", "ssm_param", "model"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_std=("best_val_accuracy", "std"),
            best_test_accuracy_mean=("best_test_accuracy", "mean"),
            best_test_accuracy_std=("best_test_accuracy", "std"),
            best_test_accuracy_min=("best_test_accuracy", "min"),
            best_test_accuracy_max=("best_test_accuracy", "max"),
            best_val_epoch_mean=("best_val_epoch", "mean"),
            trainable_parameters_mean=("trainable_parameters", "mean"),
        )
        .sort_values(["dataset", "model"])
    )
    return grouped


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def to_markdown_table(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    rows = data.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows)) if rows else len(str(header))
        for index, header in enumerate(headers)
    ]

    def fmt_row(values: list[object]) -> str:
        cells = [str(value).ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def plot_dataset_bars(summary_plot: pd.DataFrame, out_path: Path) -> None:
    datasets = list(summary_plot["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 7.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = summary_plot[summary_plot["dataset"] == dataset]
        if data.duplicated("model").any():
            data = data.groupby("model", as_index=False, observed=False).agg(
                best_test_accuracy_mean=("best_test_accuracy_mean", "mean"),
                best_test_accuracy_std=("best_test_accuracy_std", "mean"),
            )
        data = data.set_index("model").reindex(PLOT_ORDER)
        x = range(len(PLOT_ORDER))
        means = data["best_test_accuracy_mean"].to_numpy()
        stds = data["best_test_accuracy_std"].fillna(0).to_numpy()
        ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in PLOT_ORDER])
        ax.set_title(dataset.replace("ucr-", "").replace("-classification", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(PLOT_ORDER, rotation=28, ha="right")
        ymin = max(0.0, float((data["best_test_accuracy_mean"] - data["best_test_accuracy_std"]).min()) - 0.035)
        ax.set_ylim(ymin, 1.01)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(means):
            ax.text(idx, value + 0.006, f"{100 * value:.1f}", ha="center", va="bottom", fontsize=16)

    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Frequency classification: test accuracy by dataset (selected by validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_strip(runs_plot: pd.DataFrame, out_path: Path) -> None:
    datasets = list(runs_plot["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 7.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    offsets = [-0.18, -0.09, 0.0, 0.09, 0.18]
    for ax, dataset in zip(axes, datasets):
        data = runs_plot[runs_plot["dataset"] == dataset]
        for model_idx, model in enumerate(PLOT_ORDER):
            model_data = data[data["model"] == model].sort_values("seed")
            xs = [model_idx + offsets[i % len(offsets)] for i in range(len(model_data))]
            ax.scatter(xs, model_data["best_test_accuracy"], color=COLORS[model], s=34, alpha=0.85)
        ax.set_title(dataset.replace("ucr-", "").replace("-classification", ""))
        ax.set_xticks(range(len(PLOT_ORDER)))
        ax.set_xticklabels(PLOT_ORDER, rotation=28, ha="right")
        ymin = max(0.0, float(data["best_test_accuracy"].min()) - 0.035)
        ax.set_ylim(ymin, 1.01)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Test accuracy per seed")
    fig.suptitle("Frequency classification: seed-level spread (selected by validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overall(summary_plot: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    overall = (
        summary_plot.groupby("model", as_index=False, observed=False)
        .agg(
            mean_best_test_accuracy=("best_test_accuracy_mean", "mean"),
            mean_best_val_accuracy=("best_val_accuracy_mean", "mean"),
            mean_std_across_seeds=("best_test_accuracy_std", "mean"),
        )
        .set_index("model")
        .reindex(PLOT_ORDER)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(
        overall["model"],
        overall["mean_best_test_accuracy"],
        yerr=overall["mean_std_across_seeds"].fillna(0),
        capsize=4,
        color=[COLORS[m] for m in overall["model"]],
    )
    ymin = max(0.0, float(overall["mean_best_test_accuracy"].min()) - 0.045)
    ax.set_ylim(ymin, 1.0)
    ax.set_ylabel("Mean test accuracy")
    ax.set_title("Average test accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(overall["mean_best_test_accuracy"]):
        ax.text(idx, value + 0.004, f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return overall


def plot_layer_trends(summary_plot: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    for model in PLOT_ORDER:
        data = summary_plot[summary_plot["model"] == model].sort_values("num_layers")
        ax.errorbar(
            data["num_layers"],
            data["best_test_accuracy_mean"],
            yerr=data["best_test_accuracy_std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=model,
            color=COLORS[model],
        )
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Frequency classification: test accuracy vs layers")
    ymin = max(0.0, float(summary_plot["best_test_accuracy_mean"].min()) - 0.05)
    ax.set_ylim(ymin, 1.01)
    ax.set_xticks(sorted(summary_plot["num_layers"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_layer_heatmap(summary_plot: pd.DataFrame, out_path: Path) -> None:
    pivot = summary_plot.pivot_table(
        index="model",
        columns="num_layers",
        values="best_test_accuracy_mean",
        observed=False,
    ).reindex(PLOT_ORDER)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=pivot.min().min(), vmax=pivot.max().max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(layer)) for layer in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of layers")
    ax.set_title("Mean test accuracy heatmap")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{100 * value:.1f}", ha="center", va="center", color="white", fontsize=16)
    fig.colorbar(im, ax=ax, label="Test accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    summary_plot: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    display_summary = summary_plot.copy()
    for col in [
        "best_val_accuracy_mean",
        "best_val_accuracy_std",
        "best_test_accuracy_mean",
        "best_test_accuracy_std",
        "best_test_accuracy_min",
        "best_test_accuracy_max",
    ]:
        display_summary[col] = display_summary[col].map(_format_pct)

    best_by_dataset = (
        summary_plot.sort_values(["dataset", "best_test_accuracy_mean"], ascending=[True, False])
        .groupby("dataset")
        .head(1)[["dataset", "num_layers", "model", "best_test_accuracy_mean", "best_test_accuracy_std"]]
    )
    best_by_dataset = best_by_dataset.copy()
    best_by_dataset["best_test_accuracy_mean"] = best_by_dataset["best_test_accuracy_mean"].map(_format_pct)
    best_by_dataset["best_test_accuracy_std"] = best_by_dataset["best_test_accuracy_std"].map(_format_pct)

    overall_display = overall.copy()
    for col in ["mean_best_test_accuracy", "mean_best_val_accuracy", "mean_std_across_seeds"]:
        overall_display[col] = overall_display[col].map(_format_pct)

    has_multiple_layers = summary_plot["num_layers"].nunique() > 1
    figure_lines = [
        "![Dataset bars](best_test_accuracy_by_dataset.png)",
        "",
        "![Seed spread](seed_level_spread.png)",
        "",
        "![Overall average](overall_average.png)",
        "",
    ]
    if has_multiple_layers:
        figure_lines.extend(
            [
                "![Layer trend](best_test_accuracy_by_layer.png)",
                "",
                "![Layer heatmap](layer_model_heatmap.png)",
                "",
            ]
        )

    lines = [
        "# Frequency Classification Comparison",
        "",
        f"Source runs: {len(runs)}",
        f"Datasets: {', '.join(sorted(runs['dataset'].unique()))}",
        f"Seeds: {', '.join(map(str, sorted(runs['seed'].unique())))}",
        f"Layers in log: {', '.join(map(str, sorted(runs['num_layers'].dropna().astype(int).unique())))}",
        "",
        "Visualization excludes `S5 origin`; `original_no_D` is displayed as `S5`.",
        "Reported test accuracy is taken at the checkpoint selected by best validation accuracy.",
        "",
        "## Figures",
        "",
        *figure_lines,
        "## Top Model Per Dataset",
        "",
        to_markdown_table(best_by_dataset),
        "",
        "## Overall Average",
        "",
        to_markdown_table(overall_display),
        "",
        "## Dataset Summary",
        "",
        to_markdown_table(display_summary),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/frequency_classification_33869"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_log(args.log)
    summary = summarize(runs)
    runs_plot = runs[runs["ssm_param"] != "original"].copy()
    summary_plot = summary[summary["ssm_param"] != "original"].copy()
    summary_plot["model"] = pd.Categorical(summary_plot["model"], categories=PLOT_ORDER, ordered=True)
    summary_plot = summary_plot.sort_values(["dataset", "model"])
    runs_plot["model"] = pd.Categorical(runs_plot["model"], categories=PLOT_ORDER, ordered=True)
    runs_plot = runs_plot.sort_values(["dataset", "model", "seed"])

    runs.to_csv(args.out_dir / "runs.csv", index=False)
    summary.to_csv(args.out_dir / "summary_with_s5_origin.csv", index=False)
    runs_plot.to_csv(args.out_dir / "runs_for_plots_no_origin.csv", index=False)
    summary_plot.to_csv(args.out_dir / "summary_for_plots_no_origin.csv", index=False)

    plot_dataset_bars(summary_plot, args.out_dir / "best_test_accuracy_by_dataset.png")
    plot_seed_strip(runs_plot, args.out_dir / "seed_level_spread.png")
    overall = plot_overall(summary_plot, args.out_dir / "overall_average.png")
    if summary_plot["num_layers"].nunique() > 1:
        plot_layer_trends(summary_plot, args.out_dir / "best_test_accuracy_by_layer.png")
        plot_layer_heatmap(summary_plot, args.out_dir / "layer_model_heatmap.png")
    overall.to_csv(args.out_dir / "overall_average_no_origin.csv", index=False)
    write_report(args.out_dir, runs, summary, summary_plot, overall)

    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
