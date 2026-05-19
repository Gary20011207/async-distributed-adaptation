from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "sync_fedavg",
    "naive_async",
    "staleness_async",
    "fedbuff_async",
    "agreement_fedbuff_async",
    "caa_fedbuff_v2",
]
METHOD_LABELS = {
    "sync_fedavg": "Sync",
    "naive_async": "Naive",
    "staleness_async": "Staleness",
    "fedbuff_async": "FedBuff",
    "agreement_fedbuff_async": "CAA",
    "caa_fedbuff_v2": "CAA-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed FL results.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--outdir", default="figures/report")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--partition", default="iid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(path) for path in sorted(Path(args.result_dir).glob("*_summary.json"))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No summary files found.")

    frame = frame[(frame["partition"] == args.partition) & (frame["model"] == args.model)]
    frame = frame[frame["method"].isin(METHOD_ORDER)]
    if frame.empty:
        raise SystemExit("No matching summary rows found.")

    seeded = _select_seed_representatives(frame)
    seeded.to_csv(outdir / "seeded_summary.csv", index=False)
    mean_std = _mean_std(seeded)
    mean_std.to_csv(outdir / "mean_std_summary.csv", index=False)
    _write_fairness_protocol(outdir)
    _write_existing_vs_ours(outdir)

    _plot_accuracy_mean_std(mean_std, outdir)
    _plot_gap_errorbar(seeded, outdir, value="best_gap", filename="async_sync_gap_errorbar.png")
    _plot_gap_errorbar(seeded, outdir, value="final_gap", filename="final_gap_errorbar.png")
    _plot_stability_drop(seeded, outdir)
    _plot_method_rank(seeded, outdir)
    _plot_dataset_method_heatmap(seeded, outdir)
    print(f"wrote seeded summary plots to {outdir}")


def _summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    method = summary.get("method", "")
    return {
        "dataset": _dataset_from_summary(path, method, config),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": int(config.get("seed", 42)),
        "model": str(config.get("model", "resnet18")),
        "partition": str(config.get("partition", "iid") or "iid"),
        "budget": _update_budget(summary, config),
        "best_acc": _to_float(summary.get("best_test_acc")),
        "final_acc": _to_float(summary.get("final_test_acc")),
        "best_step": _to_float(summary.get("best_round_or_event")),
        "progress": _to_float(summary.get("total_rounds_or_events")),
        "path": str(path),
    }


def _select_seed_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["method_rank"] = frame["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    frame["stability_drop"] = frame["best_acc"] - frame["final_acc"]
    frame = frame.sort_values(
        ["dataset", "method", "seed", "model", "budget", "best_acc", "final_acc", "stability_drop"],
        ascending=[True, True, True, True, True, False, False, True],
    )
    return frame.groupby(["dataset", "method", "seed", "model"], as_index=False).first()


def _mean_std(seeded: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seeded.groupby(["dataset", "method", "method_label", "model", "partition"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            budget=("budget", "max"),
            best_acc_mean=("best_acc", "mean"),
            best_acc_std=("best_acc", "std"),
            final_acc_mean=("final_acc", "mean"),
            final_acc_std=("final_acc", "std"),
            stability_drop_mean=("stability_drop", "mean"),
            stability_drop_std=("stability_drop", "std"),
        )
    )
    for column in ["best_acc_std", "final_acc_std", "stability_drop_std"]:
        grouped[column] = grouped[column].fillna(0.0)
    return grouped


def _plot_accuracy_mean_std(mean_std: pd.DataFrame, outdir: Path) -> None:
    if mean_std.empty:
        return
    datasets = sorted(mean_std["dataset"].unique())
    methods = [method for method in METHOD_ORDER if method in set(mean_std["method"])]
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    _new_figure(max(9.0, len(datasets) * 1.1 + 4.0), 5.0)
    for idx, method in enumerate(methods):
        subset = mean_std[mean_std["method"] == method].set_index("dataset")
        means = [subset.loc[d, "best_acc_mean"] if d in subset.index else np.nan for d in datasets]
        stds = [subset.loc[d, "best_acc_std"] if d in subset.index else 0.0 for d in datasets]
        plt.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )
    plt.ylabel("Best accuracy mean +/- std")
    plt.xlabel("Dataset")
    plt.title("Multi-seed best accuracy")
    plt.xticks(x, datasets, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_mean_std_by_dataset.png", dpi=180)
    plt.close()


def _plot_gap_errorbar(seeded: pd.DataFrame, outdir: Path, *, value: str, filename: str) -> None:
    merged = _with_sync_gaps(seeded)
    if merged.empty:
        return
    grouped = (
        merged.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=(value, "mean"), std=(value, "std"))
        .fillna({"std": 0.0})
    )
    _plot_grouped_errorbar(
        grouped,
        outdir / filename,
        ylabel=value.replace("_", " "),
        title=value.replace("_", " ").title(),
    )


def _plot_stability_drop(seeded: pd.DataFrame, outdir: Path) -> None:
    grouped = (
        seeded.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=("stability_drop", "mean"), std=("stability_drop", "std"))
        .fillna({"std": 0.0})
    )
    _plot_grouped_errorbar(
        grouped,
        outdir / "stability_drop_errorbar.png",
        ylabel="Best acc - final acc",
        title="Stability Drop",
    )


def _plot_method_rank(seeded: pd.DataFrame, outdir: Path) -> None:
    ranked = seeded.copy()
    ranked["rank"] = ranked.groupby(["dataset", "seed"])["best_acc"].rank(
        method="min",
        ascending=False,
    )
    grouped = (
        ranked.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=("rank", "mean"), std=("rank", "std"))
        .fillna({"std": 0.0})
    )
    _plot_grouped_errorbar(
        grouped,
        outdir / "method_rank_mean_std.png",
        ylabel="Rank, lower is better",
        title="Method Rank Mean +/- Std",
    )


def _plot_dataset_method_heatmap(seeded: pd.DataFrame, outdir: Path) -> None:
    if seeded.empty:
        return
    grouped = (
        seeded.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=("best_acc", "mean"))
    )
    pivot = grouped.pivot(index="dataset", columns="method_label", values="mean")
    ordered = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in pivot.columns]
    pivot = pivot[ordered]
    if pivot.empty:
        return

    _new_figure(max(8.5, 1.2 * len(pivot.columns) + 4.0), max(4.8, 0.45 * len(pivot.index) + 2.8))
    ax = plt.gca()
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(image, ax=ax, label="Best accuracy mean")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    ax.set_title("Dataset x Method Accuracy Heatmap")
    for row_idx, dataset in enumerate(pivot.index):
        for col_idx, method in enumerate(pivot.columns):
            value = pivot.loc[dataset, method]
            if pd.notna(value):
                color = "white" if value < 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(outdir / "dataset_method_heatmap.png", dpi=180)
    plt.close()


def _plot_grouped_errorbar(grouped: pd.DataFrame, path: Path, *, ylabel: str, title: str) -> None:
    datasets = sorted(grouped["dataset"].unique())
    methods = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in set(grouped["method_label"])]
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    _new_figure(max(9.0, len(datasets) * 1.1 + 4.0), 5.0)
    for idx, label in enumerate(methods):
        subset = grouped[grouped["method_label"] == label].set_index("dataset")
        means = [subset.loc[d, "mean"] if d in subset.index else np.nan for d in datasets]
        stds = [subset.loc[d, "std"] if d in subset.index else 0.0 for d in datasets]
        plt.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
        )
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")
    plt.title(title)
    plt.xticks(x, datasets, rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _with_sync_gaps(seeded: pd.DataFrame) -> pd.DataFrame:
    sync = seeded[seeded["method"] == "sync_fedavg"][
        ["dataset", "seed", "model", "best_acc", "final_acc"]
    ].rename(columns={"best_acc": "sync_best", "final_acc": "sync_final"})
    merged = seeded[seeded["method"] != "sync_fedavg"].merge(
        sync,
        on=["dataset", "seed", "model"],
        how="inner",
    )
    if merged.empty:
        return merged
    merged["best_gap"] = merged["sync_best"] - merged["best_acc"]
    merged["final_gap"] = merged["sync_final"] - merged["final_acc"]
    return merged


def _write_fairness_protocol(outdir: Path) -> None:
    rows = [
        ("clients", "10"),
        ("local_epochs", "1"),
        ("batch_size", "128"),
        ("lr", "0.01"),
        ("lr_scheduler", "cosine"),
        ("min_lr", "0.0001"),
        ("augment", "true"),
        ("partition", "iid unless explicitly marked dirichlet"),
        ("async_delay_mode", "heterogeneous"),
        ("fair_budget", "async events = sync rounds * clients"),
        ("seed", "controls split, client partition, delay sampling, and init"),
    ]
    pd.DataFrame(rows, columns=["control", "value"]).to_csv(
        outdir / "fairness_protocol.csv",
        index=False,
    )


def _write_existing_vs_ours(outdir: Path) -> None:
    rows = [
        ("Sync FedAvg", "existing baseline", "Barrier aggregation; server waits for every client in each round."),
        ("Naive Async", "existing baseline", "Applies each arriving client update immediately with constant alpha."),
        ("Staleness-aware decay", "existing baseline", "Uses logical version gap to reduce stale update impact."),
        ("FedBuff-style buffering", "existing baseline", "Aggregates a buffer of asynchronous updates instead of one update at a time."),
        ("MedMNIST benchmark", "existing benchmark", "Medical image classification datasets used to evaluate the distributed-learning setting."),
        ("ResNet18 / MobileNetV3", "existing backbone", "Standard image classifiers used as model backbones."),
        ("Clockless simulator and logging", "our implementation", "Event-driven async simulator with logical versions, simulated time, CSV summaries, and plots."),
        ("CAA agreement weighting", "our design", "Weights buffered deltas by direction agreement without using a physical global clock."),
        ("CAA-v2 server trajectory EMA", "our design", "Compares updates with recent accepted server direction to reject conflicting movement."),
        ("CAA-v2 client fairness credit", "our design", "Reduces domination by frequently arriving fast clients using only client ids and contribution counts."),
        ("Fair-budget analysis pipeline", "our implementation", "Compares sync rounds and async events under the same client-update budget with multi-seed reports."),
    ]
    pd.DataFrame(rows, columns=["component", "source", "role"]).to_csv(
        outdir / "existing_vs_ours_table.csv",
        index=False,
    )


def _dataset_from_summary(path: Path, method: str, config: dict[str, Any]) -> str:
    if config.get("dataset"):
        return str(config["dataset"])
    name = path.name
    prefix = f"{method}_"
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        parts = rest.split("_")
        if len(parts) >= 4:
            return "_".join(parts[:-3])
    return "pathmnist"


def _update_budget(summary: dict[str, Any], config: dict[str, Any]) -> float | None:
    budget = _to_float(config.get("update_budget"))
    if budget is not None:
        return budget
    progress = _to_float(summary.get("total_rounds_or_events"))
    if progress is None:
        return None
    if summary.get("method") == "sync_fedavg":
        clients = _to_float(config.get("clients")) or 10.0
        return progress * clients
    return progress


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _new_figure(width: float, height: float) -> None:
    plt.figure(figsize=(width, height))


if __name__ == "__main__":
    main()
