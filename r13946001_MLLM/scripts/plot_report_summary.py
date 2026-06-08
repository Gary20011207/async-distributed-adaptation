from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report figures for federated LLM/MLLM runs.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--outdir", default="figures/report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [_row(path) for path in sorted(Path(args.result_dir).glob("*_summary.json"))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        _empty(outdir / "accuracy_mean_std_by_dataset.png", "No completed runs yet")
        return
    frame.to_csv(outdir / "seeded_summary.csv", index=False)
    mean_std = frame.groupby(["dataset", "model", "method"], as_index=False).agg(
        best_acc_mean=("best_acc", "mean"),
        best_acc_std=("best_acc", "std"),
        final_acc_mean=("final_acc", "mean"),
        final_acc_std=("final_acc", "std"),
        stability_drop_mean=("stability_drop", "mean"),
        seeds=("seed", "nunique"),
    )
    mean_std.to_csv(outdir / "mean_std_summary.csv", index=False)
    _plot_accuracy(mean_std, outdir)
    _plot_stability(mean_std, outdir)
    _plot_invalid(frame, outdir)
    _plot_gaps(frame, outdir)
    _plot_method_rank(mean_std, outdir)
    _write_tables(outdir)
    print(f"wrote report figures to {outdir}")


def _row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    return {
        "dataset": config.get("dataset", "unknown"),
        "task": config.get("task", ""),
        "model": config.get("model", ""),
        "method": summary.get("method", ""),
        "seed": config.get("seed", 0),
        "budget": config.get("update_budget", summary.get("total_rounds_or_events", 0)),
        "best_acc": _float(summary.get("best_test_acc")),
        "final_acc": _float(summary.get("final_test_acc")),
        "stability_drop": _float(summary.get("best_test_acc")) - _float(summary.get("final_test_acc")),
        "invalid_answer_rate": _float(summary.get("final_invalid_answer_rate")),
        "sim_time": _float(summary.get("total_simulated_time")),
    }


def _plot_accuracy(frame: pd.DataFrame, outdir: Path) -> None:
    pivot = frame.pivot_table(index="dataset", columns="method", values="best_acc_mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5), width=0.82)
    ax.set_ylabel("Best accuracy")
    ax.set_title("Mean best accuracy by dataset/method")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_mean_std_by_dataset.png", dpi=180)
    plt.close()


def _plot_stability(frame: pd.DataFrame, outdir: Path) -> None:
    pivot = frame.pivot_table(index="dataset", columns="method", values="stability_drop_mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5), width=0.82)
    ax.set_ylabel("Best - final accuracy")
    ax.set_title("Stability drop by dataset/method")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "stability_drop_errorbar.png", dpi=180)
    plt.close()


def _plot_invalid(frame: pd.DataFrame, outdir: Path) -> None:
    pivot = frame.pivot_table(index="dataset", columns="method", values="invalid_answer_rate")
    if pivot.empty:
        _empty(outdir / "invalid_answer_rate_by_method.png", "Invalid answer rate pending")
        return
    ax = pivot.plot(kind="bar", figsize=(10, 4), width=0.82)
    ax.set_ylabel("Invalid answer rate")
    ax.set_title("Closed-ended answer validity")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "invalid_answer_rate_by_method.png", dpi=180)
    plt.close()


def _plot_gaps(frame: pd.DataFrame, outdir: Path) -> None:
    sync = frame[frame["method"] == "sync_fedavg"][["dataset", "seed", "best_acc", "final_acc"]].rename(
        columns={"best_acc": "sync_best", "final_acc": "sync_final"}
    )
    merged = frame.merge(sync, on=["dataset", "seed"], how="left")
    merged = merged[merged["method"] != "sync_fedavg"].copy()
    if merged.empty:
        _empty(outdir / "async_sync_gap_errorbar.png", "Async-Sync gap pending")
        _empty(outdir / "final_gap_errorbar.png", "Final gap pending")
        return
    merged["best_gap"] = merged["sync_best"] - merged["best_acc"]
    merged["final_gap"] = merged["sync_final"] - merged["final_acc"]
    for col, filename, title in [
        ("best_gap", "async_sync_gap_errorbar.png", "Async-Sync best gap"),
        ("final_gap", "final_gap_errorbar.png", "Async-Sync final gap"),
    ]:
        pivot = merged.pivot_table(index="dataset", columns="method", values=col)
        ax = pivot.plot(kind="bar", figsize=(11, 5), width=0.82)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Sync - Async accuracy")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def _plot_method_rank(frame: pd.DataFrame, outdir: Path) -> None:
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby("dataset")["best_acc_mean"].rank(ascending=False, method="average")
    summary = ranked.groupby("method", as_index=False)["rank"].mean().sort_values("rank")
    summary.to_csv(outdir / "method_rank_mean_std.csv", index=False)
    ax = summary.plot(kind="bar", x="method", y="rank", legend=False, figsize=(8, 4))
    ax.invert_yaxis()
    ax.set_ylabel("Average rank lower is better")
    ax.set_title("Method rank across datasets")
    plt.tight_layout()
    plt.savefig(outdir / "method_rank_mean_std.png", dpi=180)
    plt.close()


def _write_tables(outdir: Path) -> None:
    pd.DataFrame(
        [
            {"Control": "clients", "Value": "10"},
            {"Control": "local_epochs", "Value": "1"},
            {"Control": "fair budget", "Value": "async events = sync rounds x clients"},
            {"Control": "delay", "Value": "same delay mode within each comparison"},
        ]
    ).to_csv(outdir / "fairness_protocol.csv", index=False)
    pd.DataFrame(
        [
            {"Existing": "FedAvg / FedBuff / staleness-aware FL", "Ours": "Clockless LLM/MLLM adapter simulator"},
            {"Existing": "LoRA / Qwen / Qwen2.5-VL", "Ours": "Sequential client LoRA aggregation protocol"},
            {"Existing": "MMLU / MedMCQA / VQA datasets", "Ours": "Fair-budget distributed-systems metrics"},
        ]
    ).to_csv(outdir / "existing_vs_ours_table.csv", index=False)


def _empty(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
