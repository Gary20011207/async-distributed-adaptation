from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    "naive_async": "Naive async",
    "staleness_async": "Staleness async",
    "fedbuff_async": "FedBuff-lite",
    "agreement_fedbuff_async": "CAA-FedBuff",
    "caa_fedbuff_v2": "CAA-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create slide-friendly summary plots.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--outdir", default="figures/report")
    parser.add_argument("--partition", default="iid")
    parser.add_argument("--model", default="resnet18")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_summary_row(path) for path in sorted(result_dir.glob("*_summary.json"))]
    rows = [
        row
        for row in rows
        if row["partition"] in ("", args.partition) and row["model"] == args.model
    ]
    best_rows = _best_rows(rows)
    if best_rows.empty:
        raise SystemExit("No summary rows found.")

    best_rows.to_csv(outdir / "best_method_summary.csv", index=False)
    _plot_accuracy(best_rows, outdir, "best_acc", "best_accuracy_by_dataset.png", "Best Accuracy")
    _plot_accuracy(best_rows, outdir, "final_acc", "final_accuracy_by_dataset.png", "Final Accuracy")
    _plot_async_sync_gap(best_rows, outdir)
    _plot_stability_drop(best_rows, outdir)
    _plot_caa_gap(best_rows, outdir)
    print(f"wrote report plots to {outdir}")


def _summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    method = summary.get("method", "")
    return {
        "dataset": _dataset_from_summary(path, method, config),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "model": str(config.get("model", "resnet18")),
        "partition": config.get("partition", "iid") or "iid",
        "budget": _update_budget(summary, config),
        "best_acc": _to_float(summary.get("best_test_acc")),
        "final_acc": _to_float(summary.get("final_test_acc")),
        "best_step": _to_float(summary.get("best_round_or_event")),
        "progress": _to_float(summary.get("total_rounds_or_events")),
        "sim_time": _to_float(summary.get("total_simulated_time")),
        "path": str(path),
    }


def _best_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    candidates = [row for row in rows if row["method"] in METHOD_ORDER and row["best_acc"] is not None]
    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates)
    frame["method_rank"] = frame["method"].map({method: index for index, method in enumerate(METHOD_ORDER)})
    frame = frame.sort_values(
        ["dataset", "method_rank", "best_acc", "final_acc"],
        ascending=[True, True, False, False],
    )
    return frame.groupby(["dataset", "method"], as_index=False).first()


def _plot_accuracy(frame: pd.DataFrame, outdir: Path, column: str, filename: str, title: str) -> None:
    pivot = _method_pivot(frame, column)
    if pivot.empty:
        return
    _new_figure(max(8.0, 1.1 * len(pivot.index) + 4.0), 4.8)
    pivot.plot(kind="bar", ax=plt.gca(), width=0.82)
    plt.ylim(0.0, min(1.0, max(0.6, float(pivot.max().max()) + 0.08)))
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=180)
    plt.close()


def _plot_async_sync_gap(frame: pd.DataFrame, outdir: Path) -> None:
    sync = frame[frame["method"] == "sync_fedavg"][["dataset", "best_acc", "final_acc"]].rename(
        columns={"best_acc": "sync_best", "final_acc": "sync_final"}
    )
    async_rows = frame[frame["method"] != "sync_fedavg"].merge(sync, on="dataset", how="inner")
    if async_rows.empty:
        return
    async_rows["best_gap"] = async_rows["sync_best"] - async_rows["best_acc"]
    pivot = async_rows.pivot(index="dataset", columns="method_label", values="best_gap")
    ordered = [METHOD_LABELS[method] for method in METHOD_ORDER[1:] if METHOD_LABELS[method] in pivot.columns]
    pivot = pivot[ordered]
    _new_figure(max(8.0, 1.1 * len(pivot.index) + 4.0), 4.8)
    pivot.plot(kind="bar", ax=plt.gca(), width=0.82)
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.xlabel("Dataset")
    plt.ylabel("Sync best - async best")
    plt.title("Async-Sync Best Gap")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "async_sync_best_gap_by_dataset.png", dpi=180)
    plt.close()


def _plot_stability_drop(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame.copy()
    data["stability_drop"] = data["best_acc"] - data["final_acc"]
    pivot = _method_pivot(data, "stability_drop")
    if pivot.empty:
        return
    _new_figure(max(8.0, 1.1 * len(pivot.index) + 4.0), 4.8)
    pivot.plot(kind="bar", ax=plt.gca(), width=0.82)
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.xlabel("Dataset")
    plt.ylabel("Best acc - final acc")
    plt.title("Stability Drop")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "stability_drop_by_dataset.png", dpi=180)
    plt.close()


def _plot_caa_gap(frame: pd.DataFrame, outdir: Path) -> None:
    sync = frame[frame["method"] == "sync_fedavg"][["dataset", "best_acc", "final_acc"]].rename(
        columns={"best_acc": "sync_best", "final_acc": "sync_final"}
    )
    caa_family = frame[frame["method"].isin({"agreement_fedbuff_async", "caa_fedbuff_v2"})].copy()
    if caa_family.empty:
        return
    caa_family["family_rank"] = caa_family["method"].map(
        {"agreement_fedbuff_async": 0, "caa_fedbuff_v2": 1}
    )
    caa_family = caa_family.sort_values(
        ["dataset", "best_acc", "final_acc", "family_rank"],
        ascending=[True, False, False, False],
    ).groupby("dataset", as_index=False).first()
    caa = caa_family.merge(sync, on="dataset", how="inner")
    if caa.empty:
        return
    caa["best_gap"] = caa["sync_best"] - caa["best_acc"]
    caa["final_gap"] = caa["sync_final"] - caa["final_acc"]
    caa["stability_drop"] = caa["best_acc"] - caa["final_acc"]
    plot_frame = caa.set_index("dataset")[["best_gap", "final_gap", "stability_drop"]]
    plot_frame = plot_frame.rename(
        columns={
            "best_gap": "Best gap",
            "final_gap": "Final gap",
            "stability_drop": "Stability drop",
        }
    )
    _new_figure(max(8.0, 1.1 * len(plot_frame.index) + 4.0), 4.8)
    plot_frame.plot(kind="bar", ax=plt.gca(), width=0.82)
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy difference")
    plt.title("Best CAA-Family Gap and Stability")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "caa_gap_and_stability_by_dataset.png", dpi=180)
    plt.close()


def _method_pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    pivot = frame.pivot(index="dataset", columns="method_label", values=column)
    ordered = [METHOD_LABELS[method] for method in METHOD_ORDER if METHOD_LABELS[method] in pivot.columns]
    return pivot[ordered]


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
