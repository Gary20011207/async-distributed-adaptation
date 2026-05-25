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

CORE_DATASETS = {"pathmnist", "pneumoniamnist", "bloodmnist", "organamnist"}
STRESS_DATASETS = {"pathmnist", "bloodmnist", "organamnist"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create distributed-systems focused plots.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--outdir", default="figures/report")
    parser.add_argument("--model", default="resnet18")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_summary_row(path) for path in sorted(Path(args.result_dir).glob("*_summary.json"))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        _write_empty(outdir)
        print(f"wrote empty distributed-systems summary to {outdir}")
        return

    frame = frame[frame["model"] == args.model]
    frame = frame[frame["method"].isin(METHOD_ORDER)]
    if frame.empty:
        _write_empty(outdir)
        print(f"wrote empty distributed-systems summary to {outdir}")
        return

    frame = _select_representatives(frame)
    frame.to_csv(outdir / "distributed_systems_summary.csv", index=False)

    _write_ablation_components(frame, outdir)
    _plot_non_iid_gap(frame, outdir)
    _plot_non_iid_stability(frame, outdir)
    _plot_straggler_staleness(frame, outdir)
    _plot_straggler_acc_vs_time(frame, outdir)
    _plot_client_contribution_gini(frame, outdir)
    _plot_time_to_accuracy(frame, outdir)
    _plot_ablation(frame, outdir, metric="best_acc", filename="caa_v2_ablation_best_acc.png")
    _plot_ablation(
        frame,
        outdir,
        metric="stability_drop",
        filename="caa_v2_ablation_stability_drop.png",
    )
    print(f"wrote distributed-systems plots to {outdir}")


def _summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    method = str(summary.get("method", ""))
    dataset = _dataset_from_summary(path, method, config)
    csv_path = Path(str(summary.get("csv_path", "")))
    if not csv_path.is_absolute():
        csv_path = path.parent / csv_path.name
    csv_metrics = _csv_metrics(csv_path)
    best_acc = _to_float(summary.get("best_test_acc"))
    final_acc = _to_float(summary.get("final_test_acc"))
    progress = _to_float(summary.get("total_rounds_or_events"))
    clients = _to_float(config.get("clients")) or 10.0
    budget = _update_budget(summary, config, progress, clients)
    partition = str(config.get("partition", "iid") or "iid")
    delay_label = _delay_label(config)
    variant = _variant_label(method, config)

    return {
        "summary_path": str(path),
        "csv_path": str(csv_path),
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "variant": variant,
        "seed": int(config.get("seed", 42)),
        "model": str(config.get("model", "resnet18")),
        "partition": partition,
        "dirichlet_alpha": _to_float(config.get("dirichlet_alpha")),
        "delay_label": delay_label,
        "delay_mode": str(config.get("delay_mode", "")),
        "straggler_ratio": _to_float(config.get("straggler_ratio")),
        "straggler_multiplier": _to_float(config.get("straggler_multiplier")),
        "update_budget": budget,
        "sync_equivalent_rounds": _to_float(config.get("sync_equivalent_rounds")),
        "best_acc": best_acc,
        "final_acc": final_acc,
        "stability_drop": _safe_sub(best_acc, final_acc),
        "best_step": _to_float(summary.get("best_round_or_event")),
        "progress": progress,
        "total_simulated_time": _to_float(summary.get("total_simulated_time")),
        "avg_staleness": csv_metrics["avg_staleness"],
        "p95_staleness": csv_metrics["p95_staleness"],
        "max_staleness": csv_metrics["max_staleness"],
        "avg_effective_alpha": csv_metrics["avg_effective_alpha"],
        "avg_buffer_alpha": csv_metrics["avg_buffer_alpha"],
        "avg_agreement": csv_metrics["avg_agreement"],
        "avg_server_momentum_agreement": csv_metrics["avg_server_momentum_agreement"],
        "avg_fairness_weight": csv_metrics["avg_fairness_weight"],
        "client_contribution_gini": csv_metrics["client_contribution_gini"],
        "client_min_max_ratio": csv_metrics["client_min_max_ratio"],
        "time_to_80pct_best_acc": csv_metrics["time_to_80pct_best_acc"],
        "time_to_90pct_best_acc": csv_metrics["time_to_90pct_best_acc"],
    }


def _csv_metrics(path: Path) -> dict[str, float | None]:
    empty = {
        "avg_staleness": None,
        "p95_staleness": None,
        "max_staleness": None,
        "avg_effective_alpha": None,
        "avg_buffer_alpha": None,
        "avg_agreement": None,
        "avg_server_momentum_agreement": None,
        "avg_fairness_weight": None,
        "client_contribution_gini": None,
        "client_min_max_ratio": None,
        "time_to_80pct_best_acc": None,
        "time_to_90pct_best_acc": None,
    }
    if not path.exists():
        return empty
    frame = pd.read_csv(path)
    for column in [
        "staleness",
        "effective_alpha",
        "buffer_alpha",
        "mean_agreement",
        "agreement",
        "server_momentum_agreement",
        "fairness_weight",
        "client_id",
        "test_acc",
        "simulated_time",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    metrics = dict(empty)
    metrics["avg_staleness"] = _mean(frame, "staleness")
    metrics["p95_staleness"] = _quantile(frame, "staleness", 0.95)
    metrics["max_staleness"] = _max(frame, "staleness")
    metrics["avg_effective_alpha"] = _mean(frame, "effective_alpha")
    metrics["avg_buffer_alpha"] = _mean(frame, "buffer_alpha")
    metrics["avg_agreement"] = _mean(frame, "mean_agreement")
    if metrics["avg_agreement"] is None:
        metrics["avg_agreement"] = _mean(frame, "agreement")
    metrics["avg_server_momentum_agreement"] = _mean(frame, "server_momentum_agreement")
    metrics["avg_fairness_weight"] = _mean(frame, "fairness_weight")
    metrics["client_contribution_gini"], metrics["client_min_max_ratio"] = _client_imbalance(frame)
    metrics["time_to_80pct_best_acc"] = _time_to_fraction_of_best(frame, 0.8)
    metrics["time_to_90pct_best_acc"] = _time_to_fraction_of_best(frame, 0.9)
    return metrics


def _select_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["method_rank"] = data["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    sort_columns = [
        "dataset",
        "method",
        "variant",
        "seed",
        "model",
        "partition",
        "dirichlet_alpha",
        "delay_label",
        "update_budget",
        "best_acc",
        "final_acc",
        "stability_drop",
    ]
    data = data.sort_values(
        sort_columns,
        ascending=[True, True, True, True, True, True, True, True, True, False, False, True],
        na_position="last",
    )
    group_columns = [
        "dataset",
        "method",
        "variant",
        "seed",
        "model",
        "partition",
        "dirichlet_alpha",
        "delay_label",
        "update_budget",
    ]
    return data.groupby(group_columns, as_index=False, dropna=False).first()


def _plot_non_iid_gap(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[(frame["partition"] == "dirichlet") & (frame["dataset"].isin(CORE_DATASETS))]
    data = data[data["method"].isin(METHOD_ORDER)]
    merged = _with_sync_reference(data)
    if merged.empty:
        return
    grouped = (
        merged.groupby(["dataset", "dirichlet_alpha", "method_label"], as_index=False)
        .agg(mean=("best_gap", "mean"), std=("best_gap", "std"))
        .fillna({"std": 0.0})
    )
    grouped["x_label"] = grouped["dataset"] + "\nalpha=" + grouped["dirichlet_alpha"].map(_fmt_alpha)
    _grouped_bar(
        grouped,
        outdir / "non_iid_async_sync_gap.png",
        value="mean",
        error="std",
        x="x_label",
        hue="method_label",
        ylabel="Sync best - async best",
        title="Non-IID Async-Sync Best Gap",
    )


def _plot_non_iid_stability(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[(frame["partition"] == "dirichlet") & (frame["dataset"].isin(CORE_DATASETS))]
    if data.empty:
        return
    grouped = (
        data.groupby(["dataset", "dirichlet_alpha", "method_label"], as_index=False)
        .agg(mean=("stability_drop", "mean"), std=("stability_drop", "std"))
        .fillna({"std": 0.0})
    )
    grouped["x_label"] = grouped["dataset"] + "\nalpha=" + grouped["dirichlet_alpha"].map(_fmt_alpha)
    _grouped_bar(
        grouped,
        outdir / "non_iid_stability_drop.png",
        value="mean",
        error="std",
        x="x_label",
        hue="method_label",
        ylabel="Best acc - final acc",
        title="Non-IID Stability Drop",
    )


def _plot_straggler_staleness(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[
        (frame["partition"] == "iid")
        & (frame["dataset"].isin(STRESS_DATASETS))
        & (frame["method"].isin(["naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"]))
    ]
    if data.empty:
        return
    grouped = (
        data.groupby(["delay_label", "method_label"], as_index=False)
        .agg(mean=("p95_staleness", "mean"), std=("p95_staleness", "std"))
        .fillna({"std": 0.0})
    )
    _grouped_bar(
        grouped,
        outdir / "straggler_staleness_distribution.png",
        value="mean",
        error="std",
        x="delay_label",
        hue="method_label",
        ylabel="p95 staleness",
        title="Staleness Under Delay Stress",
    )


def _plot_straggler_acc_vs_time(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[
        (frame["partition"] == "iid")
        & (frame["dataset"].isin(STRESS_DATASETS))
        & (frame["method"].isin(["naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"]))
    ]
    if data.empty:
        return
    grouped = (
        data.groupby(["delay_label", "method_label"], as_index=False)
        .agg(mean_time=("time_to_90pct_best_acc", "mean"), std=("time_to_90pct_best_acc", "std"))
        .fillna({"std": 0.0})
    )
    _grouped_bar(
        grouped,
        outdir / "straggler_acc_vs_simulated_time.png",
        value="mean_time",
        error="std",
        x="delay_label",
        hue="method_label",
        ylabel="Simulated time to 90% of best",
        title="Accuracy Progress Under Delay Stress",
    )


def _plot_client_contribution_gini(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[
        (frame["method"].isin(["naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"]))
        & frame["client_contribution_gini"].notna()
    ]
    if data.empty:
        return
    grouped = (
        data.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=("client_contribution_gini", "mean"), std=("client_contribution_gini", "std"))
        .fillna({"std": 0.0})
    )
    _grouped_bar(
        grouped,
        outdir / "client_contribution_gini.png",
        value="mean",
        error="std",
        x="dataset",
        hue="method_label",
        ylabel="Client contribution Gini",
        title="Fast-Client Contribution Imbalance",
    )


def _plot_time_to_accuracy(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[frame["time_to_90pct_best_acc"].notna()]
    data = data[data["method"].isin(["naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"])]
    if data.empty:
        return
    grouped = (
        data.groupby(["dataset", "method_label"], as_index=False)
        .agg(mean=("time_to_90pct_best_acc", "mean"), std=("time_to_90pct_best_acc", "std"))
        .fillna({"std": 0.0})
    )
    _grouped_bar(
        grouped,
        outdir / "time_to_accuracy.png",
        value="mean",
        error="std",
        x="dataset",
        hue="method_label",
        ylabel="Simulated time to 90% of best",
        title="Time-to-Accuracy",
    )


def _plot_ablation(frame: pd.DataFrame, outdir: Path, *, metric: str, filename: str) -> None:
    data = frame[frame["variant"].isin(_ablation_order())]
    data = data[data["dataset"].isin({"pathmnist", "bloodmnist", "organamnist"})]
    if data.empty:
        return
    grouped = (
        data.groupby(["dataset", "variant"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
        .fillna({"std": 0.0})
    )
    _grouped_bar(
        grouped,
        outdir / filename,
        value="mean",
        error="std",
        x="dataset",
        hue="variant",
        ylabel=metric.replace("_", " "),
        title=f"CAA-v2 Ablation: {metric.replace('_', ' ').title()}",
        hue_order=_ablation_order(),
    )


def _write_ablation_components(frame: pd.DataFrame, outdir: Path) -> None:
    data = frame[frame["variant"].isin(_ablation_order())].copy()
    if data.empty:
        pd.DataFrame().to_csv(outdir / "caa_v2_ablation_components.csv", index=False)
        return
    grouped = (
        data.groupby(["dataset", "variant"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            best_acc_mean=("best_acc", "mean"),
            best_acc_std=("best_acc", "std"),
            final_acc_mean=("final_acc", "mean"),
            stability_drop_mean=("stability_drop", "mean"),
            avg_agreement=("avg_agreement", "mean"),
            avg_server_momentum_agreement=("avg_server_momentum_agreement", "mean"),
            avg_fairness_weight=("avg_fairness_weight", "mean"),
        )
        .fillna(0.0)
    )
    grouped.to_csv(outdir / "caa_v2_ablation_components.csv", index=False)


def _with_sync_reference(data: pd.DataFrame) -> pd.DataFrame:
    sync = data[data["method"] == "sync_fedavg"][
        ["dataset", "seed", "model", "partition", "dirichlet_alpha", "update_budget", "best_acc", "final_acc"]
    ].rename(columns={"best_acc": "sync_best", "final_acc": "sync_final"})
    async_rows = data[data["method"] != "sync_fedavg"]
    if sync.empty or async_rows.empty:
        return pd.DataFrame()
    merged = async_rows.merge(
        sync,
        on=["dataset", "seed", "model", "partition", "dirichlet_alpha", "update_budget"],
        how="inner",
    )
    if merged.empty:
        return merged
    merged["best_gap"] = merged["sync_best"] - merged["best_acc"]
    merged["final_gap"] = merged["sync_final"] - merged["final_acc"]
    return merged


def _grouped_bar(
    data: pd.DataFrame,
    path: Path,
    *,
    value: str,
    error: str,
    x: str,
    hue: str,
    ylabel: str,
    title: str,
    hue_order: list[str] | None = None,
) -> None:
    if data.empty:
        return
    x_values = list(dict.fromkeys(data[x].astype(str).tolist()))
    if hue_order is None:
        hue_values = list(dict.fromkeys(data[hue].astype(str).tolist()))
    else:
        available = set(data[hue].astype(str))
        hue_values = [item for item in hue_order if item in available]
    positions = np.arange(len(x_values))
    width = 0.82 / max(len(hue_values), 1)
    plt.figure(figsize=(max(9.0, len(x_values) * 1.2 + 4.0), 5.2))
    for idx, hue_value in enumerate(hue_values):
        subset = data[data[hue].astype(str) == hue_value].copy()
        subset["_x"] = subset[x].astype(str)
        subset = subset.set_index("_x")
        means = [subset.loc[item, value] if item in subset.index else np.nan for item in x_values]
        stds = [subset.loc[item, error] if item in subset.index else 0.0 for item in x_values]
        plt.bar(
            positions + (idx - (len(hue_values) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=hue_value,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions, x_values, rotation=25, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_empty(outdir: Path) -> None:
    pd.DataFrame().to_csv(outdir / "distributed_systems_summary.csv", index=False)
    pd.DataFrame().to_csv(outdir / "caa_v2_ablation_components.csv", index=False)


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


def _update_budget(
    summary: dict[str, Any],
    config: dict[str, Any],
    progress: float | None,
    clients: float,
) -> float | None:
    budget = _to_float(config.get("update_budget"))
    if budget is not None:
        return budget
    if progress is None:
        return None
    if summary.get("method") == "sync_fedavg":
        return progress * clients
    return progress


def _delay_label(config: dict[str, Any]) -> str:
    mode = str(config.get("delay_mode", "none") or "none")
    if mode == "heterogeneous":
        ratio = _to_float(config.get("straggler_ratio"))
        multiplier = _to_float(config.get("straggler_multiplier"))
        if ratio is not None and multiplier is not None:
            return f"hetero_r{ratio:g}_x{multiplier:g}"
    if mode == "lognormal":
        mean = _to_float(config.get("lognormal_mean"))
        sigma = _to_float(config.get("lognormal_sigma"))
        if mean is not None and sigma is not None:
            return f"lognormal_m{mean:g}_s{sigma:g}"
    return mode


def _variant_label(method: str, config: dict[str, Any]) -> str:
    if method == "agreement_fedbuff_async":
        return "old_caa"
    if method != "caa_fedbuff_v2":
        return method
    alpha = _to_float(config.get("alpha"))
    adaptive_min = _to_float(config.get("adaptive_alpha_min"))
    adaptive_max = _to_float(config.get("adaptive_alpha_max"))
    if _to_float(config.get("history_agreement_blend")) == 0.0:
        return "no_server_ema"
    if _to_float(config.get("client_fairness_power")) == 0.0:
        return "no_fairness"
    clip = _to_float(config.get("delta_clip_multiplier"))
    if clip is not None and clip >= 1e5:
        return "no_clipping"
    if (
        alpha is not None
        and adaptive_min is not None
        and adaptive_max is not None
        and abs(adaptive_min - alpha) < 1e-9
        and abs(adaptive_max - alpha) < 1e-9
    ):
        return "static_alpha"
    return "full_caa_v2"


def _ablation_order() -> list[str]:
    return [
        "full_caa_v2",
        "no_server_ema",
        "no_fairness",
        "no_clipping",
        "static_alpha",
        "old_caa",
    ]


def _client_imbalance(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if "client_id" not in frame.columns:
        return None, None
    values = frame["client_id"].dropna()
    if values.empty:
        return None, None
    counts = values.astype(int).value_counts().sort_index().to_numpy(dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return None, None
    gini = _gini(counts)
    min_count = counts.min()
    max_count = counts.max()
    ratio = min_count / max_count if max_count > 0 else None
    return gini, ratio


def _gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values)
    n = len(sorted_values)
    if n == 0:
        return 0.0
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n)


def _time_to_fraction_of_best(frame: pd.DataFrame, fraction: float) -> float | None:
    if "test_acc" not in frame.columns or "simulated_time" not in frame.columns:
        return None
    evals = frame.dropna(subset=["test_acc", "simulated_time"])
    if evals.empty:
        return None
    best = float(evals["test_acc"].max())
    threshold = fraction * best
    reached = evals[evals["test_acc"] >= threshold]
    if reached.empty:
        return None
    return float(reached["simulated_time"].iloc[0])


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return float(values.mean()) if not values.empty else None


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return float(values.quantile(q)) if not values.empty else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return float(values.max()) if not values.empty else None


def _to_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _fmt_alpha(value: Any) -> str:
    number = _to_float(value)
    return "na" if number is None else f"{number:g}"


if __name__ == "__main__":
    main()
