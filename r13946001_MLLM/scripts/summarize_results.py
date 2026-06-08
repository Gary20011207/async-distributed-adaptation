from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize federated LLM/MLLM runs.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--out", default="REPORT_NOTES.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = [_load_summary(path) for path in sorted(result_dir.glob("*_summary.json"))]
    rows = [row for row in rows if not str(row.get("csv", "")).startswith("/tmp/")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(rows), encoding="utf-8")
    print(f"wrote {out} rows={len(rows)}")


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    csv_path = Path(summary.get("csv_path", ""))
    if not csv_path.is_absolute():
        csv_path = path.parent / csv_path.name
    metrics = _csv_metrics(csv_path)
    return {
        "dataset": config.get("dataset", _dataset_from_name(path.name)),
        "task": config.get("task", ""),
        "model": config.get("model", ""),
        "method": summary.get("method", ""),
        "seed": config.get("seed", ""),
        "partition": config.get("partition", ""),
        "budget": config.get("update_budget", _budget(summary, config)),
        "best_acc": _to_float(summary.get("best_test_acc")),
        "final_acc": _to_float(summary.get("final_test_acc")),
        "final_loss": _to_float(summary.get("final_test_loss")),
        "stability_drop": _to_float(summary.get("best_test_acc")) - _to_float(summary.get("final_test_acc")),
        "invalid_answer_rate": _to_float(summary.get("final_invalid_answer_rate")),
        "sim_time": _to_float(summary.get("total_simulated_time")),
        "avg_staleness": metrics["avg_staleness"],
        "p95_staleness": metrics["p95_staleness"],
        "avg_alpha": metrics["avg_alpha"],
        "avg_agreement": metrics["avg_agreement"],
        "client_gini": metrics["client_gini"],
        "csv": str(csv_path),
    }


def _csv_metrics(path: Path) -> dict[str, float]:
    empty = {"avg_staleness": float("nan"), "p95_staleness": float("nan"), "avg_alpha": float("nan"), "avg_agreement": float("nan"), "client_gini": float("nan")}
    if not path.exists() or path.stat().st_size == 0:
        return empty
    frame = pd.read_csv(path)
    for column in ["staleness", "effective_alpha", "mean_agreement", "agreement", "client_id"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    staleness = frame["staleness"].dropna() if "staleness" in frame else pd.Series(dtype=float)
    alpha = frame["effective_alpha"].dropna() if "effective_alpha" in frame else pd.Series(dtype=float)
    agreement_col = "mean_agreement" if "mean_agreement" in frame else "agreement"
    agreement = frame[agreement_col].dropna() if agreement_col in frame else pd.Series(dtype=float)
    counts = frame.dropna(subset=["client_id"]).groupby("client_id").size() if "client_id" in frame else pd.Series(dtype=float)
    return {
        "avg_staleness": float(staleness.mean()) if not staleness.empty else float("nan"),
        "p95_staleness": float(staleness.quantile(0.95)) if not staleness.empty else float("nan"),
        "avg_alpha": float(alpha.mean()) if not alpha.empty else float("nan"),
        "avg_agreement": float(agreement.mean()) if not agreement.empty else float("nan"),
        "client_gini": _gini(counts.to_numpy(dtype=float)) if not counts.empty else float("nan"),
    }


def _render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Federated Medical LLM/MLLM Experiment Notes",
        "",
        "## Project Question",
        "",
        "Can a clockless asynchronous federated server adapt LLM/MLLM adapters for closed-ended medical QA/VQA while approaching Sync FedAvg under the same update budget?",
        "",
        "## Current Scope",
        "",
        "- Text MCQA: MMLU and MedMCQA adapters.",
        "- Medical VQA: closed-ended VQA-RAD / PathVQA-style adapters.",
        "- Methods: Sync FedAvg, Naive Async, Staleness Async, FedBuff, CAA-FedBuff, CAA-v2.",
        "- Fairness rule: async events = sync rounds x clients.",
        "",
        "## Existing vs Ours",
        "",
        "| Existing | Ours |",
        "|---|---|",
        "| FedAvg, staleness-aware async, FedBuff | Clockless event-driven simulator for LLM/MLLM adapters |",
        "| LoRA/QLoRA, Qwen, Qwen2.5-VL | Sequential client trainer that aggregates adapter weights only |",
        "| MMLU, MedMCQA, VQA-RAD, PathVQA | Fair-budget async-vs-sync protocol and CAA-v2 system metrics |",
        "",
        "## Completed Run Summary",
        "",
        "| Dataset | Task | Model | Method | Seed | Budget | Best | Final | Drop | Invalid | Avg stale | P95 stale | Gini |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['task']} | {row['model']} | {row['method']} | {row['seed']} | {_fmt(row['budget'],0)} | {_fmt(row['best_acc'])} | {_fmt(row['final_acc'])} | {_fmt(row['stability_drop'])} | {_fmt(row['invalid_answer_rate'])} | {_fmt(row['avg_staleness'])} | {_fmt(row['p95_staleness'])} | {_fmt(row['client_gini'])} |"
        )
    lines.extend(_mean_std_table(rows))
    lines.extend([
        "",
        "## Interpretation Template",
        "",
        "- Naive Async tests pure no-barrier throughput but can amplify stale/conflicting adapter updates.",
        "- Staleness Async is safer but may be conservative when useful late updates are downweighted too strongly.",
        "- FedBuff reduces update noise by batching arrivals but does not inspect update direction.",
        "- CAA-v2 adds agreement with buffered/server trajectory direction and client fairness credit while remaining clockless.",
        "",
    ])
    return "\n".join(lines)


def _mean_std_table(rows: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"], row["method"]), []).append(row)
    lines = ["", "## Mean ± Std by Dataset/Method", "", "| Dataset | Model | Method | Seeds | Best Acc | Final Acc | Stability Drop |", "|---|---|---|---:|---:|---:|---:|"]
    for (dataset, model, method), group in sorted(groups.items()):
        best = [row["best_acc"] for row in group]
        final = [row["final_acc"] for row in group]
        drop = [row["stability_drop"] for row in group]
        lines.append(
            f"| {dataset} | {model} | {method} | {len(group)} | {_mean_std(best)} | {_mean_std(final)} | {_mean_std(drop)} |"
        )
    return lines


def _mean_std(values: list[float]) -> str:
    values = [value for value in values if value == value]
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.4f} ± 0.0000"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _budget(summary: dict[str, Any], config: dict[str, Any]) -> float:
    if summary.get("method") == "sync_fedavg":
        return float(config.get("rounds", 0)) * float(config.get("clients", 1))
    return float(config.get("events", summary.get("total_rounds_or_events", 0)))


def _dataset_from_name(name: str) -> str:
    parts = name.split("_")
    return parts[1] if len(parts) > 2 else "unknown"


def _gini(values) -> float:
    values = [float(v) for v in values if float(v) >= 0]
    if not values or sum(values) <= 0:
        return float("nan")
    sorted_values = sorted(values)
    n = len(sorted_values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    return (2 * weighted) / (n * sum(sorted_values)) - (n + 1) / n


if __name__ == "__main__":
    main()
