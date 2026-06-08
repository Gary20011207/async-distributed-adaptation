from __future__ import annotations

import argparse
import html
import json
import math
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
    "naive_async": "Naive Async",
    "staleness_async": "Staleness",
    "fedbuff_async": "FedBuff",
    "agreement_fedbuff_async": "CAA-v1",
    "caa_fedbuff_v2": "CAA-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polish MLLM experiment reports and slides.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--outdir", default="figures/report")
    parser.add_argument("--presentation-dir", default="presentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)
    presentation_dir = Path(args.presentation_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(result_dir)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No official summaries found")
    frame = frame.sort_values(["task_family", "dataset", "backend_group", "model", "method", "seed", "run_id"])
    frame.to_csv(outdir / "all_seeded_summary.csv", index=False)
    primary_raw = primary_headline_frame(frame)
    non_iid = frame[frame["partition"].fillna("").astype(str).ne("iid")].copy()
    stress = frame[
        frame["partition"].fillna("").astype(str).eq("iid")
        & ~frame.index.isin(primary_raw.index)
    ].copy()
    primary, duplicate_primary = dedupe_headline_runs(primary_raw)
    primary, incomplete_primary = keep_complete_headline_groups(primary)
    primary_raw.to_csv(outdir / "primary_seeded_summary_raw.csv", index=False)
    duplicate_primary.to_csv(outdir / "duplicate_primary_runs.csv", index=False)
    incomplete_primary.to_csv(outdir / "incomplete_primary_groups.csv", index=False)
    primary.to_csv(outdir / "seeded_summary.csv", index=False)
    stress.to_csv(outdir / "delay_stress_summary.csv", index=False)
    non_iid.to_csv(outdir / "non_iid_summary.csv", index=False)

    mean_std = compute_mean_std(primary)
    mean_std.to_csv(outdir / "mean_std_summary.csv", index=False)
    fairness = compute_fairness_audit(primary)
    fairness.to_csv(outdir / "fairness_audit.csv", index=False)
    proxy = compute_real_vs_proxy(primary)
    proxy.to_csv(outdir / "qwenvl_real_vs_proxy_summary.csv", index=False)
    closed_qa = primary[primary["dataset"].isin(["medqa_usmle", "pubmedqa"])].copy()
    closed_qa.to_csv(outdir / "closed_qa_method_comparison.csv", index=False)

    write_tables(outdir)
    plot_accuracy(mean_std, outdir)
    plot_closed_qa(closed_qa, outdir)
    plot_real_qwenvl(primary, outdir)
    plot_invalid(primary, outdir)
    plot_qwenvl_diagnostics(outdir)
    ensure_method_diagnostics_plot(outdir)
    plot_stability(mean_std, outdir)
    plot_gaps(primary, outdir)
    write_markdown_reports(primary, mean_std, fairness, proxy, outdir, total_rows=len(frame), stress_rows=len(stress), non_iid_rows=len(non_iid))
    write_presentation(primary, mean_std, fairness, proxy, outdir, presentation_dir)
    print(f"polished rows={len(primary)} primary_rows all_rows={len(frame)} summaries={result_dir}")


def primary_headline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return IID fair-budget headline rows. Stress and non-IID stay separate."""
    if frame.empty:
        return frame.copy()
    is_iid = frame["partition"].fillna("").astype(str).eq("iid")
    is_sync = frame["method"].eq("sync_fedavg")
    is_primary_async_delay = frame["delay_mode"].fillna("").astype(str).isin(["", "heterogeneous"])
    primary = frame[is_iid & (is_sync | is_primary_async_delay)].copy()
    return primary


def dedupe_headline_runs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one run per seed/config in the headline table."""
    if frame.empty:
        return frame.copy(), frame.copy()
    keys = [
        "task_family",
        "dataset",
        "backend_group",
        "model",
        "backend",
        "method",
        "seed",
        "partition",
        "delay_mode",
        "budget",
    ]
    sorted_frame = frame.sort_values(keys + ["run_id"])
    duplicate_mask = sorted_frame.duplicated(keys, keep="last")
    duplicates = sorted_frame[duplicate_mask].copy()
    deduped = sorted_frame[~duplicate_mask].copy()
    return deduped, duplicates


def keep_complete_headline_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop headline groups that do not have a Sync baseline."""
    if frame.empty:
        return frame.copy(), frame.copy()
    keys = ["dataset", "backend_group", "model", "backend", "seed", "partition", "budget"]
    keep_indices: list[int] = []
    drop_indices: list[int] = []
    for _, group in frame.groupby(keys, dropna=False):
        if "sync_fedavg" in set(group["method"].astype(str)):
            keep_indices.extend(group.index.tolist())
        else:
            drop_indices.extend(group.index.tolist())
    return frame.loc[keep_indices].copy(), frame.loc[drop_indices].copy()


def load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary.get("config", {})
        csv_path = Path(summary.get("csv_path", ""))
        if not csv_path.is_absolute():
            csv_path = path.parent / csv_path.name
        if str(csv_path).startswith("/tmp/"):
            continue
        metrics = csv_metrics(csv_path)
        model = str(config.get("model", ""))
        backend = str(config.get("model_backend", model))
        dataset = str(config.get("dataset", dataset_from_name(path.name)))
        method = str(summary.get("method", config.get("method", "")))
        best = to_float(summary.get("best_test_acc"))
        final = to_float(summary.get("final_test_acc"))
        seed = config.get("seed", "")
        rows.append(
            {
                "run_id": path.stem.replace("_summary", ""),
                "dataset": dataset,
                "task": config.get("task", ""),
                "task_family": task_family(dataset, model, backend),
                "model": model,
                "backend": backend,
                "backend_group": backend_group(model, backend),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": seed,
                "partition": config.get("partition", ""),
                "delay_mode": config.get("delay_mode", ""),
                "clients": config.get("clients", ""),
                "local_epochs": config.get("local_epochs", ""),
                "batch_size": config.get("batch_size", ""),
                "budget": config.get("update_budget", summary.get("total_rounds_or_events", "")),
                "sync_equivalent_rounds": config.get("sync_equivalent_rounds", ""),
                "best_acc": best,
                "final_acc": final,
                "final_loss": to_float(summary.get("final_test_loss")),
                "stability_drop": best - final,
                "invalid_answer_rate": to_float(summary.get("final_invalid_answer_rate")),
                "simulated_time": to_float(summary.get("total_simulated_time")),
                "avg_staleness": metrics["avg_staleness"],
                "p95_staleness": metrics["p95_staleness"],
                "avg_effective_alpha": metrics["avg_effective_alpha"],
                "avg_agreement": metrics["avg_agreement"],
                "client_gini": metrics["client_gini"],
                "csv_path": str(csv_path),
                "summary_path": str(path),
            }
        )
    return rows


def csv_metrics(path: Path) -> dict[str, float]:
    output = {
        "avg_staleness": float("nan"),
        "p95_staleness": float("nan"),
        "avg_effective_alpha": float("nan"),
        "avg_agreement": float("nan"),
        "client_gini": float("nan"),
    }
    if not path.exists() or path.stat().st_size == 0:
        return output
    frame = pd.read_csv(path)
    for column in ["staleness", "effective_alpha", "mean_agreement", "agreement", "client_id"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "staleness" in frame:
        vals = frame["staleness"].dropna()
        if not vals.empty:
            output["avg_staleness"] = float(vals.mean())
            output["p95_staleness"] = float(vals.quantile(0.95))
    if "effective_alpha" in frame:
        vals = frame["effective_alpha"].dropna()
        if not vals.empty:
            output["avg_effective_alpha"] = float(vals.mean())
    agreement_col = "mean_agreement" if "mean_agreement" in frame.columns else "agreement"
    if agreement_col in frame:
        vals = frame[agreement_col].dropna()
        if not vals.empty:
            output["avg_agreement"] = float(vals.mean())
    if "client_id" in frame:
        counts = frame.dropna(subset=["client_id"]).groupby("client_id").size().to_numpy(dtype=float)
        output["client_gini"] = gini(counts)
    return output


def compute_mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["task_family", "dataset", "backend_group", "model", "backend", "method", "method_label"], as_index=False)
    out = grouped.agg(
        runs=("run_id", "count"),
        seeds=("seed", "nunique"),
        budget_min=("budget", "min"),
        budget_max=("budget", "max"),
        best_acc_mean=("best_acc", "mean"),
        best_acc_std=("best_acc", "std"),
        final_acc_mean=("final_acc", "mean"),
        final_acc_std=("final_acc", "std"),
        stability_drop_mean=("stability_drop", "mean"),
        stability_drop_std=("stability_drop", "std"),
        invalid_answer_rate_mean=("invalid_answer_rate", "mean"),
        avg_staleness_mean=("avg_staleness", "mean"),
        client_gini_mean=("client_gini", "mean"),
    )
    return out.fillna({"best_acc_std": 0.0, "final_acc_std": 0.0, "stability_drop_std": 0.0})


def compute_fairness_audit(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "backend_group", "model", "seed"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        budgets = sorted(set(map(str, group["budget"].tolist())))
        methods = sorted(set(group["method"].tolist()), key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        delays = sorted(set(map(str, group["delay_mode"].tolist())))
        partitions = sorted(set(map(str, group["partition"].tolist())))
        rows.append(
            {
                "dataset": key[0],
                "backend_group": key[1],
                "model": key[2],
                "seed": key[3],
                "methods": ";".join(methods),
                "method_count": len(methods),
                "budgets": ";".join(budgets),
                "budget_fair": len(budgets) == 1,
                "delay_modes": ";".join(delays),
                "partition": ";".join(partitions),
                "notes": fairness_notes(group, budgets, delays),
            }
        )
    return pd.DataFrame(rows)


def compute_real_vs_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["dataset"].isin(["vqa_rad_closed", "path_vqa_closed"])].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset.groupby(["dataset", "backend_group", "model", "backend", "method_label"], as_index=False).agg(
        runs=("run_id", "count"),
        best_acc_mean=("best_acc", "mean"),
        final_acc_mean=("final_acc", "mean"),
        stability_drop_mean=("stability_drop", "mean"),
        invalid_answer_rate_mean=("invalid_answer_rate", "mean"),
    )


def write_tables(outdir: Path) -> None:
    pd.DataFrame(
        [
            {"Control": "clients", "Value": "10 for text/proxy matrix; 2 for real Qwen2.5-VL small feasibility matrix"},
            {"Control": "local_epochs", "Value": "1"},
            {"Control": "fair budget", "Value": "async events = sync rounds x clients within each comparison group"},
            {"Control": "delay", "Value": "same delay mode within each async comparison group"},
            {"Control": "model separation", "Value": "proxy VQA and real Qwen2.5-VL are reported separately"},
        ]
    ).to_csv(outdir / "fairness_protocol.csv", index=False)
    pd.DataFrame(
        [
            {"Existing": "FedAvg / FedBuff / staleness-aware FL", "Ours": "Clockless event-driven LLM/MLLM adapter simulator"},
            {"Existing": "LoRA / QLoRA / Qwen / Qwen2.5-VL", "Ours": "Sequential client adapter aggregation under fair update budgets"},
            {"Existing": "MMLU / MedMCQA / MedQA / PubMedQA / VQA-RAD / PathVQA", "Ours": "Closed-ended QA/VQA distributed-system evaluation pipeline"},
            {"Existing": "Accuracy-only ML evaluation", "Ours": "Staleness, invalid answers, client imbalance, stability drop"},
        ]
    ).to_csv(outdir / "existing_vs_ours_table.csv", index=False)


def plot_accuracy(mean_std: pd.DataFrame, outdir: Path) -> None:
    for family, name in [("text_mcqa", "accuracy_mean_std_by_dataset.png"), ("real_qwenvl_vqa", "real_qwenvl_vqa_accuracy.png")]:
        subset = mean_std[mean_std["task_family"] == family]
        if subset.empty:
            empty_plot(outdir / name, f"No {family} results")
            continue
        pivot = subset.pivot_table(index="dataset", columns="method_label", values="best_acc_mean")
        ax = pivot.plot(kind="bar", figsize=(13, 6), width=0.78)
        ax.set_ylabel("Best accuracy")
        ax.set_xlabel("")
        ax.set_title(f"{family.replace('_', ' ').title()} best accuracy")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        plt.savefig(outdir / name, dpi=200)
        plt.close()


def plot_closed_qa(frame: pd.DataFrame, outdir: Path) -> None:
    if frame.empty:
        empty_plot(outdir / "closed_qa_method_comparison.png", "Closed QA pending")
        return
    pivot = frame.pivot_table(index="dataset", columns="method_label", values="best_acc", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(12, 5.5), width=0.76)
    ax.set_ylabel("Best accuracy")
    ax.set_xlabel("")
    ax.set_title("Closed-ended QA comparison: MedQA-USMLE / PubMedQA")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(outdir / "closed_qa_method_comparison.png", dpi=200)
    plt.close()


def plot_real_qwenvl(frame: pd.DataFrame, outdir: Path) -> None:
    subset = frame[frame["backend_group"] == "real_qwenvl"].copy()
    if subset.empty:
        empty_plot(outdir / "real_qwenvl_vqa_accuracy.png", "Real Qwen2.5-VL pending")
        return
    pivot = subset.pivot_table(index="dataset", columns="method_label", values="best_acc", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(12, 5.5), width=0.76)
    ax.set_ylabel("Best accuracy")
    ax.set_xlabel("")
    ax.set_title("Real Qwen2.5-VL 4-bit QLoRA small VQA matrix")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(outdir / "real_qwenvl_vqa_accuracy.png", dpi=200)
    plt.close()


def plot_invalid(frame: pd.DataFrame, outdir: Path) -> None:
    subset = frame.copy()
    pivot = subset.pivot_table(index="dataset", columns="method_label", values="invalid_answer_rate", aggfunc="mean")
    if pivot.empty:
        empty_plot(outdir / "invalid_answer_rate_by_method.png", "Invalid answer rate pending")
        return
    ax = pivot.plot(kind="bar", figsize=(13, 5.5), width=0.78)
    ax.set_ylabel("Invalid answer rate")
    ax.set_xlabel("")
    ax.set_title("Closed-answer validity by method")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(outdir / "invalid_answer_rate_by_method.png", dpi=200)
    plt.close()


def plot_stability(mean_std: pd.DataFrame, outdir: Path) -> None:
    pivot = mean_std.pivot_table(index="dataset", columns="method_label", values="stability_drop_mean")
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(13, 5.5), width=0.78)
    ax.set_ylabel("Best - final accuracy")
    ax.set_xlabel("")
    ax.set_title("Stability drop by dataset/method")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(outdir / "stability_drop_errorbar.png", dpi=200)
    plt.close()


def plot_gaps(frame: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for keys, group in frame.groupby(["dataset", "backend_group", "model", "seed"], dropna=False):
        sync = group[group["method"] == "sync_fedavg"]
        if sync.empty:
            continue
        sync_best = float(sync.iloc[0]["best_acc"])
        sync_final = float(sync.iloc[0]["final_acc"])
        for _, row in group[group["method"] != "sync_fedavg"].iterrows():
            rows.append({**{k: v for k, v in zip(["dataset", "backend_group", "model", "seed"], keys)}, "method_label": row["method_label"], "best_gap": sync_best - row["best_acc"], "final_gap": sync_final - row["final_acc"]})
    gap = pd.DataFrame(rows)
    if gap.empty:
        empty_plot(outdir / "async_sync_gap_errorbar.png", "Async-Sync gap pending")
        empty_plot(outdir / "final_gap_errorbar.png", "Final gap pending")
        return
    for col, filename, title in [("best_gap", "async_sync_gap_errorbar.png", "Async-Sync Best Gap"), ("final_gap", "final_gap_errorbar.png", "Async-Sync Final Gap")]:
        pivot = gap.pivot_table(index="dataset", columns="method_label", values=col, aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(13, 5.5), width=0.78)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylabel("Sync - Async accuracy")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout(rect=(0, 0.1, 1, 1))
        plt.savefig(outdir / filename, dpi=200)
        plt.close()


def plot_qwenvl_diagnostics(outdir: Path) -> None:
    path = outdir / "qwenvl_closed_answer_diagnostics.csv"
    if not path.exists():
        empty_plot(outdir / "qwenvl_prediction_validity.png", "Qwen2.5-VL prediction diagnostics pending")
        return
    frame = pd.read_csv(path)
    if frame.empty or "valid" not in frame.columns:
        empty_plot(outdir / "qwenvl_prediction_validity.png", "Qwen2.5-VL prediction diagnostics pending")
        return
    summary = frame.groupby("dataset", as_index=False).agg(valid_rate=("valid", "mean"), accuracy=("correct", "mean"), samples=("valid", "count"))
    summary.to_csv(outdir / "qwenvl_closed_answer_diagnostics_summary.csv", index=False)
    ax = summary.plot(kind="bar", x="dataset", y=["valid_rate", "accuracy"], figsize=(9, 4.8), width=0.72)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_title("Real Qwen2.5-VL zero-shot closed-answer diagnostics")
    ax.legend(["Valid answer", "Accuracy"], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(outdir / "qwenvl_prediction_validity.png", dpi=200)
    plt.close()


def write_markdown_reports(frame: pd.DataFrame, mean_std: pd.DataFrame, fairness: pd.DataFrame, proxy: pd.DataFrame, outdir: Path, total_rows: int | None = None, stress_rows: int = 0, non_iid_rows: int = 0) -> None:
    report = [
        "# R13946001_MLLM Final Results Summary",
        "",
        "## Headline",
        "",
        f"- Official summaries: {total_rows if total_rows is not None else len(frame)} total rows; headline statistics use {len(frame)} deduplicated IID primary fair-budget rows, with {stress_rows} delay-stress rows and {non_iid_rows} non-IID rows reported separately.",
        "- Text MCQA matrix is the strongest evidence: MMLU, MedMCQA, MedQA-USMLE, and PubMedQA use real Qwen1.5-0.5B LoRA adapters.",
        "- Real Qwen2.5-VL is now implemented as 4-bit QLoRA; current VQA matrix is a small fair-budget feasibility run, not a large multi-seed claim.",
        "- Proxy VQA results are kept for algorithm debugging but separated from headline MLLM claims.",
        "",
        "## Result Groups",
        "",
        markdown_table(mean_std[["task_family", "dataset", "backend_group", "method_label", "runs", "seeds", "best_acc_mean", "final_acc_mean", "stability_drop_mean"]].sort_values(["task_family", "dataset", "method_label"]), max_rows=80),
        "",
        "## Real vs Proxy VQA",
        "",
        markdown_table(proxy.sort_values(["dataset", "backend_group", "method_label"]), max_rows=80),
        "",
        "## Fairness Audit",
        "",
        markdown_table(fairness.sort_values(["dataset", "backend_group", "seed"]), max_rows=80),
        "",
        "## Base Qwen2.5-VL Closed-Answer Parsing Diagnostics",
        "",
        diagnostics_markdown(outdir),
        "",
        "## Real Qwen2.5-VL Checkpoint Method Diagnostics",
        "",
        method_diagnostics_markdown(outdir),
        "",
    ]
    Path("FINAL_RESULTS_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    audit = [
        "# Experiment Completeness Audit",
        "",
        f"- Headline primary rows: {len(frame)}",
        f"- Total official rows before separating delay stress: {total_rows if total_rows is not None else len(frame)}",
        f"- Delay-stress rows kept outside headline mean/std: {stress_rows}",
        f"- Non-IID rows kept outside headline mean/std: {non_iid_rows}",
        f"- Datasets: {', '.join(sorted(frame['dataset'].unique()))}",
        f"- Real Qwen2.5-VL rows: {int((frame['backend_group'] == 'real_qwenvl').sum())}",
        f"- Proxy VQA rows: {int((frame['backend_group'] == 'proxy_vqa').sum())}",
        f"- Rows with invalid answer rate column: {int(frame['invalid_answer_rate'].notna().sum())}",
        "- Smoke `/tmp` outputs are excluded by construction.",
        "",
        "## Fairness Table",
        "",
        markdown_table(fairness, max_rows=100),
    ]
    Path("EXPERIMENT_COMPLETENESS_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    # Keep REPORT_NOTES compact and aligned with final summary.
    Path("REPORT_NOTES.md").write_text("\n".join(report), encoding="utf-8")


def write_presentation(frame: pd.DataFrame, mean_std: pd.DataFrame, fairness: pd.DataFrame, proxy: pd.DataFrame, outdir: Path, presentation_dir: Path) -> None:
    best_text = top_line(mean_std[mean_std["task_family"] == "text_mcqa"])
    best_real = top_line(mean_std[mean_std["backend_group"] == "real_qwenvl"])
    md = f"""# Clockless Federated Adaptation for Medical LLM/MLLM\n\n## 1. Motivation\n\n- 醫療資料無法集中，但不同醫院的硬體、網路、工作負載不同。\n- 同步 FL 會被慢節點拖住；非同步 FL 提高吞吐量，但會引入 stale update 與 fast-client domination。\n- 我們把問題限制在 closed-ended QA/VQA，讓答案空間有限，指標可以清楚比較。\n\n## 2. Methodology\n\n- Sync FedAvg：server 等齊所有 client update。\n- Naive Async：server 不等待，收到 update 就混合。\n- Staleness Async：用 logical version 估計 `server_version - client_start_version`。\n- FedBuff：buffer 到 B 個 update 再聚合。\n- CAA-v2：在 FedBuff 上加入 direction agreement、server trajectory EMA、delta clipping、client fairness credit。\n\n## 3. What Is Ours vs Existing\n\n| Existing | Ours |\n|---|---|\n| FedAvg / FedBuff / staleness-aware FL | Clockless event-driven simulator for LLM/MLLM adapters |\n| LoRA / QLoRA / Qwen / Qwen2.5-VL | Sequential client adapter aggregation protocol |\n| Accuracy-only report | Accuracy + stability + staleness + invalid answer + client imbalance |\n\n## 4. Experiment Design\n\n- Fair update budget：`async events = sync rounds × clients`。\n- Text MCQA：MMLU, MedMCQA, MedQA-USMLE, PubMedQA。\n- VQA：VQA-RAD, PathVQA。\n- 分開報告：`qwen_vl_proxy` 不混入 real `qwen2_5_vl_3b_qlora` headline。\n\n## 5. Text MCQA Result\n\n![Closed QA](../figures/report/closed_qa_method_comparison.png)\n\n{best_text}\n\n## 6. Real Qwen2.5-VL VQA Result\n\n![Real Qwen2.5-VL](../figures/report/real_qwenvl_vqa_accuracy.png)\n\n{best_real}\n\n## 7. Stability and Validity\n\n![Stability](../figures/report/stability_drop_errorbar.png)\n\n![Invalid Answer](../figures/report/invalid_answer_rate_by_method.png)\n\n![Qwen2.5-VL Closed-Answer Diagnostics](../figures/report/qwenvl_prediction_validity.png)\n\n> 這張圖是 base Qwen2.5-VL 的 closed-answer parsing sanity check；正式 method comparison 仍以上方 summary CSV 為準。\n\n![Qwen2.5-VL Method Diagnostics](../figures/report/qwenvl_method_validity.png)\n\n## 8. Challenges\n\n- 真 Qwen2.5-VL QLoRA 已可跑，但目前只適合 small fair-budget matrix；目前不宣稱大規模 multi-seed SOTA。\n- VQA closed-answer generation 需要解析生成文字，因此 invalid answer rate 很重要。\n- Proxy VQA 可用來 debug algorithm，但不能當 real MLLM headline。\n\n## 9. Conclusion\n\n- 專案貢獻是把 clockless async FL、LoRA/QLoRA adapter aggregation、closed-ended medical QA/VQA 串成可重現系統。\n- CAA-v2 是 course-project-level design extension：用 agreement/fairness/staleness 控制 async update 的穩定性。\n- 最保守主張：在 fair budget 下，async adapter FL 可以接近 Sync；CAA-v2 在部分 QA 場景改善 peak 或穩定性，但不是所有資料集都支配 baseline。\n"""
    md_path = presentation_dir / "mllm_report_zh.md"
    html_path = presentation_dir / "mllm_report_zh.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(render_html(md), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.head(max_rows).copy()
    for col in subset.columns:
        if pd.api.types.is_float_dtype(subset[col]):
            subset[col] = subset[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = list(subset.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)


def ensure_method_diagnostics_plot(outdir: Path) -> None:
    path = outdir / "qwenvl_method_validity.png"
    csv_path = outdir / "qwenvl_method_validity.csv"
    if path.exists():
        return
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            frame = pd.read_csv(csv_path)
            if not frame.empty:
                pivot = frame.pivot_table(index="dataset", columns="method_label", values="accuracy")
                ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.78)
                ax.set_ylabel("Sample accuracy")
                ax.set_xlabel("")
                ax.set_title("Real Qwen2.5-VL checkpoint closed-answer diagnostics")
                ax.set_ylim(0.0, 1.0)
                ax.grid(axis="y", alpha=0.25)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
                plt.xticks(rotation=15, ha="right")
                plt.tight_layout(rect=(0, 0.10, 1, 1))
                plt.savefig(path, dpi=200)
                plt.close()
                return
        except Exception:
            pass
    empty_plot(path, "Qwen2.5-VL checkpoint method diagnostics pending")


def method_diagnostics_markdown(outdir: Path) -> str:
    path = outdir / "qwenvl_method_validity.csv"
    if not path.exists() or path.stat().st_size == 0:
        return "No checkpoint-level method diagnostics yet."
    frame = pd.read_csv(path)
    if frame.empty:
        return "No checkpoint-level method diagnostics yet."
    return markdown_table(frame.sort_values(["dataset", "method_label"]), max_rows=80)


def diagnostics_markdown(outdir: Path) -> str:
    path = outdir / "qwenvl_closed_answer_diagnostics_summary.csv"
    if not path.exists():
        return "Prediction diagnostics pending."
    frame = pd.read_csv(path)
    return markdown_table(frame, max_rows=20)


def render_html(markdown_text: str) -> str:
    lines = []
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<section><h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line:
            alt = line.split("](")[0][2:]
            src = line.split("](")[1].rstrip(")")
            lines.append(f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">')
        elif line.startswith("- "):
            lines.append(f"<p>• {html.escape(line[2:])}</p>")
        elif line.startswith("|"):
            lines.append(f"<pre>{html.escape(line)}</pre>")
        elif line:
            lines.append(f"<p>{html.escape(line)}</p>")
    return """<!doctype html><html><head><meta charset=\"utf-8\"><title>MLLM Report</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans TC','Segoe UI',sans-serif;margin:0;background:#f6f7f9;color:#18202a}section,h1{max-width:1120px;margin:28px auto;background:white;padding:28px;border-radius:8px;box-shadow:0 1px 4px #d7dce3}h1{font-size:34px}h2{font-size:25px;color:#17324d}p{font-size:18px;line-height:1.55}img{width:100%;max-height:560px;object-fit:contain;border:1px solid #e3e7ee;border-radius:6px;background:white}pre{white-space:pre-wrap;font-size:13px;background:#f1f3f6;padding:8px;border-radius:4px}
</style></head><body>""" + "\n".join(lines) + "</section></body></html>"


def top_line(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed rows yet."
    row = frame.sort_values("best_acc_mean", ascending=False).iloc[0]
    return f"Best mean row: {row['dataset']} / {row['method_label']} / best={row['best_acc_mean']:.4f}, final={row['final_acc_mean']:.4f}."


def fairness_notes(group: pd.DataFrame, budgets: list[str], delays: list[str]) -> str:
    notes = []
    if len(budgets) != 1:
        notes.append("mixed_budget")
    if len(set(group["model"].astype(str))) != 1:
        notes.append("mixed_model")
    async_group = group[group["method"] != "sync_fedavg"]
    if not async_group.empty and len(set(async_group["delay_mode"].astype(str))) != 1:
        notes.append("mixed_async_delay")
    methods = set(group["method"])
    if "sync_fedavg" not in methods:
        notes.append("missing_sync")
    if not any(method in methods for method in ["naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"]):
        notes.append("missing_async")
    return ";".join(notes) if notes else "ok"


def dataset_from_name(name: str) -> str:
    for known in ["path_vqa_closed", "vqa_rad_closed", "medqa_usmle", "pubmedqa", "medmcqa", "mmlu", "synthetic_vqa", "synthetic_mcqa"]:
        if known in name:
            return known
    return "unknown"


def task_family(dataset: str, model: str, backend: str) -> str:
    if backend_group(model, backend) == "real_qwenvl":
        return "real_qwenvl_vqa"
    if dataset in {"vqa_rad_closed", "path_vqa_closed"}:
        return "proxy_vqa"
    return "text_mcqa"


def backend_group(model: str, backend: str) -> str:
    if backend == "qwen2_5_vl_3b_4bit_lora_generative":
        return "real_qwenvl"
    if model == "qwen_vl_proxy" or backend in {"compact_vqa_proxy", "compact_vqa_proxy_for_qwen_vl"}:
        return "proxy_vqa"
    return "text_llm"


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def gini(values) -> float:
    vals = [float(v) for v in values if float(v) >= 0]
    if not vals or sum(vals) <= 0:
        return float("nan")
    vals = sorted(vals)
    n = len(vals)
    weighted = sum((idx + 1) * val for idx, val in enumerate(vals))
    return (2 * weighted) / (n * sum(vals)) - (n + 1) / n


def empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=16)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
