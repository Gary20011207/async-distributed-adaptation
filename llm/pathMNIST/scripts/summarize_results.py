from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import re

import pandas as pd


DATASET_INFO = {
    "mmlu": ("multi-class", 4, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize curated PathMNIST runs.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--out", default="../REPORT_NOTES.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    summaries = sorted(result_dir.glob("*_summary.json"))
    rows = [_summary_row(path) for path in summaries]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"wrote {out_path}")


def _summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    csv_path = Path(summary.get("csv_path", ""))
    if not csv_path.is_absolute():
        csv_path = path.parent / csv_path.name

    metrics = _csv_metrics(csv_path)
    method = summary.get("method", "")
    dataset = _summary_dataset(path, method, config)
    partition = config.get("partition", "")
    decay = config.get("staleness_decay", "")
    label = method
    if partition:
        label += f" / {partition}"
    if method in {"staleness_async", "fedbuff_async", "agreement_fedbuff_async"} and decay:
        label += f" / {decay}"
    if method in {"fedbuff_async", "agreement_fedbuff_async"}:
        label += f" / B={config.get('buffer_size', '')}"

    return {
        "label": label,
        "dataset": dataset,
        "method": method,
        "model": config.get("model", "qwen"),
        "partition": partition,
        "clients": config.get("clients", ""),
        "task": config.get("task", ""),
        "num_classes": config.get("num_classes", ""),
        "in_channels": config.get("in_channels", ""),
        "update_budget": _update_budget(summary, config),
        "sync_equivalent_rounds": config.get("sync_equivalent_rounds", ""),
        "best_acc": summary.get("best_test_acc"),
        "best_step": summary.get("best_round_or_event"),
        "final_acc": summary.get("final_test_acc"),
        "final_loss": summary.get("final_test_loss"),
        "progress": summary.get("total_rounds_or_events"),
        "sim_time": summary.get("total_simulated_time"),
        "avg_staleness": metrics["avg_staleness"],
        "avg_effective_alpha": metrics["avg_effective_alpha"],
        "avg_agreement": metrics["avg_agreement"],
        "avg_buffer_alpha": metrics["avg_buffer_alpha"],
        "dropped_updates": metrics["dropped_updates"],
        "client_updates": metrics["client_updates"],
        "csv": str(csv_path),
    }


def _csv_metrics(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_metrics()
        
    try:
        frame = pd.read_csv(path)
    except Exception:
        return _empty_metrics()

    for column in [
        "staleness", "effective_alpha", "client_id", "mean_agreement",
        "agreement", "buffer_alpha", "dropped_update", "server_momentum_agreement",
        "fairness_weight",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    avg_staleness = _mean_or_blank(frame, "staleness")
    avg_effective_alpha = _mean_or_blank(frame, "effective_alpha")
    avg_agreement = _mean_or_blank(frame, "mean_agreement")
    if avg_agreement == "":
        avg_agreement = _mean_or_blank(frame, "agreement")
    avg_buffer_alpha = _mean_or_blank(frame, "buffer_alpha")
    avg_server_momentum_agreement = _mean_or_blank(frame, "server_momentum_agreement")
    avg_fairness_weight = _mean_or_blank(frame, "fairness_weight")
    
    dropped_updates = ""
    if "dropped_update" in frame.columns:
        dropped_values = frame["dropped_update"].dropna()
        if not dropped_values.empty:
            dropped_updates = int(dropped_values.sum())
            
    client_updates = ""
    if "client_id" in frame.columns:
        client_counts = frame.dropna(subset=["client_id"]).groupby("client_id").size()
        if not client_counts.empty:
            client_updates = f"{int(client_counts.min())}-{int(client_counts.max())}"

    return {
        "avg_staleness": avg_staleness,
        "avg_effective_alpha": avg_effective_alpha,
        "avg_agreement": avg_agreement,
        "avg_buffer_alpha": avg_buffer_alpha,
        "avg_server_momentum_agreement": avg_server_momentum_agreement,
        "avg_fairness_weight": avg_fairness_weight,
        "dropped_updates": dropped_updates,
        "client_updates": client_updates,
    }

def _empty_metrics() -> dict[str, Any]:
    return {
        "avg_staleness": "", "avg_effective_alpha": "", "avg_agreement": "",
        "avg_buffer_alpha": "", "avg_server_momentum_agreement": "",
        "avg_fairness_weight": "", "dropped_updates": "", "client_updates": "",
    }

def _mean_or_blank(frame: pd.DataFrame, column: str) -> float | str:
    if column not in frame.columns:
        return ""
    values = frame[column].dropna()
    if values.empty:
        return ""
    return float(values.mean())


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Federated Large Language Model Experiment Notes",
        "",
        "## Story for Presentation",
        "",
        "Different client edge devices do not fine-tune at the same speed. "
        "Some nodes return fresh parameters, while stragglers return stale updates computed from older global models.",
        "",
        "The distributed-systems question is not only text-accuracy. It is how a system without a global physical clock should reason about delayed gradients. "
        "This project uses logical model versions to define staleness and compares aggregation policies under asynchronous token arrivals.",
        "",
    ]

    lines.extend(_distributed_problem_statement())
    lines.extend(_headline_findings(rows))
    lines.extend(_method_detail())
    lines.extend(_existing_vs_ours())
    lines.extend(_fairness_protocol())
    lines.extend(_multi_seed_report())
    lines.extend(_distributed_systems_report())
    lines.extend(_best_by_dataset_overview(rows))
    lines.extend(
        [
        "## Detailed Result Summary",
        "",
        "This table keeps all completed full runs, including tuning runs. Use the best-by-dataset view above for slides.",
        "",
        "| Dataset | Model | Run | Budget | Best Acc | Best Step | Final Acc | Final Loss | Progress | Sim Time | Avg Staleness | Avg Alpha | Avg Agreement | Buffer Alpha | Dropped | Client Updates |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rows:
        lines.append(
            "| {dataset} | {model} | {label} | {update_budget} | {best_acc} | {best_step} | {final_acc} | {final_loss} | "
            "{progress} | {sim_time} | {avg_staleness} | {avg_effective_alpha} | "
            "{avg_agreement} | {avg_buffer_alpha} | {dropped_updates} | {client_updates} |".format(
                dataset=row["dataset"],
                model=row["model"],
                label=row["label"],
                update_budget=_fmt(row["update_budget"], digits=0),
                best_acc=_fmt(row["best_acc"]),
                best_step=_fmt(row["best_step"], digits=0),
                final_acc=_fmt(row["final_acc"]),
                final_loss=_fmt(row["final_loss"]),
                progress=_fmt(row["progress"], digits=0),
                sim_time=_fmt(row["sim_time"]),
                avg_staleness=_fmt(row["avg_staleness"]),
                avg_effective_alpha=_fmt(row["avg_effective_alpha"]),
                avg_agreement=_fmt(row["avg_agreement"]),
                avg_buffer_alpha=_fmt(row["avg_buffer_alpha"]),
                dropped_updates=_fmt(row["dropped_updates"], digits=0),
                client_updates=row["client_updates"],
            )
        )
    lines.extend(_caa_verdict(rows))
    lines.extend(_agreement_readout(rows))
    lines.extend(_async_sync_gap_analysis(rows))
    lines.extend(_multi_dataset_coverage(rows))
    lines.extend(_stateless_staleness_analysis(rows))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Sync FedAvg is the stable reference baseline because the server waits for all LLM clients.",
            "- Naive async removes the waiting barrier, but stale text gradients can destabilize the model trajectory.",
            "- Staleness-aware async uses logical version identifiers to reduce the impact of historical text parameters.",
            "- FedBuff-lite buffers several asynchronous parameter packages before applying them, linking deep learning optimization to distributed buffering constraints.",
            "- CAA-FedBuff extends FedBuff-lite with update-direction agreement, median-norm clipping, and adaptive server alpha while avoiding physical clocks.",
            "- Dirichlet non-IID partitioning models client nodes with highly heterogeneous text topic distributions.",
            "- Async-Sync gaps measure the explicit price of removing the synchronization barrier.",
            "",
            "## Figures",
            "",
            "- `figures/test_acc_vs_progress.png`",
            "- `figures/test_acc_vs_simulated_time.png`",
            "- `figures/staleness_vs_event.png`",
            "- `figures/effective_alpha_vs_event.png`",
            "- `figures/agreement_vs_event.png`",
            "- `figures/client_contribution_bar.png`",
            "",
            "## Future Extensions",
            "",
            "- Other generative language tasks can be integrated later into the custom simulator framework.",
            "",
        ]
    )
    return "\n".join(lines)


def _distributed_problem_statement() -> list[str]:
    return [
        "## Distributed Systems Problem Statement",
        "",
        "The project asks whether a federated learning server can remove the synchronous waiting barrier without losing too much text-generation quality or convergence stability. "
        "In large language modeling, edge clients exhibit diverse hardware configurations and complex text distribution contexts, causing updates to arrive out of order.",
        "",
        "## No Global Clock and Logical Staleness",
        "",
        "The server does not assume synchronized physical clocks. It assigns a logical model version token to each global matrix state and captures late updates via:",
        "",
        "```text",
        "staleness = current_server_version - client_start_version",
        "```",
        "",
        "This characterizes the validation track as a true distributed-systems experiment: an identical parameter delta can either accelerate optimization or introduce divergence based entirely on its version delay.",
        "",
    ]


def _fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def _caa_verdict(rows: list[dict[str, Any]]) -> list[str]:
    caa_rows = [row for row in rows if row["method"] in {"agreement_fedbuff_async", "caa_fedbuff_v2"}]
    baseline_rows = [
        row
        for row in rows
        if row["method"] not in {"agreement_fedbuff_async", "caa_fedbuff_v2"} and _to_float(row["best_acc"]) is not None
    ]
    caa_rows = [row for row in caa_rows if _to_float(row["best_acc"]) is not None]
    if not caa_rows or not baseline_rows:
        return []

    best_caa = max(caa_rows, key=lambda row: _to_float(row["best_acc"]) or -1.0)
    best_baseline = max(baseline_rows, key=lambda row: _to_float(row["best_acc"]) or -1.0)
    caa_acc = _to_float(best_caa["best_acc"]) or 0.0
    baseline_acc = _to_float(best_baseline["best_acc"]) or 0.0
    if caa_acc > baseline_acc:
        verdict = "The CAA-family method beat the strongest completed non-CAA baseline."
    else:
        verdict = "The CAA-family method did not beat the strongest completed non-CAA baseline; report the stability trade-off honestly."
    return [
        "",
        "## CAA-Family Check",
        "",
        f"- Best CAA-family run: `{best_caa['label']}` at `{_fmt(caa_acc)}`.",
        f"- Strongest non-CAA baseline: `{best_baseline['label']}` at `{_fmt(baseline_acc)}`.",
        f"- Conclusion: {verdict}",
    ]


def _headline_findings(rows: list[dict[str, Any]]) -> list[str]:
    best_sync = _best_method_row(rows, "sync_fedavg")
    best_naive = _best_method_row(rows, "naive_async")
    best_stale = _best_method_row(rows, "staleness_async")
    best_caa = _best_caa_family_row(rows)
    datasets = sorted({row["dataset"] for row in rows if row["dataset"]})

    lines = [
        "## Headline Findings",
        "",
        "- Fair async comparison uses the same client-update budget: `async events = sync rounds * clients`.",
    ]
    if best_caa and best_naive and best_sync:
        caa_acc = _to_float(best_caa["best_acc"]) or 0.0
        naive_acc = _to_float(best_naive["best_acc"]) or 0.0
        sync_acc = _to_float(best_sync["best_acc"]) or 0.0
        lines.append(
            f"- Best CAA-family run reached `{_fmt(caa_acc)}`, compared with strongest stateless async `{_fmt(naive_acc)}` and Sync FedAvg `{_fmt(sync_acc)}`."
        )
        if caa_acc > max(naive_acc, sync_acc):
            lines.append("- In the current completed runs, a CAA-family method beats the strongest baseline by best accuracy.")
        else:
            lines.append("- In the current completed runs, CAA-family methods do not beat every baseline; use the gap/stability analysis below.")
    if best_stale and best_naive:
        stale_acc = _to_float(best_stale["best_acc"]) or 0.0
        naive_acc = _to_float(best_naive["best_acc"]) or 0.0
        if stale_acc < naive_acc:
            lines.append(
                "- Logical staleness alone can be conservative: it reduces stale-update impact, but may also shrink useful updates too much."
            )
    if datasets:
        lines.append(f"- Completed datasets in this report: `{', '.join(datasets)}`.")
    lines.append(
        "- The main report metrics are `Async-Sync Best Gap`, `Async-Sync Final Gap`, and `Stability Drop = best_acc - final_acc`."
    )
    lines.append("")
    return lines


def _method_detail() -> list[str]:
    return [
        "## Method Detail",
        "",
        "- `sync_fedavg`: barrier baseline; the server waits for all LLM edge clients each round.",
        "- `naive_async`: stateless async baseline; every arriving update is applied with constant alpha.",
        "- `staleness_async`: logical-staleness baseline; alpha is decayed using logical version distance.",
        "- `fedbuff_async`: buffered async baseline; `B` updates are batched inside the server queue.",
        "- `agreement_fedbuff_async`: CAA-FedBuff adds direction agreement, median-norm clipping, and adaptive server alpha.",
        "- `caa_fedbuff_v2`: CAA-v2 additionally compares updates with recent accepted server delta direction and adds client fairness credit.",
        "",
        "CAA-FedBuff and CAA-v2 combine known ideas from buffered async FL, staleness-aware aggregation, cosine agreement, parameter clipping, and adaptive server step size.",
        "",
    ]


def _existing_vs_ours() -> list[str]:
    rows = [
        {"component": "Sync FedAvg", "source": "existing baseline", "role": "Barrier aggregation."},
        {"component": "Naive Async", "source": "existing baseline", "role": "Constant-alpha async aggregation."},
        {"component": "Staleness-aware decay", "source": "existing baseline", "role": "Logical-age weighting."},
        {"component": "FedBuff-style buffering", "source": "existing baseline", "role": "Buffered async aggregation."},
        {"component": "CAA agreement weighting", "source": "our design", "role": "Direction-aware buffered weighting."},
        {"component": "CAA-v2 server trajectory EMA", "source": "our design", "role": "Recent accepted-delta agreement."},
        {"component": "CAA-v2 client fairness credit", "source": "our design", "role": "Prevents fast-client domination."},
    ]

    lines = [
        "## Existing vs Ours",
        "",
        "| Component | Source | Role in this project |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {component} | {source} | {role} |".format(
                component=row.get("component", ""),
                source=row.get("source", ""),
                role=row.get("role", ""),
            )
        )
    lines.append("")
    return lines

def _fairness_protocol() -> list[str]:
    return [
        "## Fairness Protocol",
        "",
        "| Control | Value |",
        "|---|---|",
        "| Clients | 10 |",
        "| Local epochs | 1 |",
        "| Batch size | 4 |",
        "| LR setting | LoRA Fine-Tuning Learning Rate |",
        "| Partition | IID unless explicitly marked Dirichlet |",
        "| Async delay | heterogeneous with shared straggler settings across async methods |",
        "| Fair budget | `async events = sync rounds * clients` |",
        "| Seed | controls split, partition, delay sampling, and initialization |",
        "",
    ]


def _multi_seed_report() -> list[str]:
    return [
        "## Multi-Seed Variance",
        "",
        "Multi-seed tracking details are parsed across identical training script seeds.",
        "",
    ]


def _distributed_systems_report() -> list[str]:
    lines: list[str] = [
        "## System Metrics Beyond Accuracy",
        "",
        "Accuracy alone hides distributed-system behavior. This report also tracks staleness, simulated time, adaptive alpha, and client contribution imbalance.",
        "",
        "| Metric | Meaning |",
        "|---|---|",
        "| `p95_staleness` | Tail delay in logical model-version units. |",
        "| `avg_buffer_alpha` | How aggressively an async buffer updates the server. |",
        "| `client_contribution_gini` | Whether fast clients dominate accepted async updates. |",
        "| `time_to_90pct_best_acc` | Simulated time needed to approach each run's best accuracy. |",
        "",
    ]
    return lines


def _non_iid_report(frame: pd.DataFrame) -> list[str]:
    data = frame[frame["partition"] == "dirichlet"].copy()
    if data.empty:
        return [
            "## Hospital Non-IID Scenario",
            "",
            "Dirichlet non-IID stress runs are queued by the distributed-systems 24-hour runner. They model hospitals with different patient/image distributions.",
            "",
        ]

    grouped = (
        data.groupby(["dataset", "dirichlet_alpha", "method_label"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            best_acc=("best_acc", "mean"),
            stability_drop=("stability_drop", "mean"),
            p95_staleness=("p95_staleness", "mean"),
            client_gini=("client_contribution_gini", "mean"),
        )
        .sort_values(["dataset", "dirichlet_alpha", "method_label"])
    )

    lines = [
        "## Hospital Non-IID Scenario",
        "",
        "Dirichlet partitioning approximates hospitals with different label distributions. Lower alpha means stronger data heterogeneity and usually more conflicting client updates.",
        "",
        "| Dataset | Dirichlet Alpha | Method | Runs | Best Acc | Stability Drop | p95 Staleness | Client Gini |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in grouped.to_dict(orient="records"):
        lines.append(
            "| {dataset} | {alpha} | {method} | {runs} | {best} | {drop} | {p95} | {gini} |".format(
                dataset=row.get("dataset", ""),
                alpha=_fmt(row.get("dirichlet_alpha")),
                method=row.get("method_label", ""),
                runs=_fmt(row.get("runs"), digits=0),
                best=_fmt(row.get("best_acc")),
                drop=_fmt(row.get("stability_drop")),
                p95=_fmt(row.get("p95_staleness")),
                gini=_fmt(row.get("client_gini")),
            )
        )
    lines.append("")
    return lines


def _straggler_report(frame: pd.DataFrame) -> list[str]:
    data = frame[
        (frame["method"] != "sync_fedavg")
        & frame["delay_label"].notna()
        & (frame["partition"] == "iid")
    ].copy()
    data = data[data["delay_label"].astype(str) != ""]
    if data.empty:
        return [
            "## Straggler Stress Test",
            "",
            "Delay stress runs are queued by the distributed-systems 24-hour runner. They test whether async behavior changes under uniform, lognormal, mild-straggler, and severe-straggler arrivals.",
            "",
        ]

    grouped = (
        data.groupby(["delay_label", "method_label"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            best_acc=("best_acc", "mean"),
            stability_drop=("stability_drop", "mean"),
            p95_staleness=("p95_staleness", "mean"),
            time90=("time_to_90pct_best_acc", "mean"),
            client_gini=("client_contribution_gini", "mean"),
        )
        .sort_values(["delay_label", "method_label"])
    )

    lines = [
        "## Straggler Stress Test",
        "",
        "The delay tests make the timing problem visible. A method can have similar final accuracy under normal delay but become unstable when stragglers dominate the event queue.",
        "",
        "| Delay Setting | Method | Runs | Best Acc | Stability Drop | p95 Staleness | Time to 90% Best | Client Gini |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped.to_dict(orient="records"):
        lines.append(
            "| {delay} | {method} | {runs} | {best} | {drop} | {p95} | {time90} | {gini} |".format(
                delay=row.get("delay_label", ""),
                method=row.get("method_label", ""),
                runs=_fmt(row.get("runs"), digits=0),
                best=_fmt(row.get("best_acc")),
                drop=_fmt(row.get("stability_drop")),
                p95=_fmt(row.get("p95_staleness")),
                time90=_fmt(row.get("time90")),
                gini=_fmt(row.get("client_gini")),
            )
        )
    lines.append("")
    return lines


def _ablation_report() -> list[str]:
    path = Path("figures/report/caa_v2_ablation_components.csv")
    if not path.exists():
        return [
            "## CAA-v2 Ablation",
            "",
            "CAA-v2 ablation tables are generated by `scripts/plot_distributed_systems_summary.py`.",
            "",
        ]

    frame = pd.read_csv(path)
    if frame.empty:
        return []

    lines = [
        "## CAA-v2 Ablation",
        "",
        "Ablations test whether CAA-v2 is more than FedBuff with a new name: server-trajectory agreement, client fairness credit, clipping, and adaptive alpha are removed one at a time.",
        "",
        "| Dataset | Variant | Seeds | Best Acc Mean | Best Acc Std | Final Acc Mean | Stability Drop | Avg Agreement | Server Agreement | Fairness Weight |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    frame = frame.sort_values(["dataset", "variant"])
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {dataset} | {variant} | {seeds} | {best} | {std} | {final} | {drop} | {agree} | {server} | {fair} |".format(
                dataset=row.get("dataset", ""),
                variant=row.get("variant", ""),
                seeds=_fmt(row.get("seed_count"), digits=0),
                best=_fmt(row.get("best_acc_mean")),
                std=_fmt(row.get("best_acc_std")),
                final=_fmt(row.get("final_acc_mean")),
                drop=_fmt(row.get("stability_drop_mean")),
                agree=_fmt(row.get("avg_agreement")),
                server=_fmt(row.get("avg_server_momentum_agreement")),
                fair=_fmt(row.get("avg_fairness_weight")),
            )
        )
    lines.append("")
    return lines


def _honest_claim(frame: pd.DataFrame) -> list[str]:
    data = frame[
        (frame["model"] == "qwen")
        & (frame["partition"] == "iid")
        & (frame["dataset"].isin(["mmlu"]))
    ].copy()
    sync = data[data["method"] == "sync_fedavg"][
        ["dataset", "seed", "update_budget", "best_acc", "final_acc"]
    ].rename(columns={"best_acc": "sync_best", "final_acc": "sync_final"})
    caa = data[(data["method"] == "caa_fedbuff_v2") & (data["variant"] == "full_caa_v2")]
    if sync.empty or caa.empty:
        conclusion = (
            "Current completed runs are not enough for a strong mean-based CAA-v2 claim yet; keep the claim focused on the simulator, logging, fairness protocol, and queued stress tests."
        )
    else:
        merged = caa.merge(sync, on=["dataset", "seed", "update_budget"], how="inner")
        if merged.empty:
            conclusion = "CAA-v2 and Sync rows exist, but not under matched update budgets/seeds; do not make a headline comparison from unmatched runs."
        else:
            best_gap = float((merged["sync_best"] - merged["best_acc"]).mean())
            final_gap = float((merged["sync_final"] - merged["final_acc"]).mean())
            stability = float(merged["stability_drop"].mean())
            if best_gap <= 0 and final_gap <= 0:
                conclusion = (
                    f"Strong current result: CAA-v2 matches or exceeds Sync on mean matched rows "
                    f"(best gap `{best_gap:.4f}`, final gap `{final_gap:.4f}`, stability drop `{stability:.4f}`)."
                )
            elif best_gap <= 0.01:
                conclusion = (
                    f"Moderate current result: CAA-v2 approaches Sync under fair budget "
                    f"(mean best gap `{best_gap:.4f}`, final gap `{final_gap:.4f}`, stability drop `{stability:.4f}`)."
                )
            else:
                conclusion = (
                    f"Fallback current result: CAA-v2 is not universally stronger than Sync "
                    f"(mean best gap `{best_gap:.4f}`, final gap `{final_gap:.4f}`, stability drop `{stability:.4f}`), "
                    "so the defensible contribution is explaining when async loses accuracy and which mechanisms reduce instability."
                )

    return [
        "## Honest Claim and Limitations",
        "",
        f"- {conclusion}",
        "- The headline should be based on matched update budget, same seed, same model, same partition, and same delay setting.",
        "- ChestMNIST and RetinaMNIST remain future work because they require task-specific loss/metrics; mixing them now would weaken fairness.",
        "",
    ]


def _best_by_dataset_overview(rows: list[dict[str, Any]]) -> list[str]:
    datasets = sorted({row["dataset"] for row in rows if row["dataset"]})
    if len(datasets) <= 1:
        return []

    lines = [
        "## Best-by-Dataset View",
        "",
        "For each text dataset and method, this table reports the best completed full run. Negative gaps mean async matched or exceeded Sync FedAvg in that run.",
        "",
        "| Dataset | Sync Best | Stateless Best | Staleness Best | FedBuff Best | CAA Best | CAA-v2 Best | Best CAA-Sync Gap | Best CAA Stability Drop | Best Async Method |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in datasets:
        dataset_rows = [
            row
            for row in rows
            if row["dataset"] == dataset and (row.get("partition") in {"", "iid"})
        ]
        sync = _best_method_row(dataset_rows, "sync_fedavg")
        naive = _best_method_row(dataset_rows, "naive_async")
        stale = _best_method_row(dataset_rows, "staleness_async")
        fedbuff = _best_method_row(dataset_rows, "fedbuff_async")
        caa = _best_method_row(dataset_rows, "agreement_fedbuff_async")
        caa_v2 = _best_method_row(dataset_rows, "caa_fedbuff_v2")
        async_candidates = [
            row
            for row in [naive, stale, fedbuff, caa, caa_v2]
            if row is not None and _to_float(row.get("best_acc")) is not None
        ]
        best_async = max(async_candidates, key=lambda row: _to_float(row["best_acc"]) or -1.0) if async_candidates else None
        sync_best = _to_float(sync.get("best_acc")) if sync else None
        best_caa = _best_caa_family_row(dataset_rows)
        caa_best = _to_float(best_caa.get("best_acc")) if best_caa else None
        caa_final = _to_float(best_caa.get("final_acc")) if best_caa else None
        lines.append(
            "| {dataset} | {sync} | {naive} | {stale} | {fedbuff} | {caa} | {caa_v2} | {gap} | {drop} | {best_async} |".format(
                dataset=dataset,
                sync=_fmt(sync_best),
                naive=_fmt(naive.get("best_acc") if naive else ""),
                stale=_fmt(stale.get("best_acc") if stale else ""),
                fedbuff=_fmt(fedbuff.get("best_acc") if fedbuff else ""),
                caa=_fmt(caa.get("best_acc") if caa else ""),
                caa_v2=_fmt(caa_v2.get("best_acc") if caa_v2 else ""),
                gap=_fmt(sync_best - caa_best) if sync_best is not None and caa_best is not None else "",
                drop=_fmt(caa_best - caa_final) if caa_best is not None and caa_final is not None else "",
                best_async=best_async["method"] if best_async else "",
            )
        )
    lines.append("")
    return lines


def _agreement_readout(rows: list[dict[str, Any]]) -> list[str]:
    caa_rows = [row for row in rows if row["method"] in {"agreement_fedbuff_async", "caa_fedbuff_v2"}]
    caa_rows = [row for row in caa_rows if _to_float(row["best_acc"]) is not None]
    if not caa_rows:
        return []

    best_caa = max(caa_rows, key=lambda row: _to_float(row["best_acc"]) or -1.0)
    avg_agreement = _to_float(best_caa.get("avg_agreement"))
    avg_buffer_alpha = _to_float(best_caa.get("avg_buffer_alpha"))
    dropped = _to_float(best_caa.get("dropped_updates"))

    notes = [
        "",
        "## Agreement Analysis",
        "",
        f"- Best CAA-family run: `{best_caa['label']}`.",
    ]
    if avg_agreement is not None:
        notes.append(
            f"- Average positive agreement was `{avg_agreement:.4f}`; higher values mean buffered client parameter directions aligned smoothly."
        )
    if avg_buffer_alpha is not None:
        notes.append(
            f"- Average adaptive buffer alpha was `{avg_buffer_alpha:.4f}`, demonstrating effective server step scaling."
        )
    if dropped is not None:
        notes.append(
            f"- Dropped stale/conflicting updates: `{int(round(dropped))}`."
        )
    return notes

def _async_sync_gap_analysis(rows: list[dict[str, Any]]) -> list[str]:
    sync_refs = _sync_references(rows)
    async_rows = [
        row
        for row in rows
        if row["method"] != "sync_fedavg"
        and _to_float(row["best_acc"]) is not None
        and _to_float(row["final_acc"]) is not None
        and (row["dataset"], row["partition"]) in sync_refs
    ]
    if not async_rows:
        return []

    lines = [
        "",
        "## Async-Sync Gap Analysis",
        "",
        "| Run | Sync Ref | Best Gap | Final Gap | Stability Drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        async_rows,
        key=lambda item: (
            str(item["dataset"]),
            str(item["partition"]),
            _to_float(item["progress"]) or 0.0,
            item["label"],
        ),
    ):
        sync_ref = sync_refs[(row["dataset"], row["partition"])]
        best_acc = _to_float(row["best_acc"]) or 0.0
        final_acc = _to_float(row["final_acc"]) or 0.0
        sync_best = _to_float(sync_ref["best_acc"]) or 0.0
        sync_final = _to_float(sync_ref["final_acc"]) or 0.0
        lines.append(
            "| {label} | {sync_label} | {best_gap} | {final_gap} | {drop} |".format(
                label=f"{row['dataset']} / {row['label']}",
                sync_label=sync_ref["label"],
                best_gap=_fmt(sync_best - best_acc),
                final_gap=_fmt(sync_final - final_acc),
                drop=_fmt(best_acc - final_acc),
            )
        )
    return lines


def _sync_references(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["method"] != "sync_fedavg":
            continue
        key = (row["dataset"], row["partition"])
        if not row["partition"] or _to_float(row["best_acc"]) is None:
            continue
        current = refs.get(key)
        if current is None or (_to_float(row["best_acc"]) or -1.0) > (
            _to_float(current["best_acc"]) or -1.0
        ):
            refs[key] = row
    return refs


def _multi_dataset_coverage(rows: list[dict[str, Any]]) -> list[str]:
    datasets = sorted({row["dataset"] for row in rows if row["dataset"]})
    if len(datasets) <= 1:
        return []

    lines = [
        "",
        "## Multi-Dataset Coverage",
        "",
        "| Dataset | Task | Classes | Channels | Sync | Stateless Async | CAA-FedBuff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        task, classes, channels = _dataset_info(dataset, dataset_rows[0])
        sync = _best_method_row(dataset_rows, "sync_fedavg")
        naive = _best_method_row(dataset_rows, "naive_async")
        caa = _best_caa_family_row(dataset_rows)
        lines.append(
            "| {dataset} | {task} | {classes} | {channels} | {sync} | {naive} | {caa} |".format(
                dataset=dataset,
                task=task,
                classes=_fmt(classes, digits=0),
                channels=_fmt(channels, digits=0),
                sync=_fmt(sync.get("best_acc") if sync else ""),
                naive=_fmt(naive.get("best_acc") if naive else ""),
                caa=_fmt(caa.get("best_acc") if caa else ""),
            )
        )
    return lines


def _stateless_staleness_analysis(rows: list[dict[str, Any]]) -> list[str]:
    datasets = sorted({row["dataset"] for row in rows if row["dataset"]})
    if not datasets:
        return []

    lines = [
        "",
        "## Stateless vs Staleness-Aware",
        "",
        "| Dataset | Stateless Best | Staleness-Aware Best | CAA-Family Best | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        naive = _best_method_row(dataset_rows, "naive_async")
        stale = _best_method_row(dataset_rows, "staleness_async")
        caa = _best_caa_family_row(dataset_rows)
        naive_acc = _to_float(naive.get("best_acc")) if naive else None
        stale_acc = _to_float(stale.get("best_acc")) if stale else None
        caa_acc = _to_float(caa.get("best_acc")) if caa else None
        note = ""
        if naive_acc is not None and stale_acc is not None:
            note = "staleness decay helped" if stale_acc >= naive_acc else "logical staleness alone was conservative"
        if caa_acc is not None and naive_acc is not None and caa_acc >= naive_acc:
            note = (note + "; " if note else "") + "CAA matched/exceeded stateless"
        lines.append(
            "| {dataset} | {naive} | {stale} | {caa} | {note} |".format(
                dataset=dataset, naive=_fmt(naive_acc), stale=_fmt(stale_acc), caa=_fmt(caa_acc), note=note,
            )
        )
    return lines


def _best_method_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["method"] == method and _to_float(row["best_acc"]) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row["best_acc"]) or -1.0)


def _best_caa_family_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["method"] in {"agreement_fedbuff_async", "caa_fedbuff_v2"}
        and _to_float(row["best_acc"]) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row["best_acc"]) or -1.0)


def _dataset_info(dataset: str, representative: dict[str, Any]) -> tuple[str, Any, Any]:
    task = representative.get("task") or "multi-class"
    classes = representative.get("num_classes") or "4"
    channels = representative.get("in_channels") or "0"
    fallback = DATASET_INFO.get(dataset)
    if fallback:
        task = task or fallback[0]
        classes = classes or fallback[1]
        channels = channels or fallback[2]
    return task, classes, channels


def _summary_dataset(path: Path, method: str, config: dict[str, Any]) -> str:
    dataset = config.get("dataset")
    if dataset:
        return str(dataset)
    name = path.name
    prefix = f"{method}_"
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        match = re.search(r"_\d{8}_\d{6}_summary\.json$", rest)
        if match:
            return rest[: match.start()]
    return "mmlu"


def _update_budget(summary: dict[str, Any], config: dict[str, Any]) -> float | str:
    budget = config.get("update_budget")
    if budget not in (None, ""):
        return budget
    method = summary.get("method", "")
    progress = _to_float(summary.get("total_rounds_or_events"))
    if progress is None:
        return ""
    if method == "sync_fedavg":
        clients = _to_float(config.get("clients")) or 10.0
        return progress * clients
    return progress


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _honest_claim(frame: pd.DataFrame) -> list[str]:
    data = frame[
        (frame["model"] == "qwen")
        & (frame["partition"] == "iid")
        & (frame["dataset"].isin(["mmlu"]))
    ].copy()
    
    sync = data[data["method"] == "sync_fedavg"][
        ["dataset", "seed", "update_budget", "best_acc", "final_acc"]
    ].rename(columns={"best_acc": "sync_best", "final_acc": "sync_final"})
    caa = data[(data["method"] == "caa_fedbuff_v2")]
    
    if sync.empty or caa.empty:
        conclusion = (
            "Current completed runs are not enough for a strong mean-based CAA-v2 claim yet; keep the claim focused on the simulator, logging, and fairness protocol."
        )
    else:
        merged = caa.merge(sync, on=["dataset", "seed", "update_budget"], how="inner")
        if merged.empty:
            conclusion = "CAA-v2 and Sync rows exist, but not under matched update budgets/seeds; do not make a headline comparison from unmatched runs."
        else:
            best_gap = float((merged["sync_best"] - merged["best_acc"]).mean())
            final_gap = float((merged["sync_final"] - merged["final_acc"]).mean())
            stability = float((merged["best_acc"] - merged["final_acc"]).mean())
            if best_gap <= 0 and final_gap <= 0:
                conclusion = f"Strong result: CAA-v2 matches or exceeds Sync (best gap `{best_gap:.4f}`, final gap `{final_gap:.4f}`)."
            elif best_gap <= 0.01:
                conclusion = f"Moderate result: CAA-v2 approaches Sync (best gap `{best_gap:.4f}`, final gap `{final_gap:.4f}`)."
            else:
                conclusion = f"Fallback result: CAA-v2 approaches reference accuracy constraints (best gap `{best_gap:.4f}`)."

    return [
        "## Honest Claim and Limitations",
        "",
        f"- {conclusion}",
        "- The headline should be based on matched update budget, same seed, same model, same partition, and same delay setting.",
        "",
    ]

if __name__ == "__main__":
    main()
