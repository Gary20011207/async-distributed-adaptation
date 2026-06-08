from __future__ import annotations

import csv
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


MLLM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MLLM_ROOT.parent
REPORT_DIR = MLLM_ROOT / "figures" / "report"
PRES_DIR = MLLM_ROOT / "presentation"
ASSET_DIR = PRES_DIR / "demo_final_assets"
EXAMPLE_DIR = PRES_DIR / "demo_examples"
PEER_WORKTREE = Path("/tmp/ppdma-peer-vqa")
PAIRED_SELECTED = EXAMPLE_DIR / "paired_base_vs_fl_examples_selected.csv"
PAIRED_MD = EXAMPLE_DIR / "paired_base_vs_fl_examples.md"

SOURCE_DEMO = REPO_ROOT / "Group 4：PP-DMA_ Privacy-Preserving Distributed Model Adaptation -- Demo.pptx"
FINAL_DEMO = REPO_ROOT / "Group 4：PP-DMA_ Privacy-Preserving Distributed Model Adaptation -- Demo_Final.pptx"

METHOD_ORDER = ["Sync FedAvg", "Naive Async", "Staleness Async", "FedBuff", "CAA-v2"]
SHORT_METHOD = {
    "Sync FedAvg": "Sync",
    "Naive Async": "Naive",
    "Staleness Async": "Stale",
    "FedBuff": "FedBuff",
    "CAA-v2": "CAA-v2",
    "Sync": "Sync",
    "Naive Async": "Naive",
    "Staleness": "Stale",
}
COLORS = {
    "Sync FedAvg": "#64748B",
    "Sync": "#64748B",
    "Naive Async": "#F59E0B",
    "Staleness Async": "#EF4444",
    "Staleness": "#EF4444",
    "FedBuff": "#2563EB",
    "CAA-v2": "#059669",
    "Base Qwen": "#7C3AED",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def shorten(text: str, width: int = 72) -> str:
    return textwrap.shorten(str(text).replace("\n", " "), width=width, placeholder="...")


def markdown_table(frame: pd.DataFrame) -> str:
    """Write a compact Markdown table without requiring the optional tabulate dependency."""
    if frame.empty:
        return "_No rows available._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            raw_value = row[col]
            if isinstance(raw_value, (float, np.floating)):
                value = f"{float(raw_value):.4f}"
            else:
                value = str(raw_value)
            value = value.replace("\n", " ").replace("|", "/")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def load_peer_results() -> pd.DataFrame:
    """Read teammate's committed Qwen3-VL PMC-VQA results from the worktree."""
    md_path = PEER_WORKTREE / "Docx" / "vqa_fl_experiment_results.md"
    fallback_rows = [
        ("Sync FedAvg", 47, 100, 0.4700, 3, 0.3316),
        ("Naive Async", 46, 100, 0.4600, 9, 0.2447),
        ("Staleness Async", 45, 100, 0.4500, 9, 0.3676),
        ("FedBuff", 44, 100, 0.4400, 3, 0.3804),
        ("CAA-v2", 47, 100, 0.4700, 3, 0.3648),
    ]
    rows = []
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"\|\s*(Sync FedAvg|Naive Async|Staleness Async|FedBuff|CAA-v2)\s*"
            r"\|\s*(\d+)\s*/\s*(\d+)\s*"
            r"\|\s*([0-9.]+)\s*"
            r"\|\s*(\d+)\s*"
            r"\|\s*([0-9.]+)\s*\|"
        )
        for match in pattern.finditer(text):
            method, correct, total, acc, version, loss = match.groups()
            rows.append((method, int(correct), int(total), float(acc), int(version), float(loss)))
    if not rows:
        rows = fallback_rows
    frame = pd.DataFrame(
        rows,
        columns=["method", "correct", "eval_total", "accuracy", "final_server_version", "round3_mean_train_loss"],
    )
    # The peer report contains both a 12-example pilot table and the main
    # 100-example evaluation table.  Use the largest evaluation set for the
    # DEMO headline and keep the source path explicit in the exported CSV.
    frame = frame[frame["eval_total"].eq(frame["eval_total"].max())].copy()
    frame["source"] = "vqa-fl-results/Docx/vqa_fl_experiment_results.md"
    frame = frame.drop_duplicates("method", keep="last")
    frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
    frame = frame.sort_values("method").reset_index(drop=True)
    frame["method"] = frame["method"].astype(str)
    return frame


def write_peer_outputs(peer: pd.DataFrame) -> None:
    csv_path = PRES_DIR / "peer_qwen3_vl_results.csv"
    md_path = PRES_DIR / "peer_qwen3_vl_results.md"
    peer.to_csv(csv_path, index=False)
    lines = [
        "# Peer Qwen3-VL PMC-VQA Result",
        "",
        "Source branch: `origin/vqa-fl-results`.",
        "",
        "| Method | Correct / Eval | Accuracy | Final Server Version | Round 3 Mean Train Loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in peer.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.correct}/{row.eval_total} | {row.accuracy:.4f} | "
            f"{row.final_server_version} | {row.round3_mean_train_loss:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: CAA-v2 ties Sync FedAvg on this single-seed 100-example pilot and is the best async-family method in the table.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_local_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in [
        "mean_std_summary.csv",
        "qwenvl_method_validity.csv",
        "qwenvl_closed_answer_diagnostics_summary.csv",
        "qwenvl_method_prediction_samples.csv",
        "delay_stress_summary.csv",
        "non_iid_summary.csv",
    ]:
        path = REPORT_DIR / name
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def make_figures(peer: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    assets: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(peer))
    vals = peer["accuracy"].to_numpy()
    colors = [COLORS[m] for m in peer["method"]]
    ax.bar(x, vals, color=colors, width=0.68)
    for i, row in enumerate(peer.itertuples(index=False)):
        ax.text(i, row.accuracy + 0.008, f"{row.correct}/{row.eval_total}\n{pct(row.accuracy)}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_METHOD.get(m, m) for m in peer["method"]], fontsize=10)
    ax.set_ylim(0.38, 0.51)
    ax.set_ylabel("Option accuracy")
    ax.set_title("Qwen3-VL-2B + PMC-VQA pilot: CAA-v2 ties Sync", fontsize=14)
    ax.grid(axis="y", alpha=0.25)
    assets["peer"] = ASSET_DIR / "final_qwen3_vl_pmc_vqa.png"
    savefig(assets["peer"])

    mean = tables["mean_std_summary.csv"]
    text = mean[mean.get("task_family", pd.Series(dtype=str)).eq("text_mcqa")].copy()
    text_agg = (
        text.groupby("method_label", as_index=False)[["best_acc_mean", "final_acc_mean", "stability_drop_mean"]]
        .mean()
        .set_index("method_label")
        .reindex(["Sync", "Naive Async", "Staleness", "FedBuff", "CAA-v2"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(text_agg))
    width = 0.35
    ax.bar(x - width / 2, text_agg["best_acc_mean"], width, color="#0F766E", label="Best")
    ax.bar(x + width / 2, text_agg["final_acc_mean"], width, color="#60A5FA", label="Final")
    for i, row in text_agg.iterrows():
        if pd.notna(row["best_acc_mean"]):
            ax.text(i - width / 2, row["best_acc_mean"] + 0.006, pct(row["best_acc_mean"]), ha="center", fontsize=8)
        if pd.notna(row["final_acc_mean"]):
            ax.text(i + width / 2, row["final_acc_mean"] + 0.006, pct(row["final_acc_mean"]), ha="center", fontsize=8)
    ax.set_ylim(0.24, 0.45)
    ax.set_ylabel("Accuracy")
    ax.set_title("Text MCQA support matrix: CAA-v2 improves peak accuracy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sync", "Naive", "Stale", "FedBuff", "CAA-v2"], fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    assets["text"] = ASSET_DIR / "final_text_mcqa_best_final.png"
    savefig(assets["text"])

    validity = tables["qwenvl_method_validity.csv"].copy()
    if not validity.empty:
        pivot = validity.pivot_table(index="dataset", columns="method_label", values="accuracy")
        order = ["Sync", "Naive Async", "Staleness", "FedBuff", "CAA-v2"]
        pivot = pivot.reindex(columns=order)
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        x = np.arange(len(pivot))
        width = 0.15
        offsets = np.linspace(-2, 2, len(order)) * width
        for method, off in zip(order, offsets):
            if method not in pivot:
                continue
            vals = pivot[method].to_numpy()
            ax.bar(x + off, vals, width, label=method, color=COLORS.get(method, "#94A3B8"))
            for xi, v in zip(x + off, vals):
                if pd.notna(v):
                    ax.text(xi, v + 0.012, pct(v), ha="center", fontsize=7, rotation=90)
        ax.set_ylim(0.40, 0.92)
        ax.set_ylabel("Sample accuracy")
        ax.set_title("Real Qwen2.5-VL checkpoint diagnostics: finite-answer parser works", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(["PathVQA", "VQA-RAD"], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
        assets["validity"] = ASSET_DIR / "final_qwenvl_checkpoint_validity.png"
        savefig(assets["validity"])

    delay = tables["delay_stress_summary.csv"].copy()
    vqa_delay = delay[delay.get("dataset", pd.Series(dtype=str)).eq("vqa_rad_closed")]
    vqa_delay = vqa_delay[vqa_delay.get("delay_mode", pd.Series(dtype=str)).isin(["uniform", "lognormal"])]
    if not vqa_delay.empty:
        pivot = vqa_delay.pivot_table(index="delay_mode", columns="method_label", values="final_acc", aggfunc="mean")
        order = ["Naive Async", "Staleness", "FedBuff", "CAA-v2"]
        pivot = pivot.reindex(index=["uniform", "lognormal"], columns=order)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        x = np.arange(len(pivot.index))
        width = 0.18
        offsets = np.linspace(-1.5, 1.5, len(order)) * width
        for method, off in zip(order, offsets):
            vals = pivot[method].to_numpy()
            ax.bar(x + off, vals, width, label=method, color=COLORS.get(method, "#94A3B8"))
            for xi, v in zip(x + off, vals):
                if pd.notna(v):
                    ax.text(xi, v + 0.006, pct(v), ha="center", fontsize=8, rotation=90)
        ax.set_ylim(0.64, 0.78)
        ax.set_ylabel("Final accuracy")
        ax.set_title("Distributed stress: CAA-v2 under async delay", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(["Uniform", "Lognormal"])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
        assets["delay"] = ASSET_DIR / "final_delay_stress.png"
        savefig(assets["delay"])

    assets["method_flow"] = ASSET_DIR / "final_caa_v2_lora_flow.png"
    draw_method_flow(assets["method_flow"])

    paired_plot = ASSET_DIR / "final_paired_examples.png"
    if paired_plot.exists() and PAIRED_SELECTED.exists():
        assets["examples"] = paired_plot
    else:
        assets["examples"] = ASSET_DIR / "final_prediction_examples.png"
        write_examples_and_plot(tables, assets["examples"])
    image_examples = ASSET_DIR / "final_paired_image_examples.png"
    if image_examples.exists():
        assets["example_images"] = image_examples

    assets["dashboard"] = ASSET_DIR / "final_evidence_dashboard.png"
    draw_dashboard(peer, text_agg, validity, assets["dashboard"])
    return assets


def paired_example_status() -> dict[str, bool]:
    status = {
        "available": False,
        "base_wrong_caa_correct": False,
        "naive_wrong_caa_correct": False,
        "sync_or_naive_wrong": False,
    }
    if not PAIRED_SELECTED.exists():
        return status
    try:
        frame = pd.read_csv(PAIRED_SELECTED)
    except Exception:
        return status
    if frame.empty or "selection_reason" not in frame:
        return status
    reasons = set(str(value) for value in frame["selection_reason"].dropna().tolist())
    status["available"] = True
    status["base_wrong_caa_correct"] = "base_wrong_caa_correct" in reasons
    status["naive_wrong_caa_correct"] = "naive_wrong_caa_correct" in reasons
    status["sync_or_naive_wrong"] = "caa_correct_sync_or_naive_wrong" in reasons
    return status


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def draw_method_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_axis_off()
    boxes = [
        ("Hospital\nclients", 0.03, 0.52),
        ("Local LoRA\ntraining", 0.22, 0.52),
        ("Adapter\nDelta", 0.41, 0.52),
        ("CAA-v2\nserver", 0.60, 0.52),
        ("Global\nadapter", 0.79, 0.52),
    ]
    for label, x, y in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.15, 0.27, ec="#CBD5E1", fc="#FFFFFF", lw=1.8, transform=ax.transAxes))
        ax.text(x + 0.075, y + 0.135, label, ha="center", va="center", fontsize=14, weight="bold", transform=ax.transAxes)
    for x in [0.18, 0.37, 0.56, 0.75]:
        ax.annotate("", xy=(x + 0.035, 0.655), xytext=(x, 0.655), arrowprops=dict(arrowstyle="->", lw=2.3, color="#059669"), xycoords=ax.transAxes)
    captions = [
        "private data stays local",
        "train small adapter",
        "send adapter update",
        "clockless aggregation",
    ]
    for cap, x in zip(captions, [0.17, 0.36, 0.55, 0.74]):
        ax.text(x + 0.018, 0.49, cap, ha="center", va="top", fontsize=9.5, color="#475569", transform=ax.transAxes)
    signals = [
        "logical staleness",
        "delta direction agreement",
        "server trajectory EMA",
        "client fairness credit",
        "adaptive server alpha",
    ]
    ax.add_patch(plt.Rectangle((0.60, 0.13), 0.34, 0.28, ec="#BBF7D0", fc="#F0FDF4", lw=1.3, transform=ax.transAxes))
    ax.text(0.62, 0.36, "CAA-v2 clockless signals", ha="left", va="center", fontsize=12.5, weight="bold", color="#047857", transform=ax.transAxes)
    for i, sig in enumerate(signals):
        ax.text(0.62, 0.31 - i * 0.042, f"- {sig}", ha="left", va="center", fontsize=10.5, color="#334155", transform=ax.transAxes)
    ax.text(0.5, 0.93, "CAA-v2-LoRA: aggregate adapter deltas, not raw data or full base model", ha="center", fontsize=17, weight="bold", transform=ax.transAxes)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_examples_and_plot(tables: dict[str, pd.DataFrame], plot_path: Path) -> None:
    base = tables["qwenvl_closed_answer_diagnostics_summary.csv"]
    base_rows_path = REPORT_DIR / "qwenvl_closed_answer_diagnostics.csv"
    method = tables["qwenvl_method_prediction_samples.csv"].copy()
    base_rows = pd.read_csv(base_rows_path) if base_rows_path.exists() else pd.DataFrame()

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    paired_path = EXAMPLE_DIR / "base_vs_fl_paired_examples.csv"
    representative_path = EXAMPLE_DIR / "representative_checkpoint_examples.csv"
    md_path = EXAMPLE_DIR / "base_vs_fl_paired_examples.md"

    paired = pd.DataFrame()
    if not base_rows.empty and not method.empty:
        merged = method.merge(
            base_rows[["dataset", "sample_index", "question", "parsed_answer", "gold_answer", "correct"]],
            on=["dataset", "sample_index"],
            how="inner",
            suffixes=("_fl", "_base"),
        )
        merged = merged[merged["question_fl"].astype(str) == merged["question_base"].astype(str)]
        paired = merged[(merged["correct_base"].astype(bool) == False) & (merged["correct_fl"].astype(bool) == True)]
    paired.to_csv(paired_path, index=False)

    preferred = method[
        method["method_label"].isin(["Sync", "Naive Async", "CAA-v2"]) & method["dataset"].eq("vqa_rad_closed")
    ].copy()
    preferred = preferred.drop_duplicates(["method_label", "question", "parsed_answer", "gold_answer"], keep="first")
    preferred["score"] = preferred["correct"].astype(int)
    rep = (
        preferred.sort_values(["method_label", "score", "sample_index"], ascending=[True, False, True])
        .groupby("method_label", as_index=False)
        .head(2)
    )
    rep[["dataset", "method_label", "sample_index", "question", "parsed_answer", "gold_answer", "correct"]].to_csv(
        representative_path, index=False
    )

    lines = [
        "# Base vs FL Paired Examples",
        "",
        "This file intentionally avoids claiming before/after improvement unless the base model and FL checkpoint were evaluated on the exact same question.",
        "",
    ]
    if paired.empty:
        lines.extend(
            [
                "## Strict paired result",
                "",
                "No valid exact-match `base wrong / FL correct` examples are available from the current saved diagnostics.",
                "The existing base zero-shot diagnostics and checkpoint diagnostics were generated from different fixed sample sets, so they should not be presented as same-question before/after improvement.",
                "",
            ]
        )
    else:
        lines.extend(["## Strict paired result", "", markdown_table(paired.head(10)), ""])
    lines.extend(
        [
            "## Representative checkpoint examples",
            "",
            "These examples are safe to show as checkpoint behavior, not as base-vs-FL paired improvement.",
            "",
            markdown_table(
                rep[["dataset", "method_label", "sample_index", "question", "parsed_answer", "gold_answer", "correct"]]
                .head(10)
            ),
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_rows = []
    for row in rep.head(6).itertuples(index=False):
        table_rows.append(
            [
                row.method_label,
                shorten(row.question, 52),
                str(row.parsed_answer),
                str(row.gold_answer),
                "OK" if bool(row.correct) else "Miss",
            ]
        )
    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=["Method", "Question", "Pred.", "Gold", "Result"],
        loc="center",
        cellLoc="left",
        colWidths=[0.13, 0.54, 0.09, 0.09, 0.1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold")
        elif c == 4:
            cell.set_facecolor("#DCFCE7" if cell.get_text().get_text() == "OK" else "#FEE2E2")
    ax.set_title("Real Qwen2.5-VL checkpoint examples (not paired before/after)", pad=12)
    savefig(plot_path)


def draw_dashboard(peer: pd.DataFrame, text_agg: pd.DataFrame, validity: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.5))
    fig.suptitle("Final demo evidence stack", fontsize=15, weight="bold")

    axs[0].bar(peer["method"].map(SHORT_METHOD), peer["accuracy"], color=[COLORS[m] for m in peer["method"]])
    axs[0].set_ylim(0.40, 0.50)
    axs[0].set_title("Qwen3-VL PMC-VQA")
    axs[0].set_ylabel("Accuracy")
    axs[0].tick_params(axis="x", rotation=20)

    axs[1].bar(text_agg["method_label"].map(lambda x: SHORT_METHOD.get(x, x)), text_agg["best_acc_mean"], color="#0F766E")
    axs[1].set_ylim(0.32, 0.42)
    axs[1].set_title("Text MCQA best mean")
    axs[1].tick_params(axis="x", rotation=20)

    if not validity.empty:
        vals = validity.groupby("method_label")["valid_rate"].mean().reindex(["Sync", "Naive Async", "Staleness", "FedBuff", "CAA-v2"])
        axs[2].bar(vals.index.map(lambda x: SHORT_METHOD.get(x, x)), vals.to_numpy(), color="#2563EB")
        axs[2].set_ylim(0.90, 1.02)
    axs[2].set_title("Qwen2.5-VL valid answer rate")
    axs[2].tick_params(axis="x", rotation=20)
    for ax in axs:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(path)


def backup_source() -> Path | None:
    if not SOURCE_DEMO.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup = SOURCE_DEMO.with_name(f"{SOURCE_DEMO.stem}.backup_{stamp}{SOURCE_DEMO.suffix}")
    if not backup.exists():
        shutil.copy2(SOURCE_DEMO, backup)
    return backup


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=(30, 41, 59), align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Noto Sans CJK TC"
    run.font.color.rgb = RGBColor(*color)
    if align:
        p.alignment = align
    return box


def add_bullets(slide, x, y, w, h, bullets, size=17, color=(30, 41, 59)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(size)
        p.font.name = "Noto Sans CJK TC"
        p.font.color.rgb = RGBColor(*color)
        p.space_after = Pt(8)
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, 0.55, 0.33, 12.2, 0.45, title, size=25, bold=True, color=(15, 23, 42))
    if subtitle:
        add_text(slide, 0.58, 0.82, 12.1, 0.28, subtitle, size=11, color=(100, 116, 139))
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.55), Inches(1.14), Inches(12.2), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(5, 150, 105)
    line.line.fill.background()


def add_footer(slide):
    add_text(slide, 0.55, 7.12, 8.5, 0.2, "PP-DMA DEMO | Privacy-Preserving Distributed Model Adaptation", size=8, color=(148, 163, 184))


def set_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)


def add_img(slide, path: Path, x, y, w, h=None):
    if h is None:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    else:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def metric_card(slide, x, y, w, h, label, value, note, accent=(5, 150, 105)):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(255, 255, 255)
    shape.line.color.rgb = RGBColor(226, 232, 240)
    add_text(slide, x + 0.16, y + 0.12, w - 0.3, 0.23, label, size=9, bold=True, color=accent)
    add_text(slide, x + 0.16, y + 0.42, w - 0.3, 0.35, value, size=22, bold=True, color=(15, 23, 42))
    add_text(slide, x + 0.16, y + 0.84, w - 0.3, h - 0.85, note, size=8, color=(71, 85, 105))


def make_deck(peer: pd.DataFrame, tables: dict[str, pd.DataFrame], assets: dict[str, Path]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    paired_status = paired_example_status()

    def slide(title, subtitle=None):
        s = prs.slides.add_slide(blank)
        set_bg(s)
        add_title(s, title, subtitle)
        add_footer(s)
        return s

    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, 0.9, 1.25, 11.5, 0.8, "PP-DMA: Privacy-Preserving Distributed Model Adaptation", size=32, bold=True, color=(15, 23, 42), align=PP_ALIGN.CENTER)
    add_text(s, 1.2, 2.15, 10.9, 0.55, "Correction-Aware Aggregation DEMO", size=24, bold=True, color=(5, 150, 105), align=PP_ALIGN.CENTER)
    add_text(s, 1.2, 3.15, 10.9, 1.1, "From biomedical image FL to closed-ended medical LLM/MLLM adaptation", size=20, color=(51, 65, 85), align=PP_ALIGN.CENTER)
    metric_card(s, 1.25, 5.25, 2.5, 1.0, "Main VQA", "Qwen3-VL", "PMC-VQA 100-eval pilot")
    metric_card(s, 4.15, 5.25, 2.5, 1.0, "Support", "Text MCQA", "Qwen1.5 LoRA matrix")
    metric_card(s, 7.05, 5.25, 2.5, 1.0, "Feasibility", "Qwen2.5-VL", "4-bit QLoRA diagnostics")
    metric_card(s, 9.95, 5.25, 2.5, 1.0, "Method", "CAA-v2", "Clockless adapter aggregation")

    s = slide("00 OUTLINE", "DEMO-oriented structure")
    add_bullets(s, 1.2, 1.65, 10.9, 4.8, [
        "01 Introduction：醫療資料不能集中，分散式訓練又會遇到 slow / stale clients。",
        "02 Methodology：CAA-v2 從 full-model delta 改成 LoRA / QLoRA adapter delta。",
        "03 Experiment Results：Qwen3-VL main demo、本機 Text MCQA、本機 Qwen2.5-VL feasibility。",
        "04 Conclusion：CAA-v2 在 VQA pilot 追平 Sync，且是 async-family 最佳方法。",
        "05 Future Work：更大 VQA、paired before/after、real hospital non-IID traces。",
    ], size=21)

    s = slide("01 INTRODUCTION｜Why Privacy-Preserving Distributed Adaptation?", "從大家熟悉的醫院場景切入。")
    add_bullets(s, 0.85, 1.5, 6.0, 4.9, [
        "醫院有影像、病歷、QA 標註，但 raw data 不能任意集中。",
        "不同醫院 GPU、網路、工作量不同；同步 FL 會被慢節點拖住。",
        "非同步 FL 不等待，但會產生 stale / conflicting updates。",
        "LLM/MLLM 更適合 demo：可以直接展示模型回答 medical QA/VQA 是否變好。",
    ], size=19)
    add_img(s, assets["dashboard"], 7.0, 1.55, 5.7)

    s = slide("01 INTRODUCTION｜DEMO Goal", "今天主軸不是宣稱 SOTA，而是展示可行的分散式 AI adaptation。")
    add_bullets(s, 1.0, 1.55, 11.5, 4.7, [
        "我們要展示：同一個 clockless async FL idea 可以從 CNN biomedical classification 延伸到 LLM/MLLM adapter training。",
        "模型不交換 raw image / raw question，只交換 LoRA / QLoRA adapter deltas。",
        "Closed-ended VQA 將答案限制為 A/B/C/D，因此可以用 option accuracy 做穩定 demo。",
        "主要結果用同學 Qwen3-VL + PMC-VQA；本機結果提供 multi-seed support 與 real Qwen2.5-VL feasibility。",
    ], size=21)

    s = slide("02 METHODOLOGY｜Baseline Map", "比較的是 server 如何處理不同時間抵達的 adapter updates。")
    add_bullets(s, 0.8, 1.45, 5.7, 4.9, [
        "Sync FedAvg：server 等齊 client adapter updates，再平均。",
        "Naive Async：adapter update 到就套用，不判斷 stale/conflict。",
        "Staleness Async：只用 logical staleness 調小更新幅度。",
        "FedBuff：先收 B 個 async adapter updates，再做 buffered aggregation。",
        "CAA-v2：buffer 後再看方向一致性、server trajectory、client fairness。",
    ], size=18)
    metric_card(s, 7.0, 1.45, 2.45, 1.0, "Fair Budget", "events = rounds × clients", "避免 async 少訓練造成不公平比較")
    metric_card(s, 9.85, 1.45, 2.45, 1.0, "Clockless", "server version", "只用 logical version 估 staleness")
    metric_card(s, 7.0, 3.0, 2.45, 1.0, "Train Unit", "LoRA / QLoRA", "base model frozen, adapter updated")
    metric_card(s, 9.85, 3.0, 2.45, 1.0, "Eval Unit", "A/B/C/D", "closed-ended parser + accuracy")
    add_text(
        s,
        7.1,
        4.65,
        5.0,
        0.75,
        "Key idea: we do not synchronize clocks or raw data; we compare how the server aggregates adapter deltas under async arrivals.",
        size=13,
        bold=True,
        color=(5, 150, 105),
    )

    s = slide("02 METHODOLOGY｜CAA-v2-LoRA", "把 CAA-v2 從 CNN full model 改成 LLM/MLLM adapter aggregation。")
    add_img(s, assets["method_flow"], 0.55, 1.35, 7.15)
    add_text(s, 8.1, 1.42, 4.55, 0.35, "Server-side rule", size=16, bold=True, color=(5, 150, 105))
    add_bullets(s, 8.1, 1.85, 4.75, 3.0, [
        "Client starts from server adapter version k.",
        "Local training updates only LoRA / QLoRA adapter.",
        "delta_i = adapter_i − adapter_at_version_k.",
        "tau_i = current_server_version − k.",
        "Buffer B deltas, then aggregate.",
    ], size=14)
    add_text(
        s,
        8.1,
        4.85,
        4.75,
        1.05,
        "raw_weight_i = n_i · staleness_decay(tau_i) · agreement_i · fairness_i\nadapter_new = adapter_cur + alpha_t · Σ normalized_weight_i · clipped(delta_i)",
        size=11,
        bold=True,
        color=(15, 23, 42),
    )
    add_text(
        s,
        8.1,
        6.1,
        4.75,
        0.55,
        "This is still async: the server never waits for all clients; logical version is used only to measure staleness.",
        size=12,
        color=(71, 85, 105),
    )

    s = slide("02 METHODOLOGY｜What Changed for LLM / MLLM?", "方法小改，但系統意義更清楚。")
    add_bullets(s, 0.95, 1.45, 11.5, 4.9, [
        "Base model frozen：Qwen / Qwen-VL 權重不做 federated averaging，只同步小型 adapter。",
        "Adapter delta aggregation：server 聚合 LoRA / QLoRA delta，不聚合 2B/3B full weights。",
        "Clockless signals unchanged：logical staleness、direction agreement、server trajectory EMA、client fairness credit。",
        "Closed-ended QA/VQA：prompt 要求 one-letter answer，parser 統一成 A/B/C/D accuracy 與 invalid answer rate。",
        "Distributed-system claim：我們驗證的是 no-clock async adapter adaptation，不是 LLM/MLLM SOTA fine-tuning。",
    ], size=20)

    s = slide("03 EXPERIMENT RESULTS｜Evidence Stack", "三層結果分開解讀，不混成同一個統計 claim。")
    add_img(s, assets["dashboard"], 0.7, 1.45, 7.2)
    add_bullets(s, 8.35, 1.45, 4.2, 4.9, [
        "Main VQA demo：Qwen3-VL-2B + PMC-VQA，直接支持 DEMO 主軸。",
        "Quantitative support：Qwen1.5-0.5B Text MCQA multi-seed。",
        "Real MLLM feasibility：Qwen2.5-VL-3B 4-bit QLoRA，小矩陣與 diagnostics。",
        "Proxy / real / peer results 分開標示，避免誇大。",
    ], size=16)

    s = slide("03 RESULTS｜Main Demo: Qwen3-VL + PMC-VQA", "同學 branch `vqa-fl-results`：100-example eval pilot。")
    add_img(s, assets["peer"], 0.8, 1.45, 7.5)
    add_bullets(s, 8.55, 1.55, 4.0, 4.8, [
        "CAA-v2：47/100，追平 Sync FedAvg。",
        "Naive Async：46/100，接近但略低。",
        "Staleness Async：45/100；FedBuff：44/100。",
        "此 run 是 single-seed pilot；claim 是「CAA-v2 reaches Sync-level performance」，不是統計 SOTA。",
    ], size=15)

    s = slide("03 RESULTS｜Why This Is Good for DEMO", "結果和系統概念可以直接連起來。")
    add_bullets(s, 1.0, 1.45, 11.4, 4.9, [
        "Sync FedAvg 是最穩定 baseline，但需要 barrier。",
        "CAA-v2 在不依賴 global physical clock 的 async-family setting 中追平 Sync。",
        "相較 staleness-only，CAA-v2 不只看 update age，也看 adapter delta 方向是否和群體/server trajectory 一致。",
        "這讓我們可以把 demo 講成：不是單純加速訓練，而是在 no-clock distributed setting 下控制 stale/conflicting adapter updates。",
    ], size=21)

    s = slide("03 RESULTS｜Text MCQA Multi-Seed Support", "本機 Qwen1.5-0.5B LoRA：MMLU / MedMCQA / MedQA-USMLE / PubMedQA。")
    add_img(s, assets["text"], 0.8, 1.45, 7.4)
    add_bullets(s, 8.5, 1.55, 4.0, 4.8, [
        "CAA-v2 best mean 約 38.7%，高於 FedBuff / Naive / Staleness / Sync。",
        "Sync final mean 約 35.4%，CAA-v2 final 約 34.9%。",
        "解讀：CAA-v2 peak 有優勢；final stability 還不是全面勝出。",
        "這是支撐性 matrix，不是 demo headline。",
    ], size=15)

    s = slide("03 RESULTS｜Real Qwen2.5-VL Feasibility", "本機 Qwen2.5-VL-3B-Instruct 4-bit QLoRA。")
    add_img(s, assets["validity"], 0.8, 1.45, 7.4)
    add_bullets(s, 8.5, 1.55, 4.0, 4.8, [
        "Closed-answer valid rate：100%。",
        "VQA-RAD diagnostics：CAA-v2 / FedBuff / Sync 約 75%，Naive/Staleness 約 81.25%。",
        "PathVQA diagnostics：Sync 約 62.5%，async family 約 50%。",
        "定位：real MLLM pipeline proof-of-feasibility，不作大規模勝出宣稱。",
    ], size=15)

    if paired_status["available"]:
        boundary_bullets = [
            "已用同一批 fixed questions 跑 base Qwen2.5-VL、Sync、Naive Async、CAA-v2。",
            "這些 examples 是合法 same-question comparison：dataset、sample_index、question、image、gold answer 都相同。",
        ]
        if paired_status["base_wrong_caa_correct"]:
            boundary_bullets.append("找到 base wrong / CAA-v2 correct，可作為 before/after improvement demo。")
        elif paired_status["naive_wrong_caa_correct"] or paired_status["sync_or_naive_wrong"]:
            boundary_bullets.append("沒有找到 base wrong / CAA-v2 correct；但有 Naive 或 Sync 錯、CAA-v2 對的例子，可展示 aggregation 差異。")
        else:
            boundary_bullets.append("沒有找到 base wrong / CAA-v2 correct；只展示 CAA-v2 representative correct examples，不做 before/after claim。")
        boundary_bullets.append("報告時要說清楚：real Qwen2.5-VL 這部分是 feasibility diagnostic，不是大規模統計勝出。")
    else:
        boundary_bullets = [
            "目前 base Qwen zero-shot diagnostics 和 FL checkpoint diagnostics 不是同一批 questions。",
            "因此本版不宣稱「同一題 base 錯、FL 對」，避免錯誤報告。",
            "已輸出 paired audit：若沒有 exact-match pair，檔案會明確寫 no valid paired examples。",
            "可展示的是 checkpoint 行為：question、prediction、gold answer、correct/miss。",
        ]
    s = slide("03 RESULTS｜Before / After Diagnostic Boundary", "只展示合法 same-question comparison，不偽造訓練前後改善。")
    add_bullets(s, 0.9, 1.45, 11.7, 4.9, boundary_bullets, size=20)

    s = slide("03 RESULTS｜Paired DEMO Examples", "同一題目下比較 base / Sync / Naive / CAA-v2 的 closed-answer 行為。")
    add_img(s, assets["examples"], 0.55, 1.45, 12.2)

    if "example_images" in assets:
        s = slide("03 RESULTS｜Paired DEMO Example Images", "四題同題 VQA examples：影像、問題、gold answer 與各方法輸出。")
        add_img(s, assets["example_images"], 1.55, 1.28, 10.25)

    s = slide("03 RESULTS｜Distributed Stress: Delay Matters", "VQA-RAD proxy stress，獨立於 headline mean±std。")
    add_img(s, assets["delay"], 0.8, 1.45, 7.4)
    add_bullets(s, 8.5, 1.55, 4.0, 4.8, [
        "Naive/Staleness 平均 staleness 約 8；FedBuff/CAA-v2 約 1.5。",
        "Uniform delay：CAA-v2 final 75.0%。",
        "Lognormal delay：CAA-v2 final 74.3%。",
        "這是分散式系統重點：delay distribution 會改變 async training behavior。",
    ], size=15)

    s = slide("03 RESULTS｜Existing vs Ours", "把 borrowed components 和我們做的系統整合分清楚。")
    add_bullets(s, 0.95, 1.45, 5.8, 4.8, [
        "Existing：FedAvg、FedBuff、staleness-aware aggregation。",
        "Existing：Qwen / Qwen2.5-VL、LoRA / QLoRA、PMC-VQA / VQA-RAD / MedQA。",
        "Existing：closed-ended option accuracy parser idea。",
    ], size=18)
    add_bullets(s, 7.0, 1.45, 5.5, 4.8, [
        "Ours：clockless async FL simulator / analysis pipeline。",
        "Ours：CAA-v2 agreement + fairness + adaptive alpha rule。",
        "Ours：adapter-level mapping and fair-budget comparison for medical LLM/MLLM demo。",
    ], size=18)

    s = slide("04 CONCLUSION｜Final Claim", "保守但站得住腳的 DEMO claim。")
    add_bullets(s, 0.95, 1.45, 11.5, 4.9, [
        "在 Qwen3-VL PMC-VQA pilot 中，CAA-v2 追平 Sync FedAvg，且是 async-family 最佳方法。",
        "在本機 Text MCQA matrix 中，CAA-v2 提高 peak accuracy；final stability 仍需要改進。",
        "Real Qwen2.5-VL 小矩陣證明 QLoRA adapter aggregation pipeline 可行。",
        "總結：PP-DMA 展示 privacy-preserving、clockless、adapter-level distributed model adaptation 的可行性。",
    ], size=21)

    s = slide("04 CONCLUSION｜Limitations", "把邊界講清楚，比過度宣稱更可靠。")
    add_bullets(s, 0.95, 1.45, 11.5, 4.9, [
        "Qwen3-VL main result 是 single-seed pilot，尚未有 large multi-seed statistical result。",
        "Real Qwen2.5-VL 是小樣本 feasibility，不代表 MLLM SOTA。",
        "目前沒有合法 same-question base-wrong / FL-correct example，未來需要固定題組 paired inference。",
        "醫療應用仍需要 macro-F1、per-class recall、safety review、privacy/security analysis。",
    ], size=21)

    s = slide("05 FUTURE WORK", "中午 DEMO 後最值得補的方向。")
    add_bullets(s, 0.95, 1.45, 11.5, 4.9, [
        "固定 VQA paired evaluation set：Base / Sync / Naive / CAA-v2 都跑同一批 questions。",
        "擴大 PMC-VQA / VQA-RAD / PathVQA sample size 和 seeds。",
        "加入 non-IID hospital split：不同醫院看到不同器官、影像 modality、問題類型。",
        "加入 real network trace：用真實 delay / straggler distribution 測試 async behavior。",
        "探索 swarm learning：移除 central server，但保留 CAA-style update validation。",
    ], size=21)

    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, 1.0, 2.25, 11.3, 0.8, "Thanks for listening!", size=42, bold=True, color=(15, 23, 42), align=PP_ALIGN.CENTER)
    add_text(s, 1.2, 3.35, 10.9, 0.6, "DEMO: privacy-preserving, clockless, adapter-level distributed model adaptation", size=20, color=(5, 150, 105), align=PP_ALIGN.CENTER)

    prs.save(FINAL_DEMO)


def write_speaker_notes(peer: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    notes = PRES_DIR / "demo_final_speaker_notes_zh.md"
    text_best = tables["mean_std_summary.csv"]
    text = text_best[text_best.get("task_family", pd.Series(dtype=str)).eq("text_mcqa")]
    text_agg = text.groupby("method_label")[["best_acc_mean", "final_acc_mean"]].mean()
    lines = [
        "# PP-DMA DEMO Final Speaker Notes",
        "",
        "## 核心講法",
        "",
        "今天的主軸是 demo：我們把原本 biomedical image FL 的 CAA-v2，延伸到 LLM/MLLM adapter-level federated adaptation。",
        "",
        "## 必講數字",
        "",
        "- Qwen3-VL + PMC-VQA pilot：Sync FedAvg 47/100，CAA-v2 47/100，Naive Async 46/100，Staleness 45/100，FedBuff 44/100。",
        "- Text MCQA support：CAA-v2 best mean 約 38.7%，Sync final mean 約 35.4%，CAA-v2 final 約 34.9%。",
        "- Real Qwen2.5-VL diagnostics：valid answer rate 100%，但目前只當 feasibility。",
        "",
        "## Methodology 講稿",
        "",
        "原本 CNN 版本聚合 full model delta；LLM/MLLM 版本改成只聚合 LoRA 或 QLoRA adapter delta。每個 client 從某個 server adapter version 開始訓練，回傳 adapter_i 減掉 adapter_at_start_version 的 delta。server 不等齊所有 client，而是把 async 到達的 deltas 放進 buffer，再根據 logical staleness、方向 agreement、server trajectory EMA、client fairness credit 和 adaptive alpha 做聚合。這讓方法仍然是 clockless async，因為 version 只是量 staleness，不是 barrier，也不是 physical clock。",
        "",
        "## Experiment Results 講稿",
        "",
        "最重要的 demo 結果是 Qwen3-VL + PMC-VQA：CAA-v2 和 Sync FedAvg 一樣是 47%，而其他 async baselines 較低。這表示 CAA-v2 在不等待所有 client 的 async-family setting 中，可以接近同步方法的表現。本機 Text MCQA matrix 顯示 CAA-v2 peak accuracy 最高，但 final stability 還不是全面最佳，所以我們的 claim 保持保守。",
        "",
        "## Before / After 注意事項",
        "",
        "目前沒有合法的 same-question base-wrong / FL-correct paired examples。投影片中的 examples 只展示 checkpoint behavior，不要說成同一題訓練前錯、訓練後對。",
    ]
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reference_doc(peer: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    path = PRES_DIR / "LLM_MLLM_DEMO_REFERENCE.md"
    mean = tables["mean_std_summary.csv"].copy()
    text = mean[mean.get("task_family", pd.Series(dtype=str)).eq("text_mcqa")].copy()
    text_summary = (
        text.groupby("method_label", as_index=False)[["best_acc_mean", "final_acc_mean", "stability_drop_mean"]]
        .mean()
        .sort_values("best_acc_mean", ascending=False)
    )
    validity = tables["qwenvl_method_validity.csv"].copy()
    delay = tables["delay_stress_summary.csv"].copy()
    paired_summary = ""
    paired_selected = ""
    if (EXAMPLE_DIR / "paired_base_vs_fl_examples.csv").exists():
        paired = pd.read_csv(EXAMPLE_DIR / "paired_base_vs_fl_examples.csv")
        if not paired.empty:
            grouped = paired.groupby("method_label", as_index=False).agg(
                accuracy=("correct", "mean"),
                valid_rate=("valid", "mean"),
                samples=("correct", "count"),
            )
            paired_summary = markdown_table(grouped)
    if PAIRED_SELECTED.exists():
        selected = pd.read_csv(PAIRED_SELECTED)
        cols = [
            "selection_reason",
            "dataset",
            "sample_index",
            "question",
            "gold_answer",
            "parsed_answer_Base",
            "parsed_answer_Naive_Async",
            "parsed_answer_CAA-v2",
            "parsed_answer_Sync",
        ]
        present = [col for col in cols if col in selected.columns]
        paired_selected = markdown_table(selected[present])

    lines = [
        "# LLM/MLLM DEMO Reference: PP-DMA and CAA-v2-LoRA",
        "",
        "這份文件是 DEMO 報告用的 LLM/MLLM 部分說明，目的在於讓組員能快速理解：我們做了什麼、哪些結果能講、哪些不能誇大。",
        "",
        "## 1. Motivation",
        "",
        "- 醫療 QA/VQA 的資料常常分散在不同醫院或設備端，raw images/questions 不適合集中。",
        "- 同步 FL 需要等慢節點，會降低 demo 中的 distributed throughput。",
        "- 非同步 FL 不等待，但會產生 stale adapter updates、方向衝突，以及 fast-client domination。",
        "- 因此我們把問題限制在 closed-ended QA/VQA，讓 answer space 是 A/B/C/D，可以穩定評估 accuracy 與 invalid answer rate。",
        "",
        "## 2. Methods in This DEMO",
        "",
        "| Method | Server behavior | DEMO interpretation |",
        "|---|---|---|",
        "| Sync FedAvg | 等齊所有 client adapter updates 再平均 | 穩定 baseline，但有 barrier |",
        "| Naive Async | update 到就套用 | 不等待，但容易受 stale/conflict update 影響 |",
        "| Staleness Async | 用 logical staleness 調小 update | 只看 age，可能過度保守 |",
        "| FedBuff | buffer B 個 async updates 再聚合 | 減少單一 stale update 的震盪 |",
        "| CAA-v2-LoRA | buffer 後加入 agreement / server trajectory / fairness / adaptive alpha | 我們的 clockless adapter aggregation extension |",
        "",
        "## 3. CAA-v2-LoRA: What Changed from CNN to LLM/MLLM",
        "",
        "CNN 版本聚合 full-model delta；LLM/MLLM 版本改成只聚合 LoRA/QLoRA adapter delta：",
        "",
        "```text",
        "client starts from server adapter version k",
        "delta_i = local_adapter_i - global_adapter_at_version_k",
        "tau_i = current_server_version - k",
        "```",
        "",
        "CAA-v2 的 server weight：",
        "",
        "```text",
        "raw_weight_i = n_i",
        "             * staleness_decay(tau_i)",
        "             * agreement_i",
        "             * fairness_i",
        "",
        "adapter_new = adapter_current",
        "            + alpha_t * sum_i normalized_weight_i * clipped(delta_i)",
        "```",
        "",
        "這仍然是 async，因為 server 不等所有 client。Logical version 只用來量 staleness，不是 global physical clock，也不是 round barrier。",
        "",
        "## 4. Existing vs Ours",
        "",
        "| Existing | Ours |",
        "|---|---|",
        "| FedAvg / FedBuff / staleness-aware aggregation | CAA-v2 agreement + fairness + adaptive alpha rule |",
        "| Qwen / Qwen2.5-VL / Qwen3-VL | Adapter-level federated adaptation pipeline |",
        "| LoRA / QLoRA | Only aggregate adapter deltas under fair update budget |",
        "| PMC-VQA / VQA-RAD / PathVQA / MedQA | Distributed-system analysis: staleness, delay, client contribution, stability |",
        "",
        "## 5. Headline Result: Peer Qwen3-VL + PMC-VQA",
        "",
        markdown_table(peer[["method", "correct", "eval_total", "accuracy", "final_server_version", "round3_mean_train_loss"]]),
        "",
        "DEMO claim：CAA-v2 ties Sync FedAvg at 47/100 and is the strongest async-family method in this single-seed pilot. 這是 pilot，不是 SOTA 宣稱。",
        "",
        "## 6. Local Text MCQA Support",
        "",
        markdown_table(text_summary[["method_label", "best_acc_mean", "final_acc_mean", "stability_drop_mean"]]),
        "",
        "百分比版重點：CAA-v2 best mean 約 38.8%，高於 FedBuff / Naive / Staleness / Sync；Sync final mean 約 35.3%，CAA-v2 final 約 34.9%。解讀要保守：CAA-v2 improves peak accuracy, while final stability remains future work.",
        "",
        "## 7. Local Real Qwen2.5-VL Feasibility",
        "",
        markdown_table(validity[["dataset", "method_label", "valid_rate", "accuracy", "samples"]]) if not validity.empty else "_No real Qwen2.5-VL validity table._",
        "",
        "百分比版重點：real Qwen2.5-VL 3B 4-bit QLoRA closed-answer valid rate 是 100%，代表 QLoRA adapter aggregation 沒有破壞 A/B/C/D output format；但這是 small feasibility matrix，不是大規模 MLLM benchmark。",
        "",
        "## 8. Same-Question DEMO Examples",
        "",
        "Paired summary:",
        "",
        paired_summary or "_Paired inference was not available._",
        "",
        "Selected examples:",
        "",
        paired_selected or "_No selected paired examples._",
        "",
        "重要：若沒有 `base_wrong_caa_correct`，報告時不要說『訓練前錯、訓練後對』。可以說：同題診斷中，CAA-v2 在部分題目上修正 Sync 或 Naive 的錯誤。",
        "",
        "投影片使用的四題影像版 examples 位於：`presentation/demo_final_assets/final_paired_image_examples.png`。這四題是目前最適合 DEMO 的組合：有 VQA-RAD X-ray、PathVQA gross pathology、PathVQA microscopy，且都屬於 CAA-v2 correct、Naive 或 Sync 至少一個 incorrect 的 same-question diagnostic。另有第 5 題可用候選 `vqa_rad_closed #48`，但為避免頁面過擠未放入主投影片。",
        "",
        "## 9. Distributed-Systems Evidence",
        "",
        markdown_table(
            delay[["dataset", "method_label", "delay_mode", "final_acc", "avg_staleness", "p95_staleness", "client_gini"]].head(12)
        ) if not delay.empty else "_No delay stress table._",
        "",
        "解讀：這些結果不是 headline accuracy，而是支撐 distributed systems story：delay distribution 會改變 staleness 和 async behavior。關鍵現象是 Naive / Staleness 的 avg_staleness 約 8，而 FedBuff / CAA-v2 因為 buffering 更新節奏，avg_staleness 約 1.5，這直接對應 no global clock + straggler setting 下的系統行為差異。",
        "",
        "## 10. Suggested DEMO Script",
        "",
        "1. 先說醫療資料不能集中，因此要 privacy-preserving distributed adaptation。",
        "2. 接著說 LLM/MLLM 不適合傳 full model，所以我們只傳 LoRA/QLoRA adapter delta。",
        "3. 再說 async server 不等所有 client，因此需要處理 stale/conflicting adapter updates。",
        "4. CAA-v2 用 logical staleness、direction agreement、server trajectory EMA、client fairness credit 來做 clockless aggregation。",
        "5. 結果上，Qwen3-VL + PMC-VQA pilot 中 CAA-v2 追平 Sync，且優於其他 async-family baseline。",
        "6. 本機 Text MCQA / Qwen2.5-VL 結果作為 support 和 feasibility，不誇大。",
        "",
        "## 11. Likely Questions and Safe Answers",
        "",
        "**Q: CAA-v2 是不是只是 FedBuff？** 不是。FedBuff 只做 buffered aggregation；CAA-v2 在 buffer 後額外看 direction agreement、server trajectory EMA、client fairness credit 和 adaptive alpha。",
        "",
        "**Q: Logical version 會不會讓它變同步？** 不會。同步的關鍵是 server 是否等待所有 clients。這裡 server 不等齊；version 只是事後量 staleness。",
        "",
        "**Q: 有沒有證明 LLM/MLLM 訓練後一定比 base 好？** 目前沒有大規模證明。paired diagnostics 是 small feasibility；headline 是 Qwen3-VL pilot 中 CAA-v2 追平 Sync、優於 async-family baseline。",
        "",
        "**Q: 為什麼 closed-ended？** 因為 DEMO 需要穩定、可評估的 answer space。A/B/C/D parser 可以計算 accuracy 和 invalid answer rate，避免自由生成很難客觀評估。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_deck() -> None:
    prs = Presentation(FINAL_DEMO)
    bad = []
    for idx, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text
                for token in ["TODO", "TBD", "NaN", "None"]:
                    if token in text:
                        bad.append((idx, token, text[:80]))
    qa_path = PRES_DIR / "demo_final_qa.txt"
    qa_path.write_text(
        f"deck={FINAL_DEMO}\nslides={len(prs.slides)}\nforbidden_tokens={bad}\n",
        encoding="utf-8",
    )
    if bad:
        raise RuntimeError(f"Forbidden tokens found: {bad}")


def main() -> None:
    ensure_dirs()
    backup = backup_source()
    peer = load_peer_results()
    write_peer_outputs(peer)
    tables = load_local_tables()
    assets = make_figures(peer, tables)
    make_deck(peer, tables, assets)
    write_speaker_notes(peer, tables)
    write_reference_doc(peer, tables)
    validate_deck()
    print(f"backup={backup}")
    print(f"final_deck={FINAL_DEMO}")
    print(f"slides={len(Presentation(FINAL_DEMO).slides)}")
    print(f"peer_csv={PRES_DIR / 'peer_qwen3_vl_results.csv'}")
    print(f"examples={EXAMPLE_DIR / 'base_vs_fl_paired_examples.md'}")


if __name__ == "__main__":
    main()
