
from __future__ import annotations

from pathlib import Path
import re
import subprocess

import markdown as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRESENT = ROOT / "r13946001" / "presentation"
ASSETS = PRESENT / "assets"
REPORT = ROOT / "r13946001" / "pathMNIST" / "figures" / "report"

METHOD_ORDER = ["sync_fedavg", "naive_async", "staleness_async", "fedbuff_async", "agreement_fedbuff_async", "caa_fedbuff_v2"]
METHOD_LABEL = {
    "sync_fedavg": "Sync",
    "naive_async": "Naive",
    "staleness_async": "Stale",
    "fedbuff_async": "FedBuff",
    "agreement_fedbuff_async": "CAA-v1",
    "caa_fedbuff_v2": "CAA-v2",
}
COLORS = {
    "Sync": "#2563eb",
    "Naive": "#ef4444",
    "Stale": "#f59e0b",
    "FedBuff": "#7c3aed",
    "CAA-v1": "#10b981",
    "CAA-v2": "#0f766e",
}
DATASET_ORDER = ["pathmnist", "pneumoniamnist", "bloodmnist", "organamnist", "organcmnist", "dermamnist", "octmnist", "breastmnist", "tissuemnist"]
DATASET_SHORT = {
    "pathmnist": "Path",
    "pneumoniamnist": "Pneum.",
    "bloodmnist": "Blood",
    "organamnist": "OrganA",
    "organcmnist": "OrganC",
    "dermamnist": "Derma",
    "octmnist": "OCT",
    "breastmnist": "Breast",
    "tissuemnist": "Tissue",
}

REF_REPLACEMENTS = {
    "../pathMNIST/figures/report/dataset_method_heatmap.png": "assets/proposal_dataset_method_heatmap.png",
    "../pathMNIST/figures/report/async_sync_gap_errorbar.png": "assets/proposal_async_sync_gap_errorbar.png",
    "../pathMNIST/figures/report/stability_drop_errorbar.png": "assets/proposal_stability_drop_errorbar.png",
    "../pathMNIST/figures/report/non_iid_async_sync_gap.png": "assets/proposal_non_iid_async_sync_gap.png",
    "../pathMNIST/figures/report/straggler_staleness_distribution.png": "assets/proposal_straggler_staleness_distribution.png",
    "../pathMNIST/figures/report/straggler_acc_vs_simulated_time.png": "assets/proposal_straggler_acc_vs_simulated_time.png",
    "../pathMNIST/figures/report/caa_v2_ablation_best_acc.png": "assets/proposal_caa_v2_ablation_best_acc.png",
}

CSS = """@page { size: 1280px 720px; margin: 0; }
body { margin:0; background:#eef2f7; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Inter",Arial,sans-serif; }
.slide { width:1280px; min-height:720px; box-sizing:border-box; padding:48px 68px 40px; margin:28px auto; background:#fff; border:1px solid #cbd5e1; box-shadow:0 18px 44px rgba(15,23,42,.13); position:relative; overflow:hidden; page-break-after:always; }
.slide:before { content:""; position:absolute; left:0; top:0; bottom:0; width:12px; background:linear-gradient(180deg,#1d4ed8,#059669); }
.slide-no { position:absolute; top:18px; right:30px; color:#94a3b8; font-weight:700; font-size:18px; }
h1 { font-size:38px; line-height:1.15; margin:0 0 20px; color:#0f172a; }
p,li { font-size:23px; line-height:1.42; }
li { margin:6px 0; }
strong { color:#1d4ed8; }
blockquote { border-left:8px solid #059669; background:#ecfdf5; padding:14px 22px; color:#064e3b; margin:20px 0; font-size:24px; }
pre { background:#0f172a; color:#e2e8f0; padding:16px 20px; border-radius:10px; font-size:18px; line-height:1.30; white-space:pre-wrap; }
code { font-family:Consolas,"JetBrains Mono",monospace; }
table { width:100%; border-collapse:collapse; font-size:17px; }
th { background:#eff6ff; color:#1e3a8a; }
th,td { border:1px solid #cbd5e1; padding:7px 9px; vertical-align:top; }
img { display:block; max-width:100%; max-height:430px; object-fit:contain; margin:8px auto 0; }
a { color:#1d4ed8; }
@media print { body{background:white} .slide{margin:0; box-shadow:none; border:none;} }
"""


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 14,
        "axes.titlesize": 19,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
    mean = read_csv("mean_std_summary.csv")
    final = read_csv("final_method_comparison.csv")
    sys = read_csv("official_system_metrics_summary.csv")
    dist = read_csv("distributed_systems_summary.csv")
    ablation = read_csv("caa_v2_ablation_components.csv")

    plot_system_architecture()
    plot_component_stack_clean()
    plot_method_flow_clean()
    plot_decision_flow_clean()
    plot_novelty_boundary_clean()
    plot_challenge_map_clean()
    plot_results_dashboard_clean(mean)
    plot_caa_v2_gap_clean(final)
    plot_system_metrics_clean(sys, mean)
    plot_dataset_method_heatmap_clean(mean)
    plot_async_sync_gap_clean(mean)
    plot_stability_drop_clean(mean)
    plot_non_iid_gap_clean(dist)
    plot_straggler_staleness_clean(dist)
    plot_straggler_time_clean(dist)
    plot_ablation_clean(ablation)

    update_markdown_refs()
    rebuild_html()
    rebuild_pptx_pdf()


def read_csv(name: str) -> pd.DataFrame:
    path = REPORT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec="#334155", fs=14, weight="bold"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color="#0f172a", linespacing=1.25)


def arrow(ax, start, end, color="#0f172a", lw=2.1):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=5, shrinkB=5))


def plot_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.94, "Event-Driven Clockless FL Simulator", ha="center", fontsize=23, weight="bold")
    top = [(0.06, "Hospital A\nfast client"), (0.25, "Hospital B\nnormal client"), (0.44, "Hospital C\nstraggler"), (0.63, "Hospital D\nnon-IID data")]
    for x, t in top:
        box(ax, x, 0.70, 0.15, 0.14, t, "#dbeafe", fs=13)
        arrow(ax, (x + 0.075, 0.70), (0.50, 0.58), color="#2563eb")
    box(ax, 0.37, 0.43, 0.26, 0.13, "Arrival Event Queue\nordered by simulated time", "#ede9fe", fs=14)
    box(ax, 0.70, 0.43, 0.22, 0.13, "Logical Version\nstaleness = v - v_start", "#fef3c7", fs=13)
    arrow(ax, (0.63, 0.50), (0.70, 0.50), color="#7c3aed")
    box(ax, 0.14, 0.22, 0.24, 0.13, "Async Aggregator\nNaive / Stale / FedBuff / CAA", "#dcfce7", fs=13)
    box(ax, 0.47, 0.22, 0.24, 0.13, "Global Model Server\nversioned model state", "#ecfeff", fs=13)
    box(ax, 0.78, 0.22, 0.16, 0.13, "Logs + Plots\nCSV / summary", "#fee2e2", fs=13)
    arrow(ax, (0.48, 0.43), (0.31, 0.35), color="#059669")
    arrow(ax, (0.38, 0.285), (0.47, 0.285), color="#059669")
    arrow(ax, (0.71, 0.285), (0.78, 0.285), color="#ef4444")
    ax.text(0.5, 0.08, "No synchronized physical clock: the server reasons with logical model versions, arrival events, and client deltas.", ha="center", fontsize=16, weight="bold", color="#1d4ed8")
    save(fig, "system_architecture.png")


def plot_component_stack_clean() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.94, "CAA-v2: Small Server-Side Additions Over FedBuff", ha="center", fontsize=22, weight="bold")
    items = [
        ("FedBuff", "buffer B async updates", "#ede9fe"),
        ("Staleness", "downweight old logical versions", "#dbeafe"),
        ("Agreement", "prefer deltas aligned with buffer direction", "#dcfce7"),
        ("Trajectory", "compare with recent accepted server movement", "#fef3c7"),
        ("Fairness", "reduce long-term fast-client dominance", "#fee2e2"),
        ("Adaptive alpha", "step larger when updates agree, smaller when stale", "#ecfeff"),
    ]
    xs = [0.05, 0.36, 0.67]
    ys = [0.62, 0.34]
    for idx, (title, desc, color) in enumerate(items):
        x = xs[idx % 3]; y = ys[idx // 3]
        box(ax, x, y, 0.26, 0.18, f"{title}\n{desc}", color, fs=13)
    for start, end in [((0.31,0.71),(0.36,0.71)),((0.62,0.71),(0.67,0.71)),((0.80,0.62),(0.18,0.52)),((0.31,0.43),(0.36,0.43)),((0.62,0.43),(0.67,0.43))]:
        arrow(ax, start, end, color="#475569", lw=1.8)
    ax.text(0.5, 0.12, "Novelty boundary: an explicit, deterministic integration for clockless async FL; not claimed as publication-level SOTA.", ha="center", fontsize=15, weight="bold", color="#1d4ed8")
    save(fig, "proposal_component_stack.png")


def plot_method_flow_clean() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.94,"CAA-v2 Aggregation Pipeline",ha="center",fontsize=22,weight="bold")
    stages=[
        (0.05,"1. Receive\nclient delta"),
        (0.25,"2. Buffer\nB arrivals"),
        (0.45,"3. Score\nstale + agree"),
        (0.65,"4. Clip + fair\nnormalize"),
        (0.83,"5. Update\nserver model"),
    ]
    for x,t in stages:
        box(ax,x,0.60,0.14,0.16,t,"#dbeafe",fs=13)
    for i in range(len(stages)-1):
        arrow(ax,(stages[i][0]+0.14,0.68),(stages[i+1][0],0.68),color="#0f766e")
    lower=[
        (0.12,"Logical staleness\ntau = v_server - v_start", "#fef3c7"),
        (0.38,"Direction agreement\ncos(delta, reference)", "#dcfce7"),
        (0.64,"Client fairness\n1 / (1 + count)^p", "#fee2e2"),
    ]
    for x,t,c in lower:
        box(ax,x,0.29,0.22,0.15,t,c,fs=13)
    arrow(ax,(0.52,0.60),(0.23,0.44),color="#64748b",lw=1.6)
    arrow(ax,(0.52,0.60),(0.49,0.44),color="#64748b",lw=1.6)
    arrow(ax,(0.72,0.60),(0.75,0.44),color="#64748b",lw=1.6)
    ax.text(0.5,0.10,"All signals are server-observable and clockless: logical versions, deltas, accepted trajectory, and contribution counts.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_method_flow.png")


def plot_decision_flow_clean() -> None:
    fig, ax = plt.subplots(figsize=(13.8,7.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.94,"CAA-v2 Server Decision Flow",ha="center",fontsize=22,weight="bold")
    top=[(0.07,"Buffered updates"),(0.32,"Reference direction"),(0.57,"Score each update"),(0.80,"Aggregate delta")]
    for x,t in top:
        box(ax,x,0.64,0.16,0.14,t,"#e0f2fe",fs=13)
    for i in range(len(top)-1):
        arrow(ax,(top[i][0]+0.16,0.71),(top[i+1][0],0.71),color="#2563eb")
    box(ax,0.18,0.36,0.22,0.14,"Drop only if\nstale + conflicting", "#fee2e2", fs=13)
    box(ax,0.44,0.36,0.22,0.14,"Clip by median\ndelta norm", "#fef3c7", fs=13)
    box(ax,0.70,0.36,0.20,0.14,"Adaptive alpha\nagree ↑ / stale ↓", "#dcfce7", fs=13)
    arrow(ax,(0.65,0.64),(0.29,0.50),color="#64748b",lw=1.6)
    arrow(ax,(0.65,0.64),(0.55,0.50),color="#64748b",lw=1.6)
    arrow(ax,(0.88,0.64),(0.80,0.50),color="#64748b",lw=1.6)
    ax.text(0.5,0.13,"Fallback is still standard buffered delta aggregation if filtering would remove every update.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_method_decision.png")


def plot_novelty_boundary_clean() -> None:
    fig, ax = plt.subplots(figsize=(13.6,6.8))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.94,"Novelty Boundary",ha="center",fontsize=22,weight="bold")
    cols=[
        (0.05,"Existing baselines",["FedAvg", "FedAsync", "FedBuff", "staleness decay", "cosine robustness"],"#dbeafe"),
        (0.38,"Our contribution",["clockless simulator", "fair-budget protocol", "agreement weighting", "server trajectory EMA", "client fairness credit"],"#dcfce7"),
        (0.71,"Not claimed",["new SOTA theorem", "secure aggregation protocol", "real hospital deployment", "full convergence proof"],"#fee2e2"),
    ]
    for x,title,items,color in cols:
        box(ax,x,0.20,0.25,0.58,title+"\n\n"+"\n".join(items),color,fs=13)
    ax.text(0.5,0.08,"Claim: a reproducible systems-oriented design extension, not a standalone publication-level FL primitive.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_novelty_position.png")


def plot_challenge_map_clean() -> None:
    fig, ax = plt.subplots(figsize=(13.7,7.1))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.94,"Distributed-Systems Challenges Made Visible",ha="center",fontsize=22,weight="bold")
    box(ax,0.36,0.42,0.28,0.16,"Clockless Async FL\nfor Medical Imaging","#ecfdf5",fs=15)
    challenges=[
        (0.07,0.66,"No global clock\nlogical order only","#dbeafe"),
        (0.07,0.23,"Stale update\nold model version","#fef3c7"),
        (0.70,0.66,"Conflicting update\nnon-IID hospitals","#fee2e2"),
        (0.70,0.23,"Fast-client bias\narrival dominance","#ede9fe"),
    ]
    for x,y,t,c in challenges:
        box(ax,x,y,0.22,0.14,t,c,fs=13)
    # Short non-overlapping connector stubs toward the center.
    arrow(ax,(0.29,0.73),(0.36,0.54),color="#64748b",lw=1.5)
    arrow(ax,(0.29,0.30),(0.36,0.46),color="#64748b",lw=1.5)
    arrow(ax,(0.70,0.73),(0.64,0.54),color="#64748b",lw=1.5)
    arrow(ax,(0.70,0.30),(0.64,0.46),color="#64748b",lw=1.5)
    ax.text(0.5,0.08,"Therefore we report not only accuracy, but also staleness, simulated time, stability, and client imbalance.",ha="center",fontsize=15,weight="bold",color="#1d4ed8")
    save(fig,"proposal_challenge_map.png")


def plot_results_dashboard_clean(mean: pd.DataFrame) -> None:
    overall = mean.groupby("method").agg(best=("best_acc_mean","mean"), final=("final_acc_mean","mean"), drop=("stability_drop_mean","mean")).reindex(METHOD_ORDER)
    labels=[METHOD_LABEL[m] for m in overall.index]
    fig, axes = plt.subplots(1, 2, figsize=(13.8,6.6), gridspec_kw={"width_ratios":[1.5,1]})
    x=np.arange(len(labels))
    axes[0].bar(x-0.18, overall["best"], 0.36, color="#2563eb", label="Best")
    axes[0].bar(x+0.18, overall["final"], 0.36, color="#0f766e", label="Final")
    axes[0].set_title("Mean Accuracy Across 9 Datasets", weight="bold")
    axes[0].set_ylim(0.64, 0.735)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=.22); axes[0].legend(loc="upper left")
    axes[1].bar(labels, overall["drop"], color=[COLORS[l] for l in labels])
    axes[1].set_title("Stability Drop", weight="bold")
    axes[1].set_ylabel("best - final")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=.22)
    fig.text(0.5,0.01,"Takeaway: CAA-v2 improves mean final accuracy over Sync/Naive while keeping drop below CAA-v1 and Naive.",ha="center",fontsize=14,weight="bold",color="#1d4ed8")
    fig.tight_layout(rect=(0,0.04,1,1))
    save(fig,"proposal_results_dashboard.png")


def plot_caa_v2_gap_clean(final: pd.DataFrame) -> None:
    df=final.set_index("dataset").reindex(DATASET_ORDER).reset_index()
    labels=[DATASET_SHORT[d] for d in df["dataset"]]
    y=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(13.6,6.8))
    ax.barh(y+0.18, df["caa_v2_minus_sync_best"], height=.32, color="#0f766e", label="Best")
    ax.barh(y-0.18, df["caa_v2_minus_sync_final"], height=.32, color="#2563eb", label="Final")
    ax.axvline(0,color="#0f172a",lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("CAA-v2 - Sync accuracy")
    ax.set_title("CAA-v2 vs Sync FedAvg by Dataset", weight="bold")
    ax.grid(axis="x",alpha=.22); ax.legend(loc="lower right")
    save(fig,"proposal_caa_v2_gap.png")


def plot_system_metrics_clean(sys: pd.DataFrame, mean: pd.DataFrame) -> None:
    data=sys.set_index("method").reindex(METHOD_ORDER)
    labels=[METHOD_LABEL[m] for m in METHOD_ORDER]
    stability=mean.groupby("method")["stability_drop_mean"].mean().reindex(METHOD_ORDER)
    fig,axes=plt.subplots(1,2,figsize=(13.6,6.5))
    x=np.arange(len(labels))
    axes[0].bar(labels,data["p95_staleness"],color=[COLORS[l] for l in labels])
    axes[0].set_title("Tail Logical Staleness",weight="bold")
    axes[0].set_ylabel("p95 staleness")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels,rotation=15,ha="right")
    axes[0].grid(axis="y",alpha=.22)
    axes[1].bar(labels,stability,color=[COLORS[l] for l in labels])
    axes[1].set_title("Accuracy Oscillation",weight="bold")
    axes[1].set_ylabel("best - final")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels,rotation=15,ha="right")
    axes[1].grid(axis="y",alpha=.22)
    fig.tight_layout()
    save(fig,"proposal_system_metrics.png")


def plot_dataset_method_heatmap_clean(mean: pd.DataFrame) -> None:
    mat = mean.pivot_table(index="dataset", columns="method", values="best_acc_mean").reindex(DATASET_ORDER)[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(13.6,7.0))
    im = ax.imshow(mat.values, cmap="YlGnBu", vmin=0.45, vmax=0.92, aspect="auto")
    ax.set_xticks(np.arange(len(METHOD_ORDER))); ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(DATASET_ORDER))); ax.set_yticklabels([DATASET_SHORT[d] for d in DATASET_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val=mat.values[i,j]
            ax.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=12,color="#0f172a")
    ax.set_title("Official Fair Matrix: Mean Best Accuracy", weight="bold")
    cbar=fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("accuracy")
    fig.tight_layout()
    save(fig,"proposal_dataset_method_heatmap.png")


def plot_async_sync_gap_clean(mean: pd.DataFrame) -> None:
    rows=[]
    for dataset in DATASET_ORDER:
        sub=mean[mean["dataset"]==dataset].set_index("method")
        if "sync_fedavg" not in sub.index: continue
        sync=sub.loc["sync_fedavg","best_acc_mean"]
        for method in METHOD_ORDER[1:]:
            if method in sub.index:
                rows.append({"method":method,"gap":sync-sub.loc[method,"best_acc_mean"]})
    df=pd.DataFrame(rows)
    g=df.groupby("method")["gap"].agg(["mean","std"]).reindex(METHOD_ORDER[1:])
    labels=[METHOD_LABEL[m] for m in g.index]
    fig,ax=plt.subplots(figsize=(12.8,6.4))
    ax.bar(labels,g["mean"],yerr=g["std"],capsize=7,color=[COLORS[l] for l in labels],edgecolor="#334155")
    ax.axhline(0,color="#0f172a",lw=1.2)
    ax.set_title("Async-Sync Best Gap Under Fair Update Budget",weight="bold")
    ax.set_ylabel("Sync best - async best\n(lower is better; below 0 means async higher)")
    ax.grid(axis="y",alpha=.25)
    save(fig,"proposal_async_sync_gap_errorbar.png")


def plot_stability_drop_clean(mean: pd.DataFrame) -> None:
    g=mean.groupby("method")["stability_drop_mean"].agg(["mean","std"]).reindex(METHOD_ORDER)
    labels=[METHOD_LABEL[m] for m in g.index]
    fig,ax=plt.subplots(figsize=(12.8,6.4))
    ax.bar(labels,g["mean"],yerr=g["std"],capsize=7,color=[COLORS[l] for l in labels],edgecolor="#334155")
    ax.set_title("Stability Drop Across Datasets",weight="bold")
    ax.set_ylabel("best_acc - final_acc\n(lower is more stable)")
    ax.grid(axis="y",alpha=.25)
    save(fig,"proposal_stability_drop_errorbar.png")


def plot_non_iid_gap_clean(dist: pd.DataFrame) -> None:
    data=dist[(dist["partition"]=="dirichlet") & (dist["method"].isin(METHOD_ORDER))].copy()
    if data.empty:
        empty_plot("proposal_non_iid_async_sync_gap.png","Non-IID results pending"); return
    rows=[]
    for (dataset,alpha,seed),sub in data.groupby(["dataset","dirichlet_alpha","seed"]):
        sync=sub[sub["method"]=="sync_fedavg"]
        if sync.empty: continue
        sync_best=sync["best_acc"].iloc[0]
        for method in ["naive_async","staleness_async","fedbuff_async","caa_fedbuff_v2"]:
            m=sub[sub["method"]==method]
            if not m.empty:
                rows.append({"alpha":alpha,"method":method,"gap":sync_best-m["best_acc"].iloc[0]})
    df=pd.DataFrame(rows)
    piv=df.groupby(["alpha","method"])["gap"].mean().unstack().reindex(columns=["naive_async","staleness_async","fedbuff_async","caa_fedbuff_v2"])
    labels=[METHOD_LABEL[m] for m in piv.columns]
    x=np.arange(len(piv.index)); width=.18
    fig,ax=plt.subplots(figsize=(13.4,6.4))
    for j,m in enumerate(piv.columns):
        ax.bar(x+(j-1.5)*width,piv[m],width,label=METHOD_LABEL[m],color=COLORS[METHOD_LABEL[m]])
    ax.axhline(0,color="#0f172a",lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([f"Dirichlet α={a:g}" for a in piv.index])
    ax.set_ylabel("Sync best - async best")
    ax.set_title("Non-IID Hospital Scenario: Async-Sync Gap",weight="bold")
    ax.grid(axis="y",alpha=.25); ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.12))
    save(fig,"proposal_non_iid_async_sync_gap.png")


def plot_straggler_staleness_clean(dist: pd.DataFrame) -> None:
    methods=["naive_async","staleness_async","fedbuff_async","caa_fedbuff_v2"]
    labels_order=["uniform","lognormal_m1_s0.5","hetero_r0.2_x5","hetero_r0.4_x8"]
    data=dist[(dist["delay_label"].isin(labels_order)) & (dist["method"].isin(methods))].copy()
    if data.empty:
        empty_plot("proposal_straggler_staleness_distribution.png","Straggler results pending"); return
    piv=data.groupby(["delay_label","method"])["p95_staleness"].mean().unstack().reindex(labels_order)
    x=np.arange(len(piv.index)); width=.18
    fig,ax=plt.subplots(figsize=(13.4,6.5))
    for j,m in enumerate(methods):
        if m in piv:
            ax.bar(x+(j-1.5)*width,piv[m],width,label=METHOD_LABEL[m],color=COLORS[METHOD_LABEL[m]])
    ax.set_xticks(x); ax.set_xticklabels(["Uniform","Lognormal","Mild\nstraggler","Severe\nstraggler"])
    ax.set_ylabel("p95 logical staleness")
    ax.set_title("Delay Stress: Tail Staleness",weight="bold")
    ax.grid(axis="y",alpha=.25); ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.12))
    save(fig,"proposal_straggler_staleness_distribution.png")


def plot_straggler_time_clean(dist: pd.DataFrame) -> None:
    methods=["naive_async","staleness_async","fedbuff_async","caa_fedbuff_v2"]
    labels_order=["uniform","lognormal_m1_s0.5","hetero_r0.2_x5","hetero_r0.4_x8"]
    data=dist[(dist["delay_label"].isin(labels_order)) & (dist["method"].isin(methods))].copy()
    if data.empty:
        empty_plot("proposal_straggler_acc_vs_simulated_time.png","Simulated-time results pending"); return
    piv=data.groupby(["delay_label","method"])["time_to_90pct_best_acc"].mean().unstack().reindex(labels_order)
    fig,ax=plt.subplots(figsize=(13.4,6.4))
    x=np.arange(len(labels_order))
    for m in methods:
        if m in piv:
            ax.plot(x,piv[m],marker="o",lw=3,ms=8,label=METHOD_LABEL[m],color=COLORS[METHOD_LABEL[m]])
    ax.set_xticks(x); ax.set_xticklabels(["Uniform","Lognormal","Mild\nstraggler","Severe\nstraggler"])
    ax.set_ylabel("simulated time to 90% of best accuracy")
    ax.set_title("Delay Stress: Time-to-Accuracy",weight="bold")
    ax.grid(axis="y",alpha=.25); ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.12))
    save(fig,"proposal_straggler_acc_vs_simulated_time.png")


def plot_ablation_clean(ablation: pd.DataFrame) -> None:
    order=["old_caa","full_caa_v2","no_server_ema","no_fairness","no_clipping","static_alpha"]
    names={"old_caa":"CAA-v1","full_caa_v2":"Full v2","no_server_ema":"No EMA","no_fairness":"No fair","no_clipping":"No clip","static_alpha":"Static α"}
    data=ablation[(ablation["dataset"]=="pathmnist") & (ablation["variant"].isin(order))].copy()
    if data.empty:
        empty_plot("proposal_caa_v2_ablation_best_acc.png","Ablation results pending"); return
    data["variant"]=pd.Categorical(data["variant"],categories=order,ordered=True)
    data=data.sort_values("variant")
    labels=[names[v] for v in data["variant"].astype(str)]
    fig,axes=plt.subplots(1,2,figsize=(13.6,6.4),gridspec_kw={"width_ratios":[1.35,1]})
    x=np.arange(len(data))
    axes[0].bar(labels,data["best_acc_mean"],yerr=data["best_acc_std"],capsize=6,color="#0f766e",edgecolor="#334155")
    axes[0].set_ylim(0.88,0.915)
    axes[0].set_title("PathMNIST Best Accuracy",weight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels,rotation=18,ha="right")
    axes[0].grid(axis="y",alpha=.25)
    axes[1].bar(labels,data["stability_drop_mean"],color="#f59e0b",edgecolor="#334155")
    axes[1].set_title("Stability Drop",weight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels,rotation=18,ha="right")
    axes[1].grid(axis="y",alpha=.25)
    fig.tight_layout()
    save(fig,"proposal_caa_v2_ablation_best_acc.png")


def empty_plot(name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=24, weight="bold")
    save(fig, name)


def update_markdown_refs() -> None:
    for name in [
        "clockless_federated_adaptation_proposal_zh.md",
        "clockless_federated_adaptation_proposal_en.md",
    ]:
        path = PRESENT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in REF_REPLACEMENTS.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def rebuild_html() -> None:
    for stem, lang in [
        ("clockless_federated_adaptation_proposal_zh", "zh-Hant"),
        ("clockless_federated_adaptation_proposal_en", "en"),
    ]:
        md_path = PRESENT / f"{stem}.md"
        if not md_path.exists():
            continue
        slides = md_path.read_text(encoding="utf-8").strip().split("\n\n---\n\n")
        body=[]
        for idx, slide in enumerate(slides,1):
            html = md.markdown(slide, extensions=["tables", "fenced_code"])
            body.append(f'<section class="slide"><div class="slide-no">{idx:02d}</div>{html}</section>')
        page=f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Clockless Federated Adaptation</title>\n<style>\n{CSS}\n</style>\n</head>\n<body>\n{"".join(body)}\n</body>\n</html>\n'
        (PRESENT / f"{stem}.html").write_text(page, encoding="utf-8")


def rebuild_pptx_pdf() -> None:
    for stem in ["clockless_federated_adaptation_proposal_zh", "clockless_federated_adaptation_proposal_en"]:
        md_path = PRESENT / f"{stem}.md"
        if md_path.exists():
            subprocess.run(["pandoc", md_path.name, "-t", "pptx", "-o", f"{stem}.pptx"], cwd=PRESENT, check=True)
    for stem in ["clockless_federated_adaptation_proposal_zh", "clockless_federated_adaptation_proposal_en"]:
        html_path = PRESENT / f"{stem}.html"
        pdf_path = PRESENT / f"{stem}.pdf"
        subprocess.run(["/usr/bin/google-chrome", "--headless", "--disable-gpu", "--no-sandbox", f"--print-to-pdf={pdf_path.name}", f"file://{html_path}"], cwd=PRESENT, check=True)


if __name__ == "__main__":
    main()
