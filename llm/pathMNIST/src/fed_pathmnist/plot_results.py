from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "round_or_event",
    "server_version",
    "arrival_server_version",
    "simulated_time",
    "client_id",
    "client_start_version",
    "staleness",
    "base_alpha",
    "effective_alpha",
    "staleness_weight",
    "learning_rate",
    "train_loss",
    "test_loss",
    "test_acc",
    "num_examples",
    "delay",
    "buffer_size",
    "applied_updates",
    "agreement",
    "mean_agreement",
    "buffer_alpha",
    "delta_norm",
    "dropped_update",
    "server_momentum_agreement",
    "fairness_weight",
    "pending_pool_size",
    "quorum_size",
    "quorum_met",
    "selected_updates",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PathMNIST federated logs.")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV log files or globs.")
    parser.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_paths = _expand_csv_args(args.csv)
    if not csv_paths:
        raise SystemExit("No CSV files matched.")

    frames = [_load_csv(path) for path in csv_paths]
    data = pd.concat(frames, ignore_index=True)

    _plot_test_acc_vs_progress(data, outdir)
    _plot_test_acc_vs_simulated_time(data, outdir)
    _plot_line(data, outdir, "staleness", "staleness_vs_event.png", "Staleness")
    _plot_line(
        data,
        outdir,
        "effective_alpha",
        "effective_alpha_vs_event.png",
        "Effective alpha",
    )
    _plot_agreement(data, outdir)
    _plot_client_contributions(data, outdir)


def _expand_csv_args(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return [path for path in paths if path.exists() and path.suffix == ".csv"]


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["run_label"] = path.stem
    if "method" not in frame.columns:
        frame["method"] = path.stem.split("_pathmnist_")[0]
    frame["plot_label"] = frame["method"].astype(str) + " / " + frame["run_label"]
    return frame


def _plot_test_acc_vs_progress(data: pd.DataFrame, outdir: Path) -> None:
    subset = data.dropna(subset=["round_or_event", "test_acc"])
    if subset.empty:
        return
    _new_figure()
    for label, group in subset.groupby("plot_label"):
        group = group.sort_values("round_or_event")
        plt.plot(group["round_or_event"], group["test_acc"], marker="o", label=label)
    plt.xlabel("Round / event")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy vs progress")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "test_acc_vs_progress.png", dpi=180)
    plt.close()


def _plot_test_acc_vs_simulated_time(data: pd.DataFrame, outdir: Path) -> None:
    subset = data.dropna(subset=["simulated_time", "test_acc"])
    if subset.empty:
        return
    _new_figure()
    for label, group in subset.groupby("plot_label"):
        group = group.sort_values("simulated_time")
        plt.plot(group["simulated_time"], group["test_acc"], marker="o", label=label)
    plt.xlabel("Simulated time")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy vs simulated time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "test_acc_vs_simulated_time.png", dpi=180)
    plt.close()


def _plot_line(data: pd.DataFrame, outdir: Path, column: str, filename: str, ylabel: str) -> None:
    subset = data.dropna(subset=["round_or_event", column])
    subset = subset[subset["method"] != "sync_fedavg"]
    if subset.empty:
        return
    _new_figure()
    for label, group in subset.groupby("plot_label"):
        group = group.sort_values("round_or_event")
        plt.plot(group["round_or_event"], group[column], linewidth=1.5, label=label)
    plt.xlabel("Event")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs event")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=180)
    plt.close()


def _plot_client_contributions(data: pd.DataFrame, outdir: Path) -> None:
    subset = data.dropna(subset=["client_id"])
    subset = subset[subset["method"] != "sync_fedavg"]
    if subset.empty:
        return

    counts = (
        subset.groupby(["method", "client_id"])
        .size()
        .reset_index(name="updates")
        .pivot(index="client_id", columns="method", values="updates")
        .fillna(0)
        .sort_index()
    )
    _new_figure()
    counts.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Client ID")
    plt.ylabel("Number of arrived updates")
    plt.title("Client contribution count")
    plt.tight_layout()
    plt.savefig(outdir / "client_contribution_bar.png", dpi=180)
    plt.close()


def _plot_agreement(data: pd.DataFrame, outdir: Path) -> None:
    column = "mean_agreement" if "mean_agreement" in data.columns else "agreement"
    if column not in data.columns:
        return
    _plot_line(data, outdir, column, "agreement_vs_event.png", "Agreement")


def _new_figure() -> None:
    plt.figure(figsize=(8, 4.8))


if __name__ == "__main__":
    main()
