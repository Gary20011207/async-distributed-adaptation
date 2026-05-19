from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from fed_pathmnist.data import get_dataset_metadata, load_datasets, test_loader
from fed_pathmnist.model import create_model


DEFAULT_METHODS = [
    "sync_fedavg",
    "naive_async",
    "staleness_async",
    "agreement_fedbuff_async",
    "caa_fedbuff_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrices from best checkpoints.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--outdir", default="figures/classification")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_summary_row(path) for path in sorted(Path(args.result_dir).glob("*_summary.json"))]
    rows = [
        row
        for row in rows
        if row["method"] in set(args.methods)
        and row["model"] == args.model
        and row.get("checkpoint_path")
        and Path(row["checkpoint_path"]).exists()
    ]
    if args.datasets:
        rows = [row for row in rows if row["dataset"] in set(args.datasets)]
    selected = _best_rows(rows)
    for row in selected:
        _plot_one(row, args, outdir, device)
    print(f"wrote classification plots to {outdir}")


def _summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    method = summary.get("method", "")
    checkpoint_path = summary.get("checkpoint_path", "")
    return {
        "dataset": _dataset_from_summary(path, method, config),
        "method": method,
        "model": str(config.get("model", "resnet18")),
        "seed": int(config.get("seed", 42)),
        "best_acc": float(summary.get("best_test_acc", -1.0)),
        "checkpoint_path": checkpoint_path,
        "config": config,
    }


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["model"])
        current = best.get(key)
        if current is None or row["best_acc"] > current["best_acc"]:
            best[key] = row
    return list(best.values())


def _plot_one(
    row: dict[str, Any],
    args: argparse.Namespace,
    outdir: Path,
    device: torch.device,
) -> None:
    dataset = row["dataset"]
    method = row["method"]
    model_name = row["model"]
    config = row["config"]
    metadata = get_dataset_metadata(dataset)
    _train, testset = load_datasets(
        dataset_name=dataset,
        synthetic=False,
        download=True,
        max_train_samples=1,
        max_test_samples=config.get("max_test_samples"),
        seed=int(config.get("seed", 42)),
        augment=False,
    )
    loader = test_loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = create_model(
        num_classes=metadata.num_classes,
        in_channels=metadata.in_channels,
        model_name=model_name,
    ).to(device)
    checkpoint = torch.load(row["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    matrix = _confusion_matrix(model, loader, device, metadata.num_classes)
    labels = _class_labels(dataset, metadata.num_classes)
    stem = f"{dataset}_{method}_{model_name}"
    _plot_confusion(matrix, labels, outdir / f"{stem}_confusion.png")
    _plot_per_class_recall(matrix, labels, outdir / f"{stem}_per_class_recall.png")


@torch.no_grad()
def _confusion_matrix(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        preds = model(images).argmax(dim=1)
        for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=True):
            matrix[int(true), int(pred)] += 1
    return matrix


def _plot_confusion(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)
    plt.figure(figsize=(max(5, len(labels) * 0.65), max(4, len(labels) * 0.55)))
    plt.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Recall-normalized count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_per_class_recall(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    plt.figure(figsize=(max(6, len(labels) * 0.7), 4.0))
    plt.bar(range(len(labels)), recall)
    plt.ylim(0.0, 1.0)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Recall")
    plt.title("Per-class recall")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _class_labels(dataset: str, num_classes: int) -> list[str]:
    try:
        from medmnist import INFO

        label_map = INFO[dataset]["label"]
        return [str(label_map[str(idx)]) for idx in range(num_classes)]
    except Exception:
        return [str(idx) for idx in range(num_classes)]


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


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
