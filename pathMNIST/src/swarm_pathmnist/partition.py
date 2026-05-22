from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pathmnist_shared.data import ClientLoaders, iid_client_loaders

PartitionMethod = Literal["iid", "quantity_skew", "label_skew"]


@dataclass(frozen=True)
class PartitionSummary:
    sizes: list[int]
    label_counts: list[dict[int, int]]


def make_node_loaders(
    dataset: Dataset,
    *,
    num_nodes: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    partition: PartitionMethod,
    dirichlet_alpha: float,
) -> tuple[list[ClientLoaders], PartitionSummary]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if partition == "iid":
        loaders = iid_client_loaders(dataset, num_nodes, batch_size, seed, num_workers)
        splits = _loader_indices(loaders)
        return loaders, _summarize_partition(dataset, splits)

    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    labels = _extract_labels(dataset)
    rng = np.random.default_rng(seed)

    if partition == "quantity_skew":
        splits = _quantity_skew_indices(len(dataset), num_nodes, rng, dirichlet_alpha)
    elif partition == "label_skew":
        splits = _label_skew_indices(labels, num_nodes, rng, dirichlet_alpha)
    else:
        raise ValueError(f"Unsupported partition: {partition}")

    loaders = _build_loaders(dataset, splits, batch_size, seed, num_workers)
    return loaders, _summarize_partition(dataset, splits)


def format_partition_summary(summary: PartitionSummary, *, max_nodes: int = 5) -> str:
    shown = min(max_nodes, len(summary.sizes))
    label_bits = []
    for node_id in range(shown):
        counts = summary.label_counts[node_id]
        formatted = ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))
        label_bits.append(f"node{node_id}={{{formatted}}}")
    suffix = " ..." if shown < len(summary.sizes) else ""
    return f"sizes={summary.sizes} labels={' '.join(label_bits)}{suffix}"


def _quantity_skew_indices(
    dataset_size: int,
    num_nodes: int,
    rng: np.random.Generator,
    alpha: float,
) -> list[list[int]]:
    indices = rng.permutation(dataset_size)
    proportions = rng.dirichlet([alpha] * num_nodes)
    counts = rng.multinomial(dataset_size, proportions)
    splits = []
    offset = 0
    for count in counts:
        splits.append(indices[offset : offset + count].tolist())
        offset += int(count)
    return _rebalance_empty_splits(splits)


def _label_skew_indices(
    labels: np.ndarray,
    num_nodes: int,
    rng: np.random.Generator,
    alpha: float,
) -> list[list[int]]:
    splits: list[list[int]] = [[] for _ in range(num_nodes)]

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        class_indices = rng.permutation(class_indices)
        proportions = rng.dirichlet([alpha] * num_nodes)
        counts = rng.multinomial(len(class_indices), proportions)

        offset = 0
        for node_id, count in enumerate(counts):
            selected = class_indices[offset : offset + count]
            splits[node_id].extend(selected.tolist())
            offset += int(count)

    for split in splits:
        rng.shuffle(split)
    return _rebalance_empty_splits(splits)


def _rebalance_empty_splits(splits: list[list[int]]) -> list[list[int]]:
    for node_id, split in enumerate(splits):
        if split:
            continue
        donor_id = max(range(len(splits)), key=lambda idx: len(splits[idx]))
        if not splits[donor_id]:
            raise ValueError("Cannot create non-empty splits from an empty dataset")
        split.append(splits[donor_id].pop())
    return splits


def _build_loaders(
    dataset: Dataset,
    splits: list[list[int]],
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[ClientLoaders]:
    loaders: list[ClientLoaders] = []
    for node_id, split in enumerate(splits):
        subset = Subset(dataset, split)
        generator = torch.Generator().manual_seed(seed + node_id)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        loaders.append(ClientLoaders(train=loader, num_examples=len(subset)))
    return loaders


def _extract_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base_labels = _extract_labels(dataset.dataset)
        return base_labels[np.asarray(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels).reshape(-1).astype(np.int64)

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets).reshape(-1).astype(np.int64)

    labels = []
    for _, label in dataset:
        labels.append(int(torch.as_tensor(label).view(-1)[0].item()))
    return np.asarray(labels, dtype=np.int64)


def _summarize_partition(dataset: Dataset, splits: list[list[int]]) -> PartitionSummary:
    labels = _extract_labels(dataset)
    sizes = [len(split) for split in splits]
    label_counts = []
    for split in splits:
        split_labels = labels[np.asarray(split, dtype=np.int64)]
        unique, counts = np.unique(split_labels, return_counts=True)
        label_counts.append(
            {int(label): int(count) for label, count in zip(unique, counts, strict=True)}
        )
    return PartitionSummary(sizes=sizes, label_counts=label_counts)


def _loader_indices(loaders: list[ClientLoaders]) -> list[list[int]]:
    splits = []
    for loader in loaders:
        dataset = loader.train.dataset
        if not isinstance(dataset, Subset):
            raise TypeError("Expected iid_client_loaders to return Subset datasets")
        splits.append(list(dataset.indices))
    return splits

