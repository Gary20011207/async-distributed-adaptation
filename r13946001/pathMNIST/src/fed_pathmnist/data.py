from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

DATA_ROOT = "./data/medmnist"
SUPPORTED_TASKS = {"multi-class", "binary-class"}


@dataclass(frozen=True)
class ClientLoaders:
    train: DataLoader
    num_examples: int


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    task: str
    num_classes: int
    in_channels: int
    mean: list[float]
    std: list[float]


class SyntheticPathMNIST(Dataset):
    """Small PathMNIST-shaped dataset for dependency and pipeline smoke tests."""

    def __init__(self, size: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.images = rng.random((size, 3, 28, 28), dtype=np.float32)
        self.labels = rng.integers(0, 9, size=(size,), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.images[index]), torch.tensor(self.labels[index])


def get_dataset_metadata(dataset_name: str, *, synthetic: bool = False) -> DatasetMetadata:
    if synthetic:
        return DatasetMetadata(
            name="synthetic",
            task="multi-class",
            num_classes=9,
            in_channels=3,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    from medmnist import INFO

    key = dataset_name.lower()
    if key not in INFO:
        supported = ", ".join(sorted(INFO))
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset_name}. Known datasets: {supported}")

    info = INFO[key]
    task = str(info.get("task", ""))
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"{dataset_name} task={task!r} is not supported in this runner. "
            "Use multi-class or binary-class 2D MedMNIST datasets; skip multi-label datasets such as chestmnist."
        )
    if key.endswith("3d"):
        raise ValueError(f"{dataset_name} is a 3D MedMNIST dataset; this 2D ResNet18 runner skips it.")

    return DatasetMetadata(
        name=key,
        task=task,
        num_classes=len(info.get("label", {})),
        in_channels=int(info.get("n_channels", 3)),
        mean=_as_channel_list(info.get("mean", [0.5]), int(info.get("n_channels", 3))),
        std=_as_channel_list(info.get("std", [0.5]), int(info.get("n_channels", 3))),
    )


def _make_medmnist(
    dataset_name: str,
    split: str,
    download: bool,
    augment: bool,
) -> Dataset:
    import medmnist
    from medmnist import INFO

    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    key = dataset_name.lower()
    metadata = get_dataset_metadata(key)
    dataset_cls = getattr(medmnist, INFO[key]["python_class"])
    transform_steps = []
    if split == "train" and augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        )
        if metadata.in_channels == 3:
            transform_steps.append(
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
            )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=metadata.mean, std=metadata.std),
        ]
    )
    transform = transforms.Compose(transform_steps)
    return dataset_cls(split=split, transform=transform, download=download, root=DATA_ROOT)


def _limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:max_samples].tolist()
    return Subset(dataset, indices)


def iid_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[ClientLoaders]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)

    loaders: list[ClientLoaders] = []
    generator = torch.Generator().manual_seed(seed)
    for split in splits:
        subset = Subset(dataset, split.tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        loaders.append(ClientLoaders(train=loader, num_examples=len(subset)))
    return loaders


def dirichlet_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    alpha: float,
) -> list[ClientLoaders]:
    if alpha <= 0:
        raise ValueError("--dirichlet-alpha must be positive")
    if len(dataset) < num_clients:
        raise ValueError("Dirichlet partition needs at least one example per client")

    rng = np.random.default_rng(seed)
    labels = _dataset_labels(dataset)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for cid, split in enumerate(np.split(class_indices, split_points)):
            client_indices[cid].extend(split.tolist())

    _move_examples_to_empty_clients(client_indices, rng)
    for indices in client_indices:
        rng.shuffle(indices)

    return _client_loaders_from_indices(
        dataset,
        client_indices,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )


def partition_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    partition: str,
    dirichlet_alpha: float,
) -> list[ClientLoaders]:
    if partition == "iid":
        return iid_client_loaders(
            dataset,
            num_clients=num_clients,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
        )
    if partition == "dirichlet":
        return dirichlet_client_loaders(
            dataset,
            num_clients=num_clients,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            alpha=dirichlet_alpha,
        )
    raise ValueError(f"Unsupported partition: {partition}")


def _client_loaders_from_indices(
    dataset: Dataset,
    client_indices: list[list[int]],
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[ClientLoaders]:
    loaders: list[ClientLoaders] = []
    generator = torch.Generator().manual_seed(seed)
    for indices in client_indices:
        subset = Subset(dataset, list(indices))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        loaders.append(ClientLoaders(train=loader, num_examples=len(subset)))
    return loaders


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent_labels = _dataset_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices, dtype=np.int64)]

    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is not None:
        return np.asarray(labels).reshape(-1).astype(np.int64)

    # Fallback for custom datasets. This is slower but keeps the partitioner
    # usable for tiny smoke-test datasets with only __getitem__ labels.
    extracted = []
    for idx in range(len(dataset)):
        _image, label = dataset[idx]
        extracted.append(int(np.asarray(label).reshape(-1)[0]))
    return np.asarray(extracted, dtype=np.int64)


def _move_examples_to_empty_clients(
    client_indices: list[list[int]],
    rng: np.random.Generator,
) -> None:
    empty_clients = [cid for cid, indices in enumerate(client_indices) if not indices]
    for empty_cid in empty_clients:
        donor_cid = max(range(len(client_indices)), key=lambda cid: len(client_indices[cid]))
        if len(client_indices[donor_cid]) <= 1:
            raise ValueError("Could not create non-empty Dirichlet client partitions")
        donor_pos = int(rng.integers(0, len(client_indices[donor_cid])))
        client_indices[empty_cid].append(client_indices[donor_cid].pop(donor_pos))


def load_datasets(
    *,
    dataset_name: str = "pathmnist",
    synthetic: bool,
    download: bool,
    max_train_samples: int | None,
    max_test_samples: int | None,
    seed: int,
    augment: bool,
) -> tuple[Dataset, Dataset]:
    if synthetic:
        train_size = max_train_samples or 1000
        test_size = max_test_samples or 300
        return SyntheticPathMNIST(train_size, seed), SyntheticPathMNIST(test_size, seed + 1)

    train = _make_medmnist(dataset_name, "train", download=download, augment=augment)
    test = _make_medmnist(dataset_name, "test", download=download, augment=False)
    train = _limit_dataset(train, max_train_samples, seed)
    test = _limit_dataset(test, max_test_samples, seed + 1)
    return train, test


def test_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _as_channel_list(value: Any, channels: int) -> list[float]:
    if isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(item) for item in value]
    if len(values) == channels:
        return values
    if len(values) == 1:
        return values * channels
    raise ValueError(f"Expected one value or {channels} values, got {values}")
