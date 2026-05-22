from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

DATA_ROOT = "./data/medmnist"


@dataclass(frozen=True)
class ClientLoaders:
    train: DataLoader
    num_examples: int


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


def _make_pathmnist(split: str, download: bool, augment: bool) -> Dataset:
    from medmnist import INFO, PathMNIST

    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    info = INFO["pathmnist"]
    mean = info.get("mean", [0.5])
    std = info.get("std", [0.5])
    transform_steps = []
    if split == "train" and augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform = transforms.Compose(transform_steps)
    return PathMNIST(split=split, transform=transform, download=download, root=DATA_ROOT)


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


def load_datasets(
    *,
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

    train = _make_pathmnist("train", download=download, augment=augment)
    test = _make_pathmnist("test", download=download, augment=False)
    train = _limit_dataset(train, max_train_samples, seed)
    test = _limit_dataset(test, max_test_samples, seed + 1)
    return train, test


def test_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

