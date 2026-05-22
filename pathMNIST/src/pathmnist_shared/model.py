from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

NUM_CLASSES = 9


def create_model() -> nn.Module:
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in zip(keys, parameters, strict=True)}
    )
    model.load_state_dict(state_dict, strict=True)


def weighted_average(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise ValueError("Cannot average zero examples")

    averaged: list[np.ndarray] = []
    for layer_values in zip(*(params for params, _ in results), strict=True):
        layer_avg = sum(
            layer * (num_examples / total_examples)
            for layer, (_, num_examples) in zip(layer_values, results, strict=True)
        )
        averaged.append(layer_avg.astype(layer_values[0].dtype, copy=False))
    return averaged


def interpolate(
    old: list[np.ndarray],
    new: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    return [
        ((1.0 - alpha) * old_layer + alpha * new_layer).astype(old_layer.dtype, copy=False)
        for old_layer, new_layer in zip(old, new, strict=True)
    ]

