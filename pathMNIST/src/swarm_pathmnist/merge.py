from __future__ import annotations

from typing import Literal

import numpy as np

MergeMethod = Literal["mean", "weighted_mean", "coord_median"]


def merge_parameters(
    parameter_sets: list[list[np.ndarray]],
    weights: list[int | float],
    method: MergeMethod,
) -> list[np.ndarray]:
    if not parameter_sets:
        raise ValueError("Cannot merge an empty parameter set")
    if len(parameter_sets) != len(weights):
        raise ValueError("parameter_sets and weights must have the same length")

    if method == "mean":
        normalized_weights = [1.0 / len(parameter_sets)] * len(parameter_sets)
        return _weighted_mean(parameter_sets, normalized_weights)

    if method == "weighted_mean":
        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("Cannot weighted-merge zero total weight")
        normalized_weights = [float(weight) / total_weight for weight in weights]
        return _weighted_mean(parameter_sets, normalized_weights)

    if method == "coord_median":
        return _coordinate_median(parameter_sets)

    raise ValueError(f"Unsupported merge method: {method}")


def _weighted_mean(
    parameter_sets: list[list[np.ndarray]],
    normalized_weights: list[float],
) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for layer_values in zip(*parameter_sets, strict=True):
        layer_avg = sum(
            layer * weight
            for layer, weight in zip(layer_values, normalized_weights, strict=True)
        )
        merged.append(layer_avg.astype(layer_values[0].dtype, copy=False))
    return merged


def _coordinate_median(parameter_sets: list[list[np.ndarray]]) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for layer_values in zip(*parameter_sets, strict=True):
        stacked = np.stack(layer_values, axis=0)
        layer_median = np.median(stacked, axis=0)
        merged.append(layer_median.astype(layer_values[0].dtype, copy=False))
    return merged

