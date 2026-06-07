from __future__ import annotations

from collections.abc import Mapping

import torch

AdapterState = dict[str, torch.Tensor]


def clone_state(state: Mapping[str, torch.Tensor]) -> AdapterState:
    return {name: value.detach().clone() for name, value in state.items()}


def zero_like(state: Mapping[str, torch.Tensor]) -> AdapterState:
    return {name: torch.zeros_like(value) for name, value in state.items()}


def subtract_state(new: Mapping[str, torch.Tensor], old: Mapping[str, torch.Tensor]) -> AdapterState:
    return {name: new[name] - old[name] for name in old}


def add_scaled_delta(
    base: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    scale: float,
) -> AdapterState:
    return {
        name: base_tensor + float(scale) * delta[name].to(base_tensor.device)
        for name, base_tensor in base.items()
    }


def weighted_sum_deltas(
    deltas: list[Mapping[str, torch.Tensor]],
    weights: list[float],
    reference: Mapping[str, torch.Tensor],
) -> AdapterState:
    if not deltas:
        return zero_like(reference)
    total = float(sum(weights))
    if total <= 0.0:
        return zero_like(reference)

    result = zero_like(reference)
    for delta, weight in zip(deltas, weights, strict=True):
        normalized = float(weight) / total
        for name in result:
            result[name] = result[name] + normalized * delta[name].to(result[name].device)
    return result


def blend_delta(
    current: Mapping[str, torch.Tensor],
    history: Mapping[str, torch.Tensor] | None,
    blend: float,
) -> AdapterState:
    if history is None:
        return clone_state(current)
    amount = min(max(float(blend), 0.0), 1.0)
    return {
        name: (1.0 - amount) * current_delta + amount * history[name].to(current_delta.device)
        for name, current_delta in current.items()
    }


def update_ema(
    previous: Mapping[str, torch.Tensor] | None,
    accepted_delta: Mapping[str, torch.Tensor],
    momentum: float,
) -> AdapterState:
    amount = min(max(float(momentum), 0.0), 0.999)
    if previous is None:
        return clone_state(accepted_delta)
    return {
        name: amount * previous[name].to(delta.device) + (1.0 - amount) * delta
        for name, delta in accepted_delta.items()
    }


def delta_norm(delta: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(delta.values())).device)
    for value in delta.values():
        if not torch.is_floating_point(value):
            continue
        total = total + torch.sum(value.float() * value.float())
    return float(torch.sqrt(total).detach().cpu().item())


def cosine_similarity(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    dot = None
    left_norm = None
    right_norm = None

    for name, left_value in left.items():
        if not torch.is_floating_point(left_value):
            continue
        a = left_value.float().reshape(-1)
        b = right[name].to(left_value.device).float().reshape(-1)
        part_dot = torch.dot(a, b)
        part_left = torch.dot(a, a)
        part_right = torch.dot(b, b)
        dot = part_dot if dot is None else dot + part_dot
        left_norm = part_left if left_norm is None else left_norm + part_left
        right_norm = part_right if right_norm is None else right_norm + part_right

    if dot is None or left_norm is None or right_norm is None:
        return 0.0
    denom = torch.sqrt(left_norm) * torch.sqrt(right_norm)
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float((dot / denom).detach().cpu().item())

