from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from vqa_fl.adapter_math import (
    AdapterState,
    add_scaled_delta,
    blend_delta,
    clone_state,
    cosine_similarity,
    delta_norm,
    subtract_state,
    weighted_sum_deltas,
    zero_like,
)

DecayName = Literal["constant", "inverse", "polynomial", "exponential", "hinge"]


@dataclass
class CAAConfig:
    server_alpha: float = 0.62
    agreement_epsilon: float = 0.15
    agreement_power: float = 0.5
    agreement_drop_threshold: float = -0.05
    delta_clip_multiplier: float = 1.8
    adaptive_alpha_min: float = 0.20
    adaptive_alpha_max: float = 0.70
    adaptive_alpha_boost: float = 0.25
    adaptive_staleness_scale: float = 10.0
    drop_staleness_threshold: int = 5
    history_agreement_blend: float = 0.25
    client_fairness_power: float = 0.5


@dataclass
class LoRAUpdate:
    cid: int
    start_version: int
    arrival_version: int
    num_examples: int
    start_state: AdapterState
    updated_state: AdapterState
    staleness_weight: float
    delay: float = 0.0
    agreement: float = 0.0
    fairness_weight: float = 1.0
    dropped_update: bool = False

    @property
    def staleness(self) -> int:
        return max(self.arrival_version - self.start_version, 0)

    @property
    def delta(self) -> AdapterState:
        return subtract_state(self.updated_state, self.start_state)

    @property
    def delta_norm(self) -> float:
        return delta_norm(self.delta)


def staleness_decay_weight(
    staleness: int,
    decay: DecayName,
    *,
    power: float = 1.0,
    exp_rate: float = 0.1,
    hinge_b: int = 5,
    hinge_a: float = 0.1,
) -> float:
    tau = max(int(staleness), 0)
    if decay == "constant":
        return 1.0
    if decay == "inverse":
        return 1.0 / (1.0 + tau)
    if decay == "polynomial":
        return 1.0 / ((1.0 + tau) ** power)
    if decay == "exponential":
        return math.exp(-exp_rate * tau)
    if decay == "hinge":
        if tau <= hinge_b:
            return 1.0
        return 1.0 / (1.0 + hinge_a * (tau - hinge_b))
    raise ValueError(f"Unsupported staleness decay: {decay}")


def apply_fedbuff_lora(
    current_state: AdapterState,
    updates: list[LoRAUpdate],
    *,
    server_alpha: float,
) -> tuple[AdapterState, dict[str, float], AdapterState]:
    if not updates:
        empty = zero_like(current_state)
        return clone_state(current_state), _empty_stats(server_alpha), empty

    weights = [
        max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
        for update in updates
    ]
    deltas = [update.delta for update in updates]
    aggregate_delta = weighted_sum_deltas(deltas, weights, current_state)
    merged = add_scaled_delta(current_state, aggregate_delta, server_alpha)
    stats = {
        "buffer_alpha": float(server_alpha),
        "mean_agreement": 0.0,
        "mean_server_agreement": 0.0,
        "mean_fairness_weight": 1.0,
        "mean_delta_norm": float(np.mean([update.delta_norm for update in updates])),
        "dropped_updates": 0.0,
        "applied_updates": float(len(updates)),
    }
    return merged, stats, aggregate_delta


def apply_caa_v2_lora(
    current_state: AdapterState,
    updates: list[LoRAUpdate],
    *,
    server_delta_ema: AdapterState | None,
    client_apply_counts: list[int],
    config: CAAConfig,
) -> tuple[AdapterState, dict[str, float], AdapterState]:
    if not updates:
        empty = zero_like(current_state)
        return clone_state(current_state), _empty_stats(config.server_alpha), empty

    base_weights = [
        max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
        for update in updates
    ]
    deltas = [update.delta for update in updates]
    base_reference = weighted_sum_deltas(deltas, base_weights, current_state)
    reference_delta = blend_delta(
        base_reference,
        server_delta_ema,
        config.history_agreement_blend,
    )

    agreements = [cosine_similarity(delta, reference_delta) for delta in deltas]
    if server_delta_ema is None:
        server_agreements = [0.0 for _ in updates]
    else:
        server_agreements = [cosine_similarity(delta, server_delta_ema) for delta in deltas]
    norms = [delta_norm(delta) for delta in deltas]

    kept = [
        not (
            agreement < config.agreement_drop_threshold
            and update.staleness > config.drop_staleness_threshold
        )
        for update, agreement in zip(updates, agreements, strict=True)
    ]
    dropped_count = kept.count(False)
    if not any(kept):
        kept = [True for _ in updates]
        dropped_count = 0

    positive_norms = [norm for norm in norms if norm > 0.0]
    clip_norm = (
        float(np.median(positive_norms)) * max(config.delta_clip_multiplier, 0.0)
        if positive_norms
        else 0.0
    )

    raw_weights: list[float] = []
    fairness_values: list[float] = []
    for update, agreement, base_weight, is_kept in zip(
        updates,
        agreements,
        base_weights,
        kept,
        strict=True,
    ):
        count = client_apply_counts[update.cid] if update.cid < len(client_apply_counts) else 0
        fairness = 1.0 / ((1.0 + max(count, 0)) ** max(config.client_fairness_power, 0.0))
        update.agreement = agreement
        update.fairness_weight = fairness
        update.dropped_update = not is_kept
        fairness_values.append(fairness)
        if not is_kept:
            raw_weights.append(0.0)
            continue
        agreement_factor = (
            max(config.agreement_epsilon, 0.0) + max(agreement, 0.0)
        ) ** max(config.agreement_power, 0.0)
        raw_weights.append(base_weight * agreement_factor * fairness)

    if sum(raw_weights) <= 0.0:
        raw_weights = [
            base_weight * fairness if is_kept else 0.0
            for base_weight, fairness, is_kept in zip(
                base_weights,
                fairness_values,
                kept,
                strict=True,
            )
        ]
    if sum(raw_weights) <= 0.0:
        return apply_fedbuff_lora(
            current_state,
            updates,
            server_alpha=config.server_alpha,
        )

    clipped_deltas = [
        _clip_delta(delta, norm, clip_norm) for delta, norm in zip(deltas, norms, strict=True)
    ]
    accepted_delta = weighted_sum_deltas(clipped_deltas, raw_weights, current_state)

    kept_agreements = [
        max(agreement, 0.0)
        for agreement, is_kept in zip(agreements, kept, strict=True)
        if is_kept
    ]
    kept_server_agreements = [
        max(agreement, 0.0)
        for agreement, is_kept in zip(server_agreements, kept, strict=True)
        if is_kept
    ]
    kept_staleness = [
        update.staleness
        for update, is_kept in zip(updates, kept, strict=True)
        if is_kept
    ]
    kept_fairness = [
        fairness
        for fairness, is_kept in zip(fairness_values, kept, strict=True)
        if is_kept
    ]

    mean_agreement = float(np.mean(kept_agreements)) if kept_agreements else 0.0
    mean_server_agreement = (
        float(np.mean(kept_server_agreements)) if kept_server_agreements else 0.0
    )
    mean_staleness = float(np.mean(kept_staleness)) if kept_staleness else 0.0
    mean_fairness = float(np.mean(kept_fairness)) if kept_fairness else 0.0
    blend = min(max(config.history_agreement_blend, 0.0), 1.0)
    agreement_signal = (1.0 - blend) * mean_agreement + blend * mean_server_agreement
    denominator = 1.0 + mean_staleness / max(config.adaptive_staleness_scale, 1e-8)
    buffer_alpha = _clamp(
        config.server_alpha * (1.0 + config.adaptive_alpha_boost * agreement_signal) / denominator,
        config.adaptive_alpha_min,
        config.adaptive_alpha_max,
    )
    merged = add_scaled_delta(current_state, accepted_delta, buffer_alpha)
    stats = {
        "buffer_alpha": buffer_alpha,
        "mean_agreement": mean_agreement,
        "mean_server_agreement": mean_server_agreement,
        "mean_fairness_weight": mean_fairness,
        "mean_delta_norm": float(np.mean(norms)) if norms else 0.0,
        "dropped_updates": float(dropped_count),
        "applied_updates": float(len(updates) - dropped_count),
    }
    return merged, stats, accepted_delta


def _clip_delta(delta: AdapterState, norm: float, clip_norm: float) -> AdapterState:
    if clip_norm <= 0.0 or norm <= clip_norm:
        return clone_state(delta)
    scale = clip_norm / max(norm, 1e-12)
    return {name: value * scale for name, value in delta.items()}


def _empty_stats(alpha: float) -> dict[str, float]:
    return {
        "buffer_alpha": float(alpha),
        "mean_agreement": 0.0,
        "mean_server_agreement": 0.0,
        "mean_fairness_weight": 0.0,
        "mean_delta_norm": 0.0,
        "dropped_updates": 0.0,
        "applied_updates": 0.0,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))

