from __future__ import annotations

import random

import torch

from vqa_fl.adapter_math import clone_state, update_ema
from vqa_fl.caa_lora import (
    CAAConfig,
    LoRAUpdate,
    apply_caa_v2_lora,
    apply_fedbuff_lora,
    staleness_decay_weight,
)


def make_synthetic_lora_state(
    *,
    rank: int,
    hidden: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
            rank,
            hidden,
            generator=generator,
            device=device,
        )
        * 0.01,
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
            hidden,
            rank,
            generator=generator,
            device=device,
        )
        * 0.01,
    }


def run_adapter_smoke(
    *,
    clients: int,
    buffer_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    current = make_synthetic_lora_state(rank=4, hidden=16, device=device, seed=seed)
    server_delta_ema = None
    client_apply_counts = [0 for _ in range(clients)]
    config = CAAConfig()

    updates = []
    for cid in range(buffer_size):
        client_id = cid % clients
        start_state = clone_state(current)
        updated_state = clone_state(start_state)
        local_seed = seed + cid + 1
        generator = torch.Generator(device=device).manual_seed(local_seed)
        for name, value in updated_state.items():
            direction = 1.0 if cid < max(buffer_size - 1, 1) else -0.35
            noise = torch.randn(value.shape, generator=generator, device=device) * 0.005
            updated_state[name] = value + direction * noise

        staleness = rng.randint(0, 8)
        updates.append(
            LoRAUpdate(
                cid=client_id,
                start_version=10,
                arrival_version=10 + staleness,
                num_examples=32,
                start_state=start_state,
                updated_state=updated_state,
                staleness_weight=staleness_decay_weight(
                    staleness,
                    "hinge",
                    hinge_b=5,
                    hinge_a=0.05,
                ),
                delay=float(rng.uniform(1.0, 5.0)),
            )
        )

    fedbuff_state, fedbuff_stats, _fedbuff_delta = apply_fedbuff_lora(
        current,
        updates,
        server_alpha=0.5,
    )
    caa_state, caa_stats, accepted_delta = apply_caa_v2_lora(
        current,
        updates,
        server_delta_ema=server_delta_ema,
        client_apply_counts=client_apply_counts,
        config=config,
    )
    server_delta_ema = update_ema(server_delta_ema, accepted_delta, momentum=0.8)

    return {
        "fedbuff_norm": _state_norm(fedbuff_state),
        "caa_norm": _state_norm(caa_state),
        "ema_norm": _state_norm(server_delta_ema),
        **caa_stats,
    }


def _state_norm(state: dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(state.values())).device)
    for value in state.values():
        total = total + torch.sum(value.float() * value.float())
    return float(torch.sqrt(total).detach().cpu().item())

