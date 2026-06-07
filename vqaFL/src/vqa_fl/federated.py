from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from vqa_fl.adapter_math import AdapterState, clone_state, update_ema
from vqa_fl.caa_lora import (
    CAAConfig,
    LoRAUpdate,
    apply_caa_v2_lora,
    apply_fedbuff_lora,
)

FLMethod = Literal[
    "sync_fedavg",
    "naive_async",
    "staleness_async",
    "fedbuff",
    "caa_v2",
]


@dataclass(frozen=True)
class FLServerConfig:
    method: FLMethod = "caa_v2"
    clients: int = 3
    server_alpha: float = 0.5
    ema_momentum: float = 0.8
    caa: CAAConfig = field(default_factory=CAAConfig)


@dataclass
class FLApplyResult:
    state: AdapterState
    version: int
    stats: dict[str, float]


class LoRAServer:
    def __init__(self, initial_state: AdapterState, config: FLServerConfig) -> None:
        self.state = clone_state(initial_state)
        self.config = config
        self.version = 0
        self.server_delta_ema: AdapterState | None = None
        self.client_apply_counts = [0 for _ in range(config.clients)]

    def apply(self, updates: list[LoRAUpdate]) -> FLApplyResult:
        if not updates:
            return FLApplyResult(
                state=clone_state(self.state),
                version=self.version,
                stats={"applied_updates": 0.0, "dropped_updates": 0.0},
            )

        if self.config.method == "sync_fedavg":
            new_state, stats, accepted_delta = apply_fedbuff_lora(
                self.state,
                updates,
                server_alpha=1.0,
            )
        elif self.config.method == "naive_async":
            new_state, stats, accepted_delta = apply_fedbuff_lora(
                self.state,
                [updates[0]],
                server_alpha=self.config.server_alpha,
            )
        elif self.config.method == "staleness_async":
            effective_alpha = self.config.server_alpha * max(updates[0].staleness_weight, 0.0)
            new_state, stats, accepted_delta = apply_fedbuff_lora(
                self.state,
                [updates[0]],
                server_alpha=effective_alpha,
            )
        elif self.config.method == "fedbuff":
            new_state, stats, accepted_delta = apply_fedbuff_lora(
                self.state,
                updates,
                server_alpha=self.config.server_alpha,
            )
        elif self.config.method == "caa_v2":
            new_state, stats, accepted_delta = apply_caa_v2_lora(
                self.state,
                updates,
                server_delta_ema=self.server_delta_ema,
                client_apply_counts=self.client_apply_counts,
                config=self.config.caa,
            )
        else:
            raise ValueError(f"Unsupported FL method: {self.config.method}")

        self.state = new_state
        self.version += 1
        self.server_delta_ema = update_ema(
            self.server_delta_ema,
            accepted_delta,
            momentum=self.config.ema_momentum,
        )
        for update in updates:
            if update.cid < len(self.client_apply_counts) and not update.dropped_update:
                self.client_apply_counts[update.cid] += 1

        return FLApplyResult(
            state=clone_state(self.state),
            version=self.version,
            stats={**stats, "server_version": float(self.version)},
        )
