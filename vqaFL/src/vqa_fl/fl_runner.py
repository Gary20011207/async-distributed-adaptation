from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from vqa_fl.adapter_math import AdapterState, clone_state
from vqa_fl.caa_lora import CAAConfig, DecayName, LoRAUpdate, staleness_decay_weight
from vqa_fl.data import VQAChoiceExample, client_size_summary, partition_examples, split_examples
from vqa_fl.federated import FLMethod, FLServerConfig, LoRAServer
from vqa_fl.qwen_lora import set_adapter_state
from vqa_fl.trainer import EvalConfig, LocalTrainConfig, evaluate_choice_accuracy, train_local_lora

MethodArg = FLMethod | Literal["all"]


@dataclass(frozen=True)
class FLExperimentConfig:
    method: FLMethod
    clients: int
    rounds: int
    buffer_size: int
    server_alpha: float
    local_train: LocalTrainConfig
    eval_config: EvalConfig
    eval_every: int
    eval_fraction: float
    seed: int
    staleness_decay: DecayName = "hinge"
    shuffle_clients: bool = True
    skip_eval: bool = False


def run_lora_fl_experiment(
    *,
    model: Any,
    processor: Any,
    examples: list[VQAChoiceExample],
    initial_state: AdapterState,
    config: FLExperimentConfig,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL LoRA FL")
    if config.clients <= 0:
        raise ValueError("clients must be positive")
    if config.rounds <= 0:
        raise ValueError("rounds must be positive")

    train_examples, eval_examples = split_examples(
        examples,
        eval_fraction=config.eval_fraction,
        seed=config.seed,
    )
    partitions = partition_examples(
        train_examples,
        clients=config.clients,
        mode="iid",
        seed=config.seed,
    )
    server = LoRAServer(
        initial_state,
        FLServerConfig(
            method=config.method,
            clients=config.clients,
            server_alpha=config.server_alpha,
            caa=replace(CAAConfig(), server_alpha=config.server_alpha),
        ),
    )
    rng = random.Random(config.seed)

    history: list[dict[str, Any]] = []
    for round_idx in range(1, config.rounds + 1):
        round_state = clone_state(server.state)
        round_start_version = server.version
        client_order = list(range(config.clients))
        if config.shuffle_clients:
            rng.shuffle(client_order)

        pending_updates: list[LoRAUpdate] = []
        client_stats: list[dict[str, Any]] = []
        event_stats: list[dict[str, Any]] = []

        for cid in client_order:
            client_examples = list(partitions[cid])
            if not client_examples:
                client_stats.append({"cid": cid, "examples": 0, "skipped": True})
                continue

            start_state = clone_state(round_state)
            set_adapter_state(model, start_state)
            updated_state, train_stats = train_local_lora(
                model=model,
                processor=processor,
                examples=client_examples,
                config=config.local_train,
            )

            arrival_version = server.version
            update = _make_update(
                cid=cid,
                start_version=round_start_version,
                arrival_version=arrival_version,
                examples=len(client_examples),
                start_state=start_state,
                updated_state=updated_state,
                config=config,
            )
            client_record = {
                "cid": cid,
                "examples": len(client_examples),
                "start_version": round_start_version,
                "arrival_version": arrival_version,
                "staleness": update.staleness,
                **train_stats,
            }
            client_stats.append(client_record)

            if config.method in ("naive_async", "staleness_async"):
                result = server.apply([update])
                event_stats.append(
                    {
                        "cid": cid,
                        "server_version": result.version,
                        **result.stats,
                    }
                )
            else:
                pending_updates.append(update)

        if pending_updates:
            event_stats.extend(_apply_buffered_updates(server, pending_updates, config))

        set_adapter_state(model, server.state)
        record: dict[str, Any] = {
            "round": round_idx,
            "server_version": server.version,
            "train": client_stats,
            "events": event_stats,
        }
        if _should_eval(config, round_idx):
            record["eval"] = evaluate_choice_accuracy(
                model=model,
                processor=processor,
                examples=eval_examples,
                config=config.eval_config,
            )
        history.append(record)
        torch.cuda.empty_cache()

    final_eval = None
    if not config.skip_eval:
        set_adapter_state(model, server.state)
        final_eval = evaluate_choice_accuracy(
            model=model,
            processor=processor,
            examples=eval_examples,
            config=config.eval_config,
        )

    return {
        "method": config.method,
        "rounds": config.rounds,
        "clients": config.clients,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "client_summary": client_size_summary(partitions),
        "final_server_version": server.version,
        "final_eval": final_eval,
        "history": history,
    }


def default_methods() -> list[FLMethod]:
    return ["sync_fedavg", "naive_async", "staleness_async", "fedbuff", "caa_v2"]


def write_json_result(payload: dict[str, Any], path: str | Path) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _apply_buffered_updates(
    server: LoRAServer,
    updates: list[LoRAUpdate],
    config: FLExperimentConfig,
) -> list[dict[str, Any]]:
    if config.method == "sync_fedavg":
        for update in updates:
            _refresh_staleness(update, server.version, config)
        result = server.apply(updates)
        return [{"server_version": result.version, **result.stats}]

    records: list[dict[str, Any]] = []
    chunk_size = max(config.buffer_size, 1)
    for idx in range(0, len(updates), chunk_size):
        chunk = updates[idx : idx + chunk_size]
        for update in chunk:
            _refresh_staleness(update, server.version, config)
        result = server.apply(chunk)
        records.append({"server_version": result.version, **result.stats})
    return records


def _make_update(
    *,
    cid: int,
    start_version: int,
    arrival_version: int,
    examples: int,
    start_state: AdapterState,
    updated_state: AdapterState,
    config: FLExperimentConfig,
) -> LoRAUpdate:
    staleness = max(arrival_version - start_version, 0)
    decay = "constant" if config.method in ("sync_fedavg", "naive_async") else config.staleness_decay
    return LoRAUpdate(
        cid=cid,
        start_version=start_version,
        arrival_version=arrival_version,
        num_examples=examples,
        start_state=start_state,
        updated_state=updated_state,
        staleness_weight=staleness_decay_weight(staleness, decay),
    )


def _refresh_staleness(
    update: LoRAUpdate,
    arrival_version: int,
    config: FLExperimentConfig,
) -> None:
    update.arrival_version = arrival_version
    decay = "constant" if config.method in ("sync_fedavg", "naive_async") else config.staleness_decay
    update.staleness_weight = staleness_decay_weight(update.staleness, decay)


def _should_eval(config: FLExperimentConfig, round_idx: int) -> bool:
    if config.skip_eval:
        return False
    if config.eval_every <= 0:
        return False
    return round_idx % config.eval_every == 0
