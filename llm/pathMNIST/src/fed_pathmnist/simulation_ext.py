from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from fed_pathmnist.client import PathMNISTClient, evaluate_model
from fed_pathmnist.experiment_logging import ExperimentLogger, save_checkpoint
from fed_pathmnist.model import create_model, get_parameters, interpolate, set_parameters


Method = Literal[
    "sync_fedavg",
    "naive_async",
    "staleness_async",
    "fedbuff_async",
    "agreement_fedbuff_async",
    "caa_fedbuff_v2",
]
AsyncMethod = Literal[
    "naive_async",
    "staleness_async",
    "fedbuff_async",
    "agreement_fedbuff_async",
    "caa_fedbuff_v2",
]
DecayName = Literal["constant", "inverse", "polynomial", "exponential", "hinge"]
DelayMode = Literal["uniform", "lognormal", "heterogeneous"]


@dataclass(order=True)
class AsyncJob:
    finish_time: float
    job_id: int
    cid: int = field(compare=False)
    start_version: int = field(compare=False)
    start_time: float = field(compare=False)
    delay: float = field(compare=False)
    parameters: list[np.ndarray] = field(compare=False)


@dataclass
class BufferedUpdate:
    cid: int
    start_version: int
    arrival_server_version: int
    staleness: int
    delay: float
    num_examples: int
    train_loss: float
    staleness_weight: float
    effective_alpha: float
    start_parameters: list[np.ndarray]
    updated_parameters: list[np.ndarray]
    agreement: float = 0.0
    delta_norm: float = 0.0
    dropped_update: bool = False
    fairness_weight: float = 1.0


def build_clients(
    loaders: list,
    testloader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    num_classes: int = 9,
    in_channels: int = 3,
    model_name: str = "qwen",
) -> list[PathMNISTClient]:
    return [
        PathMNISTClient(
            cid=cid,
            trainloader=client_loader.train,
            testloader=testloader,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
        )
        for cid, client_loader in enumerate(loaders)
    ]


def evaluate_global(
    parameters: list[np.ndarray],
    testloader: DataLoader,
    device: torch.device,
    *,
    num_classes: int = 9,
    in_channels: int = 3,
    model_name: str = "qwen",
) -> tuple[float, float]:
    model = create_model(
        num_classes=num_classes,
        in_channels=in_channels,
        model_name=model_name,
    ).to(device)
    set_parameters(model, parameters)
    loss, accuracy, _ = evaluate_model(model, testloader, device)
    return loss, accuracy


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


def run_sync_fedavg(
    clients: list[PathMNISTClient],
    testloader: DataLoader,
    device: torch.device,
    rounds: int,
    lr_schedule: Callable[[int], float],
    *,
    logger: ExperimentLogger | None = None,
    save_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    args_config: dict[str, Any] | None = None,
    dataset_name: str = "pathmnist",
    num_classes: int = 9,
    in_channels: int = 3,
    model_name: str = "resnet18",
) -> list[np.ndarray]:
    global_model = create_model(
        num_classes=num_classes,
        in_channels=in_channels,
        model_name=model_name,
    ).to(device)
    parameters = get_parameters(global_model)
    best_acc = -1.0
    best_step = 0
    final_loss = float("nan")
    final_acc = float("nan")
    checkpoint_path = ""

    for rnd in range(1, rounds + 1):
        lr = lr_schedule(rnd)
        results = []
        train_losses = []
        for client in clients:
            updated, num_examples, metrics = client.fit(parameters, {"round": rnd, "lr": lr})
            results.append((updated, num_examples))
            train_losses.append(metrics["train_loss"])

        parameters = _weighted_average(results)
        final_loss, final_acc = evaluate_global(
            parameters,
            testloader,
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
        )
        train_loss = float(np.mean(train_losses))
        if final_acc > best_acc:
            best_acc = final_acc
            best_step = rnd
            if save_best and logger is not None:
                checkpoint_path = str(
                    save_checkpoint(
                        parameters=parameters,
                        method="sync_fedavg",
                        dataset_name=dataset_name,
                        best_test_acc=best_acc,
                        best_round_or_event=best_step,
                        args_config=args_config or {},
                        checkpoint_dir=checkpoint_dir,
                        run_id=logger.run_id,
                    )
                )

        if logger is not None:
            logger.log(
                method="sync_fedavg",
                round_or_event=rnd,
                server_version=rnd,
                arrival_server_version=rnd,
                simulated_time=0.0,
                client_id="NA",
                client_start_version="NA",
                staleness=0,
                base_alpha=1.0,
                effective_alpha=1.0,
                staleness_weight=1.0,
                learning_rate=lr,
                train_loss=train_loss,
                test_loss=final_loss,
                test_acc=final_acc,
                num_examples=sum(num_examples for _, num_examples in results),
                delay=0.0,
                buffer_size=0,
                applied_updates=len(clients),
            )

        print(
            f"round={rnd} method=sync_fedavg "
            f"lr={lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={final_loss:.4f} test_acc={final_acc:.4f}",
            flush=True,
        )

    if logger is not None:
        logger.write_summary(
            {
                "method": "sync_fedavg",
                "best_test_acc": best_acc,
                "best_round_or_event": best_step,
                "final_test_acc": final_acc,
                "final_test_loss": final_loss,
                "total_rounds_or_events": rounds,
                "total_simulated_time": 0.0,
                "csv_path": str(logger.csv_path),
                "checkpoint_path": checkpoint_path,
                "config": args_config or {},
            }
        )

    return parameters


def run_async(
    clients: list[PathMNISTClient],
    testloader: DataLoader,
    device: torch.device,
    method: AsyncMethod,
    events: int,
    alpha: float,
    seed: int,
    eval_every: int,
    lr_schedule: Callable[[int], float],
    *,
    staleness_decay: DecayName = "inverse",
    staleness_power: float = 1.0,
    staleness_exp_rate: float = 0.1,
    staleness_hinge_b: int = 5,
    staleness_hinge_a: float = 0.1,
    delay_mode: DelayMode = "uniform",
    min_delay: float = 1.0,
    max_delay: float = 5.0,
    lognormal_mean: float = 1.0,
    lognormal_sigma: float = 0.5,
    straggler_ratio: float = 0.2,
    straggler_multiplier: float = 5.0,
    buffer_size: int = 5,
    agreement_epsilon: float = 0.10,
    agreement_power: float = 1.0,
    agreement_drop_threshold: float = -0.05,
    delta_clip_multiplier: float = 1.5,
    adaptive_alpha_min: float = 0.15,
    adaptive_alpha_max: float = 0.65,
    adaptive_alpha_boost: float = 0.35,
    adaptive_staleness_scale: float = 8.0,
    server_delta_momentum: float = 0.8,
    history_agreement_blend: float = 0.25,
    client_fairness_power: float = 0.5,
    logger: ExperimentLogger | None = None,
    save_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    args_config: dict[str, Any] | None = None,
    dataset_name: str = "pathmnist",
    num_classes: int = 9,
    in_channels: int = 3,
    model_name: str = "resnet18",
) -> list[np.ndarray]:
    rng = random.Random(seed)
    global_model = create_model(
        num_classes=num_classes,
        in_channels=in_channels,
        model_name=model_name,
    ).to(device)
    global_parameters = get_parameters(global_model)
    global_version = 0
    next_job_id = 0
    now = 0.0
    jobs: list[AsyncJob] = []
    buffer: list[BufferedUpdate] = []
    server_delta_ema: list[np.ndarray] | None = None
    client_apply_counts = [0 for _ in clients]
    eval_every = max(int(eval_every), 1)
    buffer_size = max(int(buffer_size), 1)
    decay_name: DecayName = "constant" if method == "naive_async" else staleness_decay

    straggler_ids = _select_stragglers(
        rng,
        num_clients=len(clients),
        ratio=straggler_ratio,
        enabled=delay_mode == "heterogeneous",
    )

    best_acc = -1.0
    best_step = 0
    final_loss = float("nan")
    final_acc = float("nan")
    checkpoint_path = ""

    def submit(cid: int) -> None:
        nonlocal next_job_id
        delay = _sample_delay(
            rng,
            mode=delay_mode,
            cid=cid,
            straggler_ids=straggler_ids,
            min_delay=min_delay,
            max_delay=max_delay,
            lognormal_mean=lognormal_mean,
            lognormal_sigma=lognormal_sigma,
            straggler_multiplier=straggler_multiplier,
        )
        heapq.heappush(
            jobs,
            AsyncJob(
                finish_time=now + delay,
                job_id=next_job_id,
                cid=cid,
                start_version=global_version,
                start_time=now,
                delay=delay,
                parameters=_copy_parameters(global_parameters),
            ),
        )
        next_job_id += 1

    for cid in range(len(clients)):
        submit(cid)

    for event in range(1, events + 1):
        lr = lr_schedule(event)
        job = heapq.heappop(jobs)
        now = job.finish_time
        arrival_server_version = global_version
        client = clients[job.cid]
        updated, num_examples, metrics = client.fit(
            job.parameters,
            {"event": event, "start_version": job.start_version, "lr": lr},
        )

        staleness = max(arrival_server_version - job.start_version, 0)
        weight = staleness_decay_weight(
            staleness,
            decay_name,
            power=staleness_power,
            exp_rate=staleness_exp_rate,
            hinge_b=staleness_hinge_b,
            hinge_a=staleness_hinge_a,
        )
        effective_alpha = alpha * weight
        applied_updates = 0
        logged_buffer_size = len(buffer)
        agreement: float | str = ""
        mean_agreement: float | str = ""
        buffer_alpha: float | str = ""
        delta_norm: float | str = ""
        dropped_update: int | str = ""
        server_momentum_agreement: float | str = ""
        fairness_weight: float | str = ""
        pending_pool_size: int | str = ""
        quorum_size: int | str = ""
        quorum_met: int | str = ""
        selected_updates: int | str = ""

        if method in ("naive_async", "staleness_async"):
            global_parameters = interpolate(global_parameters, updated, effective_alpha)
            global_version += 1
            applied_updates = 1
        else:
            update_delta_norm = _delta_norm(job.parameters, updated)
            buffer.append(
                BufferedUpdate(
                    cid=job.cid,
                    start_version=job.start_version,
                    arrival_server_version=arrival_server_version,
                    staleness=staleness,
                    delay=job.delay,
                    num_examples=num_examples,
                    train_loss=float(metrics["train_loss"]),
                    staleness_weight=weight,
                    effective_alpha=effective_alpha,
                    start_parameters=job.parameters,
                    updated_parameters=updated,
                    delta_norm=update_delta_norm,
                )
            )
            logged_buffer_size = len(buffer)
            if len(buffer) >= buffer_size or event == events:
                if method in ("agreement_fedbuff_async", "caa_fedbuff_v2"):
                    if method == "caa_fedbuff_v2":
                        global_parameters, aggregate_stats, accepted_delta = (
                            _apply_caa_v2_buffered_delta(
                                current_parameters=global_parameters,
                                buffer=buffer,
                                server_alpha=alpha,
                                server_delta_ema=server_delta_ema,
                                client_apply_counts=client_apply_counts,
                                agreement_epsilon=agreement_epsilon,
                                agreement_power=agreement_power,
                                agreement_drop_threshold=agreement_drop_threshold,
                                delta_clip_multiplier=delta_clip_multiplier,
                                adaptive_alpha_min=adaptive_alpha_min,
                                adaptive_alpha_max=adaptive_alpha_max,
                                adaptive_alpha_boost=adaptive_alpha_boost,
                                adaptive_staleness_scale=adaptive_staleness_scale,
                                drop_staleness_threshold=staleness_hinge_b,
                                server_delta_momentum=server_delta_momentum,
                                history_agreement_blend=history_agreement_blend,
                                client_fairness_power=client_fairness_power,
                            )
                        )
                        server_delta_ema = _update_delta_ema(
                            server_delta_ema,
                            accepted_delta,
                            momentum=server_delta_momentum,
                        )
                    else:
                        accepted_delta = None
                        global_parameters, aggregate_stats = _apply_agreement_buffered_delta(
                            current_parameters=global_parameters,
                            buffer=buffer,
                            server_alpha=alpha,
                            agreement_epsilon=agreement_epsilon,
                            agreement_power=agreement_power,
                            agreement_drop_threshold=agreement_drop_threshold,
                            delta_clip_multiplier=delta_clip_multiplier,
                            adaptive_alpha_min=adaptive_alpha_min,
                            adaptive_alpha_max=adaptive_alpha_max,
                            adaptive_alpha_boost=adaptive_alpha_boost,
                            adaptive_staleness_scale=adaptive_staleness_scale,
                            drop_staleness_threshold=staleness_hinge_b,
                        )
                    agreement = aggregate_stats["mean_agreement"]
                    mean_agreement = aggregate_stats["mean_agreement"]
                    buffer_alpha = aggregate_stats["buffer_alpha"]
                    delta_norm = aggregate_stats["mean_delta_norm"]
                    dropped_update = aggregate_stats["dropped_updates"]
                    applied_updates = int(aggregate_stats["applied_updates"])
                    server_momentum_agreement = aggregate_stats.get(
                        "server_momentum_agreement",
                        "",
                    )
                    fairness_weight = aggregate_stats.get("mean_fairness_weight", "")
                    selected_updates = applied_updates
                    for update in buffer:
                        if not update.dropped_update:
                            client_apply_counts[update.cid] += 1
                else:
                    global_parameters = _apply_buffered_delta(global_parameters, buffer, alpha)
                    applied_updates = len(buffer)
                global_version += 1
                buffer.clear()

        should_eval = event == 1 or event % eval_every == 0 or event == events
        if should_eval:
            final_loss, final_acc = evaluate_global(
                global_parameters,
                testloader,
                device,
                num_classes=num_classes,
                in_channels=in_channels,
                model_name=model_name,
            )
            if final_acc > best_acc:
                best_acc = final_acc
                best_step = event
                if save_best and logger is not None:
                    checkpoint_path = str(
                        save_checkpoint(
                            parameters=global_parameters,
                            method=method,
                            dataset_name=dataset_name,
                            best_test_acc=best_acc,
                            best_round_or_event=best_step,
                            args_config=args_config or {},
                            checkpoint_dir=checkpoint_dir,
                            run_id=logger.run_id,
                        )
                    )

            print(
                f"event={event} method={method} cid={job.cid} "
                f"time={now:.2f} version={global_version} lr={lr:.6f} "
                f"staleness={staleness} alpha={effective_alpha:.4f} "
                f"train_loss={metrics['train_loss']:.4f} "
                f"test_loss={final_loss:.4f} test_acc={final_acc:.4f}"
                + (
                    f" mean_agreement={mean_agreement:.4f} "
                    f"buffer_alpha={buffer_alpha:.4f} dropped={dropped_update}"
                    if mean_agreement != ""
                    else ""
                ),
                flush=True,
            )

        if logger is not None:
            logger.log(
                method=method,
                round_or_event=event,
                server_version=global_version,
                arrival_server_version=arrival_server_version,
                simulated_time=now,
                client_id=job.cid,
                client_start_version=job.start_version,
                staleness=staleness,
                base_alpha=alpha,
                effective_alpha=effective_alpha,
                staleness_weight=weight,
                learning_rate=lr,
                train_loss=float(metrics["train_loss"]),
                test_loss=final_loss if should_eval else "",
                test_acc=final_acc if should_eval else "",
                num_examples=num_examples,
                delay=job.delay,
                buffer_size=logged_buffer_size,
                applied_updates=applied_updates,
                agreement=agreement,
                mean_agreement=mean_agreement,
                buffer_alpha=buffer_alpha,
                delta_norm=delta_norm,
                dropped_update=dropped_update,
                server_momentum_agreement=server_momentum_agreement,
                fairness_weight=fairness_weight,
                pending_pool_size=pending_pool_size,
                quorum_size=quorum_size,
                quorum_met=quorum_met,
                selected_updates=selected_updates,
            )

        submit(job.cid)

    if logger is not None:
        logger.write_summary(
            {
                "method": method,
                "best_test_acc": best_acc,
                "best_round_or_event": best_step,
                "final_test_acc": final_acc,
                "final_test_loss": final_loss,
                "total_rounds_or_events": events,
                "total_simulated_time": now,
                "csv_path": str(logger.csv_path),
                "checkpoint_path": checkpoint_path,
                "straggler_ids": sorted(straggler_ids),
                "config": args_config or {},
            }
        )

    return global_parameters


def _weighted_average(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    from fed_pathmnist.model import weighted_average

    return weighted_average(results)


def _copy_parameters(parameters: list[np.ndarray]) -> list[np.ndarray]:
    return [layer.copy() for layer in parameters]


def _select_stragglers(
    rng: random.Random,
    *,
    num_clients: int,
    ratio: float,
    enabled: bool,
) -> set[int]:
    if not enabled or ratio <= 0.0 or num_clients <= 0:
        return set()
    count = min(num_clients, max(1, round(num_clients * ratio)))
    return set(rng.sample(range(num_clients), count))


def _sample_delay(
    rng: random.Random,
    *,
    mode: DelayMode,
    cid: int,
    straggler_ids: set[int],
    min_delay: float,
    max_delay: float,
    lognormal_mean: float,
    lognormal_sigma: float,
    straggler_multiplier: float,
) -> float:
    min_delay = max(float(min_delay), 0.0)
    max_delay = max(float(max_delay), min_delay)
    if mode == "uniform":
        return rng.uniform(min_delay, max_delay)
    if mode == "lognormal":
        return max(min_delay, rng.lognormvariate(lognormal_mean, lognormal_sigma))
    if mode == "heterogeneous":
        delay = rng.uniform(min_delay, max_delay)
        if cid in straggler_ids:
            delay *= max(float(straggler_multiplier), 1.0)
        return delay
    raise ValueError(f"Unsupported delay mode: {mode}")


def _apply_buffered_delta(
    current_parameters: list[np.ndarray],
    buffer: list[BufferedUpdate],
    server_alpha: float,
) -> list[np.ndarray]:
    total_weight = sum(
        max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
        for update in buffer
    )
    if total_weight <= 0.0:
        return _copy_parameters(current_parameters)

    new_parameters: list[np.ndarray] = []
    for layer_idx, current_layer in enumerate(current_parameters):
        if not np.issubdtype(current_layer.dtype, np.floating):
            new_parameters.append(current_layer.copy())
            continue

        aggregate_delta = np.zeros_like(current_layer, dtype=np.float64)
        for update in buffer:
            raw_weight = max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
            normalized_weight = raw_weight / total_weight
            delta = update.updated_parameters[layer_idx] - update.start_parameters[layer_idx]
            aggregate_delta += normalized_weight * delta

        updated_layer = current_layer + server_alpha * aggregate_delta
        new_parameters.append(updated_layer.astype(current_layer.dtype, copy=False))

    return new_parameters


def _apply_agreement_buffered_delta(
    *,
    current_parameters: list[np.ndarray],
    buffer: list[BufferedUpdate],
    server_alpha: float,
    agreement_epsilon: float,
    agreement_power: float,
    agreement_drop_threshold: float,
    delta_clip_multiplier: float,
    adaptive_alpha_min: float,
    adaptive_alpha_max: float,
    adaptive_alpha_boost: float,
    adaptive_staleness_scale: float,
    drop_staleness_threshold: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """FedBuff delta aggregation with agreement filtering and adaptive alpha.

    The agreement score is computed against a provisional buffered direction.
    This keeps the method clockless: it uses logical versions and model deltas,
    not wall-clock timestamps, to decide how much each update should matter.
    """

    if not buffer:
        return _copy_parameters(current_parameters), _empty_agreement_stats(server_alpha)

    base_weights = [
        max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
        for update in buffer
    ]
    base_total = sum(base_weights)
    if base_total <= 0.0:
        return _copy_parameters(current_parameters), _empty_agreement_stats(server_alpha)

    reference_delta = _reference_delta(buffer, base_weights, base_total)
    agreements = [_cosine_to_reference(update, reference_delta) for update in buffer]
    norms = [update.delta_norm for update in buffer]
    kept = [
        not (
            agreement < agreement_drop_threshold
            and update.staleness > drop_staleness_threshold
        )
        for update, agreement in zip(buffer, agreements, strict=True)
    ]

    dropped_count = kept.count(False)
    if not any(kept):
        # A full rejection is too brittle for a small course-project buffer.
        # Fall back to standard FedBuff weighting for this buffer.
        kept = [True for _ in buffer]
        dropped_count = 0

    positive_norms = [norm for norm in norms if norm > 0.0]
    if positive_norms:
        clip_norm = float(np.median(positive_norms)) * max(delta_clip_multiplier, 0.0)
    else:
        clip_norm = 0.0

    raw_weights: list[float] = []
    for update, agreement, base_weight, is_kept in zip(
        buffer,
        agreements,
        base_weights,
        kept,
        strict=True,
    ):
        update.agreement = agreement
        update.dropped_update = not is_kept
        if not is_kept:
            raw_weights.append(0.0)
            continue
        factor = (max(float(agreement_epsilon), 0.0) + max(agreement, 0.0)) ** max(
            agreement_power,
            0.0,
        )
        raw_weights.append(base_weight * factor)

    raw_total = sum(raw_weights)
    if raw_total <= 0.0:
        raw_weights = [base_weight if is_kept else 0.0 for base_weight, is_kept in zip(base_weights, kept, strict=True)]
        raw_total = sum(raw_weights)
    if raw_total <= 0.0:
        return _apply_buffered_delta(current_parameters, buffer, server_alpha), {
            "mean_agreement": float(np.mean([max(value, 0.0) for value in agreements])),
            "buffer_alpha": server_alpha,
            "mean_delta_norm": float(np.mean(norms)) if norms else 0.0,
            "dropped_updates": 0.0,
            "applied_updates": float(len(buffer)),
        }

    kept_agreements = [
        max(agreement, 0.0)
        for agreement, is_kept in zip(agreements, kept, strict=True)
        if is_kept
    ]
    kept_staleness = [
        update.staleness for update, is_kept in zip(buffer, kept, strict=True) if is_kept
    ]
    mean_agreement = float(np.mean(kept_agreements)) if kept_agreements else 0.0
    mean_staleness = float(np.mean(kept_staleness)) if kept_staleness else 0.0
    denominator = 1.0 + mean_staleness / max(float(adaptive_staleness_scale), 1e-8)
    buffer_alpha = _clamp(
        server_alpha * (1.0 + adaptive_alpha_boost * mean_agreement) / denominator,
        adaptive_alpha_min,
        adaptive_alpha_max,
    )

    new_parameters: list[np.ndarray] = []
    for layer_idx, current_layer in enumerate(current_parameters):
        if not np.issubdtype(current_layer.dtype, np.floating):
            new_parameters.append(current_layer.copy())
            continue

        aggregate_delta = np.zeros_like(current_layer, dtype=np.float64)
        for update, raw_weight, norm in zip(buffer, raw_weights, norms, strict=True):
            if raw_weight <= 0.0:
                continue
            normalized_weight = raw_weight / raw_total
            delta = update.updated_parameters[layer_idx] - update.start_parameters[layer_idx]
            if clip_norm > 0.0 and norm > clip_norm:
                delta = delta * (clip_norm / max(norm, 1e-12))
            aggregate_delta += normalized_weight * delta

        updated_layer = current_layer + buffer_alpha * aggregate_delta
        new_parameters.append(updated_layer.astype(current_layer.dtype, copy=False))

    return new_parameters, {
        "mean_agreement": mean_agreement,
        "buffer_alpha": buffer_alpha,
        "mean_delta_norm": float(np.mean(norms)) if norms else 0.0,
        "dropped_updates": float(dropped_count),
        "applied_updates": float(len(buffer) - dropped_count),
    }


def _apply_caa_v2_buffered_delta(
    *,
    current_parameters: list[np.ndarray],
    buffer: list[BufferedUpdate],
    server_alpha: float,
    server_delta_ema: list[np.ndarray] | None,
    client_apply_counts: list[int],
    agreement_epsilon: float,
    agreement_power: float,
    agreement_drop_threshold: float,
    delta_clip_multiplier: float,
    adaptive_alpha_min: float,
    adaptive_alpha_max: float,
    adaptive_alpha_boost: float,
    adaptive_staleness_scale: float,
    drop_staleness_threshold: int,
    server_delta_momentum: float,
    history_agreement_blend: float,
    client_fairness_power: float,
) -> tuple[list[np.ndarray], dict[str, float], list[np.ndarray]]:
    """CAA-FedBuff v2 with server trajectory agreement and client fairness.

    The rule remains clockless. It uses logical staleness, buffered model
    deltas, recent accepted server direction, and contribution counts.
    """

    if not buffer:
        return (
            _copy_parameters(current_parameters),
            _empty_agreement_stats(server_alpha),
            _zero_delta_like(current_parameters),
        )

    base_weights = [
        max(update.num_examples, 1) * max(update.staleness_weight, 0.0)
        for update in buffer
    ]
    base_total = sum(base_weights)
    if base_total <= 0.0:
        return (
            _copy_parameters(current_parameters),
            _empty_agreement_stats(server_alpha),
            _zero_delta_like(current_parameters),
        )

    buffer_reference = _reference_delta(buffer, base_weights, base_total)
    reference_delta = _blend_reference_delta(
        buffer_reference,
        server_delta_ema,
        blend=history_agreement_blend,
    )
    agreements = [_cosine_to_reference(update, reference_delta) for update in buffer]
    if server_delta_ema is None:
        server_agreements = [0.0 for _ in buffer]
    else:
        server_agreements = [_cosine_to_reference(update, server_delta_ema) for update in buffer]
    norms = [update.delta_norm for update in buffer]

    kept = [
        not (
            agreement < agreement_drop_threshold
            and update.staleness > drop_staleness_threshold
        )
        for update, agreement in zip(buffer, agreements, strict=True)
    ]
    dropped_count = kept.count(False)
    if not any(kept):
        kept = [True for _ in buffer]
        dropped_count = 0

    positive_norms = [norm for norm in norms if norm > 0.0]
    clip_norm = (
        float(np.median(positive_norms)) * max(delta_clip_multiplier, 0.0)
        if positive_norms
        else 0.0
    )

    fairness_values: list[float] = []
    raw_weights: list[float] = []
    fairness_power = max(float(client_fairness_power), 0.0)
    for update, agreement, base_weight, is_kept in zip(
        buffer,
        agreements,
        base_weights,
        kept,
        strict=True,
    ):
        count = client_apply_counts[update.cid] if update.cid < len(client_apply_counts) else 0
        fairness = 1.0 / ((1.0 + max(count, 0)) ** fairness_power)
        update.agreement = agreement
        update.fairness_weight = fairness
        update.dropped_update = not is_kept
        fairness_values.append(fairness)
        if not is_kept:
            raw_weights.append(0.0)
            continue
        agreement_factor = (
            max(float(agreement_epsilon), 0.0) + max(agreement, 0.0)
        ) ** max(agreement_power, 0.0)
        raw_weights.append(base_weight * agreement_factor * fairness)

    raw_total = sum(raw_weights)
    if raw_total <= 0.0:
        raw_weights = [
            base_weight * fairness if is_kept else 0.0
            for base_weight, fairness, is_kept in zip(
                base_weights,
                fairness_values,
                kept,
                strict=True,
            )
        ]
        raw_total = sum(raw_weights)
    if raw_total <= 0.0:
        fallback = _apply_buffered_delta(current_parameters, buffer, server_alpha)
        return fallback, _empty_agreement_stats(server_alpha), _zero_delta_like(current_parameters)

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
        update.staleness for update, is_kept in zip(buffer, kept, strict=True) if is_kept
    ]
    kept_fairness = [
        fairness for fairness, is_kept in zip(fairness_values, kept, strict=True) if is_kept
    ]
    mean_agreement = float(np.mean(kept_agreements)) if kept_agreements else 0.0
    mean_server_agreement = (
        float(np.mean(kept_server_agreements)) if kept_server_agreements else 0.0
    )
    mean_staleness = float(np.mean(kept_staleness)) if kept_staleness else 0.0
    mean_fairness = float(np.mean(kept_fairness)) if kept_fairness else 0.0
    blend = _clamp(history_agreement_blend, 0.0, 1.0)
    agreement_signal = (1.0 - blend) * mean_agreement + blend * mean_server_agreement
    denominator = 1.0 + mean_staleness / max(float(adaptive_staleness_scale), 1e-8)
    buffer_alpha = _clamp(
        server_alpha * (1.0 + adaptive_alpha_boost * agreement_signal) / denominator,
        adaptive_alpha_min,
        adaptive_alpha_max,
    )

    new_parameters: list[np.ndarray] = []
    accepted_delta: list[np.ndarray] = []
    for layer_idx, current_layer in enumerate(current_parameters):
        if not np.issubdtype(current_layer.dtype, np.floating):
            new_parameters.append(current_layer.copy())
            accepted_delta.append(np.zeros_like(current_layer, dtype=np.float64))
            continue

        aggregate_delta = np.zeros_like(current_layer, dtype=np.float64)
        for update, raw_weight, norm in zip(buffer, raw_weights, norms, strict=True):
            if raw_weight <= 0.0:
                continue
            normalized_weight = raw_weight / raw_total
            delta = update.updated_parameters[layer_idx] - update.start_parameters[layer_idx]
            if clip_norm > 0.0 and norm > clip_norm:
                delta = delta * (clip_norm / max(norm, 1e-12))
            aggregate_delta += normalized_weight * delta

        updated_layer = current_layer + buffer_alpha * aggregate_delta
        new_parameters.append(updated_layer.astype(current_layer.dtype, copy=False))
        accepted_delta.append(aggregate_delta)

    stats = {
        "mean_agreement": mean_agreement,
        "buffer_alpha": buffer_alpha,
        "mean_delta_norm": float(np.mean(norms)) if norms else 0.0,
        "dropped_updates": float(dropped_count),
        "applied_updates": float(len(buffer) - dropped_count),
        "server_momentum_agreement": mean_server_agreement,
        "mean_fairness_weight": mean_fairness,
    }
    # Mention the momentum value in a deterministic way so summaries preserve it.
    stats["server_delta_momentum"] = _clamp(server_delta_momentum, 0.0, 0.999)
    return new_parameters, stats, accepted_delta


def _empty_agreement_stats(server_alpha: float) -> dict[str, float]:
    return {
        "mean_agreement": 0.0,
        "buffer_alpha": server_alpha,
        "mean_delta_norm": 0.0,
        "dropped_updates": 0.0,
        "applied_updates": 0.0,
        "server_momentum_agreement": 0.0,
        "mean_fairness_weight": 0.0,
    }


def _reference_delta(
    buffer: list[BufferedUpdate],
    base_weights: list[float],
    base_total: float,
) -> list[np.ndarray]:
    reference: list[np.ndarray] = []
    for layer_idx, layer in enumerate(buffer[0].updated_parameters):
        if not np.issubdtype(layer.dtype, np.floating):
            reference.append(np.zeros_like(layer, dtype=np.float64))
            continue
        aggregate_delta = np.zeros_like(layer, dtype=np.float64)
        for update, base_weight in zip(buffer, base_weights, strict=True):
            normalized_weight = base_weight / base_total
            delta = update.updated_parameters[layer_idx] - update.start_parameters[layer_idx]
            aggregate_delta += normalized_weight * delta
        reference.append(aggregate_delta)
    return reference


def _blend_reference_delta(
    buffer_reference: list[np.ndarray],
    server_delta_ema: list[np.ndarray] | None,
    *,
    blend: float,
) -> list[np.ndarray]:
    if server_delta_ema is None:
        return [layer.copy() for layer in buffer_reference]
    amount = _clamp(blend, 0.0, 1.0)
    blended: list[np.ndarray] = []
    for buffer_layer, server_layer in zip(buffer_reference, server_delta_ema, strict=True):
        if not np.issubdtype(buffer_layer.dtype, np.floating):
            blended.append(buffer_layer.copy())
            continue
        blended.append((1.0 - amount) * buffer_layer + amount * server_layer)
    return blended


def _update_delta_ema(
    previous: list[np.ndarray] | None,
    accepted_delta: list[np.ndarray],
    *,
    momentum: float,
) -> list[np.ndarray]:
    amount = _clamp(momentum, 0.0, 0.999)
    if previous is None:
        return [layer.copy() for layer in accepted_delta]
    return [
        (amount * old_layer + (1.0 - amount) * new_layer).astype(np.float64, copy=False)
        for old_layer, new_layer in zip(previous, accepted_delta, strict=True)
    ]


def _zero_delta_like(parameters: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros_like(layer, dtype=np.float64) for layer in parameters]


def _cosine_to_reference(update: BufferedUpdate, reference_delta: list[np.ndarray]) -> float:
    dot = 0.0
    update_sq = 0.0
    ref_sq = 0.0
    for layer_idx, ref_layer in enumerate(reference_delta):
        if not np.issubdtype(update.updated_parameters[layer_idx].dtype, np.floating):
            continue
        delta = update.updated_parameters[layer_idx] - update.start_parameters[layer_idx]
        delta64 = delta.astype(np.float64, copy=False)
        ref64 = ref_layer.astype(np.float64, copy=False)
        dot += float(np.sum(delta64 * ref64))
        update_sq += float(np.sum(delta64 * delta64))
        ref_sq += float(np.sum(ref64 * ref64))
    if update_sq <= 0.0 or ref_sq <= 0.0:
        return 0.0
    return dot / math.sqrt(update_sq * ref_sq)


def _delta_norm(start_parameters: list[np.ndarray], updated_parameters: list[np.ndarray]) -> float:
    total_sq = 0.0
    for start_layer, updated_layer in zip(start_parameters, updated_parameters, strict=True):
        if not np.issubdtype(updated_layer.dtype, np.floating):
            continue
        delta = updated_layer - start_layer
        delta64 = delta.astype(np.float64, copy=False)
        total_sq += float(np.sum(delta64 * delta64))
    return math.sqrt(total_sq)


def _clamp(value: float, lower: float, upper: float) -> float:
    low = min(float(lower), float(upper))
    high = max(float(lower), float(upper))
    return min(max(float(value), low), high)
