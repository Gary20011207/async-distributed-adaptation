from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from fed_pathmnist.client import PathMNISTClient, evaluate_model
from fed_pathmnist.model import create_model, get_parameters, interpolate, set_parameters


Method = Literal["sync_fedavg", "naive_async", "staleness_async"]


@dataclass(order=True)
class AsyncJob:
    finish_time: float
    job_id: int
    cid: int = field(compare=False)
    start_version: int = field(compare=False)
    parameters: list[np.ndarray] = field(compare=False)


def build_clients(
    loaders: list,
    testloader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
) -> list[PathMNISTClient]:
    return [
        PathMNISTClient(
            cid=cid,
            trainloader=client_loader.train,
            testloader=testloader,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
        )
        for cid, client_loader in enumerate(loaders)
    ]


def evaluate_global(
    parameters: list[np.ndarray],
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model = create_model().to(device)
    set_parameters(model, parameters)
    loss, accuracy, _ = evaluate_model(model, testloader, device)
    return loss, accuracy


def run_sync_fedavg(
    clients: list[PathMNISTClient],
    testloader: DataLoader,
    device: torch.device,
    rounds: int,
    lr_schedule: Callable[[int], float],
) -> list[np.ndarray]:
    global_model = create_model().to(device)
    parameters = get_parameters(global_model)

    for rnd in range(1, rounds + 1):
        lr = lr_schedule(rnd)
        results = []
        train_losses = []
        for client in clients:
            updated, num_examples, metrics = client.fit(parameters, {"round": rnd, "lr": lr})
            results.append((updated, num_examples))
            train_losses.append(metrics["train_loss"])

        parameters = _weighted_average(results)
        loss, accuracy = evaluate_global(parameters, testloader, device)
        print(
            f"round={rnd} method=sync_fedavg "
            f"lr={lr:.6f} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"test_loss={loss:.4f} test_acc={accuracy:.4f}",
            flush=True,
        )

    return parameters


def run_async(
    clients: list[PathMNISTClient],
    testloader: DataLoader,
    device: torch.device,
    method: Literal["naive_async", "staleness_async"],
    events: int,
    alpha: float,
    max_delay: float,
    seed: int,
    eval_every: int,
    lr_schedule: Callable[[int], float],
) -> list[np.ndarray]:
    rng = random.Random(seed)
    global_model = create_model().to(device)
    global_parameters = get_parameters(global_model)
    global_version = 0
    next_job_id = 0
    now = 0.0
    jobs: list[AsyncJob] = []

    def submit(cid: int) -> None:
        nonlocal next_job_id
        delay = rng.uniform(1.0, max_delay)
        heapq.heappush(
            jobs,
            AsyncJob(
                finish_time=now + delay,
                job_id=next_job_id,
                cid=cid,
                start_version=global_version,
                parameters=[layer.copy() for layer in global_parameters],
            ),
        )
        next_job_id += 1

    for cid in range(len(clients)):
        submit(cid)

    for event in range(1, events + 1):
        lr = lr_schedule(event)
        job = heapq.heappop(jobs)
        now = job.finish_time
        client = clients[job.cid]
        updated, _num_examples, metrics = client.fit(
            job.parameters,
            {"event": event, "start_version": job.start_version, "lr": lr},
        )

        staleness = max(global_version - job.start_version, 0)
        effective_alpha = alpha
        if method == "staleness_async":
            effective_alpha = alpha / (1.0 + staleness)

        global_parameters = interpolate(global_parameters, updated, effective_alpha)
        global_version += 1

        if event % eval_every == 0 or event == 1 or event == events:
            loss, accuracy = evaluate_global(global_parameters, testloader, device)
            print(
                f"event={event} method={method} cid={job.cid} "
                f"lr={lr:.6f} "
                f"staleness={staleness} alpha={effective_alpha:.4f} "
                f"train_loss={metrics['train_loss']:.4f} "
                f"test_loss={loss:.4f} test_acc={accuracy:.4f}",
                flush=True,
            )

        submit(job.cid)

    return global_parameters


def _weighted_average(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    from fed_pathmnist.model import weighted_average

    return weighted_average(results)
