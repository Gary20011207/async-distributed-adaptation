from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pathmnist_shared.data import ClientLoaders
from pathmnist_shared.model import create_model, set_parameters
from swarm_pathmnist.merge import MergeMethod, merge_parameters
from swarm_pathmnist.node import SwarmNode

Topology = Literal["fully_connected", "ring", "random"]


def build_nodes(
    loaders: list[ClientLoaders],
    *,
    device: torch.device,
    local_epochs: int,
    lr: float,
) -> list[SwarmNode]:
    return [
        SwarmNode(
            nid=nid,
            trainloader=loader.train,
            num_examples=loader.num_examples,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
        )
        for nid, loader in enumerate(loaders)
    ]


def run_swarm_sync(
    *,
    nodes: list[SwarmNode],
    testloader: DataLoader,
    device: torch.device,
    rounds: int,
    lr_schedule: Callable[[int], float],
    topology: Topology,
    merge_method: MergeMethod,
    peers: int,
    sync_frequency: int | None,
    seed: int,
    eval_every: int,
) -> list[np.ndarray]:
    if not nodes:
        raise ValueError("At least one swarm node is required")

    initial_parameters = nodes[0].get_parameters()
    for node in nodes[1:]:
        node.set_parameters([layer.copy() for layer in initial_parameters])

    rng = random.Random(seed)
    weights = [node.num_examples for node in nodes]

    for rnd in range(1, rounds + 1):
        lr = lr_schedule(rnd)
        train_metrics = [
            node.train_interval(lr=lr, sync_frequency=sync_frequency) for node in nodes
        ]

        pre_merge_parameters = [node.get_parameters() for node in nodes]
        peer_groups = _peer_groups(
            num_nodes=len(nodes),
            topology=topology,
            peers=peers,
            rng=rng,
        )

        merged_by_node = []
        for group in peer_groups:
            merged = merge_parameters(
                [pre_merge_parameters[node_id] for node_id in group],
                [weights[node_id] for node_id in group],
                merge_method,
            )
            merged_by_node.append(merged)

        for node, merged in zip(nodes, merged_by_node, strict=True):
            node.set_parameters(merged)

        if rnd % eval_every == 0 or rnd == 1 or rnd == rounds:
            monitor_parameters = merge_parameters(
                [node.get_parameters() for node in nodes],
                weights,
                "weighted_mean",
            )
            loss, accuracy = evaluate_parameters(monitor_parameters, testloader, device)
            print(
                f"round={rnd} method=swarm_sync "
                f"topology={topology} merge={merge_method} lr={lr:.6f} "
                f"sync_frequency={sync_frequency or 'full'} "
                f"peer_group_size={np.mean([len(group) for group in peer_groups]):.2f} "
                f"train_loss={np.mean([metric.loss for metric in train_metrics]):.4f} "
                f"test_loss={loss:.4f} test_acc={accuracy:.4f}",
                flush=True,
            )

    return merge_parameters([node.get_parameters() for node in nodes], weights, "weighted_mean")


@torch.no_grad()
def evaluate_parameters(
    parameters: list[np.ndarray],
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model = create_model().to(device)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def _peer_groups(
    *,
    num_nodes: int,
    topology: Topology,
    peers: int,
    rng: random.Random,
) -> list[list[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    if topology == "fully_connected":
        return [list(range(num_nodes)) for _ in range(num_nodes)]

    if topology == "ring":
        return [
            sorted({node_id, (node_id - 1) % num_nodes, (node_id + 1) % num_nodes})
            for node_id in range(num_nodes)
        ]

    if topology == "random":
        if peers < 0:
            raise ValueError("peers must be non-negative")
        groups = []
        for node_id in range(num_nodes):
            candidates = [idx for idx in range(num_nodes) if idx != node_id]
            sample_size = min(peers, len(candidates))
            selected = rng.sample(candidates, sample_size) if sample_size else []
            groups.append(sorted({node_id, *selected}))
        return groups

    raise ValueError(f"Unsupported topology: {topology}")
