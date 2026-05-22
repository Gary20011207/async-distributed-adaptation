from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable

import numpy as np
import torch

from pathmnist_shared.data import load_datasets, test_loader
from swarm_pathmnist.merge import MergeMethod
from swarm_pathmnist.partition import (
    PartitionMethod,
    format_partition_summary,
    make_node_loaders,
)
from swarm_pathmnist.simulation import Topology, build_nodes, run_swarm_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decentralized PathMNIST swarm experiments.")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--sync-frequency",
        type=int,
        default=0,
        help="Local batches between swarm merges. 0 means full local epoch interval.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "step"],
        default="none",
    )
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--partition",
        choices=["iid", "quantity_skew", "label_skew"],
        default="iid",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--topology",
        choices=["fully_connected", "ring", "random"],
        default="fully_connected",
    )
    parser.add_argument(
        "--merge-method",
        choices=["mean", "weighted_mean", "coord_median"],
        default="weighted_mean",
    )
    parser.add_argument("--peers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_schedule(args: argparse.Namespace) -> Callable[[int], float]:
    if args.lr_scheduler == "none":
        return lambda _step: args.lr

    if args.lr_scheduler == "step":
        return lambda step: args.lr * (args.gamma ** ((step - 1) // args.step_size))

    if args.lr_scheduler == "cosine":
        denominator = max(args.rounds - 1, 1)

        def cosine(step: int) -> float:
            progress = (step - 1) / denominator
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr + (args.lr - args.min_lr) * cosine_decay

        return cosine

    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    sync_frequency = args.sync_frequency if args.sync_frequency > 0 else None

    trainset, testset = load_datasets(
        synthetic=args.synthetic,
        download=not args.no_download,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        augment=args.augment,
    )
    node_loaders, partition_summary = make_node_loaders(
        trainset,
        num_nodes=args.nodes,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    testloader = test_loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    nodes = build_nodes(
        node_loaders,
        device=device,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )

    print(
        "config "
        f"method=swarm_sync nodes={args.nodes} "
        f"train_examples={len(trainset)} test_examples={len(testset)} "
        f"device={device} lr_scheduler={args.lr_scheduler} augment={args.augment} "
        f"partition={args.partition} dirichlet_alpha={args.dirichlet_alpha} "
        f"topology={args.topology} merge={args.merge_method} peers={args.peers}",
        flush=True,
    )
    print(f"partition_summary {format_partition_summary(partition_summary)}", flush=True)

    run_swarm_sync(
        nodes=nodes,
        testloader=testloader,
        device=device,
        rounds=args.rounds,
        lr_schedule=build_lr_schedule(args),
        topology=args.topology,
        merge_method=args.merge_method,
        peers=args.peers,
        sync_frequency=sync_frequency,
        seed=args.seed,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()

