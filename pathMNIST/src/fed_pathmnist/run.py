from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable

import numpy as np
import torch

from fed_pathmnist.data import iid_client_loaders, load_datasets, test_loader
from fed_pathmnist.simulation import StalenessDecay, build_clients, run_async, run_sync_fedavg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated PathMNIST experiments.")
    parser.add_argument(
        "--method",
        choices=["sync_fedavg", "naive_async", "staleness_async"],
        default="sync_fedavg",
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--events", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--staleness-decay",
        choices=["inverse", "tau_inverse", "floor_tau_inverse", "exp", "hinge"],
        default="inverse",
    )
    parser.add_argument("--staleness-tau", type=float, default=5.0)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--max-delay", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
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


def build_lr_schedule(args: argparse.Namespace, total_steps: int) -> Callable[[int], float]:
    if args.lr_scheduler == "none":
        return lambda _step: args.lr

    if args.lr_scheduler == "step":
        return lambda step: args.lr * (args.gamma ** ((step - 1) // args.step_size))

    if args.lr_scheduler == "cosine":
        denominator = max(total_steps - 1, 1)

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

    trainset, testset = load_datasets(
        synthetic=args.synthetic,
        download=not args.no_download,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        augment=args.augment,
    )
    client_loaders = iid_client_loaders(
        trainset,
        num_clients=args.clients,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    testloader = test_loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    clients = build_clients(
        client_loaders,
        testloader,
        device=device,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )
    total_steps = args.rounds if args.method == "sync_fedavg" else args.events
    lr_schedule = build_lr_schedule(args, total_steps)

    print(
        "config "
        f"method={args.method} clients={args.clients} "
        f"train_examples={len(trainset)} test_examples={len(testset)} "
        f"device={device} lr_scheduler={args.lr_scheduler} augment={args.augment} "
        f"staleness_decay={args.staleness_decay} staleness_tau={args.staleness_tau} "
        f"min_alpha={args.min_alpha}",
        flush=True,
    )

    if args.method == "sync_fedavg":
        run_sync_fedavg(
            clients=clients,
            testloader=testloader,
            device=device,
            rounds=args.rounds,
            lr_schedule=lr_schedule,
        )
    else:
        run_async(
            clients=clients,
            testloader=testloader,
            device=device,
            method=args.method,
            events=args.events,
            alpha=args.alpha,
            max_delay=args.max_delay,
            seed=args.seed,
            eval_every=args.eval_every,
            lr_schedule=lr_schedule,
            staleness_decay=args.staleness_decay,
            staleness_tau=args.staleness_tau,
            min_alpha=args.min_alpha,
        )


if __name__ == "__main__":
    main()
