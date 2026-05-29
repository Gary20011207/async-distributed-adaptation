from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable

import numpy as np
import torch

from fed_pathmnist.data import (
    get_dataset_metadata,
    load_datasets,
    partition_client_loaders,
    test_loader,
)
from fed_pathmnist.experiment_logging import ExperimentLogger
from fed_pathmnist.simulation_ext import build_clients, run_async, run_sync_fedavg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated PathMNIST experiments.")
    parser.add_argument(
        "--method",
        choices=[
            "sync_fedavg",
            "naive_async",
            "staleness_async",
            "fedbuff_async",
            "agreement_fedbuff_async",
            "caa_fedbuff_v2",
        ],
        default="sync_fedavg",
    )
    parser.add_argument(
        "--dataset",
        default="mmlu",
    )
    parser.add_argument(
        "--model",
        choices=["qwen"],
        default="qwen",
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--events", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
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
        choices=["constant", "inverse", "polynomial", "exponential", "hinge"],
        default="inverse",
        help="Decay rule for stale async updates. naive_async always uses constant.",
    )
    parser.add_argument("--staleness-power", type=float, default=1.0)
    parser.add_argument("--staleness-exp-rate", type=float, default=0.1)
    parser.add_argument("--staleness-hinge-b", type=int, default=5)
    parser.add_argument("--staleness-hinge-a", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=5)
    parser.add_argument("--agreement-epsilon", type=float, default=0.10)
    parser.add_argument("--agreement-power", type=float, default=1.0)
    parser.add_argument("--agreement-drop-threshold", type=float, default=-0.05)
    parser.add_argument("--delta-clip-multiplier", type=float, default=1.5)
    parser.add_argument("--adaptive-alpha-min", type=float, default=0.15)
    parser.add_argument("--adaptive-alpha-max", type=float, default=0.65)
    parser.add_argument("--adaptive-alpha-boost", type=float, default=0.35)
    parser.add_argument("--adaptive-staleness-scale", type=float, default=8.0)
    parser.add_argument("--server-delta-momentum", type=float, default=0.8)
    parser.add_argument("--history-agreement-blend", type=float, default=0.25)
    parser.add_argument("--client-fairness-power", type=float, default=0.5)
    parser.add_argument(
        "--delay-mode",
        choices=["uniform", "lognormal", "heterogeneous"],
        default="uniform",
    )
    parser.add_argument("--min-delay", type=float, default=1.0)
    parser.add_argument("--max-delay", type=float, default=5.0)
    parser.add_argument("--lognormal-mean", type=float, default=1.0)
    parser.add_argument("--lognormal-sigma", type=float, default=0.5)
    parser.add_argument("--straggler-ratio", type=float, default=0.2)
    parser.add_argument("--straggler-multiplier", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-best", action="store_true")
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
    metadata = get_dataset_metadata(args.dataset, synthetic=args.synthetic)
    dataset_name = metadata.name

    trainset, testset = load_datasets(
        dataset_name=args.dataset,
        synthetic=args.synthetic,
        download=not args.no_download,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        augment=args.augment,
    )
    client_loaders = partition_client_loaders(
        trainset,
        num_clients=args.clients,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    testloader = test_loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    clients = build_clients(
        client_loaders,
        testloader,
        device=device,
        local_epochs=args.local_epochs,
        lr=args.lr,
        num_classes=metadata.num_classes,
        in_channels=metadata.in_channels,
        model_name=args.model,
    )
    staleness_decay = args.staleness_decay
    staleness_hinge_a = args.staleness_hinge_a
    if args.method in {"agreement_fedbuff_async", "caa_fedbuff_v2"} and args.staleness_decay == "inverse":
        # CAA-FedBuff defaults to a mild hinge rule. Passing any explicit
        # non-inverse decay keeps that user-selected setting.
        staleness_decay = "hinge"
        staleness_hinge_a = 0.05
    total_steps = args.rounds if args.method == "sync_fedavg" else args.events
    lr_schedule = build_lr_schedule(args, total_steps)

    print(
        "config "
        f"dataset={dataset_name} task={metadata.task} "
        f"classes={metadata.num_classes} channels={metadata.in_channels} "
        f"method={args.method} clients={args.clients} "
        f"model={args.model} "
        f"partition={args.partition} dirichlet_alpha={args.dirichlet_alpha} "
        f"train_examples={len(trainset)} test_examples={len(testset)} "
        f"device={device} lr_scheduler={args.lr_scheduler} augment={args.augment} "
        f"staleness_decay={staleness_decay}",
        flush=True,
    )
    client_sizes = [loader.num_examples for loader in client_loaders]
    print(
        "client_examples "
        f"min={min(client_sizes)} max={max(client_sizes)} "
        f"mean={np.mean(client_sizes):.1f}",
        flush=True,
    )

    logger = ExperimentLogger(args.method, result_dir=args.result_dir, dataset_name=dataset_name)
    args_config = vars(args).copy()
    args_config["dataset"] = dataset_name
    args_config["task"] = metadata.task
    args_config["num_classes"] = metadata.num_classes
    args_config["in_channels"] = metadata.in_channels
    args_config["update_budget"] = (
        args.rounds * args.clients if args.method == "sync_fedavg" else args.events
    )
    args_config["sync_equivalent_rounds"] = (
        args.rounds if args.method == "sync_fedavg" else args.events / max(args.clients, 1)
    )
    args_config["staleness_decay"] = staleness_decay
    args_config["staleness_hinge_a"] = staleness_hinge_a
    try:
        if args.method == "sync_fedavg":
            run_sync_fedavg(
                clients=clients,
                testloader=testloader,
                device=device,
                rounds=args.rounds,
                lr_schedule=lr_schedule,
                logger=logger,
                save_best=args.save_best,
                checkpoint_dir=args.checkpoint_dir,
                args_config=args_config,
                dataset_name=dataset_name,
                num_classes=metadata.num_classes,
                in_channels=metadata.in_channels,
                model_name=args.model,
            )
        else:
            run_async(
                clients=clients,
                testloader=testloader,
                device=device,
                method=args.method,
                events=args.events,
                alpha=args.alpha,
                seed=args.seed,
                eval_every=args.eval_every,
                lr_schedule=lr_schedule,
                staleness_decay=staleness_decay,
                staleness_power=args.staleness_power,
                staleness_exp_rate=args.staleness_exp_rate,
                staleness_hinge_b=args.staleness_hinge_b,
                staleness_hinge_a=staleness_hinge_a,
                delay_mode=args.delay_mode,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                lognormal_mean=args.lognormal_mean,
                lognormal_sigma=args.lognormal_sigma,
                straggler_ratio=args.straggler_ratio,
                straggler_multiplier=args.straggler_multiplier,
                buffer_size=args.buffer_size,
                agreement_epsilon=args.agreement_epsilon,
                agreement_power=args.agreement_power,
                agreement_drop_threshold=args.agreement_drop_threshold,
                delta_clip_multiplier=args.delta_clip_multiplier,
                adaptive_alpha_min=args.adaptive_alpha_min,
                adaptive_alpha_max=args.adaptive_alpha_max,
                adaptive_alpha_boost=args.adaptive_alpha_boost,
                adaptive_staleness_scale=args.adaptive_staleness_scale,
                server_delta_momentum=args.server_delta_momentum,
                history_agreement_blend=args.history_agreement_blend,
                client_fairness_power=args.client_fairness_power,
                logger=logger,
                save_best=args.save_best,
                checkpoint_dir=args.checkpoint_dir,
                args_config=args_config,
                dataset_name=dataset_name,
                num_classes=metadata.num_classes,
                in_channels=metadata.in_channels,
                model_name=args.model,
            )
    finally:
        logger.close()
        print(f"log_csv={logger.csv_path}", flush=True)
        print(f"summary_json={logger.summary_path}", flush=True)


if __name__ == "__main__":
    main()
