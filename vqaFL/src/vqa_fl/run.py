from __future__ import annotations

import argparse
import json

import torch

from vqa_fl.adapter_math import clone_state
from vqa_fl.data import (
    client_size_summary,
    dump_client_jsonl,
    dump_jsonl,
    format_prompt,
    load_hf_pmc_vqa,
    load_jsonl,
    partition_examples,
    split_examples,
    synthetic_examples,
)
from vqa_fl.fl_runner import FLExperimentConfig, default_methods, run_lora_fl_experiment, write_json_result
from vqa_fl.qwen_lora import DEFAULT_MODEL_NAME, qwen_training_status
from vqa_fl.qwen_lora import (
    QwenLoRAConfig,
    get_adapter_state,
    load_qwen_lora,
    set_adapter_state,
    trainable_parameter_summary,
)
from vqa_fl.simulation import run_adapter_smoke
from vqa_fl.trainer import EvalConfig, LocalTrainConfig, evaluate_choice_accuracy, train_local_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL VQA-FL utilities.")
    parser.add_argument(
        "--mode",
        choices=[
            "adapter-smoke",
            "data-smoke",
            "export-data",
            "fl-run",
            "local-train-smoke",
            "model-smoke",
            "status",
        ],
        default="status",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--data-source", choices=["synthetic", "jsonl", "hf-pmc-vqa"], default="synthetic")
    parser.add_argument("--data-jsonl")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--partition", choices=["iid", "client_id"], default="iid")
    parser.add_argument("--hf-dataset", default="OctoMed/PMC-VQA")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-no-streaming", action="store_true")
    parser.add_argument("--output-jsonl", default="data/pmc_vqa_tiny/examples.jsonl")
    parser.add_argument("--output-dir", default="data/pmc_vqa_tiny")
    parser.add_argument("--per-client-files", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.33)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--method",
        choices=["all", "sync_fedavg", "naive_async", "staleness_async", "fedbuff", "caa_v2"],
        default="sync_fedavg",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--server-alpha", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--staleness-decay",
        choices=["constant", "inverse", "polynomial", "exponential", "hinge"],
        default="hinge",
    )
    parser.add_argument("--results-json")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.mode == "status":
        print(f"model={args.model}")
        print(qwen_training_status())
        return

    if args.mode == "data-smoke":
        print(json.dumps(run_data_smoke(args), indent=2, sort_keys=True))
        return

    if args.mode == "export-data":
        print(json.dumps(run_export_data(args), indent=2, sort_keys=True))
        return

    if args.mode == "model-smoke":
        print(json.dumps(run_model_smoke(args), indent=2, sort_keys=True))
        return

    if args.mode == "local-train-smoke":
        print(json.dumps(run_local_train_smoke(args), indent=2, sort_keys=True))
        return

    if args.mode == "fl-run":
        print(json.dumps(run_fl(args), indent=2, sort_keys=True))
        return

    device = resolve_device(args.device)
    stats = run_adapter_smoke(
        clients=args.clients,
        buffer_size=args.buffer_size,
        seed=args.seed,
        device=device,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "model": args.model,
                "device": str(device),
                "stats": stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_data_smoke(args: argparse.Namespace) -> dict[str, object]:
    examples = load_examples_from_args(args)
    partitions = partition_examples(
        examples,
        clients=args.clients,
        mode=args.partition,
        seed=args.seed,
    )
    sample_prompt = format_prompt(examples[0]) if examples else ""
    return {
        "data_source": args.data_source,
        "partition": args.partition,
        "summary": client_size_summary(partitions),
        "sample_prompt": sample_prompt,
    }


def run_export_data(args: argparse.Namespace) -> dict[str, object]:
    examples = load_examples_from_args(args)
    partitions = partition_examples(
        examples,
        clients=args.clients,
        mode=args.partition,
        seed=args.seed,
    )
    if args.per_client_files:
        dump_client_jsonl(partitions, args.output_dir)
        output = args.output_dir
    else:
        dump_jsonl(
            examples,
            args.output_jsonl,
            image_dir=f"{args.output_dir}/images",
        )
        output = args.output_jsonl
    return {
        "output": output,
        "data_source": args.data_source,
        "partition": args.partition,
        "summary": client_size_summary(partitions),
    }


def load_examples_from_args(args: argparse.Namespace):
    if args.data_source == "synthetic":
        return synthetic_examples(args.max_examples)
    elif args.data_source == "jsonl":
        if not args.data_jsonl:
            raise ValueError("--data-jsonl is required when --data-source jsonl")
        return load_jsonl(args.data_jsonl, max_examples=args.max_examples)
    elif args.data_source == "hf-pmc-vqa":
        return load_hf_pmc_vqa(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            max_examples=args.max_examples,
            seed=args.seed,
            streaming=not args.hf_no_streaming,
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")


def run_model_smoke(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL model smoke test")

    config = QwenLoRAConfig(
        model_name=args.model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        dtype=args.dtype,
        load_in_4bit=args.qlora,
    )
    loaded = load_qwen_lora(config)
    parameter_stats = trainable_parameter_summary(loaded.model)
    adapter_state = get_adapter_state(loaded.model, cpu=True)
    return {
        "model": args.model,
        "dtype": args.dtype,
        "qlora": args.qlora,
        "adapter_tensors": len(adapter_state),
        "trainable_parameters": parameter_stats,
    }


def run_local_train_smoke(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL local training")
    examples = load_examples_from_args(args)
    if len(examples) < 2:
        raise ValueError("local-train-smoke needs at least 2 examples")

    train_examples, eval_examples = split_examples(
        examples,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    if not train_examples:
        train_examples = examples
    if not eval_examples:
        eval_examples = train_examples

    config = QwenLoRAConfig(
        model_name=args.model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        dtype=args.dtype,
        load_in_4bit=args.qlora,
    )
    loaded = load_qwen_lora(config)
    train_config = LocalTrainConfig(
        epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum,
    )
    _adapter_state, train_stats = train_local_lora(
        model=loaded.model,
        processor=loaded.processor,
        examples=train_examples,
        config=train_config,
    )
    result: dict[str, object] = {
        "model": args.model,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train": train_stats,
    }
    if not args.skip_eval:
        result["eval"] = evaluate_choice_accuracy(
            model=loaded.model,
            processor=loaded.processor,
            examples=eval_examples,
            config=EvalConfig(batch_size=args.batch_size),
        )
    return result


def run_fl(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL LoRA FL")
    examples = load_examples_from_args(args)
    if len(examples) < args.clients:
        raise ValueError("FL run needs at least one example per client before train/eval split")

    model_config = QwenLoRAConfig(
        model_name=args.model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        dtype=args.dtype,
        load_in_4bit=args.qlora,
    )
    loaded = load_qwen_lora(model_config)
    initial_state = get_adapter_state(loaded.model, cpu=True)

    methods = default_methods() if args.method == "all" else [args.method]
    results = []
    for method in methods:
        set_adapter_state(loaded.model, clone_state(initial_state))
        config = FLExperimentConfig(
            method=method,
            clients=args.clients,
            rounds=args.rounds,
            buffer_size=args.buffer_size,
            server_alpha=args.server_alpha,
            local_train=LocalTrainConfig(
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=args.grad_accum,
            ),
            eval_config=EvalConfig(batch_size=args.batch_size),
            eval_every=args.eval_every,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            staleness_decay=args.staleness_decay,
            skip_eval=args.skip_eval,
        )
        results.append(
            run_lora_fl_experiment(
                model=loaded.model,
                processor=loaded.processor,
                examples=examples,
                initial_state=clone_state(initial_state),
                config=config,
            )
        )

    payload: dict[str, object] = {
        "model": args.model,
        "dtype": args.dtype,
        "qlora": args.qlora,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "data_source": args.data_source,
        "data_jsonl": args.data_jsonl,
        "max_examples": args.max_examples,
        "results": results,
    }
    if args.results_json:
        write_json_result(payload, args.results_json)
        payload["results_json"] = args.results_json
    return payload


if __name__ == "__main__":
    main()
