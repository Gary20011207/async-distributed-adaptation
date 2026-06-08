from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from fed_mllm.data import ANSWER_LABELS, get_dataset_metadata, load_datasets
from fed_mllm.model import create_model, set_parameters

METHOD_LABELS = {
    "sync_fedavg": "Sync",
    "naive_async": "Naive Async",
    "staleness_async": "Staleness",
    "fedbuff_async": "FedBuff",
    "caa_fedbuff_v2": "CAA-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real Qwen2.5-VL checkpoint closed-answer samples.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default="figures/report/qwenvl_method_prediction_samples.csv")
    parser.add_argument("--plot-out", default="figures/report/qwenvl_method_validity.png")
    parser.add_argument("--datasets", nargs="+", default=["vqa_rad_closed", "path_vqa_closed"])
    parser.add_argument("--methods", nargs="+", default=["sync_fedavg", "naive_async", "staleness_async", "fedbuff_async", "caa_fedbuff_v2"])
    parser.add_argument("--samples-per-run", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--only-latest-per-key", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_out = Path(args.plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    candidates = discover_checkpoints(Path(args.result_dir), Path(args.checkpoint_dir), args.datasets, args.methods)
    if args.only_latest_per_key:
        candidates = latest_per_key(candidates)
    rows: list[dict[str, Any]] = []
    if not candidates:
        write_rows(out, rows)
        write_empty_plot(plot_out, "No real Qwen2.5-VL checkpoints found")
        print(f"qwenvl checkpoint diagnostics rows=0 out={out}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Qwen2.5-VL diagnostics, but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        by_dataset.setdefault(str(item["dataset"]), []).append(item)
    for dataset, items in sorted(by_dataset.items()):
        rows.extend(evaluate_dataset_checkpoints(dataset, items, args.samples_per_run, args.max_length, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(out, rows)
    plot_summary(out, plot_out)
    print(f"qwenvl checkpoint diagnostics rows={len(rows)} out={out}")


def discover_checkpoints(result_dir: Path, checkpoint_dir: Path, datasets: list[str], methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary.get("config", {})
        dataset = str(config.get("dataset", ""))
        method = str(summary.get("method", config.get("method", "")))
        model = str(config.get("model", ""))
        backend = str(config.get("model_backend", ""))
        if dataset not in datasets or method not in methods:
            continue
        if model != "qwen2_5_vl_3b_qlora" or backend != "qwen2_5_vl_3b_4bit_lora_generative":
            continue
        checkpoint = str(summary.get("checkpoint_path", "") or "")
        if not checkpoint:
            continue
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = path.parents[1] / checkpoint_path
        if not checkpoint_path.exists():
            fallback = checkpoint_dir / Path(checkpoint).name
            if fallback.exists():
                checkpoint_path = fallback
            else:
                continue
        rows.append(
            {
                "summary_path": str(path),
                "checkpoint_path": str(checkpoint_path),
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": int(config.get("seed", 0) or 0),
                "run_id": path.stem.replace("_summary", ""),
                "best_test_acc": summary.get("best_test_acc"),
                "final_test_acc": summary.get("final_test_acc"),
            }
        )
    return rows


def latest_per_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["seed"])
        if key not in selected or row["run_id"] > selected[key]["run_id"]:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def evaluate_dataset_checkpoints(
    dataset: str,
    items: list[dict[str, Any]],
    samples_per_run: int,
    max_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    metadata = get_dataset_metadata(dataset, synthetic=False, task="medical_vqa")
    _, testset = load_datasets(
        dataset_name=dataset,
        task="medical_vqa",
        model_name="qwen2_5_vl_3b_qlora",
        synthetic=False,
        download=True,
        max_train_samples=8,
        max_test_samples=max(samples_per_run, 8),
        seed=min(int(item.get("seed", 0) or 0) for item in items),
        max_length=max_length,
    )
    model = create_model(num_classes=metadata.num_classes, input_mode=metadata.input_mode, model_name="qwen2_5_vl_3b_qlora")
    model.eval()
    answer_labels = list(getattr(model, "answer_labels", ANSWER_LABELS))
    processor = getattr(model, "closed_answer_processor")
    tokenizer = getattr(processor, "tokenizer", None)

    output_rows: list[dict[str, Any]] = []
    limit = min(samples_per_run, len(testset))
    for item in sorted(items, key=lambda row: (row["method"], row["seed"], row["run_id"])):
        # These checkpoints are generated locally by this project and contain
        # NumPy adapter arrays, so PyTorch 2.6's default weights_only=True is too strict.
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
        params = checkpoint.get("adapter_parameters")
        if params is None:
            raise RuntimeError(f"Checkpoint has no adapter_parameters: {item['checkpoint_path']}")
        set_parameters(model, params)
        for sample_idx in range(limit):
            example = testset[sample_idx]
            inputs = build_generation_inputs(processor, example, max_length)
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            prompt_len = int(inputs["input_ids"].shape[1])
            suffix = generated[:, prompt_len:]
            text = tokenizer.batch_decode(suffix, skip_special_tokens=True)[0] if tokenizer is not None else ""
            pred = parse_closed_answer(text, answer_labels)
            gold = int(example["answer_label"])
            valid = pred >= 0
            output_rows.append(
                {
                    "dataset": item["dataset"],
                    "method": item["method"],
                    "method_label": item["method_label"],
                    "seed": item["seed"],
                    "run_id": item["run_id"],
                    "sample_index": sample_idx,
                    "question": example.get("question", ""),
                    "prediction_text": text.strip(),
                    "parsed_answer": answer_labels[pred] if valid and pred < len(answer_labels) else "INVALID",
                    "gold_answer": answer_labels[gold] if 0 <= gold < len(answer_labels) else str(gold),
                    "valid": valid,
                    "correct": bool(pred == gold),
                    "best_test_acc": item.get("best_test_acc"),
                    "final_test_acc": item.get("final_test_acc"),
                    "checkpoint_path": item["checkpoint_path"],
                }
            )
    del model
    return output_rows


def build_generation_inputs(processor: Any, example: dict[str, Any], max_length: int) -> dict[str, torch.Tensor]:
    choices = list(example["choices"]) + [""] * max(0, 4 - len(example["choices"]))
    question = (
        f"{example['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer with exactly one option letter: A, B, C, or D."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info([messages])
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    except Exception:
        return processor(
            text=[text],
            images=[example["image"]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )


def parse_closed_answer(text: str, answer_labels: list[str]) -> int:
    normalized = text.strip().upper()
    for idx, label in enumerate(answer_labels):
        lab = str(label).upper()
        if normalized == lab or normalized.startswith(lab) or f" {lab}" in f" {normalized}":
            return idx
    return -1


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "method_label",
        "seed",
        "run_id",
        "sample_index",
        "question",
        "prediction_text",
        "parsed_answer",
        "gold_answer",
        "valid",
        "correct",
        "best_test_acc",
        "final_test_acc",
        "checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(csv_path: Path, plot_out: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        write_empty_plot(plot_out, "No prediction rows")
        return
    frame = pd.read_csv(csv_path)
    if frame.empty:
        write_empty_plot(plot_out, "No prediction rows")
        return
    summary = frame.groupby(["dataset", "method_label"], as_index=False).agg(
        valid_rate=("valid", "mean"),
        accuracy=("correct", "mean"),
        samples=("correct", "count"),
    )
    summary.to_csv(plot_out.with_suffix(".csv"), index=False)
    pivot = summary.pivot_table(index="dataset", columns="method_label", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.78)
    ax.set_ylabel("Sample accuracy")
    ax.set_xlabel("")
    ax.set_title("Real Qwen2.5-VL checkpoint closed-answer diagnostics")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(plot_out, dpi=200)
    plt.close()


def write_empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
