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
from evaluate_qwenvl_checkpoints import build_generation_inputs, parse_closed_answer


METHOD_LABELS = {
    "sync_fedavg": "Sync",
    "naive_async": "Naive Async",
    "caa_fedbuff_v2": "CAA-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run same-question base-vs-FL Qwen2.5-VL demo inference.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--datasets", nargs="+", default=["vqa_rad_closed", "path_vqa_closed"])
    parser.add_argument("--methods", nargs="+", default=["sync_fedavg", "naive_async", "caa_fedbuff_v2"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", default="presentation/demo_examples/paired_base_vs_fl_examples.csv")
    parser.add_argument("--md-out", default="presentation/demo_examples/paired_base_vs_fl_examples.md")
    parser.add_argument("--plot-out", default="presentation/demo_final_assets/final_paired_examples.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    md_out = Path(args.md_out)
    plot_out = Path(args.plot_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        write_empty_outputs(out, md_out, plot_out, "CUDA is unavailable; paired Qwen2.5-VL inference was not run.")
        print("paired inference skipped: CUDA unavailable")
        return

    device = torch.device(args.device)
    candidates = discover_checkpoints(
        Path(args.result_dir),
        Path(args.checkpoint_dir),
        args.datasets,
        args.methods,
        args.seed,
    )
    if not candidates:
        write_empty_outputs(out, md_out, plot_out, "No matching real Qwen2.5-VL checkpoints were found.")
        print("paired inference skipped: no checkpoints")
        return

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        items = [item for item in candidates if item["dataset"] == dataset]
        if not items:
            continue
        rows.extend(evaluate_dataset(dataset, items, args.samples_per_dataset, args.max_length, device, args.seed))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_prediction_rows(out, rows)
    frame = pd.DataFrame(rows)
    selected = select_demo_examples(frame)
    selected_out = out.with_name(out.stem + "_selected.csv")
    selected.to_csv(selected_out, index=False)
    write_markdown(md_out, frame, selected)
    plot_examples(plot_out, selected, frame)
    print(f"paired inference rows={len(rows)} selected={len(selected)} out={out} selected_out={selected_out}")


def discover_checkpoints(
    result_dir: Path,
    checkpoint_dir: Path,
    datasets: list[str],
    methods: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary.get("config", {})
        dataset = str(config.get("dataset", ""))
        method = str(summary.get("method", config.get("method", "")))
        model = str(config.get("model", ""))
        backend = str(config.get("model_backend", ""))
        row_seed = int(config.get("seed", 0) or 0)
        if dataset not in datasets or method not in methods or row_seed != seed:
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
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": row_seed,
                "run_id": path.stem.replace("_summary", ""),
                "checkpoint_path": str(checkpoint_path),
                "best_test_acc": summary.get("best_test_acc"),
                "final_test_acc": summary.get("final_test_acc"),
            }
        )
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        if key not in selected or row["run_id"] > selected[key]["run_id"]:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def evaluate_dataset(
    dataset: str,
    items: list[dict[str, Any]],
    samples_per_dataset: int,
    max_length: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    metadata = get_dataset_metadata(dataset, synthetic=False, task="medical_vqa")
    _, testset = load_datasets(
        dataset_name=dataset,
        task="medical_vqa",
        model_name="qwen2_5_vl_3b_qlora",
        synthetic=False,
        download=True,
        max_train_samples=8,
        max_test_samples=max(samples_per_dataset, 8),
        seed=seed,
        max_length=max_length,
    )
    model = create_model(num_classes=metadata.num_classes, input_mode=metadata.input_mode, model_name="qwen2_5_vl_3b_qlora")
    model.eval()
    answer_labels = list(getattr(model, "answer_labels", ANSWER_LABELS))
    processor = getattr(model, "closed_answer_processor")
    tokenizer = getattr(processor, "tokenizer", None)
    limit = min(samples_per_dataset, len(testset))

    output_rows: list[dict[str, Any]] = []
    output_rows.extend(evaluate_current_model(model, processor, tokenizer, answer_labels, dataset, "base", "Base", seed, "base_zero_shot", testset, limit, max_length, device, "", None, None))
    for item in sorted(items, key=lambda row: row["method"]):
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
        params = checkpoint.get("adapter_parameters")
        if params is None:
            raise RuntimeError(f"Checkpoint has no adapter_parameters: {item['checkpoint_path']}")
        set_parameters(model, params)
        output_rows.extend(
            evaluate_current_model(
                model,
                processor,
                tokenizer,
                answer_labels,
                dataset,
                item["method"],
                item["method_label"],
                item["seed"],
                item["run_id"],
                testset,
                limit,
                max_length,
                device,
                item["checkpoint_path"],
                item.get("best_test_acc"),
                item.get("final_test_acc"),
            )
        )
    del model
    return output_rows


def evaluate_current_model(
    model: Any,
    processor: Any,
    tokenizer: Any,
    answer_labels: list[str],
    dataset: str,
    method: str,
    method_label: str,
    seed: int,
    run_id: str,
    testset: Any,
    limit: int,
    max_length: int,
    device: torch.device,
    checkpoint_path: str,
    best_test_acc: Any,
    final_test_acc: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": method_label,
                "seed": seed,
                "run_id": run_id,
                "sample_index": sample_idx,
                "question": example.get("question", ""),
                "choices": " | ".join(str(choice) for choice in example.get("choices", [])[:4]),
                "prediction_text": text.strip(),
                "parsed_answer": answer_labels[pred] if valid and pred < len(answer_labels) else "INVALID",
                "gold_answer": answer_labels[gold] if 0 <= gold < len(answer_labels) else str(gold),
                "valid": bool(valid),
                "correct": bool(pred == gold),
                "best_test_acc": best_test_acc,
                "final_test_acc": final_test_acc,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def select_demo_examples(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index=["dataset", "sample_index", "question", "choices", "gold_answer"],
        columns="method_label",
        values=["parsed_answer", "correct", "prediction_text"],
        aggfunc="first",
    )
    pivot.columns = [f"{outer}_{inner}".replace(" ", "_") for outer, inner in pivot.columns]
    pivot = pivot.reset_index()
    for col in [
        "correct_Base",
        "correct_CAA-v2",
        "correct_Naive_Async",
        "correct_Sync",
    ]:
        if col not in pivot:
            pivot[col] = False
    rules = [
        ("base_wrong_caa_correct", (pivot["correct_Base"] == False) & (pivot["correct_CAA-v2"] == True)),
        ("naive_wrong_caa_correct", (pivot["correct_Naive_Async"] == False) & (pivot["correct_CAA-v2"] == True)),
        (
            "caa_correct_sync_or_naive_wrong",
            (pivot["correct_CAA-v2"] == True)
            & ((pivot["correct_Naive_Async"] == False) | (pivot["correct_Sync"] == False)),
        ),
        ("representative_caa_correct", pivot["correct_CAA-v2"] == True),
    ]
    selected_parts = []
    used: set[tuple[str, int]] = set()
    for label, mask in rules:
        candidates = pivot[mask].copy()
        for _, row in candidates.iterrows():
            key = (str(row["dataset"]), int(row["sample_index"]))
            if key in used:
                continue
            row_dict = row.to_dict()
            row_dict["selection_reason"] = label
            selected_parts.append(row_dict)
            used.add(key)
            if len(selected_parts) >= 6:
                return pd.DataFrame(selected_parts)
    return pd.DataFrame(selected_parts)


def write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "method_label",
        "seed",
        "run_id",
        "sample_index",
        "question",
        "choices",
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


def write_markdown(path: Path, frame: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Paired Base vs FL Demo Examples",
        "",
        "These rows use the same dataset, sample index, question, image, choices, and gold answer for base Qwen2.5-VL and FL checkpoints.",
        "",
    ]
    if frame.empty:
        lines.append("No paired inference rows were generated.")
    else:
        summary = frame.groupby("method_label", as_index=False).agg(
            accuracy=("correct", "mean"),
            valid_rate=("valid", "mean"),
            samples=("correct", "count"),
        )
        lines.extend(["## Summary", "", markdown_table(summary), ""])
    if selected.empty:
        lines.extend(
            [
                "## Selected Examples",
                "",
                "No valid same-question improvement examples were found. Do not claim base-wrong / FL-correct improvement.",
            ]
        )
    else:
        cols = [
            "selection_reason",
            "dataset",
            "sample_index",
            "question",
            "gold_answer",
            "parsed_answer_Base",
            "parsed_answer_CAA-v2",
            "parsed_answer_Naive_Async",
            "parsed_answer_Sync",
        ]
        present = [col for col in cols if col in selected.columns]
        lines.extend(["## Selected Examples", "", markdown_table(selected[present]), ""])
        if not (selected["selection_reason"] == "base_wrong_caa_correct").any():
            lines.append("No `base wrong / CAA-v2 correct` example was found; use the selected rows as method-comparison examples only.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            values.append(text.replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_examples(path: Path, selected: pd.DataFrame, frame: pd.DataFrame) -> None:
    if selected.empty:
        write_empty_plot(path, "No paired examples available")
        return
    rows = []
    for _, row in selected.head(4).iterrows():
        reason = str(row.get("selection_reason", ""))
        reason_label = {
            "base_wrong_caa_correct": "Base wrong\nCAA correct",
            "naive_wrong_caa_correct": "Naive wrong\nCAA correct",
            "caa_correct_sync_or_naive_wrong": "Sync/Naive wrong\nCAA correct",
            "representative_caa_correct": "CAA correct",
        }.get(reason, reason.replace("_", " "))
        rows.append(
            [
                reason_label,
                shorten(row.get("question", ""), 54),
                str(row.get("gold_answer", "")),
                str(row.get("parsed_answer_Base", "")),
                str(row.get("parsed_answer_Naive_Async", "")),
                str(row.get("parsed_answer_CAA-v2", "")),
            ]
        )
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Why selected", "Question", "Gold", "Base", "Naive", "CAA-v2"],
        loc="center",
        cellLoc="left",
        colWidths=[0.18, 0.42, 0.08, 0.08, 0.08, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.62)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold")
        elif c == 5:
            cell.set_facecolor("#DCFCE7")
    title = "Same-question base vs FL checkpoint examples"
    if not (selected["selection_reason"] == "base_wrong_caa_correct").any():
        title += " (no base-wrong/CAA-correct row found)"
    ax.set_title(title, pad=14, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def shorten(value: Any, width: int) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def write_empty_outputs(csv_path: Path, md_path: Path, plot_path: Path, message: str) -> None:
    write_prediction_rows(csv_path, [])
    md_path.write_text(f"# Paired Base vs FL Demo Examples\n\n{message}\n", encoding="utf-8")
    write_empty_plot(plot_path, message)


def write_empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
