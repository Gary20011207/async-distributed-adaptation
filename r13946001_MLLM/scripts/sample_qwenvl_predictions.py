from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

ANSWER_LABELS = ["A", "B", "C", "D"]
YES_NO_LABELS = ["yes", "no"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample real Qwen2.5-VL closed-ended VQA predictions.")
    parser.add_argument("--datasets", nargs="+", default=["vqa_rad_closed", "path_vqa_closed"])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--out", default="figures/report/qwenvl_closed_answer_diagnostics.csv")
    parser.add_argument("--cache-dir", default="data/hf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
        trust_remote_code=True,
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        ds = load_vqa_dataset(dataset, args.cache_dir)
        count = min(args.max_samples, len(ds))
        for idx in range(count):
            item = ds[idx]
            question = str(item.get("question") or item.get("Question") or "")
            choices = extract_choices(item)
            gold = extract_label(item)
            image = extract_image(item)
            messages = build_messages(question, choices, image)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            suffix = generated[:, inputs["input_ids"].shape[1]:]
            pred_text = processor.tokenizer.batch_decode(suffix, skip_special_tokens=True)[0].strip()
            pred = parse_closed_answer(pred_text)
            rows.append(
                {
                    "dataset": dataset,
                    "sample_index": idx,
                    "method": "base_qwen2_5_vl_zero_shot",
                    "prediction_text": pred_text,
                    "parsed_answer": "" if pred < 0 else ANSWER_LABELS[pred],
                    "gold_answer": ANSWER_LABELS[gold],
                    "correct": int(pred == gold),
                    "valid": int(pred >= 0),
                    "question": question[:300],
                }
            )
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


def load_vqa_dataset(dataset: str, cache_dir: str):
    if dataset == "vqa_rad_closed":
        return load_dataset("abhay2812/vqa-rad", split="test", cache_dir=cache_dir)
    if dataset == "path_vqa_closed":
        return load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=cache_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_choices(item: dict[str, Any]) -> list[str]:
    if "choices" in item and isinstance(item["choices"], list):
        choices = [str(choice) for choice in item["choices"][:4]]
        return choices + [""] * max(0, 4 - len(choices))
    answer = str(item.get("answer") or item.get("Answer") or "").strip()
    if answer.lower() in YES_NO_LABELS:
        return ["yes", "no", "", ""]
    return [answer, "not shown", "uncertain", "other"]


def extract_label(item: dict[str, Any]) -> int:
    value = item.get("answer") or item.get("Answer") or item.get("label")
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "yes":
            return 0
        if lower == "no":
            return 1
        upper = value.strip().upper()
        if upper in ANSWER_LABELS:
            return ANSWER_LABELS.index(upper)
    if isinstance(value, int):
        return int(value)
    return 0


def extract_image(item: dict[str, Any]):
    from PIL import Image
    image = item.get("image") or item.get("Image")
    if image is None:
        return Image.new("RGB", (64, 64), color=(0, 0, 0))
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def build_messages(question: str, choices: list[str], image: Any) -> list[dict[str, Any]]:
    padded = choices + [""] * max(0, 4 - len(choices))
    prompt = (
        f"{question}\n"
        f"A. {padded[0]}\n"
        f"B. {padded[1]}\n"
        f"C. {padded[2]}\n"
        f"D. {padded[3]}\n"
        "Answer with exactly one option letter: A, B, C, or D."
    )
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def parse_closed_answer(text: str) -> int:
    normalized = text.strip().upper()
    for idx, label in enumerate(ANSWER_LABELS):
        if normalized == label or normalized.startswith(label) or f" {label}" in f" {normalized}":
            return idx
    return -1


if __name__ == "__main__":
    main()
