from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast


@dataclass(frozen=True)
class VQAChoiceExample:
    image: Any
    question: str
    choices: tuple[str, str, str, str]
    answer: str
    client_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


OPTION_LETTERS = ("A", "B", "C", "D")
PartitionMode = Literal["iid", "client_id"]


def load_jsonl(path: str | Path, *, max_examples: int | None = None) -> list[VQAChoiceExample]:
    jsonl_path = Path(path)
    examples: list[VQAChoiceExample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            example = example_from_mapping(payload)
            if isinstance(example.image, str) and example.image and not Path(example.image).is_absolute():
                example = replace(example, image=str(jsonl_path.parent / example.image))
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def load_hf_pmc_vqa(
    *,
    dataset_name: str = "OctoMed/PMC-VQA",
    split: str = "train",
    max_examples: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    streaming: bool = True,
    cache_dir: str | Path | None = None,
) -> list[VQAChoiceExample]:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for Hugging Face PMC-VQA loading. "
            "Install vqaFL dependencies first."
        ) from exc

    dataset_kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if cache_dir is not None:
        dataset_kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(dataset_name, **dataset_kwargs)

    if shuffle:
        if streaming:
            buffer_size = min(max(max_examples or 1000, 100), 10000)
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            dataset = dataset.shuffle(seed=seed)

    examples: list[VQAChoiceExample] = []
    try:
        for row in dataset:
            examples.append(example_from_mapping(row))
            if max_examples is not None and len(examples) >= max_examples:
                break
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return examples


def example_from_mapping(payload: Mapping[str, Any]) -> VQAChoiceExample:
    choices = _coerce_choices(payload)
    answer = _normalize_answer(payload.get("answer"), choices)
    return VQAChoiceExample(
        image=payload.get("image") or payload.get("image_path") or payload.get("filename"),
        question=str(payload["question"]),
        choices=choices,
        answer=answer,
        client_id=_coerce_client_id(payload.get("client_id")),
        metadata={key: value for key, value in payload.items() if key not in _KNOWN_FIELDS},
    )


def dump_jsonl(
    examples: Sequence[VQAChoiceExample],
    path: str | Path,
    *,
    image_dir: str | Path | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir_path = Path(image_dir) if image_dir is not None else None
    if image_dir_path is not None:
        image_dir_path.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(examples):
            image = _serialize_image(example.image, image_dir_path, idx, output_path.parent)
            payload = {
                "image": image,
                "question": example.question,
                "choices": list(example.choices),
                "answer": example.answer,
                "client_id": example.client_id,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_client_jsonl(
    partitions: Sequence[Sequence[VQAChoiceExample]],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for cid, examples in enumerate(partitions):
        dump_jsonl(
            examples,
            output_path / f"client_{cid:02d}.jsonl",
            image_dir=output_path / "images" / f"client_{cid:02d}",
        )


def _serialize_image(
    image: Any,
    image_dir: Path | None,
    idx: int,
    jsonl_dir: Path,
) -> str:
    if image is None:
        return ""
    if isinstance(image, (str, Path)):
        return str(image)
    if image_dir is None:
        return str(image)
    if isinstance(image, Mapping):
        path = image.get("path") or image.get("filename")
        if path:
            return str(path)
    save = getattr(image, "save", None)
    if callable(save):
        image_path = image_dir / f"{idx:06d}.png"
        save(image_path)
        return os.path.relpath(image_path, jsonl_dir)
    return str(image)


def format_prompt(example: VQAChoiceExample) -> str:
    choice_lines = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(OPTION_LETTERS, example.choices, strict=True)
    )
    return (
        f"Question: {example.question}\n"
        f"{choice_lines}\n"
        "Answer with one letter only: A, B, C, or D."
    )


def partition_examples(
    examples: Sequence[VQAChoiceExample],
    *,
    clients: int,
    mode: PartitionMode = "iid",
    seed: int = 42,
) -> list[list[VQAChoiceExample]]:
    if clients <= 0:
        raise ValueError("clients must be positive")
    if mode == "client_id" and all(example.client_id is not None for example in examples):
        partitions: list[list[VQAChoiceExample]] = [[] for _ in range(clients)]
        overflow: list[VQAChoiceExample] = []
        for example in examples:
            assert example.client_id is not None
            if 0 <= example.client_id < clients:
                partitions[example.client_id].append(example)
            else:
                overflow.append(example)
        if overflow:
            iid_overflow = _iid_partitions(overflow, clients, seed)
            for cid, subset in enumerate(iid_overflow):
                partitions[cid].extend(subset)
        return partitions
    return _iid_partitions(examples, clients, seed)


def split_examples(
    examples: Sequence[VQAChoiceExample],
    *,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[VQAChoiceExample], list[VQAChoiceExample]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(round(len(shuffled) * eval_fraction))) if shuffled else 0
    eval_examples = shuffled[:eval_size]
    train_examples = shuffled[eval_size:]
    return train_examples, eval_examples


def client_size_summary(partitions: Sequence[Sequence[VQAChoiceExample]]) -> dict[str, Any]:
    sizes = [len(partition) for partition in partitions]
    answer_hist: dict[str, int] = defaultdict(int)
    for partition in partitions:
        for example in partition:
            answer_hist[example.answer] += 1
    return {
        "clients": len(partitions),
        "sizes": sizes,
        "total": sum(sizes),
        "answer_hist": dict(sorted(answer_hist.items())),
    }


def synthetic_examples(size: int = 12) -> list[VQAChoiceExample]:
    examples = []
    for idx in range(size):
        answer = OPTION_LETTERS[idx % len(OPTION_LETTERS)]
        examples.append(
            VQAChoiceExample(
                image=f"synthetic_{idx}.png",
                question=f"Which option is marked correct for synthetic case {idx}?",
                choices=("alpha", "beta", "gamma", "delta"),
                answer=answer,
                client_id=idx % 3,
            )
        )
    return examples


_KNOWN_FIELDS = {
    "image",
    "image_path",
    "filename",
    "question",
    "choices",
    "options",
    "answer",
    "client_id",
}


def _iid_partitions(
    examples: Sequence[VQAChoiceExample],
    clients: int,
    seed: int,
) -> list[list[VQAChoiceExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    partitions: list[list[VQAChoiceExample]] = [[] for _ in range(clients)]
    for idx, example in enumerate(shuffled):
        partitions[idx % clients].append(example)
    return partitions


def _coerce_choices(payload: Mapping[str, Any]) -> tuple[str, str, str, str]:
    raw = payload.get("choices", payload.get("options"))
    if raw is None:
        option_values = [payload.get(letter) for letter in OPTION_LETTERS]
        if all(value is not None for value in option_values):
            return cast(tuple[str, str, str, str], tuple(str(value) for value in option_values))
        raise ValueError("Example is missing choices/options")

    if isinstance(raw, Mapping):
        values = []
        for letter in OPTION_LETTERS:
            value = raw.get(letter) or raw.get(letter.lower())
            if value is None:
                value = raw.get(str(len(values)))
            values.append(value)
        if any(value is None for value in values):
            raise ValueError("Choice mapping must contain A-D or 0-3 entries")
        return cast(tuple[str, str, str, str], tuple(_clean_choice_text(value) for value in values))

    if isinstance(raw, str):
        return _parse_choices_from_string(raw)

    if isinstance(raw, Sequence):
        values = tuple(_clean_choice_text(choice) for choice in raw)
        if len(values) != 4:
            raise ValueError("Each closed-ended VQA example must have exactly four choices")
        return cast(tuple[str, str, str, str], values)

    raise TypeError(f"Unsupported choices/options type: {type(raw)!r}")


def _parse_choices_from_string(raw: str) -> tuple[str, str, str, str]:
    pieces = re.split(r"\s*(?:^|[\n;|])\s*[A-Da-d][\).:]\s*", raw)
    values = [piece.strip() for piece in pieces if piece.strip()]
    if len(values) == 4:
        return cast(tuple[str, str, str, str], tuple(values))

    values = [piece.strip() for piece in raw.split("|") if piece.strip()]
    if len(values) == 4:
        return cast(tuple[str, str, str, str], tuple(values))

    raise ValueError("Could not parse four choices from options string")


def _normalize_answer(answer: Any, choices: tuple[str, str, str, str]) -> str:
    if isinstance(answer, int) and 0 <= answer < len(OPTION_LETTERS):
        return OPTION_LETTERS[answer]

    answer_text = str(answer).strip()
    if not answer_text:
        raise ValueError("Example has an empty answer")

    first = answer_text[:1].upper()
    if first in OPTION_LETTERS:
        return first

    normalized = _normalize_text(answer_text)
    for letter, choice in zip(OPTION_LETTERS, choices, strict=True):
        if normalized == _normalize_text(choice):
            return letter

    raise ValueError(f"Could not map answer to A-D: {answer_text!r}")


def _clean_choice_text(value: Any) -> str:
    return re.sub(r"^[A-Da-d]\s*[\).:]\s*", "", str(value).strip()).strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _coerce_client_id(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
