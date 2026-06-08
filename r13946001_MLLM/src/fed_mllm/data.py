from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class ClientLoaders:
    train: DataLoader
    num_examples: int


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    task: str
    num_classes: int
    input_mode: str
    answer_labels: list[str]


ANSWER_LABELS = ["A", "B", "C", "D"]
PUBMEDQA_LABELS = ["yes", "no", "maybe"]
YES_NO_LABELS = ["yes", "no"]
QWEN_VL_MODEL_NAMES = {"qwen2_5_vl_3b_qlora"}


class SyntheticMCQADataset(Dataset):
    """Small closed-ended text dataset for dependency-free smoke tests."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        vocab_size: int = 512,
        seq_len: int = 64,
        num_classes: int = 4,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_ids = rng.integers(4, vocab_size, size=(size, seq_len), dtype=np.int64)
        self.attention_mask = np.ones((size, seq_len), dtype=np.int64)
        self.labels = rng.integers(0, num_classes, size=(size,), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[index]),
            "attention_mask": torch.from_numpy(self.attention_mask[index]),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class SyntheticVQADataset(Dataset):
    """Small image+question dataset for compact VQA smoke tests."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        vocab_size: int = 512,
        seq_len: int = 48,
        image_size: int = 64,
        num_classes: int = 4,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_ids = rng.integers(4, vocab_size, size=(size, seq_len), dtype=np.int64)
        self.attention_mask = np.ones((size, seq_len), dtype=np.int64)
        self.pixel_values = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(size, 3, image_size, image_size),
        ).astype(np.float32)
        self.labels = rng.integers(0, num_classes, size=(size,), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[index]),
            "attention_mask": torch.from_numpy(self.attention_mask[index]),
            "pixel_values": torch.from_numpy(self.pixel_values[index]),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class SyntheticRawVQADataset(Dataset):
    """Raw PIL-image samples for real Qwen2.5-VL smoke tests."""

    def __init__(self, size: int, *, seed: int, image_size: int = 64, num_classes: int = 4) -> None:
        from PIL import Image

        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, num_classes, size=(size,), dtype=np.int64)
        self.samples: list[dict[str, Any]] = []
        for idx, label in enumerate(self.labels.tolist()):
            pixels = rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
            self.samples.append(
                {
                    "question": f"Synthetic closed VQA question {idx}. Choose the correct option.",
                    "choices": ["A pattern", "B pattern", "C pattern", "D pattern"],
                    "answer_label": int(label),
                    "image": Image.fromarray(pixels, mode="RGB"),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class TokenizedMCQADataset(Dataset):
    """Hugging Face MCQA dataset converted to closed-set classification."""

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        *,
        dataset_name: str,
        max_length: int,
        answer_labels: list[str] | None = None,
    ) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.answer_labels = answer_labels or ANSWER_LABELS
        self.labels = np.asarray([self._extract_label(item) for item in hf_dataset], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]
        prompt = _format_mcqa_prompt(
            question=self._extract_question(item),
            choices=self._extract_choices(item),
            answer_labels=self.answer_labels,
        )
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self._extract_label(item), dtype=torch.long),
        }

    def _extract_question(self, item: dict[str, Any]) -> str:
        question = str(item.get("question") or item.get("sent1") or "")
        if self.dataset_name == "pubmedqa":
            context = item.get("context", {})
            contexts = context.get("contexts", []) if isinstance(context, dict) else []
            context_text = " ".join(str(piece) for piece in contexts[:4])
            return f"{context_text}\nQuestion: {question}"
        return question

    def _extract_choices(self, item: dict[str, Any]) -> list[str]:
        if self.dataset_name == "pubmedqa":
            return PUBMEDQA_LABELS
        if "options" in item and isinstance(item["options"], dict):
            return [str(item["options"].get(label, "")) for label in ANSWER_LABELS]
        if "choices" in item and isinstance(item["choices"], list):
            return [str(choice) for choice in item["choices"][: len(self.answer_labels)]]
        return [
            str(item.get("opa", "")),
            str(item.get("opb", "")),
            str(item.get("opc", "")),
            str(item.get("opd", "")),
        ][: len(self.answer_labels)]

    def _extract_label(self, item: dict[str, Any]) -> int:
        if self.dataset_name == "pubmedqa":
            decision = str(item.get("final_decision", "")).strip().lower()
            if decision in PUBMEDQA_LABELS:
                return PUBMEDQA_LABELS.index(decision)
            return 0
        if "answer_idx" in item and item["answer_idx"] is not None:
            answer_idx = str(item["answer_idx"]).strip().upper()
            if answer_idx in ANSWER_LABELS:
                return ANSWER_LABELS.index(answer_idx)
        for key in ("answer", "cop", "label"):
            if key in item and item[key] is not None:
                value = item[key]
                if isinstance(value, str):
                    text = value.strip().upper()
                    if text in ANSWER_LABELS:
                        return ANSWER_LABELS.index(text)
                    if text.isdigit():
                        return int(text)
                return int(value)
        raise KeyError(f"Could not extract MCQA label from keys={sorted(item)}")


class ClosedVQADataset(Dataset):
    """Closed-ended VQA dataset with a compact image+text classifier path."""

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        *,
        dataset_name: str,
        max_length: int,
        image_size: int = 64,
    ) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.image_size = image_size
        self.labels = np.asarray([self._extract_label(item) for item in hf_dataset], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]
        question = str(item.get("question") or item.get("Question") or "")
        prompt = _format_mcqa_prompt(question=question, choices=self._extract_choices(item))
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": self._extract_image_tensor(item),
            "labels": torch.tensor(self._extract_label(item), dtype=torch.long),
        }

    def _extract_choices(self, item: dict[str, Any]) -> list[str]:
        if "choices" in item and isinstance(item["choices"], list):
            choices = [str(choice) for choice in item["choices"][:4]]
            return choices + [""] * max(0, 4 - len(choices))
        answer = str(item.get("answer") or item.get("Answer") or "").strip()
        if answer.lower() in YES_NO_LABELS:
            return ["yes", "no", "", ""]
        return [answer, "not shown", "uncertain", "other"]

    def _extract_label(self, item: dict[str, Any]) -> int:
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
        if isinstance(value, (int, np.integer)):
            return int(value)
        return 0

    def _extract_image_tensor(self, item: dict[str, Any]) -> torch.Tensor:
        image = item.get("image") or item.get("Image")
        if image is None:
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        try:
            from PIL import Image
            from torchvision import transforms

            if not isinstance(image, Image.Image):
                image = Image.open(image)
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )
            return transform(image.convert("RGB")).float()
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)


class QwenVLClosedDataset(Dataset):
    """Raw closed-ended VQA samples consumed by Qwen2.5-VL chat processor."""

    def __init__(self, hf_dataset: Any, *, dataset_name: str) -> None:
        self.data = hf_dataset
        self.dataset_name = dataset_name
        self.labels = np.asarray([self._extract_label(item) for item in hf_dataset], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        return {
            "question": str(item.get("question") or item.get("Question") or ""),
            "choices": self._extract_choices(item),
            "answer_label": int(self.labels[index]),
            "image": self._extract_image(item),
        }

    def _extract_choices(self, item: dict[str, Any]) -> list[str]:
        if "choices" in item and isinstance(item["choices"], list):
            choices = [str(choice) for choice in item["choices"][:4]]
            return choices + [""] * max(0, 4 - len(choices))
        answer = str(item.get("answer") or item.get("Answer") or "").strip()
        if answer.lower() in YES_NO_LABELS:
            return ["yes", "no", "", ""]
        return [answer, "not shown", "uncertain", "other"]

    def _extract_label(self, item: dict[str, Any]) -> int:
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
        if isinstance(value, (int, np.integer)):
            return int(value)
        return 0

    def _extract_image(self, item: dict[str, Any]) -> Any:
        image = item.get("image") or item.get("Image")
        if image is None:
            from PIL import Image

            return Image.new("RGB", (64, 64), color=(0, 0, 0))
        try:
            from PIL import Image

            if isinstance(image, Image.Image):
                return image.convert("RGB")
            return Image.open(image).convert("RGB")
        except Exception:
            from PIL import Image

            return Image.new("RGB", (64, 64), color=(0, 0, 0))


class QwenVLClosedCollator:
    """Build causal-label Qwen2.5-VL batches for closed A/B/C/D answers."""

    def __init__(self, processor: Any, *, answer_labels: list[str], max_length: int) -> None:
        self.processor = processor
        self.answer_labels = answer_labels
        self.max_length = max_length
        tokenizer = getattr(processor, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        full_messages: list[list[dict[str, Any]]] = []
        prompt_messages: list[list[dict[str, Any]]] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        answer_ids: list[int] = []
        for example in examples:
            label = int(example["answer_label"])
            answer = self.answer_labels[label]
            prompt = self._messages(example, include_answer=False)
            full = self._messages(example, include_answer=True, answer=answer)
            prompt_messages.append(prompt)
            full_messages.append(full)
            prompt_texts.append(
                self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            )
            answer_ids.append(label)

        full_inputs = self._processor_call(full_messages, full_texts)
        prompt_inputs = self._processor_call(prompt_messages, prompt_texts)
        labels = full_inputs["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        for row_idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
            prompt_len = int((prompt_ids != self.pad_token_id).sum().item())
            labels[row_idx, :prompt_len] = -100
            # With image tokens and aggressive max_length, the answer can be
            # truncated. Keep one supervised token to avoid NaN loss in smoke
            # tests; full experiments should use a larger --max-length.
            if bool((labels[row_idx] != -100).sum().item() == 0):
                non_pad = torch.flatnonzero(full_inputs["input_ids"][row_idx] != self.pad_token_id)
                if len(non_pad) > 0:
                    last_pos = int(non_pad[-1].item())
                    labels[row_idx, last_pos] = full_inputs["input_ids"][row_idx, last_pos]
        full_inputs["labels"] = labels
        full_inputs["answer_label_ids"] = torch.tensor(answer_ids, dtype=torch.long)
        for key, value in prompt_inputs.items():
            if torch.is_tensor(value):
                full_inputs[f"eval_{key}"] = value
        return full_inputs

    def _messages(
        self,
        example: dict[str, Any],
        *,
        include_answer: bool,
        answer: str | None = None,
    ) -> list[dict[str, Any]]:
        choices = example["choices"] + [""] * max(0, 4 - len(example["choices"]))
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
        if include_answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
        return messages

    def _processor_call(self, messages: list[list[dict[str, Any]]], texts: list[str]) -> dict[str, torch.Tensor]:
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception:
            images = [message[0]["content"][0]["image"] for message in messages]
            return self.processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )


def get_dataset_metadata(dataset_name: str, *, synthetic: bool = False, task: str = "text_mcqa") -> DatasetMetadata:
    if synthetic:
        if task == "medical_vqa":
            return DatasetMetadata("synthetic_vqa", task, 4, "image_text", ANSWER_LABELS)
        return DatasetMetadata("synthetic_mcqa", task, 4, "text", ANSWER_LABELS)

    key = dataset_name.lower()
    if key in {"mmlu", "medmcqa", "medqa_usmle"}:
        return DatasetMetadata(key, "text_mcqa", 4, "text", ANSWER_LABELS)
    if key == "pubmedqa":
        return DatasetMetadata(key, "text_mcqa", 3, "text", PUBMEDQA_LABELS)
    if key in {"vqa_rad_closed", "path_vqa_closed", "slake_closed", "scienceqa_image"}:
        return DatasetMetadata(key, "medical_vqa", 4, "image_text", ANSWER_LABELS)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_datasets(
    *,
    dataset_name: str,
    task: str,
    model_name: str,
    synthetic: bool,
    download: bool,
    max_train_samples: int | None,
    max_test_samples: int | None,
    seed: int,
    max_length: int = 256,
) -> tuple[Dataset, Dataset]:
    if synthetic:
        size_train = max_train_samples or 128
        size_test = max_test_samples or 64
        if task == "medical_vqa" and model_name in QWEN_VL_MODEL_NAMES:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                _tokenizer_name_for_model(model_name),
                min_pixels=64 * 28 * 28,
                max_pixels=256 * 28 * 28,
                trust_remote_code=True,
            )
            train = SyntheticRawVQADataset(size_train, seed=seed)
            test = SyntheticRawVQADataset(size_test, seed=seed + 1)
            collator = QwenVLClosedCollator(processor, answer_labels=ANSWER_LABELS, max_length=max_length)
            train.collate_fn = collator
            test.collate_fn = collator
            return train, test
        if task == "medical_vqa":
            return (
                SyntheticVQADataset(size_train, seed=seed),
                SyntheticVQADataset(size_test, seed=seed + 1),
            )
        return (
            SyntheticMCQADataset(size_train, seed=seed),
            SyntheticMCQADataset(size_test, seed=seed + 1),
        )

    if not download:
        raise RuntimeError("Real datasets require download/cache access; remove --no-download.")

    from datasets import load_dataset
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_name = _tokenizer_name_for_model(model_name)
    processor = None
    tokenizer = None
    if model_name in QWEN_VL_MODEL_NAMES and task == "medical_vqa":
        processor = AutoProcessor.from_pretrained(
            tokenizer_name,
            min_pixels=64 * 28 * 28,
            max_pixels=256 * 28 * 28,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    key = dataset_name.lower()
    if key == "mmlu":
        train = load_dataset("cais/mmlu", "all", split="auxiliary_train", cache_dir="data/hf")
        test = load_dataset("cais/mmlu", "all", split="test", cache_dir="data/hf")
        trainset = TokenizedMCQADataset(train, tokenizer, dataset_name=key, max_length=max_length)
        testset = TokenizedMCQADataset(test, tokenizer, dataset_name=key, max_length=max_length)
    elif key == "medmcqa":
        train = load_dataset("openlifescienceai/medmcqa", split="train", cache_dir="data/hf")
        test = load_dataset("openlifescienceai/medmcqa", split="validation", cache_dir="data/hf")
        trainset = TokenizedMCQADataset(train, tokenizer, dataset_name=key, max_length=max_length)
        testset = TokenizedMCQADataset(test, tokenizer, dataset_name=key, max_length=max_length)
    elif key == "medqa_usmle":
        train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", cache_dir="data/hf")
        test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir="data/hf")
        trainset = TokenizedMCQADataset(train, tokenizer, dataset_name=key, max_length=max_length)
        testset = TokenizedMCQADataset(test, tokenizer, dataset_name=key, max_length=max_length)
    elif key == "pubmedqa":
        all_rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir="data/hf")
        split = all_rows.train_test_split(test_size=0.2, seed=seed)
        trainset = TokenizedMCQADataset(
            split["train"], tokenizer, dataset_name=key, max_length=max_length, answer_labels=PUBMEDQA_LABELS
        )
        testset = TokenizedMCQADataset(
            split["test"], tokenizer, dataset_name=key, max_length=max_length, answer_labels=PUBMEDQA_LABELS
        )
    elif key == "vqa_rad_closed":
        train = load_dataset("abhay2812/vqa-rad", split="train", cache_dir="data/hf")
        test = load_dataset("abhay2812/vqa-rad", split="test", cache_dir="data/hf")
        trainset, testset = _build_vqa_datasets(train, test, processor, tokenizer, key, max_length)
    elif key == "path_vqa_closed":
        train = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir="data/hf")
        test = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir="data/hf")
        trainset, testset = _build_vqa_datasets(train, test, processor, tokenizer, key, max_length)
    else:
        raise ValueError(f"Dataset adapter is not implemented yet: {dataset_name}")

    return (
        _limit_dataset(trainset, max_train_samples, seed),
        _limit_dataset(testset, max_test_samples, seed + 1),
    )


def _build_vqa_datasets(
    train: Any,
    test: Any,
    processor: Any,
    tokenizer: Any,
    key: str,
    max_length: int,
) -> tuple[Dataset, Dataset]:
    if processor is not None:
        trainset = QwenVLClosedDataset(train, dataset_name=key)
        testset = QwenVLClosedDataset(test, dataset_name=key)
        collator = QwenVLClosedCollator(processor, answer_labels=ANSWER_LABELS, max_length=max_length)
        trainset.collate_fn = collator
        testset.collate_fn = collator
        return trainset, testset
    return (
        ClosedVQADataset(train, tokenizer, dataset_name=key, max_length=max_length),
        ClosedVQADataset(test, tokenizer, dataset_name=key, max_length=max_length),
    )


def _tokenizer_name_for_model(model_name: str) -> str:
    if model_name == "qwen2_5_vl_3b_qlora":
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    if model_name == "qwen_text_0_5b_lora":
        return "Qwen/Qwen1.5-0.5B"
    return "bert-base-uncased"


def _format_mcqa_prompt(
    question: str,
    choices: list[str],
    answer_labels: list[str] | None = None,
) -> str:
    labels = answer_labels or ANSWER_LABELS
    padded = choices + [""] * max(0, len(labels) - len(choices))
    options = "\n".join(f"{label}) {padded[idx]}" for idx, label in enumerate(labels))
    return f"Question: {question}\n{options}\nAnswer:"


def _limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:max_samples].tolist()
    subset = Subset(dataset, indices)
    collate_fn = _collate_fn_for_dataset(dataset)
    if collate_fn is not None:
        subset.collate_fn = collate_fn
    return subset


def partition_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    partition: str,
    dirichlet_alpha: float,
) -> list[ClientLoaders]:
    if partition == "iid":
        return _iid_client_loaders(dataset, num_clients, batch_size, seed, num_workers)
    if partition == "dirichlet":
        return _dirichlet_client_loaders(
            dataset,
            num_clients,
            batch_size,
            seed,
            num_workers,
            dirichlet_alpha,
        )
    raise ValueError(f"Unsupported partition: {partition}")


def test_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn_for_dataset(dataset),
    )


def _iid_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[ClientLoaders]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return _client_loaders_from_indices(dataset, [split.tolist() for split in splits], batch_size, seed, num_workers)


def _dirichlet_client_loaders(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    alpha: float,
) -> list[ClientLoaders]:
    if alpha <= 0:
        raise ValueError("--dirichlet-alpha must be positive")
    rng = np.random.default_rng(seed)
    labels = _dataset_labels(dataset)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for cid, split in enumerate(np.split(class_indices, split_points)):
            client_indices[cid].extend(split.tolist())
    _move_examples_to_empty_clients(client_indices, rng)
    for indices in client_indices:
        rng.shuffle(indices)
    return _client_loaders_from_indices(dataset, client_indices, batch_size, seed, num_workers)


def _client_loaders_from_indices(
    dataset: Dataset,
    client_indices: list[list[int]],
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[ClientLoaders]:
    loaders: list[ClientLoaders] = []
    generator = torch.Generator().manual_seed(seed)
    for indices in client_indices:
        subset = Subset(dataset, list(indices))
        collate_fn = _collate_fn_for_dataset(dataset)
        if collate_fn is not None:
            subset.collate_fn = collate_fn
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            collate_fn=collate_fn,
        )
        loaders.append(ClientLoaders(train=loader, num_examples=len(subset)))
    return loaders


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return np.asarray(labels).reshape(-1).astype(np.int64)
    extracted = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = item.get("labels", item.get("answer_label", 0))
        extracted.append(int(label))
    return np.asarray(extracted, dtype=np.int64)


def _collate_fn_for_dataset(dataset: Dataset):
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is not None:
        return collate_fn
    if isinstance(dataset, Subset):
        return _collate_fn_for_dataset(dataset.dataset)
    return None


def _move_examples_to_empty_clients(
    client_indices: list[list[int]],
    rng: np.random.Generator,
) -> None:
    for empty_cid in [cid for cid, indices in enumerate(client_indices) if not indices]:
        donor_cid = max(range(len(client_indices)), key=lambda cid: len(client_indices[cid]))
        if len(client_indices[donor_cid]) <= 1:
            raise ValueError("Could not create non-empty client partitions")
        donor_pos = int(rng.integers(0, len(client_indices[donor_cid])))
        client_indices[empty_cid].append(client_indices[donor_cid].pop(donor_pos))
