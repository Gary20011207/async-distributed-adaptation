from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vqa_fl.data import VQAChoiceExample, format_prompt
from vqa_fl.metrics import option_accuracy, parse_choice
from vqa_fl.qwen_lora import get_adapter_state


@dataclass(frozen=True)
class LocalTrainConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 1
    max_new_tokens: int = 4


def train_local_lora(
    *,
    model: Any,
    processor: Any,
    examples: list[VQAChoiceExample],
    config: LocalTrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if not examples:
        raise ValueError("Cannot train on an empty client dataset")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL LoRA training")

    from torch.utils.data import DataLoader

    model.train()
    device = infer_model_device(model)
    collate = QwenVQATrainCollator(processor=processor, device=device)
    loader = DataLoader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    step_count = 0
    for _epoch in range(config.epochs):
        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss / max(config.gradient_accumulation_steps, 1)
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu().item()))
            step_count += 1

            if step_count % max(config.gradient_accumulation_steps, 1) == 0:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    if step_count % max(config.gradient_accumulation_steps, 1) != 0:
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            config.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    state = get_adapter_state(model, cpu=True)
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return state, {"train_loss": mean_loss, "train_steps": float(step_count)}


@torch.no_grad()
def evaluate_choice_accuracy(
    *,
    model: Any,
    processor: Any,
    examples: list[VQAChoiceExample],
    config: EvalConfig,
) -> dict[str, float]:
    if not examples:
        return {"accuracy": 0.0, "evaluated": 0.0}

    from torch.utils.data import DataLoader

    model.eval()
    device = infer_model_device(model)
    collate = QwenVQAEvalCollator(processor=processor, device=device)
    loader = DataLoader(
        examples,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    predictions: list[str] = []
    labels: list[str] = []
    for batch, answers in loader:
        generated = model.generate(
            **batch,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
        prompt_lens = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
        decoded = []
        for row, prompt_len in enumerate(prompt_lens):
            decoded_ids = generated[row][int(prompt_len) :]
            decoded.append(
                processor.decode(
                    decoded_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        predictions.extend(parse_choice(text) for text in decoded)
        labels.extend(answers)

    return {
        "accuracy": option_accuracy(predictions, labels),
        "evaluated": float(len(labels)),
    }


class QwenVQATrainCollator:
    def __init__(self, *, processor: Any, device: torch.device) -> None:
        self.processor = processor
        self.device = device

    def __call__(self, examples: list[VQAChoiceExample]) -> dict[str, torch.Tensor]:
        prompt_messages = [_messages_for_example(example, include_answer=False) for example in examples]
        full_messages = [_messages_for_example(example, include_answer=True) for example in examples]

        full = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

        labels = full["input_ids"].clone()
        prompt_lens = prompt["attention_mask"].sum(dim=1)
        for row, prompt_len in enumerate(prompt_lens.tolist()):
            labels[row, : min(int(prompt_len), labels.shape[1])] = -100

        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[full["input_ids"] == pad_token_id] = -100

        full["labels"] = labels
        return _move_batch_to_device(full, self.device)


class QwenVQAEvalCollator:
    def __init__(self, *, processor: Any, device: torch.device) -> None:
        self.processor = processor
        self.device = device

    def __call__(self, examples: list[VQAChoiceExample]) -> tuple[dict[str, torch.Tensor], list[str]]:
        messages = [_messages_for_example(example, include_answer=False) for example in examples]
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return _move_batch_to_device(batch, self.device), [example.answer for example in examples]


def infer_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        device = getattr(model, "device")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cuda")


def _messages_for_example(example: VQAChoiceExample, *, include_answer: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example.image},
                {"type": "text", "text": format_prompt(example)},
            ],
        }
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example.answer})
    return messages


def _move_batch_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
