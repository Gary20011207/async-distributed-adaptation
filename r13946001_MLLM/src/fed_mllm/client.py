from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fed_mllm.model import create_model, get_parameters, set_parameters

try:
    import flwr as fl
except ImportError:
    class _NumPyClient:
        pass

    class _ClientNamespace:
        NumPyClient = _NumPyClient

    class _FlowerNamespace:
        client = _ClientNamespace()

    fl = _FlowerNamespace()


class FederatedMLLMClient(fl.client.NumPyClient):
    """Sequential client wrapper that does not keep a large model resident."""

    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        lr: float,
        num_classes: int = 4,
        input_mode: str = "text",
        model_name: str = "tiny_text",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.num_classes = num_classes
        self.input_mode = input_mode
        self.model_name = model_name
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        model = self._new_model()
        params = get_parameters(model)
        self._release_model(model)
        return params

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        model = self._new_model()
        set_parameters(model, parameters)
        model.train()
        lr = float(config.get("lr", self.lr))
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

        total_loss = 0.0
        total_examples = 0
        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0
        for _ in range(self.local_epochs):
            for batch in self.trainloader:
                batch = _move_batch(batch, self.device)
                outputs = model(**_model_forward_inputs(batch))
                loss = outputs.loss
                if loss is None:
                    loss = nn.CrossEntropyLoss()(outputs.logits, batch["labels"])
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                step_in_accum += 1
                if step_in_accum % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                batch_size = _batch_size(batch)
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
        if step_in_accum % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        updated = get_parameters(model)
        self._release_model(model)
        avg_loss = total_loss / max(total_examples, 1)
        return updated, total_examples, {"train_loss": avg_loss, "lr": lr}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        model = self._new_model()
        set_parameters(model, parameters)
        loss, accuracy, num_examples, invalid_rate = evaluate_model(model, self.testloader, self.device)
        self._release_model(model)
        return loss, num_examples, {"accuracy": accuracy, "invalid_answer_rate": invalid_rate}

    def _new_model(self) -> torch.nn.Module:
        model = create_model(
            num_classes=self.num_classes,
            input_mode=self.input_mode,
            model_name=self.model_name,
        )
        if not getattr(model, "skip_to_device", False):
            model = model.to(self.device)
        return model

    def _release_model(self, model: torch.nn.Module) -> None:
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int, float]:
    if getattr(model, "is_generative_vlm", False):
        return _evaluate_generative_vlm(model, dataloader, device)

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    invalid_predictions = 0
    for batch in dataloader:
        batch = _move_batch(batch, device)
        outputs = model(**_model_forward_inputs(batch))
        logits = outputs.logits
        loss = outputs.loss
        if loss is None:
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_correct += int((preds == labels).sum().item())
        invalid_predictions += int(((preds < 0) | (preds >= logits.shape[-1])).sum().item())
        total_examples += int(labels.shape[0])
    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        total_examples,
        invalid_predictions / max(total_examples, 1),
    )


def _evaluate_generative_vlm(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int, float]:
    model.eval()
    tokenizer = getattr(getattr(model, "closed_answer_processor", None), "tokenizer", None)
    answer_labels = list(getattr(model, "answer_labels", ["A", "B", "C", "D"]))
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    invalid = 0
    for batch in dataloader:
        batch = _move_batch(batch, device)
        outputs = model(**_model_forward_inputs(batch))
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
        labels = batch["answer_label_ids"]
        gen_inputs = _generation_inputs(batch)
        generated = model.generate(**gen_inputs, max_new_tokens=4, do_sample=False)
        prompt_len = int(gen_inputs["input_ids"].shape[1])
        generated_suffix = generated[:, prompt_len:]
        if tokenizer is not None:
            decoded = tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)
        else:
            decoded = [""] * int(labels.shape[0])
        preds = torch.tensor(
            [_parse_closed_answer(text, answer_labels) for text in decoded],
            dtype=torch.long,
            device=labels.device,
        )
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_correct += int((preds == labels).sum().item())
        invalid += int((preds < 0).sum().item())
        total_examples += int(labels.shape[0])
    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        total_examples,
        invalid / max(total_examples, 1),
    )


def _parse_closed_answer(text: str, answer_labels: list[str]) -> int:
    normalized = text.strip().upper()
    for idx, label in enumerate(answer_labels):
        lab = str(label).upper()
        if normalized == lab or normalized.startswith(lab) or f" {lab}" in f" {normalized}":
            return idx
    return -1


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _model_forward_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in batch.items()
        if torch.is_tensor(value) and not key.startswith("eval_") and key != "answer_label_ids"
    }


def _generation_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("eval_"): value
        for key, value in batch.items()
        if key.startswith("eval_") and torch.is_tensor(value)
    }


def _batch_size(batch: dict[str, Any]) -> int:
    if "answer_label_ids" in batch:
        return int(batch["answer_label_ids"].shape[0])
    return int(batch["labels"].shape[0])
