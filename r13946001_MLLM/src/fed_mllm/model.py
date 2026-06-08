from __future__ import annotations

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

DEFAULT_NUM_CLASSES = 4
DEFAULT_VOCAB_SIZE = 512


class TinyTextClassifier(nn.Module):
    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 96)
        self.encoder = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ):
        hidden = self.embedding(input_ids.clamp_min(0) % self.embedding.num_embeddings)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.encoder(pooled))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SimpleOutput(loss=loss, logits=logits)


class TinyVQAModel(nn.Module):
    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        super().__init__()
        self.text = TinyTextClassifier(num_classes=128, vocab_size=vocab_size)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 48, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ):
        text_logits = self.text(input_ids=input_ids, attention_mask=attention_mask).logits
        vision_feat = self.vision(pixel_values).flatten(1)
        logits = self.classifier(torch.cat([text_logits, vision_feat], dim=1))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SimpleOutput(loss=loss, logits=logits)


class SimpleOutput:
    def __init__(self, *, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


def create_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_mode: str = "text",
    model_name: str = "tiny_text",
) -> nn.Module:
    if model_name == "tiny_text":
        return TinyTextClassifier(num_classes=num_classes)
    if model_name == "tiny_vqa":
        return TinyVQAModel(num_classes=num_classes)
    if model_name == "qwen_text_0_5b_lora":
        return _create_qwen_text_classifier(num_classes=num_classes)
    if model_name == "qwen_vl_proxy":
        return TinyVQAModel(num_classes=num_classes)
    if model_name == "qwen2_5_vl_3b_qlora":
        return _create_qwen_vl_qlora(num_classes=num_classes, input_mode=input_mode)
    raise ValueError(f"Unsupported model: {model_name}")


def _create_qwen_text_classifier(num_classes: int) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "qwen_text_0_5b_lora requires transformers and peft. "
            "Install the project with `python -m pip install -e .`."
        ) from exc

    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen1.5-0.5B",
        num_labels=num_classes,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = getattr(model.config, "eos_token_id", None)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, peft_config)


def _create_qwen_vl_qlora(num_classes: int, input_mode: str) -> nn.Module:
    if input_mode != "image_text":
        raise ValueError("qwen2_5_vl_3b_qlora is only supported for --task medical_vqa")
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "qwen2_5_vl_3b_qlora requires transformers, peft, accelerate, bitsandbytes, and qwen-vl-utils. "
            "Install the project with `python -m pip install -e .`."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Real Qwen2.5-VL QLoRA requires CUDA in this project setup.")

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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.is_generative_vlm = True
    model.answer_labels = ["A", "B", "C", "D"][:num_classes]
    model.closed_answer_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
        trust_remote_code=True,
    )
    model.skip_to_device = True
    return model


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    peft_state = _peft_state_dict(model)
    state = peft_state if peft_state is not None else model.state_dict()
    return [tensor.detach().cpu().float().numpy().copy() for tensor in state.values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    peft_state = _peft_state_dict(model)
    if peft_state is not None:
        try:
            from peft import set_peft_model_state_dict
        except ImportError as exc:
            raise RuntimeError("Setting LoRA parameters requires peft.") from exc
        keys = list(peft_state.keys())
        state = OrderedDict(
            (key, torch.as_tensor(value, dtype=peft_state[key].dtype, device=peft_state[key].device))
            for key, value in zip(keys, parameters, strict=True)
        )
        set_peft_model_state_dict(model, state)
        return

    keys = list(model.state_dict().keys())
    current = model.state_dict()
    state = OrderedDict(
        (key, torch.as_tensor(value, dtype=current[key].dtype, device=current[key].device))
        for key, value in zip(keys, parameters, strict=True)
    )
    model.load_state_dict(state, strict=True)


def weighted_average(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise ValueError("Cannot average zero examples")
    averaged: list[np.ndarray] = []
    for layer_values in zip(*(params for params, _ in results), strict=True):
        layer_avg = sum(
            layer * (num_examples / total_examples)
            for layer, (_, num_examples) in zip(layer_values, results, strict=True)
        )
        averaged.append(layer_avg.astype(layer_values[0].dtype, copy=False))
    return averaged


def interpolate(old: list[np.ndarray], new: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    return [
        ((1.0 - alpha) * old_layer + alpha * new_layer).astype(old_layer.dtype, copy=False)
        for old_layer, new_layer in zip(old, new, strict=True)
    ]


def _peft_state_dict(model: nn.Module) -> dict[str, torch.Tensor] | None:
    if not hasattr(model, "peft_config"):
        return None
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("Reading LoRA parameters requires peft.") from exc
    return get_peft_model_state_dict(model)
