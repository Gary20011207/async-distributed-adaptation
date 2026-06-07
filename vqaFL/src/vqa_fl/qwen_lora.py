from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vqa_fl.adapter_math import AdapterState, clone_state


DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DTypeName = Literal["auto", "bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class QwenLoRAConfig:
    model_name: str = DEFAULT_MODEL_NAME
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    dtype: DTypeName = "bf16"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True


@dataclass
class QwenLoRAModel:
    model: Any
    processor: Any


def require_qwen_dependencies() -> None:
    missing = missing_qwen_dependencies()
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Qwen3-VL LoRA training dependencies are missing: "
            f"{joined}. Install vqaFL dependencies before running real VQA training."
        )


def missing_qwen_dependencies() -> list[str]:
    missing = []
    for module_name in ("transformers", "peft", "datasets", "accelerate", "qwen_vl_utils"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def qwen_training_status() -> str:
    try:
        require_qwen_dependencies()
    except RuntimeError as exc:
        return str(exc)
    return "Qwen3-VL LoRA dependencies are available."


def load_qwen_lora(config: QwenLoRAConfig) -> QwenLoRAModel:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    require_qwen_dependencies()

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import transformers

    dtype = _torch_dtype(config.dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": config.trust_remote_code,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = AutoModelForImageTextToText

    model = model_cls.from_pretrained(config.model_name, **model_kwargs)
    _disable_use_cache(model)
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    _disable_use_cache(model)
    return QwenLoRAModel(model=model, processor=processor)


def get_adapter_state(model: Any, *, cpu: bool = True) -> AdapterState:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    if cpu:
        return {name: value.detach().cpu().clone() for name, value in state.items()}
    return clone_state(state)


def set_adapter_state(model: Any, state: AdapterState) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, state)


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = float(trainable / total) if total else 0.0
    return {"trainable": trainable, "total": total, "ratio": ratio}


def _torch_dtype(dtype: DTypeName) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _disable_use_cache(model: Any) -> None:
    candidates = [model]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop()
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        for attr in ("base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None:
                candidates.append(child)
