from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import resnet18

from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict

DEFAULT_NUM_CLASSES = 4
DEFAULT_IN_CHANNELS = 3


def create_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    in_channels: int = DEFAULT_IN_CHANNELS,
    model_name: str = "resnet18",
) -> nn.Module:
    # if model_name == "small_cnn":
    #     return SmallCNN(num_classes=num_classes, in_channels=in_channels)
        
    # if model_name == "mobilenet_v3_small":
    #     model = mobilenet_v3_small(
    #         weights=None,
    #         num_classes=num_classes,
    #         width_mult=0.5,
    #     )
    #     model.features[0][0] = nn.Conv2d(
    #         in_channels,
    #         model.features[0][0].out_channels,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #         bias=False,
    #     )
    #     return model
        
    if model_name == "qwen":
        model = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen1.5-0.5B",
            num_labels=num_classes,
            torch_dtype=torch.bfloat16,
        )
        model.config.pad_token_id = model.config.eos_token_id
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        return model

    # if model_name != "resnet18":
    #     raise ValueError(f"Unsupported model: {model_name}")
    # model = resnet18(weights=None, num_classes=num_classes)
    # model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool = nn.Identity()
    # return model


# class SmallCNN(nn.Module):

#     def __init__(self, num_classes: int, in_channels: int) -> None:
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.classifier = nn.Linear(128, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    if hasattr(model, "peft_config"):
        state_dict = get_peft_model_state_dict(model)
        return [val.detach().cpu().float().numpy().copy() for _, val in state_dict.items()]
        
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    if hasattr(model, "peft_config"):
        keys = list(get_peft_model_state_dict(model).keys())
        state_dict = OrderedDict(
            {key: torch.tensor(value) for key, value in zip(keys, parameters, strict=True)}
        )
        set_peft_model_state_dict(model, state_dict)
        return

    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in zip(keys, parameters, strict=True)}
    )
    model.load_state_dict(state_dict, strict=True)


def weighted_average(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
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


def interpolate(
    old: list[np.ndarray],
    new: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    return [
        ((1.0 - alpha) * old_layer + alpha * new_layer).astype(old_layer.dtype, copy=False)
        for old_layer, new_layer in zip(old, new, strict=True)
    ]