from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fed_pathmnist.model import create_model, get_parameters, set_parameters

try:
    import flwr as fl
except ImportError:
    # The local simulator only needs the NumPyClient interface shape. Keeping a
    # tiny fallback lets smoke tests run before Flower is installed.
    class _NumPyClient:
        pass

    class _ClientNamespace:
        NumPyClient = _NumPyClient

    class _FlowerNamespace:
        client = _ClientNamespace()

    fl = _FlowerNamespace()


class PathMNISTClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        lr: float,
        num_classes: int = 9,
        in_channels: int = 3,
        model_name: str = "resnet18",
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_name = model_name
        self.model = create_model(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        set_parameters(self.model, parameters)
        self.model.train()
        lr = float(config.get("lr", self.lr))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        total_loss = 0.0
        total_examples = 0
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.view(-1).long().to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

        avg_loss = total_loss / max(total_examples, 1)
        return get_parameters(self.model), total_examples, {"train_loss": avg_loss, "lr": lr}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        set_parameters(self.model, parameters)
        loss, accuracy, num_examples = evaluate_model(
            self.model,
            self.testloader,
            self.device,
        )
        return loss, num_examples, {"accuracy": accuracy}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        total_examples,
    )
