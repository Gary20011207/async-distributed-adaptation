from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pathmnist_shared.model import create_model, get_parameters, set_parameters


@dataclass
class TrainMetrics:
    loss: float
    examples: int
    batches: int


class SwarmNode:
    def __init__(
        self,
        *,
        nid: int,
        trainloader: DataLoader,
        num_examples: int,
        device: torch.device,
        local_epochs: int,
        lr: float,
    ) -> None:
        self.nid = nid
        self.trainloader = trainloader
        self.num_examples = num_examples
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = create_model().to(device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self) -> list[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        set_parameters(self.model, parameters)

    def train_interval(self, *, lr: float, sync_frequency: int | None) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        total_loss = 0.0
        total_examples = 0
        total_batches = 0

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
                total_batches += 1

                if sync_frequency is not None and total_batches >= sync_frequency:
                    avg_loss = total_loss / max(total_examples, 1)
                    return TrainMetrics(avg_loss, total_examples, total_batches)

        avg_loss = total_loss / max(total_examples, 1)
        return TrainMetrics(avg_loss, total_examples, total_batches)

