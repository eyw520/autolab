from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class HarnessConfig:
    time_budget: int
    seq_len: int
    primary_metric: str
    cache_dir: str


@dataclass
class TrainingConfig:
    total_batch_size: int
    device_batch_size: int


@dataclass
class TrainingResult:
    total_training_time: float
    total_tokens: int
    num_steps: int
    final_loss: float


class Harness(Protocol):
    @property
    def config(self) -> HarnessConfig: ...

    def prepare(self) -> None: ...

    def get_vocab_size(self) -> int: ...

    def make_dataloader(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]: ...

    def evaluate(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, float]: ...


class Experiment(Protocol):
    def get_training_config(self, device: torch.device) -> TrainingConfig: ...

    def build_model(
        self,
        vocab_size: int,
        seq_len: int,
        device: torch.device,
    ) -> nn.Module: ...

    def build_optimizer(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> optim.Optimizer: ...

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        progress: float,
        device: torch.device,
    ) -> float: ...
