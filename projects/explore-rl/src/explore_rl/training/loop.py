from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from explore_rl.core.config import TrainingConfig
from explore_rl.core.types import TrainingMetrics
from explore_rl.training.logging import ConsoleLogger, MetricsTracker
import torch


@dataclass
class CheckpointState:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    metrics: dict[str, float]


class Trainer(ABC):
    @abstractmethod
    def train_step(self, batch: Any) -> TrainingMetrics:
        pass


class TrainingLoop:
    def __init__(
        self,
        trainer: Trainer,
        data_iterator: Iterator[Any],
        config: TrainingConfig,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        console_logger: ConsoleLogger | None = None,
        extra_loggers: list[Any] | None = None,
    ) -> None:
        self.trainer = trainer
        self.data_iterator = data_iterator
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.console_logger = console_logger or ConsoleLogger(config.log_interval)
        self.extra_loggers = extra_loggers or []

        self.tracker = MetricsTracker()
        self._current_step = 0
        self._should_stop = False

    def run(self) -> MetricsTracker:
        self._should_stop = False

        for step in range(self._current_step, self.config.max_steps):
            if self._should_stop:
                break

            try:
                batch = next(self.data_iterator)
            except StopIteration:
                print(f"Data iterator exhausted at step {step}")
                break

            metrics = self.trainer.train_step(batch)
            self.tracker.log(metrics)
            self._current_step = step + 1

            self.console_logger.log(self.tracker)
            for logger in self.extra_loggers:
                logger.log(self.tracker)

            if self.config.eval_interval > 0 and (step + 1) % self.config.eval_interval == 0:
                self._on_eval()

            if self.config.save_interval > 0 and (step + 1) % self.config.save_interval == 0:
                self._on_save()

        return self.tracker

    def stop(self) -> None:
        self._should_stop = True

    def _on_eval(self) -> None:
        pass

    def _on_save(self) -> None:
        if self.model is None or self.optimizer is None:
            return

        checkpoint = CheckpointState(
            step=self._current_step,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            metrics=self.tracker.get_all_averages(),
        )

        torch.save(
            {
                "step": checkpoint.step,
                "model_state_dict": checkpoint.model_state,
                "optimizer_state_dict": checkpoint.optimizer_state,
                "metrics": checkpoint.metrics,
            },
            f"checkpoint_{checkpoint.step}.pt",
        )

    def load_checkpoint(self, path: str) -> None:
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer required to load checkpoint")

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_step = checkpoint["step"]

    @property
    def current_step(self) -> int:
        return self._current_step


def create_prompt_iterator(
    prompts: list[str],
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[list[str]]:
    import random

    indices = list(range(len(prompts)))

    while True:
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield [prompts[i] for i in batch_indices]
