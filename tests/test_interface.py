from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import (
    Experiment,
    Harness,
    HarnessConfig,
    TrainingConfig,
    TrainingResult,
)


class _Harness:
    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(time_budget=0, seq_len=8, primary_metric="val_loss", cache_dir="/tmp")

    def prepare(self) -> None:
        pass

    def get_vocab_size(self) -> int:
        return 16

    def make_dataloader(self, split, batch_size, seq_len, device):  # type: ignore[no-untyped-def]
        while True:
            yield torch.zeros(1), torch.zeros(1), 1

    def evaluate(self, model, batch_size, device):  # type: ignore[no-untyped-def]
        return {"val_loss": 0.0}


class _Experiment:
    def get_training_config(self, device):  # type: ignore[no-untyped-def]
        return TrainingConfig(total_batch_size=8, device_batch_size=1)

    def build_model(self, vocab_size, seq_len, device):  # type: ignore[no-untyped-def]
        return nn.Linear(1, 1)

    def build_optimizer(self, model, training_config, device):  # type: ignore[no-untyped-def]
        return optim.SGD(model.parameters(), lr=0.1)

    def train_step(self, model, optimizer, x, y, step, progress, device):  # type: ignore[no-untyped-def]
        return 0.0


def test_dataclasses_construct() -> None:
    cfg = HarnessConfig(time_budget=300, seq_len=1024, primary_metric="val_bpb", cache_dir="/tmp")
    assert cfg.time_budget == 300
    tc = TrainingConfig(total_batch_size=2**12, device_batch_size=8)
    assert tc.device_batch_size == 8
    tr = TrainingResult(total_training_time=1.0, total_tokens=10, num_steps=2, final_loss=0.5)
    assert tr.num_steps == 2


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_Harness(), Harness)
    assert isinstance(_Experiment(), Experiment)


def test_incomplete_implementations_fail_isinstance() -> None:
    class Partial:
        def prepare(self) -> None:
            pass

    assert not isinstance(Partial(), Harness)
    assert not isinstance(Partial(), Experiment)


def _iter_type_ok() -> None:
    # static sanity that the dataloader return annotation matches usage
    _: Iterator[tuple[torch.Tensor, torch.Tensor, int]] = _Harness().make_dataloader("train", 1, 8, torch.device("cpu"))
