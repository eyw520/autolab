import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import HarnessConfig, TrainingConfig
from autoresearch.runner import get_device, run


VOCAB = 16
SEQ = 8


class _TinyModel(nn.Module):
    def __init__(self, vocab: int, dim: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.emb(x))
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


class FakeHarness:
    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(time_budget=1, seq_len=SEQ, primary_metric="val_loss", cache_dir="/tmp")

    def prepare(self) -> None:
        pass

    def get_vocab_size(self) -> int:
        return VOCAB

    def make_dataloader(self, split, batch_size, seq_len, device):  # type: ignore[no-untyped-def]
        gen = torch.Generator().manual_seed(0)
        while True:
            x = torch.randint(0, VOCAB, (batch_size, seq_len), generator=gen).to(device)
            y = torch.randint(0, VOCAB, (batch_size, seq_len), generator=gen).to(device)
            yield x, y, 1

    def evaluate(self, model, batch_size, device):  # type: ignore[no-untyped-def]
        x = torch.randint(0, VOCAB, (batch_size, SEQ)).to(device)
        with torch.no_grad():
            loss = model(x, x)
        return {"val_loss": float(loss.item())}


class FakeExperiment:
    def get_training_config(self, device):  # type: ignore[no-untyped-def]
        b = 4
        return TrainingConfig(total_batch_size=b * SEQ, device_batch_size=b)

    def build_model(self, vocab_size, seq_len, device):  # type: ignore[no-untyped-def]
        return _TinyModel(vocab_size).to(device)

    def build_optimizer(self, model, training_config, device):  # type: ignore[no-untyped-def]
        return optim.SGD(model.parameters(), lr=0.1)

    def train_step(self, model, optimizer, x, y, step, progress, device):  # type: ignore[no-untyped-def]
        loss = model(x, y)
        loss.backward()
        return float(loss.item())


def test_get_device_returns_torch_device() -> None:
    assert isinstance(get_device(), torch.device)


def test_run_completes_and_reports_metrics(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("autoresearch.runner.get_device", lambda: torch.device("cpu"))
    result = run(FakeHarness(), FakeExperiment())
    assert "val_loss" in result
    assert result["num_steps"] > 0
    assert result["total_seconds"] >= 0
    assert "total_tokens_M" in result
