import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import Budget, HarnessSpec, RunContext
from autoresearch.runner import run
from autoresearch.util import get_device


class FakeHarness:
    def __init__(self, primary: str = "val_loss") -> None:
        self._primary = primary

    @property
    def spec(self) -> HarnessSpec:
        return HarnessSpec(self._primary, "min", Budget(steps=5))

    def prepare(self) -> None:
        pass

    def evaluate(self, artifact: nn.Module, ctx: RunContext) -> dict[str, float]:
        x = torch.randn(8, 4, device=ctx.device)
        with torch.no_grad():
            loss = nn.functional.cross_entropy(artifact(x), torch.zeros(8, dtype=torch.long, device=ctx.device))
        return {self._primary: float(loss.item())}


class FakeExperiment:
    def run(self, harness: FakeHarness, ctx: RunContext) -> nn.Module:
        model = nn.Linear(4, 2).to(ctx.device)
        opt = optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(8, 4, device=ctx.device)
        target = torch.randint(0, 2, (8,), device=ctx.device)
        step = 0
        while not ctx.budget.exceeded(steps=step):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), target).backward()
            opt.step()
            step += 1
        ctx.record({"num_steps": float(step)})
        return model


class CrashingExperiment:
    def run(self, harness: FakeHarness, ctx: RunContext) -> nn.Module:
        raise RuntimeError("diverged")


def test_get_device_returns_torch_device() -> None:
    assert isinstance(get_device(), torch.device)


def test_run_completes_and_merges_telemetry(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("autoresearch.runner.get_device", lambda: torch.device("cpu"))
    record = run(FakeHarness(), FakeExperiment())
    assert "val_loss" in record
    assert record["num_steps"] == 5.0
    assert "total_seconds" in record
    assert "peak_vram_mb" in record


def test_run_handles_crash(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("autoresearch.runner.get_device", lambda: torch.device("cpu"))
    record = run(FakeHarness(), CrashingExperiment())
    assert "val_loss" in record
    import math

    assert math.isnan(record["val_loss"])
