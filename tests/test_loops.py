import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import Budget, RunContext
from autoresearch.loops import SupervisedConfig, supervised_loop


SEQ = 4
B = 2


def _dataloader(device):  # type: ignore[no-untyped-def]
    while True:
        yield (
            torch.randn(B, SEQ, device=device),
            torch.randn(B, SEQ, device=device),
            1,
        )


def _step_fn(model, optimizer, x, y, step, progress, device):  # type: ignore[no-untyped-def]
    loss = model(x).pow(2).mean()
    loss.backward()
    return float(loss.item())


def test_supervised_loop_runs_and_records_telemetry() -> None:
    device = torch.device("cpu")
    ctx = RunContext(device=device, budget=Budget(steps=15), seed=0)
    model = nn.Linear(SEQ, SEQ)
    opt = optim.SGD(model.parameters(), lr=0.001)

    supervised_loop(model, opt, _dataloader(device), _step_fn, ctx, SupervisedConfig(B * SEQ, B, SEQ))

    assert ctx.telemetry["num_steps"] >= 15
    assert "training_seconds" in ctx.telemetry
    assert "final_loss" in ctx.telemetry


def test_supervised_loop_detects_divergence() -> None:
    device = torch.device("cpu")
    ctx = RunContext(device=device, budget=Budget(steps=50), seed=0)
    model = nn.Linear(SEQ, SEQ)
    opt = optim.SGD(model.parameters(), lr=0.001)

    def diverge(model, optimizer, x, y, step, progress, device):  # type: ignore[no-untyped-def]
        return float("nan")

    try:
        supervised_loop(model, opt, _dataloader(device), diverge, ctx, SupervisedConfig(B * SEQ, B, SEQ))
    except RuntimeError as e:
        assert "diverged" in str(e)
    else:
        raise AssertionError("expected RuntimeError on divergence")
