import torch

from autoresearch.interface import Budget, Experiment, Harness, HarnessSpec, RunContext
from autoresearch.util import is_better


class _Harness:
    @property
    def spec(self) -> HarnessSpec:
        return HarnessSpec("m", "min", Budget(steps=1))

    def prepare(self) -> None:
        pass

    def evaluate(self, artifact, ctx):  # type: ignore[no-untyped-def]
        return {"m": 0.0}


class _Experiment:
    def run(self, harness, ctx):  # type: ignore[no-untyped-def]
        return object()


def test_budget_exceeded_any_axis() -> None:
    b = Budget(wall_clock_s=10, steps=100, env_steps=1000)
    assert not b.exceeded(elapsed=5, steps=50, env_steps=500)
    assert b.exceeded(elapsed=10)
    assert b.exceeded(steps=100)
    assert b.exceeded(env_steps=1000)


def test_budget_progress_is_max_fraction_clamped() -> None:
    b = Budget(wall_clock_s=100, steps=10)
    assert b.progress(elapsed=50, steps=0) == 0.5
    assert b.progress(elapsed=0, steps=5) == 0.5
    assert b.progress(elapsed=200) == 1.0
    assert Budget().progress(elapsed=5) == 0.0


def test_run_context_record_accumulates() -> None:
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(steps=1), seed=0)
    ctx.record({"a": 1.0})
    ctx.record({"b": 2.0, "a": 3.0})
    assert ctx.telemetry == {"a": 3.0, "b": 2.0}


def test_is_better_respects_direction() -> None:
    assert is_better(0.9, 1.0, "min")
    assert not is_better(1.1, 1.0, "min")
    assert is_better(500.0, 480.0, "max")
    assert not is_better(470.0, 480.0, "max")


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_Harness(), Harness)
    assert isinstance(_Experiment(), Experiment)

    class Partial:
        def prepare(self) -> None:
            pass

    assert not isinstance(Partial(), Harness)
    assert not isinstance(Partial(), Experiment)
