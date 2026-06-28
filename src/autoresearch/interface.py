from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch


Direction = Literal["min", "max"]


@dataclass
class Budget:
    wall_clock_s: float | None = None
    steps: int | None = None
    env_steps: int | None = None

    def exceeded(self, *, elapsed: float = 0.0, steps: int = 0, env_steps: int = 0) -> bool:
        if self.wall_clock_s is not None and elapsed >= self.wall_clock_s:
            return True
        if self.steps is not None and steps >= self.steps:
            return True
        if self.env_steps is not None and env_steps >= self.env_steps:
            return True
        return False

    def progress(self, *, elapsed: float = 0.0, steps: int = 0, env_steps: int = 0) -> float:
        fracs: list[float] = []
        if self.wall_clock_s:
            fracs.append(elapsed / self.wall_clock_s)
        if self.steps:
            fracs.append(steps / self.steps)
        if self.env_steps:
            fracs.append(env_steps / self.env_steps)
        if not fracs:
            return 0.0
        return min(1.0, max(fracs))


@dataclass
class HarnessSpec:
    primary_metric: str
    direction: Direction
    budget: Budget
    domain: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    device: torch.device
    budget: Budget
    seed: int
    telemetry: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)

    def record(self, metrics: dict[str, float]) -> None:
        self.telemetry.update(metrics)

    def log_step(self, metrics: dict[str, float]) -> None:
        self.history.append(dict(metrics))
        self.telemetry.update(metrics)


@runtime_checkable
class Harness(Protocol):
    @property
    def spec(self) -> HarnessSpec: ...

    def prepare(self) -> None: ...

    def evaluate(self, artifact: Any, ctx: RunContext) -> dict[str, float]: ...


@runtime_checkable
class Experiment(Protocol):
    def run(self, harness: Any, ctx: RunContext) -> Any: ...
