from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0

    @property
    def num_steps(self) -> int:
        return len(self.transitions)


@runtime_checkable
class Policy(Protocol):
    def act(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class TrainablePolicy(Policy, Protocol):
    def action_log_prob(self, observation: dict[str, Any], action: dict[str, Any]) -> Any: ...

    def parameters(self) -> Iterable[Any]: ...
