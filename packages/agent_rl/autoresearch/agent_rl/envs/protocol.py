from typing import Any, Protocol, runtime_checkable

from autoresearch.agent_rl.types import StepResult


@runtime_checkable
class AgentEnv(Protocol):
    def reset(self, seed: int | None = None) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> StepResult: ...

    def close(self) -> None: ...
