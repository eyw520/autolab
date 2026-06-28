from collections.abc import Callable
from typing import Protocol, runtime_checkable

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.types import Trajectory
from autoresearch.agent_rl.verifiers.state import StateSnapshot, VerificationError


@runtime_checkable
class SnapshotEnv(AgentEnv, Protocol):
    def snapshot(self, which: str) -> StateSnapshot: ...


Verifier = Callable[[AgentEnv], None]


class VerifierReward:
    def __init__(self, verifier: Verifier) -> None:
        self._verifier = verifier

    def __call__(self, env: AgentEnv, trajectory: Trajectory) -> float:
        try:
            self._verifier(env)
        except VerificationError:
            return 0.0
        return 1.0
