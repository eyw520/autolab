from typing import Protocol, runtime_checkable

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.types import Trajectory


@runtime_checkable
class Reward(Protocol):
    def __call__(self, env: AgentEnv, trajectory: Trajectory) -> float: ...
