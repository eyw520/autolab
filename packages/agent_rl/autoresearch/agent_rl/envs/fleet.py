from typing import Any

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.rollout import EnvFactory
from autoresearch.agent_rl.types import StepResult


class FleetEnv:
    def __init__(self, instance: Any) -> None:
        self._instance = instance

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        response = self._instance.reset(seed=seed)
        if isinstance(response, dict):
            return response
        return {"reset": response}

    def step(self, action: dict[str, Any]) -> StepResult:
        observation, reward, done = self._instance.step(action)
        return StepResult(observation=observation, reward=float(reward), done=bool(done))

    def close(self) -> None:
        self._instance.close()


def make_fleet_env_factory(env_key: str) -> EnvFactory:
    def factory(seed: int) -> AgentEnv:
        try:
            import fleet
        except ImportError as error:
            raise ImportError(
                "Fleet backend requires the 'fleet' extra: pip install 'autoresearch-agent-rl[fleet]'"
            ) from error
        instance = fleet.Fleet().make(env_key)
        return FleetEnv(instance)

    return factory
