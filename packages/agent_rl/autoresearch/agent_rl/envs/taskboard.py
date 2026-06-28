from collections.abc import Callable
import random
from typing import Any

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.types import StepResult


class TaskBoardEnv:
    def __init__(self, seed: int, num_items: int) -> None:
        self._num_items = num_items
        self._max_steps = num_items
        self._goal: list[int] = []
        self._statuses: list[int] = []
        self._steps = 0
        self._reset_state(seed)

    def _reset_state(self, seed: int) -> None:
        rng = random.Random(seed)
        goal = [0] * self._num_items
        while not any(goal):
            goal = [rng.randint(0, 1) for _ in range(self._num_items)]
        self._goal = goal
        self._statuses = [0] * self._num_items
        self._steps = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._reset_state(seed)
        return self._observe()

    def step(self, action: dict[str, Any]) -> StepResult:
        item = int(action["item"])
        self._steps += 1
        off_goal = self._goal[item] == 0
        self._statuses[item] = 1
        done = off_goal or self._statuses == self._goal or self._steps >= self._max_steps
        return StepResult(observation=self._observe(), reward=0.0, done=done)

    def close(self) -> None:
        pass

    def is_solved(self) -> bool:
        return self._statuses == self._goal

    def _observe(self) -> dict[str, Any]:
        return {"statuses": list(self._statuses), "goal": list(self._goal), "step": self._steps}


def make_taskboard_env_factory(num_items: int = 3) -> Callable[[int], AgentEnv]:
    def factory(seed: int) -> AgentEnv:
        return TaskBoardEnv(seed=seed, num_items=num_items)

    return factory
