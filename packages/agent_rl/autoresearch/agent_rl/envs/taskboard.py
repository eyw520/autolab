from collections.abc import Callable
import random
from typing import Any

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.types import StepResult
from autoresearch.agent_rl.verifiers.state import Change, StateSnapshot


class TaskBoardEnv:
    def __init__(
        self,
        seed: int,
        num_items: int,
        goal_size: int | None = None,
        ordered: bool = False,
    ) -> None:
        self._num_items = num_items
        self._goal_size = goal_size
        self._ordered = ordered
        self._max_steps = num_items
        self._goal: list[int] = []
        self._statuses: list[int] = []
        self._steps = 0
        self._reset_state(seed)

    def _reset_state(self, seed: int) -> None:
        rng = random.Random(seed)
        goal_size = self._goal_size
        if goal_size is None:
            goal = [0] * self._num_items
            while not any(goal):
                goal = [rng.randint(0, 1) for _ in range(self._num_items)]
        else:
            chosen = set(rng.sample(range(self._num_items), goal_size))
            goal = [1 if i in chosen else 0 for i in range(self._num_items)]
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
        out_of_order = (
            self._ordered
            and self._goal[item] == 1
            and any(self._goal[j] == 1 and self._statuses[j] == 0 for j in range(item))
        )
        self._statuses[item] = 1
        done = off_goal or out_of_order or self._statuses == self._goal or self._steps >= self._max_steps
        return StepResult(observation=self._observe(), reward=0.0, done=done)

    def close(self) -> None:
        pass

    def is_solved(self) -> bool:
        return self._statuses == self._goal

    @property
    def goal(self) -> list[int]:
        return list(self._goal)

    def snapshot(self, which: str = "current") -> StateSnapshot:
        statuses = [0] * self._num_items if which == "seed" else self._statuses
        return StateSnapshot({"items": {str(i): {"done": statuses[i]} for i in range(self._num_items)}})

    def _observe(self) -> dict[str, Any]:
        return {"statuses": list(self._statuses), "goal": list(self._goal), "step": self._steps}


def make_taskboard_env_factory(
    num_items: int = 3,
    goal_size: int | None = None,
    ordered: bool = False,
) -> Callable[[int], AgentEnv]:
    def factory(seed: int) -> AgentEnv:
        return TaskBoardEnv(seed=seed, num_items=num_items, goal_size=goal_size, ordered=ordered)

    return factory


def taskboard_verifier(env: AgentEnv) -> None:
    if not isinstance(env, TaskBoardEnv):
        raise TypeError("taskboard_verifier requires a TaskBoardEnv")
    expected = [Change("items", str(i), "done", 1) for i, g in enumerate(env.goal) if g]
    env.snapshot("seed").diff(env.snapshot("current")).expect_only(expected)
