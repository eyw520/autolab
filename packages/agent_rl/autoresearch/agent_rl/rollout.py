from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.reward.protocol import Reward
from autoresearch.agent_rl.types import Policy, Trajectory, Transition


EnvFactory = Callable[[int], AgentEnv]


class RolloutEngine:
    def __init__(
        self,
        env_factory: EnvFactory,
        reward: Reward,
        max_steps: int,
        num_parallel: int = 1,
    ) -> None:
        self._env_factory = env_factory
        self._reward = reward
        self._max_steps = max_steps
        self._num_parallel = num_parallel

    def collect(self, policy: Policy, num_episodes: int, base_seed: int) -> list[Trajectory]:
        seeds = [base_seed + i for i in range(num_episodes)]
        if self._num_parallel <= 1:
            return [self._episode(policy, seed) for seed in seeds]
        with ThreadPoolExecutor(max_workers=self._num_parallel) as pool:
            return list(pool.map(lambda seed: self._episode(policy, seed), seeds))

    def _episode(self, policy: Policy, seed: int) -> Trajectory:
        env = self._env_factory(seed)
        trajectory = Trajectory()
        try:
            observation = env.reset(seed=seed)
            for _ in range(self._max_steps):
                action = policy.act(observation)
                result = env.step(action)
                trajectory.transitions.append(
                    Transition(
                        observation=observation,
                        action=action,
                        reward=result.reward,
                        done=result.done,
                    )
                )
                observation = result.observation
                if result.done:
                    break
            trajectory.total_reward = self._reward(env, trajectory)
        finally:
            env.close()
        return trajectory
