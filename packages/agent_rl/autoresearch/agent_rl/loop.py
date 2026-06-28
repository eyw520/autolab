from dataclasses import dataclass

from autoresearch.agent_rl.algo.grpo import GRPOTrainer, UnconfiguredTrainer
from autoresearch.agent_rl.reward.protocol import Reward
from autoresearch.agent_rl.rollout import EnvFactory, RolloutEngine
from autoresearch.agent_rl.types import Policy

from autoresearch.interface import RunContext


@dataclass
class AgentRLConfig:
    episodes_per_batch: int
    max_steps_per_episode: int
    num_parallel_envs: int = 1
    group_size: int = 1


def agent_rl_loop(
    policy: Policy,
    env_factory: EnvFactory,
    reward: Reward,
    ctx: RunContext,
    config: AgentRLConfig,
    trainer: GRPOTrainer | None = None,
) -> Policy:
    trainer = trainer or UnconfiguredTrainer()
    engine = RolloutEngine(env_factory, reward, config.max_steps_per_episode, config.num_parallel_envs)

    total_env_steps = 0
    iteration = 0
    while not ctx.budget.exceeded(env_steps=total_env_steps):
        base_seed = ctx.seed + iteration * config.episodes_per_batch
        groups = engine.collect_groups(policy, config.episodes_per_batch, config.group_size, base_seed)
        trajectories = [t for group in groups for t in group]
        total_env_steps += sum(t.num_steps for t in trajectories)
        metrics = trainer.update(policy, groups)
        mean_reward = sum(t.total_reward for t in trajectories) / max(1, len(trajectories))
        iteration += 1
        ctx.log_step({"train_reward": mean_reward, "env_steps": float(total_env_steps), **metrics})
        print(
            f"\riter {iteration:03d} | env_steps {total_env_steps:6d} | train_reward {mean_reward:6.2f}    ",
            end="",
            flush=True,
        )
    print()
    return policy
