from dataclasses import dataclass

from autoresearch.agent_rl.algo.grpo import GRPOTrainer
from autoresearch.agent_rl.loop import AgentRLConfig, agent_rl_loop
from autoresearch.agent_rl.reward.protocol import Reward
from autoresearch.agent_rl.rollout import EnvFactory, RolloutEngine
from autoresearch.agent_rl.types import Policy

from autoresearch.interface import Budget, RunContext


@dataclass
class Stage:
    label: str
    env_factory: EnvFactory
    max_steps: int
    budget: Budget


def run_curriculum(
    policy: Policy,
    reward: Reward,
    stages: list[Stage],
    ctx: RunContext,
    trainer: GRPOTrainer,
    *,
    episodes_per_batch: int,
    group_size: int = 1,
    num_parallel_envs: int = 1,
    eval_episodes: int = 40,
    eval_seed_base: int = 10_000,
    advance_threshold: float = 0.8,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for stage in stages:
        stage_ctx = RunContext(device=ctx.device, budget=stage.budget, seed=ctx.seed)
        agent_rl_loop(
            policy,
            stage.env_factory,
            reward,
            stage_ctx,
            AgentRLConfig(episodes_per_batch, stage.max_steps, num_parallel_envs, group_size),
            trainer,
        )
        engine = RolloutEngine(stage.env_factory, reward, stage.max_steps, num_parallel_envs)
        trajectories = engine.collect(policy, eval_episodes, eval_seed_base)
        score = sum(t.total_reward for t in trajectories) / max(1, len(trajectories))
        results.append((stage.label, score))
        ctx.record({f"return_{stage.label}": score})
        print(f"stage {stage.label}: eval_return {score:.3f}")
        if score < advance_threshold:
            break
    return results
