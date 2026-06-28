import pytest
import torch

from autoresearch.agent_rl.algo.grpo import UnconfiguredTrainer
from autoresearch.agent_rl.envs.taskboard import make_taskboard_env_factory
from autoresearch.agent_rl.loop import AgentRLConfig, agent_rl_loop
from autoresearch.agent_rl.rollout import RolloutEngine
from autoresearch.agent_rl.types import Trajectory
from autoresearch.interface import Budget, RunContext


class _FixedPolicy:
    def act(self, observation):
        return {"tool": "complete", "item": 0}


class _NoopTrainer:
    def update(self, policy, groups):
        return {}


def _reward(env, trajectory):
    return 1.0


def test_collect_groups_shape():
    engine = RolloutEngine(make_taskboard_env_factory(3), _reward, max_steps=3)
    groups = engine.collect_groups(_FixedPolicy(), num_prompts=4, group_size=5, base_seed=0)
    assert len(groups) == 4
    assert all(len(group) == 5 for group in groups)


def test_collect_shape():
    engine = RolloutEngine(make_taskboard_env_factory(3), _reward, max_steps=3)
    trajectories = engine.collect(_FixedPolicy(), num_episodes=6, base_seed=10)
    assert len(trajectories) == 6
    assert all(isinstance(t, Trajectory) for t in trajectories)


def test_loop_respects_env_step_budget():
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(env_steps=30), seed=0)
    agent_rl_loop(
        _FixedPolicy(),
        make_taskboard_env_factory(3),
        _reward,
        ctx,
        AgentRLConfig(episodes_per_batch=4, max_steps_per_episode=3, group_size=2),
        trainer=_NoopTrainer(),
    )
    assert ctx.telemetry["env_steps"] >= 30


def test_unconfigured_trainer_raises():
    with pytest.raises(NotImplementedError):
        UnconfiguredTrainer().update(_FixedPolicy(), [])
