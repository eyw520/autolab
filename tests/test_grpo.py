import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from autoresearch.agent_rl.algo.grpo_torch import TorchGRPOTrainer
from autoresearch.agent_rl.envs.taskboard import make_taskboard_env_factory
from autoresearch.agent_rl.loop import AgentRLConfig, agent_rl_loop
from autoresearch.agent_rl.rollout import RolloutEngine
from autoresearch.agent_rl.types import TrainablePolicy
from autoresearch.interface import Budget, RunContext


class _LinearPolicy(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.net = nn.Linear(2 * num_items, num_items)

    def _logits(self, observation):
        features = torch.tensor(observation["statuses"] + observation["goal"], dtype=torch.float32)
        return self.net(features)

    def act(self, observation):
        item = int(Categorical(logits=self._logits(observation)).sample().item())
        return {"tool": "complete", "item": item}

    def action_log_prob(self, observation, action):
        return Categorical(logits=self._logits(observation)).log_prob(torch.tensor(int(action["item"])))


def _reward(env, trajectory):
    return 1.0 if env.is_solved() else 0.0


def test_linear_policy_is_trainable():
    assert isinstance(_LinearPolicy(3), TrainablePolicy)


def test_grpo_learns_taskboard():
    torch.manual_seed(0)
    random.seed(0)
    num_items = 3
    factory = make_taskboard_env_factory(num_items)
    policy = _LinearPolicy(num_items)
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(env_steps=4000), seed=0)
    agent_rl_loop(
        policy,
        factory,
        _reward,
        ctx,
        AgentRLConfig(episodes_per_batch=16, max_steps_per_episode=num_items, num_parallel_envs=1, group_size=8),
        trainer=TorchGRPOTrainer(lr=0.1),
    )
    trajectories = RolloutEngine(factory, _reward, max_steps=num_items).collect(policy, 40, base_seed=10_000)
    eval_return = sum(t.total_reward for t in trajectories) / len(trajectories)
    assert eval_return >= 0.8


def test_grpo_with_reference_kl():
    torch.manual_seed(0)
    random.seed(0)
    num_items = 3
    factory = make_taskboard_env_factory(num_items)
    policy = _LinearPolicy(num_items)
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(env_steps=4000), seed=0)
    agent_rl_loop(
        policy,
        factory,
        _reward,
        ctx,
        AgentRLConfig(episodes_per_batch=16, max_steps_per_episode=num_items, num_parallel_envs=1, group_size=8),
        trainer=TorchGRPOTrainer(lr=0.1, kl_coef=0.05),
    )
    assert "kl" in ctx.telemetry
    trajectories = RolloutEngine(factory, _reward, max_steps=num_items).collect(policy, 40, base_seed=10_000)
    eval_return = sum(t.total_reward for t in trajectories) / len(trajectories)
    assert eval_return >= 0.7
