import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from autoresearch.agent_rl.algo.grpo_torch import TorchGRPOTrainer
from autoresearch.agent_rl.curriculum import Stage, run_curriculum
from autoresearch.agent_rl.envs.taskboard import make_taskboard_env_factory, taskboard_verifier
from autoresearch.agent_rl.verifiers.reward import VerifierReward
from autoresearch.interface import Budget, RunContext


class _Policy(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * num_items, 32), nn.Tanh(), nn.Linear(32, num_items))

    def _logits(self, observation):
        return self.net(torch.tensor(observation["statuses"] + observation["goal"], dtype=torch.float32))

    def act(self, observation):
        return {"tool": "complete", "item": int(Categorical(logits=self._logits(observation)).sample().item())}

    def action_log_prob(self, observation, action):
        return Categorical(logits=self._logits(observation)).log_prob(torch.tensor(int(action["item"])))


def test_curriculum_advances_and_learns_first_stage():
    torch.manual_seed(0)
    random.seed(0)
    num_items = 4
    reward = VerifierReward(taskboard_verifier)
    policy = _Policy(num_items)
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(env_steps=1), seed=0)
    stages = [
        Stage(f"goal{k}", make_taskboard_env_factory(num_items, goal_size=k, ordered=True), num_items, Budget(env_steps=2500))
        for k in range(1, 3)
    ]
    results = run_curriculum(
        policy,
        reward,
        stages,
        ctx,
        TorchGRPOTrainer(lr=0.1),
        episodes_per_batch=16,
        group_size=8,
        num_parallel_envs=1,
        eval_episodes=40,
        eval_seed_base=10_000,
        advance_threshold=0.7,
    )
    assert len(results) >= 1
    assert results[0][0] == "goal1"
    assert results[0][1] >= 0.8
