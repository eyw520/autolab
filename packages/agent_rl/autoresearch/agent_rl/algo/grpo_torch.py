import copy
from typing import Any

from autoresearch.agent_rl.types import TrainablePolicy, Trajectory


class TorchGRPOTrainer:
    def __init__(self, lr: float = 0.1, kl_coef: float = 0.0) -> None:
        self._lr = lr
        self._kl_coef = kl_coef
        self._optimizer: Any = None
        self._reference: Any = None

    def update(self, policy: TrainablePolicy, groups: list[list[Trajectory]]) -> dict[str, float]:
        import torch

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(policy.parameters(), lr=self._lr)
            if self._kl_coef > 0.0:
                self._reference = copy.deepcopy(policy)

        log_probs = []
        advantages = []
        kls = []
        rewards = []
        for group in groups:
            group_rewards = torch.tensor([t.total_reward for t in group], dtype=torch.float32)
            rewards.append(group_rewards)
            group_advantage = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            for trajectory, advantage in zip(group, group_advantage):
                if not trajectory.transitions:
                    continue
                current = [policy.action_log_prob(t.observation, t.action) for t in trajectory.transitions]
                log_probs.append(torch.stack(current).sum())
                advantages.append(advantage)
                if self._reference is not None:
                    for transition, current_log_prob in zip(trajectory.transitions, current):
                        with torch.no_grad():
                            ref_log_prob = self._reference.action_log_prob(transition.observation, transition.action)
                        log_ratio = ref_log_prob - current_log_prob
                        kls.append(torch.exp(log_ratio) - log_ratio - 1.0)

        if not log_probs:
            return {"loss": 0.0, "mean_reward": 0.0}

        policy_loss = -(torch.stack(log_probs) * torch.stack(advantages).detach()).mean()
        metrics = {"mean_reward": float(torch.cat(rewards).mean().item())}
        loss = policy_loss
        if kls:
            kl = torch.stack(kls).mean()
            loss = policy_loss + self._kl_coef * kl
            metrics["kl"] = float(kl.item())

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        metrics["loss"] = float(loss.item())
        return metrics
