from typing import Any

from autoresearch.agent_rl.types import TrainablePolicy, Trajectory


class TorchGRPOTrainer:
    def __init__(self, lr: float = 0.1) -> None:
        self._lr = lr
        self._optimizer: Any = None

    def update(self, policy: TrainablePolicy, groups: list[list[Trajectory]]) -> dict[str, float]:
        import torch

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(policy.parameters(), lr=self._lr)

        log_probs = []
        advantages = []
        rewards = []
        for group in groups:
            group_rewards = torch.tensor([t.total_reward for t in group], dtype=torch.float32)
            rewards.append(group_rewards)
            group_advantage = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            for trajectory, advantage in zip(group, group_advantage):
                if not trajectory.transitions:
                    continue
                log_prob = torch.stack(
                    [policy.action_log_prob(t.observation, t.action) for t in trajectory.transitions]
                ).sum()
                log_probs.append(log_prob)
                advantages.append(advantage)

        if not log_probs:
            return {"loss": 0.0, "mean_reward": 0.0}

        loss = -(torch.stack(log_probs) * torch.stack(advantages).detach()).mean()
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        return {"loss": float(loss.item()), "mean_reward": float(torch.cat(rewards).mean().item())}
