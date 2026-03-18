from explore_rl.rewards.base import RewardFunction
import torch
from torch import Tensor


class CompositeReward(RewardFunction):
    def __init__(
        self,
        rewards: list[tuple[RewardFunction, float]],
        normalize: bool = False,
    ) -> None:
        self.rewards = rewards
        self.normalize = normalize
        self._total_weight = sum(w for _, w in rewards)

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        combined = torch.zeros(len(responses), dtype=torch.float32)

        for reward_fn, weight in self.rewards:
            reward = reward_fn(prompts, responses)
            combined = combined + weight * reward

        if self.normalize and self._total_weight > 0:
            combined = combined / self._total_weight

        return combined

    def get_breakdown(self, prompts: list[str], responses: list[str]) -> dict[str, Tensor]:
        breakdown = {}
        for reward_fn, weight in self.rewards:
            reward = reward_fn(prompts, responses)
            breakdown[reward_fn.name] = reward
            breakdown[f"{reward_fn.name}_weighted"] = weight * reward
        breakdown["total"] = self(prompts, responses)
        return breakdown


class ThresholdReward(RewardFunction):
    def __init__(
        self,
        base_reward: RewardFunction,
        threshold: float,
        above_threshold: float = 1.0,
        below_threshold: float = 0.0,
    ) -> None:
        self.base_reward = base_reward
        self.threshold = threshold
        self.above_threshold = above_threshold
        self.below_threshold = below_threshold

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        base = self.base_reward(prompts, responses)
        return torch.where(
            base >= self.threshold,
            torch.full_like(base, self.above_threshold),
            torch.full_like(base, self.below_threshold),
        )


class ClippedReward(RewardFunction):
    def __init__(
        self,
        base_reward: RewardFunction,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ) -> None:
        self.base_reward = base_reward
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        base = self.base_reward(prompts, responses)
        return torch.clamp(base, self.min_value, self.max_value)


class ScaledReward(RewardFunction):
    def __init__(
        self,
        base_reward: RewardFunction,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        self.base_reward = base_reward
        self.scale = scale
        self.offset = offset

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        base = self.base_reward(prompts, responses)
        return self.scale * base + self.offset
