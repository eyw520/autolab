import re

from explore_rl.rewards.base import RewardFunction
import torch
from torch import Tensor


class LengthReward(RewardFunction):
    def __init__(
        self,
        target_length: int = 100,
        tolerance: int = 20,
        penalty_scale: float = 0.01,
    ) -> None:
        self.target_length = target_length
        self.tolerance = tolerance
        self.penalty_scale = penalty_scale

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        rewards = []
        for response in responses:
            length = len(response)
            diff = abs(length - self.target_length)
            if diff <= self.tolerance:
                reward = 1.0
            else:
                reward = 1.0 - self.penalty_scale * (diff - self.tolerance)
            rewards.append(max(0.0, reward))
        return torch.tensor(rewards, dtype=torch.float32)


class FormatReward(RewardFunction):
    def __init__(
        self,
        pattern: str,
        reward_match: float = 1.0,
        reward_no_match: float = 0.0,
    ) -> None:
        self.pattern = re.compile(pattern)
        self.reward_match = reward_match
        self.reward_no_match = reward_no_match

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        rewards = []
        for response in responses:
            if self.pattern.search(response):
                rewards.append(self.reward_match)
            else:
                rewards.append(self.reward_no_match)
        return torch.tensor(rewards, dtype=torch.float32)


class KeywordReward(RewardFunction):
    def __init__(
        self,
        required_keywords: list[str] | None = None,
        forbidden_keywords: list[str] | None = None,
        reward_per_keyword: float = 0.2,
        penalty_per_keyword: float = 0.5,
        case_sensitive: bool = False,
    ) -> None:
        self.required_keywords = required_keywords or []
        self.forbidden_keywords = forbidden_keywords or []
        self.reward_per_keyword = reward_per_keyword
        self.penalty_per_keyword = penalty_per_keyword
        self.case_sensitive = case_sensitive

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        rewards = []
        for response in responses:
            check_text = response if self.case_sensitive else response.lower()
            reward = 0.0

            for keyword in self.required_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword in check_text:
                    reward += self.reward_per_keyword

            for keyword in self.forbidden_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword in check_text:
                    reward -= self.penalty_per_keyword

            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


class LengthPenaltyReward(RewardFunction):
    def __init__(self, max_length: int = 500, penalty_per_char: float = 0.001) -> None:
        self.max_length = max_length
        self.penalty_per_char = penalty_per_char

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        rewards = []
        for response in responses:
            excess = max(0, len(response) - self.max_length)
            reward = -excess * self.penalty_per_char
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


class RepetitionPenalty(RewardFunction):
    def __init__(self, ngram_size: int = 3, penalty_scale: float = 0.1) -> None:
        self.ngram_size = ngram_size
        self.penalty_scale = penalty_scale

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        rewards = []
        for response in responses:
            words = response.lower().split()
            if len(words) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams: list[tuple[str, ...]] = []
            for i in range(len(words) - self.ngram_size + 1):
                ngrams.append(tuple(words[i : i + self.ngram_size]))

            unique_ngrams = set(ngrams)
            if len(ngrams) > 0:
                repetition_ratio = 1 - len(unique_ngrams) / len(ngrams)
                reward = -repetition_ratio * self.penalty_scale
            else:
                reward = 0.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
