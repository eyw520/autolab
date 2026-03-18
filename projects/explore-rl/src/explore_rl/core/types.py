from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class TokenSequence:
    input_ids: Tensor
    attention_mask: Tensor

    def __post_init__(self) -> None:
        assert self.input_ids.dim() == 2
        assert self.attention_mask.shape == self.input_ids.shape

    @property
    def batch_size(self) -> int:
        return self.input_ids.size(0)

    @property
    def seq_len(self) -> int:
        return self.input_ids.size(1)

    def to(self, device: torch.device) -> "TokenSequence":
        return TokenSequence(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


@dataclass
class GenerationOutput:
    sequences: Tensor
    prompt_length: int
    log_probs: Tensor | None = None

    @property
    def response_ids(self) -> Tensor:
        return self.sequences[:, self.prompt_length :]

    @property
    def response_log_probs(self) -> Tensor | None:
        if self.log_probs is None:
            return None
        return self.log_probs[:, self.prompt_length - 1 : -1]


@dataclass
class RolloutBatch:
    prompt_ids: Tensor
    prompt_mask: Tensor
    response_ids: Tensor
    response_mask: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    advantages: Tensor
    returns: Tensor
    values: Tensor | None = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            prompt_ids=self.prompt_ids.to(device),
            prompt_mask=self.prompt_mask.to(device),
            response_ids=self.response_ids.to(device),
            response_mask=self.response_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            rewards=self.rewards.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            values=self.values.to(device) if self.values is not None else None,
        )


@dataclass
class PreferencePair:
    prompt_ids: Tensor
    prompt_mask: Tensor
    chosen_ids: Tensor
    chosen_mask: Tensor
    rejected_ids: Tensor
    rejected_mask: Tensor

    def to(self, device: torch.device) -> "PreferencePair":
        return PreferencePair(
            prompt_ids=self.prompt_ids.to(device),
            prompt_mask=self.prompt_mask.to(device),
            chosen_ids=self.chosen_ids.to(device),
            chosen_mask=self.chosen_mask.to(device),
            rejected_ids=self.rejected_ids.to(device),
            rejected_mask=self.rejected_mask.to(device),
        )


@dataclass
class TrainingMetrics:
    loss: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)
