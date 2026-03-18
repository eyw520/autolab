from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from explore_rl.core.config import GenerationConfig
from explore_rl.core.types import GenerationOutput
import torch
from torch import Tensor
import torch.nn as nn


@runtime_checkable
class LanguageModel(Protocol):
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor: ...

    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput: ...

    def log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class LanguageModelWithValueHead(LanguageModel, Protocol):
    def forward_with_value(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class BaseLanguageModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        pass

    @abstractmethod
    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        pass

    def log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        logits = self.forward(input_ids, attention_mask)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return token_log_probs


class BaseLanguageModelWithValueHead(BaseLanguageModel):
    @abstractmethod
    def forward_with_value(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pass
