from typing import Any

from explore_rl.rewards.base import RewardFunction
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class RewardModelHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedRewardModel(RewardFunction, nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        head: nn.Module | None = None,
        hidden_size: int = 768,
    ) -> None:
        nn.Module.__init__(self)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head or RewardModelHead(hidden_size)

    def forward_encoded(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if hasattr(self.backbone, "output_hidden_states"):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = self.backbone(input_ids, attention_mask)
            if hasattr(hidden_states, "last_hidden_state"):
                hidden_states = hidden_states.last_hidden_state

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]

        rewards = self.head(last_hidden).squeeze(-1)
        return rewards

    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        texts = [p + r for p, r in zip(prompts, responses)]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            rewards = self.forward_encoded(input_ids, attention_mask)

        return rewards.cpu()


class BradleyTerryTrainer:
    def __init__(
        self,
        model: LearnedRewardModel,
        optimizer: Optimizer,
        margin: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.margin = margin

    def train_step(
        self,
        chosen_ids: Tensor,
        chosen_mask: Tensor,
        rejected_ids: Tensor,
        rejected_mask: Tensor,
    ) -> dict[str, float]:
        self.model.train()

        chosen_rewards = self.model.forward_encoded(chosen_ids, chosen_mask)
        rejected_rewards = self.model.forward_encoded(rejected_ids, rejected_mask)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards - self.margin).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

    def train_step_from_pairs(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[str],
    ) -> dict[str, float]:
        chosen_texts = [p + r for p, r in zip(prompts, chosen_responses)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]

        tokenizer = self.model.tokenizer
        device = next(self.model.parameters()).device

        chosen_encoded = tokenizer(chosen_texts, padding=True, truncation=True, return_tensors="pt")
        rejected_encoded = tokenizer(rejected_texts, padding=True, truncation=True, return_tensors="pt")

        return self.train_step(
            chosen_encoded["input_ids"].to(device),
            chosen_encoded["attention_mask"].to(device),
            rejected_encoded["input_ids"].to(device),
            rejected_encoded["attention_mask"].to(device),
        )
