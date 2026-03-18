from collections.abc import Callable
from typing import Any

from explore_rl.core.config import GenerationConfig, GRPOConfig
from explore_rl.core.types import TrainingMetrics
from explore_rl.models.base import BaseLanguageModel
from explore_rl.rewards.base import RewardFunction
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class GRPOTrainer:
    def __init__(
        self,
        model: BaseLanguageModel,
        ref_model: nn.Module | None,
        optimizer: Optimizer,
        reward_fn: RewardFunction,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: GRPOConfig,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config
        self.generation_config = generation_config or GenerationConfig()

    def get_log_probs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)

        logits: Any = model(full_ids, full_mask)

        prompt_length = prompt_ids.size(1)
        response_logits = logits[:, prompt_length - 1 : -1, :]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        return token_log_probs

    @torch.no_grad()
    def get_ref_log_probs(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        if self.ref_model is None:
            return torch.zeros_like(response_ids, dtype=torch.float32)

        return self.get_log_probs(
            self.ref_model,
            prompt_ids,
            prompt_mask,
            response_ids,
            response_mask,
        )

    @torch.no_grad()
    def generate_group(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, list[str], list[str]]:
        batch_size = len(prompts)
        group_size = self.config.group_size

        expanded_prompts = [p for p in prompts for _ in range(group_size)]

        encoded = [self.tokenizer_encode(p) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)

        prompt_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
        prompt_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)

        for i, e in enumerate(encoded):
            prompt_ids[i, -len(e) :] = torch.tensor(e, dtype=torch.long)
            prompt_mask[i, -len(e) :] = 1

        expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
        expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

        gen_output = self.model.generate(expanded_prompt_ids, expanded_prompt_mask, self.generation_config)

        response_ids = gen_output.response_ids
        response_mask = torch.ones_like(response_ids)

        if self.generation_config.eos_token_id is not None:
            eos_positions = (response_ids == self.generation_config.eos_token_id).long()
            for i in range(response_ids.size(0)):
                eos_idx = eos_positions[i].nonzero()
                if len(eos_idx) > 0:
                    response_mask[i, eos_idx[0] + 1 :] = 0

        responses = [self.tokenizer_decode(ids.tolist()) for ids in response_ids]

        return (
            expanded_prompt_ids,
            expanded_prompt_mask,
            response_ids,
            response_mask,
            gen_output.log_probs,
            expanded_prompts,
            responses,
        )

    def compute_group_advantages(
        self,
        rewards: Tensor,
        batch_size: int,
    ) -> Tensor:
        group_size = self.config.group_size
        rewards_grouped = rewards.view(batch_size, group_size)

        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)

        if self.config.normalize_advantages:
            advantages = (rewards_grouped - group_mean) / (group_std + 1e-8)
        else:
            advantages = rewards_grouped - group_mean

        return advantages.view(-1)

    def train_step(self, prompts: list[str]) -> TrainingMetrics:
        self.model.train()
        device = next(self.model.parameters()).device
        batch_size = len(prompts)

        with torch.no_grad():
            (
                prompt_ids,
                prompt_mask,
                response_ids,
                response_mask,
                old_log_probs,
                expanded_prompts,
                responses,
            ) = self.generate_group(prompts, device)

            rewards = self.reward_fn(expanded_prompts, responses).to(device)
            advantages = self.compute_group_advantages(rewards, batch_size)

            ref_log_probs = self.get_ref_log_probs(prompt_ids, prompt_mask, response_ids, response_mask)

        token_log_probs = self.get_log_probs(self.model, prompt_ids, prompt_mask, response_ids, response_mask)

        masked_token_log_probs = token_log_probs * response_mask
        seq_log_probs = masked_token_log_probs.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)

        masked_old_log_probs = old_log_probs * response_mask if old_log_probs is not None else None
        if masked_old_log_probs is not None:
            old_seq_log_probs = masked_old_log_probs.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
        else:
            old_seq_log_probs = seq_log_probs.detach()

        ratio = torch.exp(seq_log_probs - old_seq_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        if self.ref_model is not None:
            masked_ref_log_probs = ref_log_probs * response_mask
            ref_seq_log_probs = masked_ref_log_probs.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
            kl = (ref_seq_log_probs - seq_log_probs).mean()
        else:
            kl = torch.tensor(0.0, device=device)

        loss = policy_loss + self.config.beta * kl

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return TrainingMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_divergence=kl.item(),
            extras={
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "mean_advantage": advantages.mean().item(),
            },
        )
