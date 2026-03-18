from typing import Any

from explore_rl.core.config import DPOConfig
from explore_rl.core.types import PreferencePair, TrainingMetrics
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class DPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        optimizer: Optimizer,
        config: DPOConfig,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config

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

        masked_log_probs = token_log_probs * response_mask
        seq_log_probs = masked_log_probs.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)

        return seq_log_probs

    @torch.no_grad()
    def get_ref_log_probs(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        if self.ref_model is None:
            return torch.zeros(prompt_ids.size(0), device=prompt_ids.device)

        return self.get_log_probs(
            self.ref_model,
            prompt_ids,
            prompt_mask,
            response_ids,
            response_mask,
        )

    def compute_dpo_loss(
        self,
        pi_chosen: Tensor,
        pi_rejected: Tensor,
        ref_chosen: Tensor,
        ref_rejected: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected

        if self.config.reference_free:
            logits = self.config.beta * pi_logratios
        else:
            logits = self.config.beta * (pi_logratios - ref_logratios)

        if self.config.label_smoothing > 0:
            losses = (
                -F.logsigmoid(logits) * (1 - self.config.label_smoothing)
                - F.logsigmoid(-logits) * self.config.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        chosen_rewards = self.config.beta * (pi_chosen - ref_chosen).detach()
        rejected_rewards = self.config.beta * (pi_rejected - ref_rejected).detach()

        metrics = {
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "accuracy": (logits > 0).float().mean(),
            "logits": logits.mean(),
        }

        return losses.mean(), metrics

    def train_step(self, batch: PreferencePair) -> TrainingMetrics:
        self.model.train()

        pi_chosen = self.get_log_probs(
            self.model,
            batch.prompt_ids,
            batch.prompt_mask,
            batch.chosen_ids,
            batch.chosen_mask,
        )

        pi_rejected = self.get_log_probs(
            self.model,
            batch.prompt_ids,
            batch.prompt_mask,
            batch.rejected_ids,
            batch.rejected_mask,
        )

        ref_chosen = self.get_ref_log_probs(
            batch.prompt_ids,
            batch.prompt_mask,
            batch.chosen_ids,
            batch.chosen_mask,
        )

        ref_rejected = self.get_ref_log_probs(
            batch.prompt_ids,
            batch.prompt_mask,
            batch.rejected_ids,
            batch.rejected_mask,
        )

        loss, metrics = self.compute_dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return TrainingMetrics(
            loss=loss.item(),
            extras={
                "chosen_rewards": metrics["chosen_rewards"].item(),
                "rejected_rewards": metrics["rejected_rewards"].item(),
                "reward_margin": metrics["reward_margin"].item(),
                "accuracy": metrics["accuracy"].item(),
            },
        )

    def get_implicit_reward(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            pi_log_probs = self.get_log_probs(
                self.model,
                prompt_ids,
                prompt_mask,
                response_ids,
                response_mask,
            )

            ref_log_probs = self.get_ref_log_probs(
                prompt_ids,
                prompt_mask,
                response_ids,
                response_mask,
            )

            reward = self.config.beta * (pi_log_probs - ref_log_probs)

        return reward
