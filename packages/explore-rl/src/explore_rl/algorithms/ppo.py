from explore_rl.core.config import PPOConfig
from explore_rl.core.types import RolloutBatch, TrainingMetrics
from explore_rl.models.base import BaseLanguageModelWithValueHead
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer


class PPOTrainer:
    def __init__(
        self,
        model: BaseLanguageModelWithValueHead,
        ref_model: nn.Module | None,
        optimizer: Optimizer,
        config: PPOConfig,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config

    def compute_policy_loss(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ratio = torch.exp(log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2)

        masked_loss = (policy_loss * mask).sum() / mask.sum()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_eps).float()
            clip_fraction = (clip_fraction * mask).sum() / mask.sum()

        return masked_loss, clip_fraction

    def compute_value_loss(
        self,
        values: Tensor,
        old_values: Tensor,
        returns: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.config.value_clip_eps is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip_eps,
                self.config.value_clip_eps,
            )
            value_loss_1 = (values - returns) ** 2
            value_loss_2 = (values_clipped - returns) ** 2
            value_loss = torch.max(value_loss_1, value_loss_2)
        else:
            value_loss = (values - returns) ** 2

        return (value_loss * mask).sum() / mask.sum() * 0.5

    def compute_entropy(self, logits: Tensor, mask: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return (entropy * mask).sum() / mask.sum()

    def compute_kl_divergence(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        kl = ref_log_probs - log_probs
        return (kl * mask).sum() / mask.sum()

    @torch.no_grad()
    def get_ref_log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_ids: Tensor,
        prompt_length: int,
    ) -> Tensor:
        if self.ref_model is None:
            return torch.zeros_like(response_ids, dtype=torch.float32)

        logits = self.ref_model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        response_log_probs = log_probs[:, prompt_length - 1 : -1, :].gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
        return response_log_probs

    def train_step(self, batch: RolloutBatch) -> TrainingMetrics:
        self.model.train()

        prompt_length = batch.prompt_ids.size(1)
        full_ids = torch.cat([batch.prompt_ids, batch.response_ids], dim=1)
        full_mask = torch.cat([batch.prompt_mask, batch.response_mask], dim=1)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        ref_log_probs = self.get_ref_log_probs(full_ids, full_mask, batch.response_ids, prompt_length)

        for _ in range(self.config.epochs_per_update):
            indices = torch.randperm(batch.prompt_ids.size(0))

            for start in range(0, len(indices), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]

                mb_full_ids = full_ids[mb_indices]
                mb_full_mask = full_mask[mb_indices]
                mb_response_ids = batch.response_ids[mb_indices]
                mb_response_mask = batch.response_mask[mb_indices]
                mb_old_log_probs = batch.old_log_probs[mb_indices]
                mb_advantages = batch.advantages[mb_indices]
                mb_returns = batch.returns[mb_indices]
                mb_old_values = batch.values[mb_indices] if batch.values is not None else None
                mb_ref_log_probs = ref_log_probs[mb_indices]

                logits, values = self.model.forward_with_value(mb_full_ids, mb_full_mask)

                response_logits = logits[:, prompt_length - 1 : -1, :]
                response_values = values[:, prompt_length - 1 : -1]

                log_probs = torch.log_softmax(response_logits, dim=-1)
                token_log_probs = log_probs.gather(2, mb_response_ids.unsqueeze(-1)).squeeze(-1)

                policy_loss, clip_frac = self.compute_policy_loss(
                    token_log_probs,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_response_mask.float(),
                )

                if mb_old_values is not None:
                    value_loss = self.compute_value_loss(
                        response_values,
                        mb_old_values,
                        mb_returns,
                        mb_response_mask.float(),
                    )
                else:
                    value_loss = torch.tensor(0.0, device=policy_loss.device)

                entropy = self.compute_entropy(response_logits, mb_response_mask.float())

                kl = self.compute_kl_divergence(
                    token_log_probs,
                    mb_ref_log_probs,
                    mb_response_mask.float(),
                )

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.kl_coef * kl
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                total_clip_frac += clip_frac.item()
                n_updates += 1

            if self.config.target_kl is not None:
                if total_kl / n_updates > self.config.target_kl:
                    break

        return TrainingMetrics(
            loss=total_policy_loss / n_updates + total_value_loss / n_updates,
            policy_loss=total_policy_loss / n_updates,
            value_loss=total_value_loss / n_updates,
            entropy=total_entropy / n_updates,
            kl_divergence=total_kl / n_updates,
            clip_fraction=total_clip_frac / n_updates,
        )
