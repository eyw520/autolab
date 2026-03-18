from collections.abc import Callable

from explore_rl.core.config import GenerationConfig, PPOConfig
from explore_rl.core.types import RolloutBatch
from explore_rl.models.base import BaseLanguageModelWithValueHead
from explore_rl.rewards.base import RewardFunction
import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = torch.zeros(batch_size, device=rewards.device)
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


class RolloutCollector:
    def __init__(
        self,
        model: BaseLanguageModelWithValueHead,
        reward_fn: RewardFunction,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: PPOConfig,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config
        self.generation_config = generation_config or GenerationConfig()

    @torch.no_grad()
    def collect(
        self,
        prompts: list[str],
        device: torch.device | None = None,
    ) -> RolloutBatch:
        device = device or next(self.model.parameters()).device

        encoded = [self.tokenizer_encode(p) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)

        prompt_ids = torch.zeros(len(prompts), max_prompt_len, dtype=torch.long, device=device)
        prompt_mask = torch.zeros(len(prompts), max_prompt_len, dtype=torch.long, device=device)

        for i, e in enumerate(encoded):
            prompt_ids[i, -len(e) :] = torch.tensor(e, dtype=torch.long)
            prompt_mask[i, -len(e) :] = 1

        gen_output = self.model.generate(prompt_ids, prompt_mask, self.generation_config)

        response_ids = gen_output.response_ids
        response_mask = torch.ones_like(response_ids)

        if self.generation_config.eos_token_id is not None:
            eos_positions = (response_ids == self.generation_config.eos_token_id).long()
            for i in range(response_ids.size(0)):
                eos_idx = eos_positions[i].nonzero()
                if len(eos_idx) > 0:
                    response_mask[i, eos_idx[0] + 1 :] = 0

        responses = [self.tokenizer_decode(ids.tolist()) for ids in response_ids]
        rewards_final = self.reward_fn(prompts, responses)

        seq_len = response_ids.size(1)
        rewards = torch.zeros(len(prompts), seq_len, device=device)
        rewards[:, -1] = rewards_final.to(device)

        full_ids = gen_output.sequences
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)

        logits, values = self.model.forward_with_value(full_ids, full_mask)

        response_values = values[:, max_prompt_len - 1 : -1]

        log_probs = torch.log_softmax(logits, dim=-1)
        response_log_probs = log_probs[:, max_prompt_len - 1 : -1, :].gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        dones = torch.zeros_like(rewards)
        dones[:, -1] = 1.0
        if self.generation_config.eos_token_id is not None:
            eos_mask = response_ids == self.generation_config.eos_token_id
            dones = dones | eos_mask.float()

        advantages, returns = compute_gae(
            rewards,
            response_values,
            dones,
            self.config.gamma,
            self.config.gae_lambda,
        )

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            old_log_probs=response_log_probs,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
            values=response_values,
        )
