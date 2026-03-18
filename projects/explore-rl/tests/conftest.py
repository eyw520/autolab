import pytest
import torch

from explore_rl.core.config import (
    DPOConfig,
    GenerationConfig,
    GRPOConfig,
    ModelConfig,
    PPOConfig,
)
from explore_rl.models.gpt import GPT, GPTWithValueHead


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=64,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def tiny_gpt(tiny_config: ModelConfig) -> GPT:
    return GPT(tiny_config)


@pytest.fixture
def tiny_gpt_with_value(tiny_config: ModelConfig) -> GPTWithValueHead:
    return GPTWithValueHead(tiny_config)


@pytest.fixture
def generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=1,
    )


@pytest.fixture
def ppo_config() -> PPOConfig:
    return PPOConfig(
        clip_eps=0.2,
        value_clip_eps=0.2,
        kl_coef=0.1,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=1.0,
        gae_lambda=0.95,
        normalize_advantages=True,
        max_grad_norm=0.5,
        epochs_per_update=2,
        minibatch_size=2,
    )


@pytest.fixture
def dpo_config() -> DPOConfig:
    return DPOConfig(
        beta=0.1,
        label_smoothing=0.0,
        reference_free=False,
        max_grad_norm=1.0,
    )


@pytest.fixture
def grpo_config() -> GRPOConfig:
    return GRPOConfig(
        beta=0.1,
        group_size=2,
        normalize_advantages=True,
        max_grad_norm=1.0,
        clip_eps=0.2,
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def simple_tokenizer():
    class SimpleTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 100 for c in text[:20]]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i + 32) for i in ids if i > 0)

    return SimpleTokenizer()
