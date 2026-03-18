import torch

from explore_rl.core.config import GenerationConfig, ModelConfig
from explore_rl.models.gpt import GPT, GPTWithValueHead


class TestGPT:
    def test_forward(self, tiny_gpt: GPT) -> None:
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        logits = tiny_gpt(input_ids, attention_mask)

        assert logits.shape == (batch_size, seq_len, 100)

    def test_forward_no_mask(self, tiny_gpt: GPT) -> None:
        input_ids = torch.randint(0, 100, (2, 10))
        logits = tiny_gpt(input_ids)
        assert logits.shape == (2, 10, 100)

    def test_generate(self, tiny_gpt: GPT) -> None:
        prompt_ids = torch.randint(0, 100, (2, 5))
        config = GenerationConfig(max_new_tokens=10, do_sample=True)

        output = tiny_gpt.generate(prompt_ids, config=config)

        assert output.sequences.shape[0] == 2
        assert output.sequences.shape[1] == 15
        assert output.prompt_length == 5
        assert output.log_probs is not None
        assert output.log_probs.shape == (2, 10)

    def test_generate_greedy(self, tiny_gpt: GPT) -> None:
        prompt_ids = torch.randint(0, 100, (1, 5))
        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        output = tiny_gpt.generate(prompt_ids, config=config)

        assert output.sequences.shape == (1, 10)

    def test_log_probs(self, tiny_gpt: GPT) -> None:
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        log_probs = tiny_gpt.log_probs(input_ids, attention_mask)

        assert log_probs.shape == (2, 9)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()


class TestGPTWithValueHead:
    def test_forward_with_value(self, tiny_gpt_with_value: GPTWithValueHead) -> None:
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        logits, values = tiny_gpt_with_value.forward_with_value(input_ids, attention_mask)

        assert logits.shape == (2, 10, 100)
        assert values.shape == (2, 10)

    def test_forward(self, tiny_gpt_with_value: GPTWithValueHead) -> None:
        input_ids = torch.randint(0, 100, (2, 10))
        logits = tiny_gpt_with_value(input_ids)
        assert logits.shape == (2, 10, 100)

    def test_generate(self, tiny_gpt_with_value: GPTWithValueHead) -> None:
        prompt_ids = torch.randint(0, 100, (2, 5))
        config = GenerationConfig(max_new_tokens=5, do_sample=True)

        output = tiny_gpt_with_value.generate(prompt_ids, config=config)

        assert output.sequences.shape == (2, 10)
        assert output.prompt_length == 5


class TestModelConfig:
    def test_default_config(self) -> None:
        config = ModelConfig()
        assert config.vocab_size == 50257
        assert config.n_embd == 768
        assert config.n_head == 12
        assert config.n_layer == 12

    def test_custom_config(self) -> None:
        config = ModelConfig(vocab_size=1000, n_embd=128, n_head=4, n_layer=4)
        model = GPT(config)
        input_ids = torch.randint(0, 1000, (1, 10))
        logits = model(input_ids)
        assert logits.shape == (1, 10, 1000)
