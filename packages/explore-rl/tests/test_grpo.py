import torch

from explore_rl.algorithms.grpo import GRPOTrainer
from explore_rl.core.config import GenerationConfig, GRPOConfig
from explore_rl.models.gpt import GPT
from explore_rl.rewards.rule_based import LengthReward


class TestGRPOTrainer:
    def test_train_step(
        self,
        tiny_gpt: GPT,
        grpo_config: GRPOConfig,
        simple_tokenizer,
    ) -> None:
        ref_model = GPT(tiny_gpt.config)
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)
        reward_fn = LengthReward(target_length=20, tolerance=10)

        generation_config = GenerationConfig(
            max_new_tokens=5,
            do_sample=True,
            temperature=1.0,
        )

        trainer = GRPOTrainer(
            model=tiny_gpt,
            ref_model=ref_model,
            optimizer=optimizer,
            reward_fn=reward_fn,
            tokenizer_encode=simple_tokenizer.encode,
            tokenizer_decode=simple_tokenizer.decode,
            config=grpo_config,
            generation_config=generation_config,
        )

        prompts = ["test prompt one", "test prompt two"]

        metrics = trainer.train_step(prompts)

        assert metrics.loss != 0
        assert metrics.policy_loss != 0
        assert "mean_reward" in metrics.extras

    def test_group_advantages(
        self,
        tiny_gpt: GPT,
        grpo_config: GRPOConfig,
        simple_tokenizer,
    ) -> None:
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)
        reward_fn = LengthReward(target_length=20, tolerance=10)

        trainer = GRPOTrainer(
            model=tiny_gpt,
            ref_model=None,
            optimizer=optimizer,
            reward_fn=reward_fn,
            tokenizer_encode=simple_tokenizer.encode,
            tokenizer_decode=simple_tokenizer.decode,
            config=grpo_config,
        )

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        batch_size = 4
        advantages = trainer.compute_group_advantages(rewards, batch_size)

        assert advantages.shape == (8,)
        assert advantages.mean().abs() < 0.1

    def test_no_ref_model(
        self,
        tiny_gpt: GPT,
        grpo_config: GRPOConfig,
        simple_tokenizer,
    ) -> None:
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)
        reward_fn = LengthReward(target_length=20, tolerance=10)

        generation_config = GenerationConfig(
            max_new_tokens=5,
            do_sample=True,
        )

        trainer = GRPOTrainer(
            model=tiny_gpt,
            ref_model=None,
            optimizer=optimizer,
            reward_fn=reward_fn,
            tokenizer_encode=simple_tokenizer.encode,
            tokenizer_decode=simple_tokenizer.decode,
            config=grpo_config,
            generation_config=generation_config,
        )

        prompts = ["test"]

        metrics = trainer.train_step(prompts)

        assert metrics.loss != 0
        assert metrics.kl_divergence == 0.0

    def test_different_group_sizes(
        self,
        tiny_gpt: GPT,
        simple_tokenizer,
    ) -> None:
        for group_size in [2, 4]:
            config = GRPOConfig(group_size=group_size)
            optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)
            reward_fn = LengthReward(target_length=20, tolerance=10)

            generation_config = GenerationConfig(
                max_new_tokens=5,
                do_sample=True,
            )

            trainer = GRPOTrainer(
                model=tiny_gpt,
                ref_model=None,
                optimizer=optimizer,
                reward_fn=reward_fn,
                tokenizer_encode=simple_tokenizer.encode,
                tokenizer_decode=simple_tokenizer.decode,
                config=config,
                generation_config=generation_config,
            )

            prompts = ["test"]
            metrics = trainer.train_step(prompts)

            assert metrics.loss != 0
