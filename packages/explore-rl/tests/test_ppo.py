import torch

from explore_rl.algorithms.ppo import PPOTrainer
from explore_rl.core.config import PPOConfig
from explore_rl.core.types import RolloutBatch
from explore_rl.models.gpt import GPTWithValueHead
from explore_rl.training.rollout import compute_gae


class TestComputeGAE:
    def test_basic_gae(self) -> None:
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        dones = torch.tensor([[0.0, 0.0, 1.0]])

        advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)

        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)
        assert returns[0, 2].item() == 1.0

    def test_discount(self) -> None:
        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.zeros(1, 3)
        dones = torch.tensor([[0.0, 0.0, 1.0]])

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.5, gae_lambda=1.0)

        assert advantages[0, 2].item() == 1.0
        assert advantages[0, 1].item() == 1.0 + 0.5 * 1.0
        assert advantages[0, 0].item() == 1.0 + 0.5 * (1.0 + 0.5 * 1.0)


class TestPPOTrainer:
    def test_train_step(
        self,
        tiny_gpt_with_value: GPTWithValueHead,
        ppo_config: PPOConfig,
    ) -> None:
        ref_model = GPTWithValueHead(tiny_gpt_with_value.config)
        optimizer = torch.optim.Adam(tiny_gpt_with_value.parameters(), lr=1e-4)

        trainer = PPOTrainer(
            model=tiny_gpt_with_value,
            ref_model=ref_model,
            optimizer=optimizer,
            config=ppo_config,
        )

        batch_size = 4
        prompt_len = 5
        response_len = 10

        batch = RolloutBatch(
            prompt_ids=torch.randint(0, 100, (batch_size, prompt_len)),
            prompt_mask=torch.ones(batch_size, prompt_len, dtype=torch.long),
            response_ids=torch.randint(0, 100, (batch_size, response_len)),
            response_mask=torch.ones(batch_size, response_len, dtype=torch.long),
            old_log_probs=torch.randn(batch_size, response_len) * 0.1,
            rewards=torch.zeros(batch_size, response_len),
            advantages=torch.randn(batch_size, response_len),
            returns=torch.randn(batch_size, response_len),
            values=torch.randn(batch_size, response_len),
        )

        metrics = trainer.train_step(batch)

        assert metrics.loss != 0
        assert metrics.policy_loss != 0
        assert isinstance(metrics.clip_fraction, float)

    def test_policy_loss_computation(
        self,
        tiny_gpt_with_value: GPTWithValueHead,
        ppo_config: PPOConfig,
    ) -> None:
        optimizer = torch.optim.Adam(tiny_gpt_with_value.parameters(), lr=1e-4)
        trainer = PPOTrainer(
            model=tiny_gpt_with_value,
            ref_model=None,
            optimizer=optimizer,
            config=ppo_config,
        )

        log_probs = torch.tensor([[-1.0, -1.0]])
        old_log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])

        loss, clip_frac = trainer.compute_policy_loss(log_probs, old_log_probs, advantages, mask)

        assert loss.shape == ()
        assert clip_frac.item() == 0.0

    def test_value_loss_computation(
        self,
        tiny_gpt_with_value: GPTWithValueHead,
        ppo_config: PPOConfig,
    ) -> None:
        optimizer = torch.optim.Adam(tiny_gpt_with_value.parameters(), lr=1e-4)
        trainer = PPOTrainer(
            model=tiny_gpt_with_value,
            ref_model=None,
            optimizer=optimizer,
            config=ppo_config,
        )

        values = torch.tensor([[0.5, 0.6]])
        old_values = torch.tensor([[0.5, 0.5]])
        returns = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])

        loss = trainer.compute_value_loss(values, old_values, returns, mask)

        assert loss.shape == ()
        assert loss.item() > 0
