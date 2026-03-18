import torch

from explore_rl.algorithms.dpo import DPOTrainer
from explore_rl.core.config import DPOConfig
from explore_rl.core.types import PreferencePair
from explore_rl.models.gpt import GPT


class TestDPOTrainer:
    def test_train_step(
        self,
        tiny_gpt: GPT,
        dpo_config: DPOConfig,
    ) -> None:
        ref_model = GPT(tiny_gpt.config)
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)

        trainer = DPOTrainer(
            model=tiny_gpt,
            ref_model=ref_model,
            optimizer=optimizer,
            config=dpo_config,
        )

        batch_size = 2
        prompt_len = 5
        chosen_len = 8
        rejected_len = 8

        batch = PreferencePair(
            prompt_ids=torch.randint(0, 100, (batch_size, prompt_len)),
            prompt_mask=torch.ones(batch_size, prompt_len, dtype=torch.long),
            chosen_ids=torch.randint(0, 100, (batch_size, chosen_len)),
            chosen_mask=torch.ones(batch_size, chosen_len, dtype=torch.long),
            rejected_ids=torch.randint(0, 100, (batch_size, rejected_len)),
            rejected_mask=torch.ones(batch_size, rejected_len, dtype=torch.long),
        )

        metrics = trainer.train_step(batch)

        assert metrics.loss != 0
        assert "accuracy" in metrics.extras
        assert "reward_margin" in metrics.extras

    def test_reference_free(
        self,
        tiny_gpt: GPT,
    ) -> None:
        config = DPOConfig(beta=0.1, reference_free=True)
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)

        trainer = DPOTrainer(
            model=tiny_gpt,
            ref_model=None,
            optimizer=optimizer,
            config=config,
        )

        batch = PreferencePair(
            prompt_ids=torch.randint(0, 100, (2, 5)),
            prompt_mask=torch.ones(2, 5, dtype=torch.long),
            chosen_ids=torch.randint(0, 100, (2, 8)),
            chosen_mask=torch.ones(2, 8, dtype=torch.long),
            rejected_ids=torch.randint(0, 100, (2, 8)),
            rejected_mask=torch.ones(2, 8, dtype=torch.long),
        )

        metrics = trainer.train_step(batch)
        assert metrics.loss != 0

    def test_implicit_reward(
        self,
        tiny_gpt: GPT,
        dpo_config: DPOConfig,
    ) -> None:
        ref_model = GPT(tiny_gpt.config)
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)

        trainer = DPOTrainer(
            model=tiny_gpt,
            ref_model=ref_model,
            optimizer=optimizer,
            config=dpo_config,
        )

        prompt_ids = torch.randint(0, 100, (2, 5))
        prompt_mask = torch.ones(2, 5, dtype=torch.long)
        response_ids = torch.randint(0, 100, (2, 8))
        response_mask = torch.ones(2, 8, dtype=torch.long)

        reward = trainer.get_implicit_reward(
            prompt_ids, prompt_mask, response_ids, response_mask
        )

        assert reward.shape == (2,)

    def test_dpo_loss_computation(
        self,
        tiny_gpt: GPT,
        dpo_config: DPOConfig,
    ) -> None:
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)
        trainer = DPOTrainer(
            model=tiny_gpt,
            ref_model=None,
            optimizer=optimizer,
            config=dpo_config,
        )

        pi_chosen = torch.tensor([-1.0, -1.0])
        pi_rejected = torch.tensor([-2.0, -2.0])
        ref_chosen = torch.tensor([-1.5, -1.5])
        ref_rejected = torch.tensor([-1.5, -1.5])

        loss, metrics = trainer.compute_dpo_loss(
            pi_chosen, pi_rejected, ref_chosen, ref_rejected
        )

        assert loss.shape == ()
        assert "accuracy" in metrics
        assert metrics["accuracy"].item() == 1.0

    def test_label_smoothing(
        self,
        tiny_gpt: GPT,
    ) -> None:
        config = DPOConfig(beta=0.1, label_smoothing=0.1)
        optimizer = torch.optim.Adam(tiny_gpt.parameters(), lr=1e-4)

        trainer = DPOTrainer(
            model=tiny_gpt,
            ref_model=None,
            optimizer=optimizer,
            config=config,
        )

        batch = PreferencePair(
            prompt_ids=torch.randint(0, 100, (2, 5)),
            prompt_mask=torch.ones(2, 5, dtype=torch.long),
            chosen_ids=torch.randint(0, 100, (2, 8)),
            chosen_mask=torch.ones(2, 8, dtype=torch.long),
            rejected_ids=torch.randint(0, 100, (2, 8)),
            rejected_mask=torch.ones(2, 8, dtype=torch.long),
        )

        metrics = trainer.train_step(batch)
        assert metrics.loss != 0
