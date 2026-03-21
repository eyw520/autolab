from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import HarnessConfig, TrainingConfig

from pretraining_gpt.harness.data.constants import (
    CACHE_DIR,
    MAX_SEQ_LEN,
    MAX_SHARD,
    TIME_BUDGET,
)
from pretraining_gpt.harness.data.download import download_data
from pretraining_gpt.harness.data.loader import make_dataloader
from pretraining_gpt.harness.data.tokenizer import Tokenizer, train_tokenizer
from pretraining_gpt.harness.eval.metrics import evaluate_bpb

from pretraining_gpt.experiment.model.attention import init_flash_attention
from pretraining_gpt.experiment.model.gpt import GPT, build_model_config
from pretraining_gpt.experiment.optim.muon import setup_optimizer
from pretraining_gpt.experiment.training.schedule import (
    get_lr_multiplier,
    get_muon_momentum,
    get_weight_decay,
)


@dataclass
class GPTTrainingConfig(TrainingConfig):
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple[float, float] = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    depth: int = 8
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"


class PretrainingHarness:
    def __init__(self):
        self._tokenizer: Tokenizer | None = None

    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(
            time_budget=TIME_BUDGET,
            seq_len=MAX_SEQ_LEN,
            primary_metric="val_bpb",
            cache_dir=CACHE_DIR,
        )

    def prepare(self) -> None:
        download_data(num_shards=10)
        train_tokenizer()

    def make_dataloader(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_directory()
        return make_dataloader(self._tokenizer, batch_size, seq_len, split, device=device)

    def evaluate(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, float]:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_directory()
        use_cuda = device.type == "cuda"
        if use_cuda:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, self._tokenizer, batch_size, device=device)
        return {"val_bpb": val_bpb}

    def get_vocab_size(self) -> int:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_directory()
        return self._tokenizer.get_vocab_size()


class PretrainingExperiment:
    def __init__(self):
        self._training_config: GPTTrainingConfig | None = None
        self._grad_accum_steps: int = 1

    def get_training_config(self, device: torch.device) -> GPTTrainingConfig:
        use_cuda = device.type == "cuda"
        use_mps = device.type == "mps"

        if use_cuda:
            config = GPTTrainingConfig(
                total_batch_size=2**19,
                device_batch_size=128,
                depth=8,
                aspect_ratio=64,
                head_dim=128,
                window_pattern="SSSL",
            )
        elif use_mps:
            config = GPTTrainingConfig(
                total_batch_size=2**14,
                device_batch_size=16,
                depth=4,
                aspect_ratio=64,
                head_dim=64,
                window_pattern="L",
            )
        else:
            config = GPTTrainingConfig(
                total_batch_size=2**12,
                device_batch_size=8,
                depth=4,
                aspect_ratio=64,
                head_dim=64,
                window_pattern="L",
            )

        self._training_config = config
        return config

    def build_model(
        self,
        vocab_size: int,
        seq_len: int,
        device: torch.device,
    ) -> nn.Module:
        assert self._training_config is not None
        config = self._training_config

        init_flash_attention()

        model_config = build_model_config(
            depth=config.depth,
            vocab_size=vocab_size,
            sequence_len=seq_len,
            aspect_ratio=config.aspect_ratio,
            head_dim=config.head_dim,
            window_pattern=config.window_pattern,
        )

        print(f"Model config: {model_config}")

        use_cuda = device.type == "cuda"
        if use_cuda:
            with torch.device("meta"):
                model = GPT(model_config)
            model.to_empty(device=device)
        else:
            model = GPT(model_config).to(device)

        model.init_weights()

        param_counts = model.num_scaling_params()
        print("Parameter counts:")
        for key, value in param_counts.items():
            print(f"  {key:24s}: {value:,}")
        print(f"Estimated FLOPs per token: {model.estimate_flops():e}")

        return model

    def build_optimizer(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> optim.Optimizer:
        assert isinstance(training_config, GPTTrainingConfig)
        assert isinstance(model, GPT)
        use_cuda = device.type == "cuda"

        tokens_per_fwdbwd = training_config.device_batch_size * MAX_SEQ_LEN
        self._grad_accum_steps = training_config.total_batch_size // tokens_per_fwdbwd

        return setup_optimizer(
            model,
            unembedding_lr=training_config.unembedding_lr,
            embedding_lr=training_config.embedding_lr,
            scalar_lr=training_config.scalar_lr,
            adam_betas=training_config.adam_betas,
            matrix_lr=training_config.matrix_lr,
            weight_decay=training_config.weight_decay,
            use_cuda=use_cuda,
        )

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        progress: float,
        device: torch.device,
    ) -> float:
        assert self._training_config is not None
        config = self._training_config
        use_cuda = device.type == "cuda"

        if use_cuda:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

        with autocast_ctx:
            loss = model(x, y)

        train_loss = loss.detach().item()
        scaled_loss = loss / self._grad_accum_steps
        scaled_loss.backward()

        lrm = get_lr_multiplier(progress, config.warmup_ratio, config.warmdown_ratio, config.final_lr_frac)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress, config.weight_decay)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        return train_loss


harness = PretrainingHarness()
experiment = PretrainingExperiment()
