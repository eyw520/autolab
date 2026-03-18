from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: int | None = None


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    kl_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    epochs_per_update: int = 4
    minibatch_size: int = 64


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
    max_grad_norm: float = 1.0


@dataclass
class GRPOConfig:
    beta: float = 0.1
    group_size: int = 4
    normalize_advantages: bool = True
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    eval_interval: int = 100
    save_interval: int = 1000
    log_interval: int = 10
    gradient_accumulation_steps: int = 1
