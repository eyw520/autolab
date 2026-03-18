import math

from explore_rl.core.config import GenerationConfig, ModelConfig
from explore_rl.core.types import GenerationOutput
from explore_rl.models.base import BaseLanguageModel, BaseLanguageModelWithValueHead
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        causal_mask: Tensor = self.bias[:, :, :T, :T] == 0  # type: ignore[index]
        att = att.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * float("-inf")
            att = att + padding_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(BaseLanguageModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        device = input_ids.device
        _, T = input_ids.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        config = config or GenerationConfig()
        prompt_length = prompt_ids.size(1)
        device = prompt_ids.device

        sequences = prompt_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)
        current_mask = attention_mask.clone()

        all_log_probs: list[Tensor] = []

        self.eval()
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                if sequences.size(1) > self.config.block_size:
                    input_ids = sequences[:, -self.config.block_size :]
                    mask = current_mask[:, -self.config.block_size :]
                else:
                    input_ids = sequences
                    mask = current_mask

                logits = self.forward(input_ids, mask)
                next_logits = logits[:, -1, :].clone()

                next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                if config.temperature != 1.0:
                    next_logits = next_logits / config.temperature

                if config.top_k is not None:
                    v, _ = torch.topk(next_logits, min(config.top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                if config.top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1e-8, posinf=1.0, neginf=0.0)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                log_probs = F.log_softmax(next_logits, dim=-1)

                if config.do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                token_log_prob = log_probs.gather(-1, next_token).squeeze(-1)
                all_log_probs.append(token_log_prob)

                sequences = torch.cat([sequences, next_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones((current_mask.size(0), 1), device=device)], dim=1)

                if config.eos_token_id is not None and (next_token == config.eos_token_id).all():
                    break

        log_probs_tensor = torch.stack(all_log_probs, dim=1) if all_log_probs else None

        return GenerationOutput(
            sequences=sequences,
            prompt_length=prompt_length,
            log_probs=log_probs_tensor,
        )


class GPTWithValueHead(BaseLanguageModelWithValueHead):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gpt = GPT(config)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.gpt.forward(input_ids, attention_mask)

    def forward_with_value(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = input_ids.device
        _, T = input_ids.size()

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.gpt.wte(input_ids)
        pos_emb = self.gpt.wpe(pos)
        x = self.gpt.drop(tok_emb + pos_emb)

        for block in self.gpt.blocks:
            x = block(x, attention_mask)

        x = self.gpt.ln_f(x)
        logits = self.gpt.lm_head(x)
        values = self.value_head(x).squeeze(-1)

        return logits, values

    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        return self.gpt.generate(prompt_ids, attention_mask, config)

    def log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self.gpt.log_probs(input_ids, attention_mask)
