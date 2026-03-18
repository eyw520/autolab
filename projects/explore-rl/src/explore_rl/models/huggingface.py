from typing import Any

from explore_rl.core.config import GenerationConfig
from explore_rl.core.types import GenerationOutput
from explore_rl.models.base import BaseLanguageModel, BaseLanguageModelWithValueHead
import torch
from torch import Tensor
import torch.nn as nn


class HuggingFaceModel(BaseLanguageModel):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        config = config or GenerationConfig()
        prompt_length = prompt_ids.size(1)

        gen_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else None,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "pad_token_id": config.pad_token_id or self.tokenizer.pad_token_id,
            "eos_token_id": config.eos_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        sequences = outputs.sequences

        if hasattr(outputs, "scores") and outputs.scores:
            log_probs_list = []
            for i, scores in enumerate(outputs.scores):
                log_probs = torch.log_softmax(scores, dim=-1)
                next_token = sequences[:, prompt_length + i]
                token_log_probs = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
                log_probs_list.append(token_log_probs)
            log_probs_tensor = torch.stack(log_probs_list, dim=1)
        else:
            log_probs_tensor = None

        return GenerationOutput(
            sequences=sequences,
            prompt_length=prompt_length,
            log_probs=log_probs_tensor,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "HuggingFaceModel":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers package required for HuggingFaceModel") from e

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        hf_model = hf_model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(hf_model, tokenizer)


class HuggingFaceModelWithValueHead(BaseLanguageModelWithValueHead):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        value_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hf_model = HuggingFaceModel(model, tokenizer)

        if value_head is None:
            hidden_size = model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.value_head = value_head

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.hf_model.forward(input_ids, attention_mask)

    def forward_with_value(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        values = self.value_head(last_hidden).squeeze(-1)

        return logits, values

    def generate(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        return self.hf_model.generate(prompt_ids, attention_mask, config)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "HuggingFaceModelWithValueHead":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers package required for HuggingFaceModelWithValueHead") from e

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        hf_model = hf_model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(hf_model, tokenizer)
