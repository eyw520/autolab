import math

import torch
import torch.nn as nn

from pretraining_gpt.harness.data.constants import EVAL_TOKENS, MAX_SEQ_LEN
from pretraining_gpt.harness.data.loader import make_dataloader
from pretraining_gpt.harness.data.tokenizer import Tokenizer, get_token_bytes


@torch.no_grad()
def evaluate_bpb(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> float:
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
