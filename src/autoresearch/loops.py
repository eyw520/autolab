from collections.abc import Callable, Iterator
from dataclasses import dataclass
import gc
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import RunContext


StepFn = Callable[[nn.Module, optim.Optimizer, torch.Tensor, torch.Tensor, int, float, torch.device], float]


@dataclass
class SupervisedConfig:
    total_batch_size: int
    device_batch_size: int
    seq_len: int


def supervised_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor, int]],
    step_fn: StepFn,
    ctx: RunContext,
    config: SupervisedConfig,
) -> None:
    device = ctx.device
    use_cuda = device.type == "cuda"

    tokens_per_fwdbwd = config.device_batch_size * config.seq_len
    assert config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = config.total_batch_size // tokens_per_fwdbwd

    x, y, epoch = next(dataloader)
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    smooth_train_loss = 0.0
    training_time = 0.0
    debiased_smooth_loss = 0.0
    loss = 0.0
    step = 0

    while True:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(grad_accum_steps):
            progress = ctx.budget.progress(elapsed=training_time, steps=step)
            loss = step_fn(model, optimizer, x, y, step, progress, device)
            x, y, epoch = next(dataloader)

        optimizer.step()
        model.zero_grad(set_to_none=True)

        if math.isnan(loss) or loss > 100:
            raise RuntimeError("Training diverged")

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * ctx.budget.progress(elapsed=training_time, steps=step)
        tok_per_sec = int(config.total_batch_size / dt)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch}    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and ctx.budget.exceeded(elapsed=training_time, steps=step):
            break

    print()

    ctx.record(
        {
            "training_seconds": training_time,
            "num_steps": float(step),
            "total_tokens_M": step * config.total_batch_size / 1e6,
            "final_loss": debiased_smooth_loss,
        }
    )
