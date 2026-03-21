import gc
import math
import time

import torch

from autoresearch.interface import Experiment, Harness, TrainingConfig, TrainingResult


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(harness: Harness, experiment: Experiment) -> dict[str, float]:
    t_start = time.time()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    device = get_device()
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.cuda.manual_seed(42)

    harness_config = harness.config
    training_config = experiment.get_training_config(device)

    print(f"Device: {device}")
    print(f"Time budget: {harness_config.time_budget}s")

    vocab_size = harness.get_vocab_size()
    model = experiment.build_model(vocab_size, harness_config.seq_len, device)
    optimizer = experiment.build_optimizer(model, training_config, device)

    if use_cuda:
        model = torch.compile(model, dynamic=False)

    result = _train_loop(harness, experiment, model, optimizer, training_config, device)

    model.eval()
    metrics = harness.evaluate(model, training_config.device_batch_size, device)

    t_end = time.time()

    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    print("---")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"training_seconds: {result.total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_tokens_M:   {result.total_tokens / 1e6:.1f}")
    print(f"num_steps:        {result.num_steps}")

    return {
        **metrics,
        "training_seconds": result.total_training_time,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "total_tokens_M": result.total_tokens / 1e6,
        "num_steps": result.num_steps,
    }


def _train_loop(
    harness: Harness,
    experiment: Experiment,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    use_cuda = device.type == "cuda"
    harness_config = harness.config

    tokens_per_fwdbwd = training_config.device_batch_size * harness_config.seq_len
    assert training_config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = training_config.total_batch_size // tokens_per_fwdbwd

    train_loader = harness.make_dataloader(
        "train",
        training_config.device_batch_size,
        harness_config.seq_len,
        device,
    )
    x, y, epoch = next(train_loader)

    print(f"Gradient accumulation steps: {grad_accum_steps}")

    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    while True:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(grad_accum_steps):
            progress = min(total_training_time / harness_config.time_budget, 1.0)
            loss = experiment.train_step(model, optimizer, x, y, step, progress, device)
            x, y, epoch = next(train_loader)

        optimizer.step()
        model.zero_grad(set_to_none=True)

        if math.isnan(loss) or loss > 100:
            print("FAIL")
            raise RuntimeError("Training diverged")

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        progress = min(total_training_time / harness_config.time_budget, 1.0)
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(training_config.total_batch_size / dt)
        remaining = max(0, harness_config.time_budget - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
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

        if step > 10 and total_training_time >= harness_config.time_budget:
            break

    print()

    return TrainingResult(
        total_training_time=total_training_time,
        total_tokens=step * training_config.total_batch_size,
        num_steps=step,
        final_loss=debiased_smooth_loss,
    )
