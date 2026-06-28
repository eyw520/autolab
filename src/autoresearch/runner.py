import math
import time

import torch

from autoresearch.interface import Experiment, Harness, RunContext
from autoresearch.util import get_device, seed_everything


def run(
    harness: Harness,
    experiment: Experiment,
    seed: int = 42,
) -> dict[str, float]:
    t_start = time.time()
    torch.set_float32_matmul_precision("high")

    device = get_device()
    seed_everything(seed)

    spec = harness.spec
    ctx = RunContext(device=device, budget=spec.budget, seed=seed)

    print(f"Device: {device}")
    print(f"Primary metric: {spec.primary_metric} ({spec.direction})")
    print(f"Budget: {spec.budget}")

    crashed = False
    metrics: dict[str, float] = {}
    try:
        artifact = experiment.run(harness, ctx)
        metrics = harness.evaluate(artifact, ctx)
    except RuntimeError as e:
        print(f"\nFAIL: {e}")
        crashed = True
        metrics = {spec.primary_metric: math.nan}

    t_end = time.time()

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    record: dict[str, float] = {
        **ctx.telemetry,
        **metrics,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
    }

    print("---")
    for key, value in record.items():
        print(f"{key}: {value:.6f}")
    print(f"status: {'crash' if crashed else 'ok'}")

    return record
