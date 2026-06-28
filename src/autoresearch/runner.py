from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

import torch

from autoresearch.interface import Experiment, Harness, HarnessSpec, RunContext
from autoresearch.util import get_device, seed_everything


def _git_info(path: Path) -> dict[str, str | None]:
    def query(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    return {"commit": query(["rev-parse", "--short", "HEAD"]), "branch": query(["rev-parse", "--abbrev-ref", "HEAD"])}


def _write_run(
    output_dir: Path,
    record: dict[str, float],
    spec: HarnessSpec,
    seed: int,
    status: str,
    history: list[dict[str, float]],
    started_at: datetime,
) -> Path:
    run_id = f"{started_at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment": output_dir.name,
        "run_id": run_id,
        "timestamp": started_at.isoformat(timespec="seconds"),
        "seed": seed,
        "status": status,
        "spec": {
            "primary_metric": spec.primary_metric,
            "direction": spec.direction,
            "budget": asdict(spec.budget),
            "domain": spec.domain,
        },
        "metrics": record,
        "git": _git_info(output_dir),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    with (run_dir / "metrics.jsonl").open("w") as stream:
        for entry in history:
            stream.write(json.dumps(entry, default=str) + "\n")
    return run_dir


def run(
    harness: Harness,
    experiment: Experiment,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    started_at = datetime.now()
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
    status = "crash" if crashed else "ok"
    print(f"status: {status}")

    if output_dir is not None:
        run_dir = _write_run(Path(output_dir), record, spec, seed, status, ctx.history, started_at)
        print(f"run: {run_dir}")

    return record
