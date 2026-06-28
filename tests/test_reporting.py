import json

import torch

from autoresearch.interface import Budget, HarnessSpec, RunContext
from autoresearch.runner import run


class _Harness:
    @property
    def spec(self):
        return HarnessSpec("score", "max", Budget(steps=1), {"backend": "fake"})

    def prepare(self):
        pass

    def evaluate(self, artifact, ctx):
        return {"score": 1.0}


class _Experiment:
    def run(self, harness, ctx):
        ctx.log_step({"loss": 1.0, "reward": 0.2})
        ctx.log_step({"loss": 0.5, "reward": 0.8})
        return object()


def test_log_step_accumulates_history_and_updates_telemetry():
    ctx = RunContext(device=torch.device("cpu"), budget=Budget(steps=1), seed=0)
    ctx.log_step({"a": 1.0})
    ctx.log_step({"a": 2.0})
    assert len(ctx.history) == 2
    assert ctx.history[0]["a"] == 1.0
    assert ctx.telemetry["a"] == 2.0


def test_run_persists_result_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("autoresearch.runner.get_device", lambda: torch.device("cpu"))
    run(_Harness(), _Experiment(), output_dir=tmp_path)

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "ok"
    assert result["seed"] == 42
    assert result["metrics"]["score"] == 1.0
    assert result["spec"]["primary_metric"] == "score"
    assert result["spec"]["direction"] == "max"
    assert "git" in result

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["loss"] == 0.5


def test_run_without_output_dir_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("autoresearch.runner.get_device", lambda: torch.device("cpu"))
    run(_Harness(), _Experiment())
    assert not (tmp_path / "runs").exists()
