import json
import math
import sys

from autoresearch.report import _load_runs, _metric_value, _sort_key, main


def _write_run(runs_dir, run_id, metric, value, direction="max", status="ok"):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp": run_id,
                "seed": 42,
                "status": status,
                "spec": {"primary_metric": metric, "direction": direction, "budget": {}, "domain": {}},
                "metrics": {metric: value, "total_seconds": 1.0},
                "git": {"commit": "abc1234", "branch": "main"},
            }
        )
    )


def test_no_runs_returns_empty(tmp_path):
    assert _load_runs(tmp_path) == []


def test_sort_max_best_first(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "20260101-000001-aaa", "score", 0.5)
    _write_run(runs, "20260101-000002-bbb", "score", 0.9)
    _write_run(runs, "20260101-000003-ccc", "score", 0.7)
    loaded = sorted(_load_runs(tmp_path), key=lambda r: _sort_key(r, "score", "max"))
    assert [_metric_value(r, "score") for r in loaded] == [0.9, 0.7, 0.5]


def test_sort_min_and_crashed_runs_last(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "r1", "loss", 0.2, direction="min")
    _write_run(runs, "r2", "loss", 0.1, direction="min")
    _write_run(runs, "r3", "loss", float("nan"), direction="min", status="crash")
    loaded = sorted(_load_runs(tmp_path), key=lambda r: _sort_key(r, "loss", "min"))
    values = [_metric_value(r, "loss") for r in loaded]
    assert values[0] == 0.1
    assert values[1] == 0.2
    assert math.isnan(values[2])


def test_main_prints_leaderboard(tmp_path, capsys, monkeypatch):
    runs = tmp_path / "runs"
    _write_run(runs, "r1", "score", 0.5)
    _write_run(runs, "r2", "score", 0.9)
    monkeypatch.setattr(sys, "argv", ["autoresearch-report", str(tmp_path)])
    main()
    out = capsys.readouterr().out
    assert "score (max)" in out
    assert out.index("r2") < out.index("r1")
