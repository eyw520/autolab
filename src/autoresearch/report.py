import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_runs(project: Path) -> list[dict[str, Any]]:
    runs_dir = project / "runs"
    if not runs_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for result_file in sorted(runs_dir.glob("*/result.json")):
        try:
            runs.append(json.loads(result_file.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def _metric_value(run: dict[str, Any], metric: str) -> float:
    value = run.get("metrics", {}).get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _sort_key(run: dict[str, Any], metric: str, direction: str) -> tuple[int, float]:
    value = _metric_value(run, metric)
    if math.isnan(value):
        return (1, 0.0)
    return (0, -value if direction == "max" else value)


def _row(run: dict[str, Any], metric: str) -> dict[str, Any]:
    git = run.get("git") or {}
    return {
        "run_id": run.get("run_id", ""),
        "timestamp": run.get("timestamp", ""),
        "branch": git.get("branch"),
        "commit": git.get("commit"),
        "seed": run.get("seed"),
        "status": run.get("status", ""),
        metric: _metric_value(run, metric),
        "total_seconds": _metric_value(run, "total_seconds"),
    }


def _print_table(runs: list[dict[str, Any]], metric: str, direction: str) -> None:
    header = ["#", "run_id", "branch", "commit", "seed", "status", metric, "secs"]
    rows: list[list[str]] = []
    for index, run in enumerate(runs, 1):
        value = _metric_value(run, metric)
        secs = _metric_value(run, "total_seconds")
        git = run.get("git") or {}
        rows.append(
            [
                str(index),
                str(run.get("run_id", "-")),
                str(git.get("branch") or "-"),
                str(git.get("commit") or "-"),
                str(run.get("seed", "-")),
                str(run.get("status", "-")),
                "nan" if math.isnan(value) else f"{value:.4f}",
                "nan" if math.isnan(secs) else f"{secs:.1f}",
            ]
        )
    widths = [max(len(header[col]), *(len(row[col]) for row in rows)) for col in range(len(header))]
    print(f"{metric} ({direction}) - {len(runs)} run(s), best first")
    print("  ".join(header[col].ljust(widths[col]) for col in range(len(header))))
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in range(len(header))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize autoresearch runs for an experiment")
    parser.add_argument("project", type=str, help="Path to the experiment directory")
    parser.add_argument("--metric", type=str, default=None, help="Metric to sort by (default: primary_metric)")
    parser.add_argument("--direction", choices=["min", "max"], default=None, help="Sort direction override")
    parser.add_argument("--limit", type=int, default=None, help="Show only the top N runs")
    parser.add_argument("--json", action="store_true", help="Emit aggregated JSON instead of a table")
    args = parser.parse_args()

    project = Path(args.project)
    runs = _load_runs(project)
    if not runs:
        print(f"No runs found under {project / 'runs'}")
        return

    latest = max(runs, key=lambda run: run.get("timestamp", ""))
    spec = latest.get("spec", {})
    metric = args.metric or spec.get("primary_metric", "")
    direction = args.direction or spec.get("direction", "max")

    runs.sort(key=lambda run: _sort_key(run, metric, direction))
    if args.limit is not None:
        runs = runs[: args.limit]

    if args.json:
        print(json.dumps([_row(run, metric) for run in runs], indent=2, default=str))
        return

    _print_table(runs, metric, direction)


if __name__ == "__main__":
    main()
