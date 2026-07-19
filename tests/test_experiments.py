from pathlib import Path

import pytest

from autoresearch.cli import load_project
from autoresearch.interface import Experiment, Harness


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["exp-fleet-agent-rl", "exp-fleet-taskboard", "exp-fleet-curriculum"])
def test_experiment_loads_and_conforms(name):
    path = ROOT / "experiments" / name
    if not path.exists():
        pytest.skip("experiments are local-only (gitignored); absent on a fresh checkout")
    harness, experiment = load_project(str(path))
    assert isinstance(harness, Harness)
    assert isinstance(experiment, Experiment)
    assert harness.spec.primary_metric == "eval_return"
    assert harness.spec.direction == "max"
