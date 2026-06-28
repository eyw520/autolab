from pathlib import Path

import pytest

from autoresearch.cli import load_project
from autoresearch.interface import Experiment, Harness


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["exp-fleet-agent-rl", "exp-fleet-taskboard", "exp-fleet-curriculum"])
def test_experiment_loads_and_conforms(name):
    harness, experiment = load_project(str(ROOT / "experiments" / name))
    assert isinstance(harness, Harness)
    assert isinstance(experiment, Experiment)
    assert harness.spec.primary_metric == "eval_return"
    assert harness.spec.direction == "max"
