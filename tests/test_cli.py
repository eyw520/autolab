from pathlib import Path

import pytest

from autoresearch.cli import _resolve_module_name, load_project
from autoresearch.interface import Experiment, Harness
from autoresearch.scaffold import create_experiment


def test_resolve_module_name_prefers_actual_package(tmp_path: Path) -> None:
    create_experiment("vision-cls", tmp_path)
    exp_dir = tmp_path / "exp-vision-cls"
    assert _resolve_module_name(exp_dir, exp_dir / "src") == "vision_cls"


def test_resolve_module_name_strips_exp_prefix_when_no_package(tmp_path: Path) -> None:
    empty = tmp_path / "exp-foo-bar"
    empty.mkdir()
    assert _resolve_module_name(empty, empty) == "foo_bar"


def test_load_project_loads_scaffolded_experiment(tmp_path: Path) -> None:
    create_experiment("loadme", tmp_path)
    harness, experiment = load_project(str(tmp_path / "exp-loadme"))
    assert isinstance(harness, Harness)
    assert isinstance(experiment, Experiment)


def test_load_project_rejects_nonconforming(tmp_path: Path) -> None:
    pkg = tmp_path / "exp-bad" / "src" / "bad"
    pkg.mkdir(parents=True)
    (pkg / "interface.py").write_text("class H:\n    pass\nclass E:\n    pass\nharness = H()\nexperiment = E()\n")
    with pytest.raises(TypeError):
        load_project(str(tmp_path / "exp-bad"))


def test_load_project_missing_objects(tmp_path: Path) -> None:
    pkg = tmp_path / "exp-empty" / "src" / "empty"
    pkg.mkdir(parents=True)
    (pkg / "interface.py").write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_project(str(tmp_path / "exp-empty"))
