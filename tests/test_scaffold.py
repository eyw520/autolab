from pathlib import Path

import pytest

from autoresearch.scaffold import create_experiment


def test_create_experiment_writes_expected_layout(tmp_path: Path) -> None:
    exp_dir = create_experiment("my-task", tmp_path)
    assert exp_dir == tmp_path / "exp-my-task"
    assert (exp_dir / "EXPERIMENT.md").exists()
    assert (exp_dir / ".gitignore").exists()
    interface = exp_dir / "src" / "my_task" / "interface.py"
    assert interface.exists()
    text = interface.read_text()
    assert "harness = " in text
    assert "experiment = " in text


def test_create_experiment_rejects_invalid_names(tmp_path: Path) -> None:
    for bad in ["My-Task", "1task", "my_task", "exp space"]:
        with pytest.raises(ValueError):
            create_experiment(bad, tmp_path)


def test_create_experiment_rejects_existing(tmp_path: Path) -> None:
    create_experiment("dup", tmp_path)
    with pytest.raises(FileExistsError):
        create_experiment("dup", tmp_path)
