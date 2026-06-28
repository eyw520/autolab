import importlib.util
import sys

import pytest

from autoresearch.launch import CloudLauncher, Launcher, LocalLauncher, resolve_launcher


def test_resolve_local_and_cloud():
    assert isinstance(resolve_launcher("local"), LocalLauncher)
    assert isinstance(resolve_launcher("cloud"), CloudLauncher)


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_launcher("nope")


def test_local_launcher_is_launcher():
    assert isinstance(LocalLauncher(), Launcher)


def test_local_launcher_dispatches(monkeypatch):
    captured = {}

    class _Result:
        returncode = 0

    def fake_run(command, check):
        captured["command"] = command
        return _Result()

    monkeypatch.setattr("autoresearch.launch.subprocess.run", fake_run)
    code = LocalLauncher().run("experiments/exp-x", prepare=True)
    assert code == 0
    assert captured["command"] == [sys.executable, "-m", "autoresearch.cli", "experiments/exp-x", "--prepare"]


def test_cloud_launcher_fails_fast_without_skypilot():
    if importlib.util.find_spec("sky") is not None:
        pytest.skip("skypilot installed")
    with pytest.raises(ImportError):
        CloudLauncher().run("experiments/exp-x")
