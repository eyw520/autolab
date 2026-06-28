import pytest

from autoresearch.agent_rl.types import Trajectory
from autoresearch.agent_rl.verifiers.reward import VerifierReward
from autoresearch.agent_rl.verifiers.state import Change, IgnoreConfig, StateSnapshot, VerificationError


def _snap(done):
    return StateSnapshot({"items": {str(i): {"done": d} for i, d in enumerate(done)}})


def test_expect_only_passes_on_match():
    expected = [Change("items", "0", "done", 1), Change("items", "2", "done", 1)]
    _snap([0, 0, 0]).diff(_snap([1, 0, 1])).expect_only(expected)


def test_expect_only_raises_on_unexpected_change():
    with pytest.raises(VerificationError):
        _snap([0, 0, 0]).diff(_snap([1, 1, 0])).expect_only([Change("items", "0", "done", 1)])


def test_expect_only_raises_on_missing_change():
    expected = [Change("items", "0", "done", 1), Change("items", "1", "done", 1)]
    with pytest.raises(VerificationError):
        _snap([0, 0, 0]).diff(_snap([1, 0, 0])).expect_only(expected)


def test_record_assert_eq():
    _snap([1, 0, 0]).table("items").row("0").assert_eq("done", 1)
    with pytest.raises(VerificationError):
        _snap([1, 0, 0]).table("items").row("1").assert_eq("done", 1)


def test_ignore_config_skips_field():
    diff = _snap([0, 0, 0]).diff(_snap([1, 0, 0]), IgnoreConfig(fields={"items": {"done"}}))
    assert diff.changes == []


def test_verifier_reward_pass_and_fail():
    def good(env):
        return None

    def bad(env):
        raise VerificationError("nope")

    assert VerifierReward(good)(object(), Trajectory()) == 1.0
    assert VerifierReward(bad)(object(), Trajectory()) == 0.0
