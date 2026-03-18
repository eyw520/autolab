import torch

from explore_rl.rewards.base import RewardFunction
from explore_rl.rewards.composite import (
    ClippedReward,
    CompositeReward,
    ScaledReward,
    ThresholdReward,
)
from explore_rl.rewards.rule_based import (
    FormatReward,
    KeywordReward,
    LengthPenaltyReward,
    LengthReward,
    RepetitionPenalty,
)


class TestLengthReward:
    def test_exact_target(self) -> None:
        reward = LengthReward(target_length=10, tolerance=5)
        prompts = ["test"]
        responses = ["a" * 10]
        result = reward(prompts, responses)
        assert result[0] == 1.0

    def test_within_tolerance(self) -> None:
        reward = LengthReward(target_length=10, tolerance=5)
        prompts = ["test"]
        responses = ["a" * 13]
        result = reward(prompts, responses)
        assert result[0] == 1.0

    def test_outside_tolerance(self) -> None:
        reward = LengthReward(target_length=10, tolerance=5, penalty_scale=0.1)
        prompts = ["test"]
        responses = ["a" * 20]
        result = reward(prompts, responses)
        assert result[0] < 1.0
        assert result[0] >= 0.0


class TestFormatReward:
    def test_match(self) -> None:
        reward = FormatReward(pattern=r"\d+")
        prompts = ["test"]
        responses = ["answer is 42"]
        result = reward(prompts, responses)
        assert result[0] == 1.0

    def test_no_match(self) -> None:
        reward = FormatReward(pattern=r"\d+")
        prompts = ["test"]
        responses = ["no numbers here"]
        result = reward(prompts, responses)
        assert result[0] == 0.0


class TestKeywordReward:
    def test_required_keywords(self) -> None:
        reward = KeywordReward(
            required_keywords=["hello", "world"],
            reward_per_keyword=0.5,
        )
        prompts = ["test"]
        responses = ["hello world"]
        result = reward(prompts, responses)
        assert result[0] == 1.0

    def test_forbidden_keywords(self) -> None:
        reward = KeywordReward(
            forbidden_keywords=["bad"],
            penalty_per_keyword=1.0,
        )
        prompts = ["test"]
        responses = ["this is bad"]
        result = reward(prompts, responses)
        assert result[0] == -1.0

    def test_case_insensitive(self) -> None:
        reward = KeywordReward(
            required_keywords=["HELLO"],
            reward_per_keyword=1.0,
            case_sensitive=False,
        )
        prompts = ["test"]
        responses = ["hello there"]
        result = reward(prompts, responses)
        assert result[0] == 1.0


class TestLengthPenaltyReward:
    def test_within_limit(self) -> None:
        reward = LengthPenaltyReward(max_length=100)
        prompts = ["test"]
        responses = ["short"]
        result = reward(prompts, responses)
        assert result[0] == 0.0

    def test_exceeds_limit(self) -> None:
        reward = LengthPenaltyReward(max_length=10, penalty_per_char=0.1)
        prompts = ["test"]
        responses = ["a" * 20]
        result = reward(prompts, responses)
        assert result[0] == -1.0


class TestRepetitionPenalty:
    def test_no_repetition(self) -> None:
        reward = RepetitionPenalty(ngram_size=2)
        prompts = ["test"]
        responses = ["one two three four five"]
        result = reward(prompts, responses)
        assert result[0] == 0.0

    def test_with_repetition(self) -> None:
        reward = RepetitionPenalty(ngram_size=2, penalty_scale=1.0)
        prompts = ["test"]
        responses = ["hello world hello world"]
        result = reward(prompts, responses)
        assert result[0] < 0.0


class TestCompositeReward:
    def test_weighted_combination(self) -> None:
        length_reward = LengthReward(target_length=10, tolerance=5)
        keyword_reward = KeywordReward(required_keywords=["test"], reward_per_keyword=1.0)

        composite = CompositeReward([
            (length_reward, 0.5),
            (keyword_reward, 0.5),
        ])

        prompts = ["query"]
        responses = ["test" + "a" * 6]

        result = composite(prompts, responses)
        assert result[0] == 1.0

    def test_breakdown(self) -> None:
        length_reward = LengthReward(target_length=10, tolerance=5)
        keyword_reward = KeywordReward(required_keywords=["test"], reward_per_keyword=1.0)

        composite = CompositeReward([
            (length_reward, 1.0),
            (keyword_reward, 1.0),
        ])

        prompts = ["query"]
        responses = ["test" + "a" * 6]

        breakdown = composite.get_breakdown(prompts, responses)
        assert "LengthReward" in breakdown
        assert "KeywordReward" in breakdown
        assert "total" in breakdown


class TestThresholdReward:
    def test_above_threshold(self) -> None:
        base = LengthReward(target_length=10, tolerance=5)
        threshold = ThresholdReward(base, threshold=0.5, above_threshold=1.0, below_threshold=0.0)
        prompts = ["test"]
        responses = ["a" * 10]
        result = threshold(prompts, responses)
        assert result[0] == 1.0


class TestClippedReward:
    def test_clipping(self) -> None:
        base = KeywordReward(
            required_keywords=["a", "b", "c", "d", "e"],
            reward_per_keyword=1.0,
        )
        clipped = ClippedReward(base, min_value=-1.0, max_value=2.0)
        prompts = ["test"]
        responses = ["a b c d e"]
        result = clipped(prompts, responses)
        assert result[0] == 2.0


class TestScaledReward:
    def test_scaling(self) -> None:
        base = LengthReward(target_length=10, tolerance=5)
        scaled = ScaledReward(base, scale=2.0, offset=1.0)
        prompts = ["test"]
        responses = ["a" * 10]
        result = scaled(prompts, responses)
        assert result[0] == 3.0
