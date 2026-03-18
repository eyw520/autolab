from rl.core import Action, Direction, State
from rl.gridworld import GridPosition
from rl.policies import EpsilonGreedy, GreedyPolicy, RandomPolicy


def test_random_policy_uniform_distribution():
    policy = RandomPolicy()
    state = State(GridPosition(0, 0))
    actions = [Action(d) for d in Direction]
    probs = policy.probabilities(state, actions)
    assert all(abs(p - 0.25) < 1e-6 for p in probs.values())


def test_greedy_policy_selects_max_q():
    q_values = {
        (State(GridPosition(0, 0)), Action(Direction.UP)): 1.0,
        (State(GridPosition(0, 0)), Action(Direction.DOWN)): 2.0,
        (State(GridPosition(0, 0)), Action(Direction.LEFT)): 0.5,
        (State(GridPosition(0, 0)), Action(Direction.RIGHT)): 3.0,
    }
    policy = GreedyPolicy(lambda s, a: q_values.get((s, a), 0.0))
    state = State(GridPosition(0, 0))
    actions = [Action(d) for d in Direction]
    action = policy.select(state, actions)
    assert action.value == Direction.RIGHT


def test_greedy_policy_probability_one_for_best():
    q_values = {
        (State(GridPosition(0, 0)), Action(Direction.RIGHT)): 5.0,
    }
    policy = GreedyPolicy(lambda s, a: q_values.get((s, a), 0.0))
    state = State(GridPosition(0, 0))
    actions = [Action(d) for d in Direction]
    probs = policy.probabilities(state, actions)
    assert probs[Action(Direction.RIGHT)] == 1.0
    assert sum(probs.values()) == 1.0


def test_epsilon_greedy_probability_distribution():
    q_values = {
        (State(GridPosition(0, 0)), Action(Direction.RIGHT)): 5.0,
    }
    policy = EpsilonGreedy(lambda s, a: q_values.get((s, a), 0.0), epsilon=0.2)
    state = State(GridPosition(0, 0))
    actions = [Action(d) for d in Direction]
    probs = policy.probabilities(state, actions)
    best_prob = 1.0 - 0.2 + 0.2 / 4
    other_prob = 0.2 / 4
    assert abs(probs[Action(Direction.RIGHT)] - best_prob) < 1e-6
    for d in [Direction.UP, Direction.DOWN, Direction.LEFT]:
        assert abs(probs[Action(d)] - other_prob) < 1e-6
