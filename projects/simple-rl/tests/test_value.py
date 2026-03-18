from rl.core import Action, Direction, State
from rl.gridworld import GridPosition, GridWorld
from rl.policies import RandomPolicy
from rl.value import (
    QTable,
    ValueFunction,
    collect_episode,
    compute_advantage,
    compute_returns,
    estimate_value_from_q,
)


def test_q_table_default_value():
    q = QTable(default=0.0)
    state = State(GridPosition(0, 0))
    action = Action(Direction.UP)
    assert q(state, action) == 0.0


def test_q_table_update():
    q = QTable()
    state = State(GridPosition(0, 0))
    action = Action(Direction.UP)
    q.update(state, action, 5.0)
    assert q(state, action) == 5.0


def test_value_function_default_and_update():
    v = ValueFunction(default=-1.0)
    state = State(GridPosition(0, 0))
    assert v(state) == -1.0
    v.update(state, 10.0)
    assert v(state) == 10.0


def test_compute_advantage():
    q = lambda s, a: 5.0
    v = lambda s: 3.0
    state = State(GridPosition(0, 0))
    action = Action(Direction.UP)
    advantage = compute_advantage(q, v, state, action)
    assert advantage == 2.0


def test_compute_advantage_negative():
    q = lambda s, a: 1.0
    v = lambda s: 4.0
    state = State(GridPosition(0, 0))
    action = Action(Direction.UP)
    advantage = compute_advantage(q, v, state, action)
    assert advantage == -3.0


def test_estimate_value_from_q_uniform():
    q_values = {
        (State(GridPosition(0, 0)), Action(Direction.UP)): 2.0,
        (State(GridPosition(0, 0)), Action(Direction.DOWN)): 4.0,
        (State(GridPosition(0, 0)), Action(Direction.LEFT)): 6.0,
        (State(GridPosition(0, 0)), Action(Direction.RIGHT)): 8.0,
    }
    q = lambda s, a: q_values.get((s, a), 0.0)
    policy = RandomPolicy()
    state = State(GridPosition(0, 0))
    actions = [Action(d) for d in Direction]
    v = estimate_value_from_q(q, policy, state, actions)
    assert abs(v - 5.0) < 1e-6


def test_collect_episode_terminates_at_goal():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(0, 1))
    policy = RandomPolicy()
    trajectory = collect_episode(env, policy, max_steps=1000)
    assert trajectory[-1].done is True
    assert trajectory[-1].next_state.value == GridPosition(0, 1)


def test_compute_returns_single_step():
    from rl.core import Transition

    trajectory = [
        Transition(
            State(GridPosition(0, 0)),
            Action(Direction.RIGHT),
            10.0,
            State(GridPosition(0, 1)),
            True,
        )
    ]
    returns = compute_returns(trajectory, gamma=0.99)
    assert returns == [10.0]


def test_compute_returns_multiple_steps():
    from rl.core import Transition

    trajectory = [
        Transition(
            State(GridPosition(0, 0)),
            Action(Direction.RIGHT),
            -1.0,
            State(GridPosition(0, 1)),
            False,
        ),
        Transition(
            State(GridPosition(0, 1)),
            Action(Direction.RIGHT),
            10.0,
            State(GridPosition(0, 2)),
            True,
        ),
    ]
    returns = compute_returns(trajectory, gamma=0.5)
    assert returns[1] == 10.0
    assert returns[0] == -1.0 + 0.5 * 10.0
