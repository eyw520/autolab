from rl.core import Action, Direction, State, Transition
from rl.gridworld import GridPosition


def test_state_is_hashable():
    s1 = State(GridPosition(0, 0))
    s2 = State(GridPosition(0, 0))
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


def test_action_is_hashable():
    a1 = Action(Direction.UP)
    a2 = Action(Direction.UP)
    assert a1 == a2
    assert hash(a1) == hash(a2)


def test_transition_captures_full_step():
    t = Transition(
        state=State(GridPosition(0, 0)),
        action=Action(Direction.RIGHT),
        reward=-0.1,
        next_state=State(GridPosition(0, 1)),
        done=False,
    )
    assert t.reward == -0.1
    assert t.next_state.value.col == 1
