from rl.core import Action, Direction
from rl.gridworld import GridPosition, GridWorld


def test_reset_returns_start_position():
    env = GridWorld(start=(1, 1))
    state = env.reset()
    assert state.value == GridPosition(1, 1)


def test_step_moves_agent():
    env = GridWorld(start=(0, 0))
    env.reset()
    state, _, _ = env.step(Action(Direction.RIGHT))
    assert state.value == GridPosition(0, 1)


def test_wall_blocks_movement():
    env = GridWorld(start=(0, 0), walls={(0, 1)})
    env.reset()
    state, _, _ = env.step(Action(Direction.RIGHT))
    assert state.value == GridPosition(0, 0)


def test_boundary_blocks_movement():
    env = GridWorld(start=(0, 0))
    env.reset()
    state, _, _ = env.step(Action(Direction.LEFT))
    assert state.value == GridPosition(0, 0)


def test_reaching_goal_gives_reward_and_terminates():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(0, 1), goal_reward=10.0)
    env.reset()
    state, reward, done = env.step(Action(Direction.RIGHT))
    assert state.value == GridPosition(0, 1)
    assert reward == 10.0
    assert done is True


def test_non_goal_step_gives_penalty():
    env = GridWorld(start=(0, 0), goal=(3, 3), step_penalty=-0.5)
    env.reset()
    _, reward, done = env.step(Action(Direction.RIGHT))
    assert reward == -0.5
    assert done is False


def test_available_actions_are_all_directions():
    env = GridWorld()
    state = env.reset()
    actions = env.actions(state)
    assert len(actions) == 4
    assert {a.value for a in actions} == set(Direction)


def test_render_shows_agent_and_goal():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    env.reset()
    grid = env.render()
    assert "A" in grid
    assert "G" in grid
