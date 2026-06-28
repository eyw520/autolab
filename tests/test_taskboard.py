from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.envs.taskboard import TaskBoardEnv, make_taskboard_env_factory
from autoresearch.agent_rl.verifiers.state import Change


def _mixed_env():
    for seed in range(100):
        env = TaskBoardEnv(seed=seed, num_items=3)
        if 0 in env.goal and 1 in env.goal:
            return env
    raise AssertionError("no mixed-goal seed found")


def test_reset_is_deterministic_per_seed():
    a = TaskBoardEnv(seed=7, num_items=3).reset(seed=7)
    b = TaskBoardEnv(seed=7, num_items=3).reset(seed=7)
    assert a["goal"] == b["goal"]


def test_factory_returns_agentenv():
    assert isinstance(make_taskboard_env_factory(3)(0), AgentEnv)


def test_completing_goal_solves_and_verifies():
    env = TaskBoardEnv(seed=5, num_items=3)
    env.reset(seed=5)
    for i, g in enumerate(env.goal):
        if g:
            env.step({"tool": "complete", "item": i})
    assert env.is_solved()
    expected = [Change("items", str(i), "done", 1) for i, g in enumerate(env.goal) if g]
    env.snapshot("seed").diff(env.snapshot("current")).expect_only(expected)


def test_off_goal_completion_ends_unsolved():
    env = _mixed_env()
    result = env.step({"tool": "complete", "item": env.goal.index(0)})
    assert result.done
    assert not env.is_solved()


def test_goal_size_fixes_goal_count():
    env = TaskBoardEnv(seed=1, num_items=5, goal_size=3)
    assert sum(env.goal) == 3


def test_ordered_out_of_order_fails():
    env = TaskBoardEnv(seed=1, num_items=5, goal_size=3, ordered=True)
    goal_items = [i for i, g in enumerate(env.goal) if g]
    result = env.step({"tool": "complete", "item": goal_items[-1]})
    assert result.done
    assert not env.is_solved()


def test_ordered_in_order_solves():
    env = TaskBoardEnv(seed=1, num_items=5, goal_size=3, ordered=True)
    for i in sorted(i for i, g in enumerate(env.goal) if g):
        env.step({"tool": "complete", "item": i})
    assert env.is_solved()
