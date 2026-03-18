from rl.algorithms import evaluate_policy, monte_carlo, q_learning, sarsa
from rl.gridworld import GridWorld


def test_q_learning_learns_to_reach_goal():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    q = q_learning(env, episodes=200, epsilon=0.3)
    avg_reward = evaluate_policy(env, q, episodes=50)
    assert avg_reward > 0


def test_sarsa_learns_to_reach_goal():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    q = sarsa(env, episodes=200, epsilon=0.3)
    avg_reward = evaluate_policy(env, q, episodes=50)
    assert avg_reward > 0


def test_monte_carlo_learns_to_reach_goal():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    q = monte_carlo(env, episodes=200, epsilon=0.3)
    avg_reward = evaluate_policy(env, q, episodes=50)
    assert avg_reward > 0


def test_q_learning_handles_walls():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), walls={(1, 1)})
    q = q_learning(env, episodes=300, epsilon=0.3)
    avg_reward = evaluate_policy(env, q, episodes=50)
    assert avg_reward > 0
