from rl.core import Environment, S, A
from rl.policies import EpsilonGreedy, GreedyPolicy
from rl.value import QTable, collect_episode, compute_returns


def q_learning(
    env: Environment[S, A],
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> QTable:
    q = QTable()
    policy = EpsilonGreedy(q, epsilon)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.actions(state)
            action = policy.select(state, actions)
            next_state, reward, done = env.step(action)
            next_actions = env.actions(next_state)
            max_next_q = max(q(next_state, a) for a in next_actions)
            td_target = reward + gamma * max_next_q * (1 - int(done))
            td_error = td_target - q(state, action)
            q.update(state, action, q(state, action) + alpha * td_error)
            state = next_state

    return q


def sarsa(
    env: Environment[S, A],
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> QTable:
    q = QTable()
    policy = EpsilonGreedy(q, epsilon)

    for _ in range(episodes):
        state = env.reset()
        actions = env.actions(state)
        action = policy.select(state, actions)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_actions = env.actions(next_state)
            next_action = policy.select(next_state, next_actions)
            td_target = reward + gamma * q(next_state, next_action) * (1 - int(done))
            td_error = td_target - q(state, action)
            q.update(state, action, q(state, action) + alpha * td_error)
            state = next_state
            action = next_action

    return q


def monte_carlo(
    env: Environment[S, A],
    episodes: int = 500,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> QTable:
    q = QTable()
    counts: dict[tuple, int] = {}
    policy = EpsilonGreedy(q, epsilon)

    for _ in range(episodes):
        trajectory = collect_episode(env, policy)
        returns = compute_returns(trajectory, gamma)
        visited = set()
        for t, g in zip(trajectory, returns):
            key = (t.state, t.action)
            if key in visited:
                continue
            visited.add(key)
            counts[key] = counts.get(key, 0) + 1
            old_q = q(t.state, t.action)
            q.update(t.state, t.action, old_q + (g - old_q) / counts[key])

    return q


def evaluate_policy(
    env: Environment[S, A],
    q: QTable,
    episodes: int = 100,
    max_steps: int = 100,
) -> float:
    policy = GreedyPolicy(q)
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            actions = env.actions(state)
            action = policy.select(state, actions)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / episodes
