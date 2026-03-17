from collections import defaultdict
from typing import Callable

from rl.core import Action, Environment, Policy, State, Transition, S, A


class QTable:
    def __init__(self, default: float = 0.0):
        self._table: dict[tuple[State, Action], float] = defaultdict(lambda: default)

    def __call__(self, state: State, action: Action) -> float:
        return self._table[(state, action)]

    def update(self, state: State, action: Action, value: float) -> None:
        self._table[(state, action)] = value

    def items(self):
        return self._table.items()


class ValueFunction:
    def __init__(self, default: float = 0.0):
        self._table: dict[State, float] = defaultdict(lambda: default)

    def __call__(self, state: State) -> float:
        return self._table[state]

    def update(self, state: State, value: float) -> None:
        self._table[state] = value


def compute_advantage(
    q: Callable[[State[S], Action[A]], float],
    v: Callable[[State[S]], float],
    state: State[S],
    action: Action[A],
) -> float:
    return q(state, action) - v(state)


def estimate_value_from_q(
    q: Callable[[State[S], Action[A]], float],
    policy: Policy[S, A],
    state: State[S],
    actions: list[Action[A]],
) -> float:
    probs = policy.probabilities(state, actions)
    return sum(probs[a] * q(state, a) for a in actions)


def collect_episode(
    env: Environment[S, A],
    policy: Policy[S, A],
    max_steps: int = 1000,
) -> list[Transition[S, A]]:
    trajectory = []
    state = env.reset()
    for _ in range(max_steps):
        actions = env.actions(state)
        action = policy.select(state, actions)
        next_state, reward, done = env.step(action)
        trajectory.append(Transition(state, action, reward, next_state, done))
        if done:
            break
        state = next_state
    return trajectory


def compute_returns(
    trajectory: list[Transition[S, A]],
    gamma: float = 0.99,
) -> list[float]:
    returns = []
    g = 0.0
    for t in reversed(trajectory):
        g = t.reward + gamma * g
        returns.append(g)
    return list(reversed(returns))
