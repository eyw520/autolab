import random
from typing import Callable

from rl.core import Action, Policy, State, S, A


class RandomPolicy(Policy[S, A]):
    def select(self, state: State[S], available_actions: list[Action[A]]) -> Action[A]:
        return random.choice(available_actions)

    def probabilities(
        self, state: State[S], available_actions: list[Action[A]]
    ) -> dict[Action[A], float]:
        p = 1.0 / len(available_actions)
        return {a: p for a in available_actions}


class EpsilonGreedy(Policy[S, A]):
    def __init__(
        self,
        q_function: Callable[[State[S], Action[A]], float],
        epsilon: float = 0.1,
    ):
        self.q = q_function
        self.epsilon = epsilon

    def select(self, state: State[S], available_actions: list[Action[A]]) -> Action[A]:
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        return max(available_actions, key=lambda a: self.q(state, a))

    def probabilities(
        self, state: State[S], available_actions: list[Action[A]]
    ) -> dict[Action[A], float]:
        n = len(available_actions)
        best = max(available_actions, key=lambda a: self.q(state, a))
        probs = {}
        for a in available_actions:
            if a == best:
                probs[a] = 1.0 - self.epsilon + self.epsilon / n
            else:
                probs[a] = self.epsilon / n
        return probs


class GreedyPolicy(Policy[S, A]):
    def __init__(self, q_function: Callable[[State[S], Action[A]], float]):
        self.q = q_function

    def select(self, state: State[S], available_actions: list[Action[A]]) -> Action[A]:
        return max(available_actions, key=lambda a: self.q(state, a))

    def probabilities(
        self, state: State[S], available_actions: list[Action[A]]
    ) -> dict[Action[A], float]:
        best = max(available_actions, key=lambda a: self.q(state, a))
        return {a: 1.0 if a == best else 0.0 for a in available_actions}
