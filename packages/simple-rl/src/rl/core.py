from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass(frozen=True)
class State(Generic[S]):
    value: S


@dataclass(frozen=True)
class Action(Generic[A]):
    value: A


@dataclass
class Transition(Generic[S, A]):
    state: State[S]
    action: Action[A]
    reward: float
    next_state: State[S]
    done: bool


class Environment(ABC, Generic[S, A]):
    @abstractmethod
    def reset(self) -> State[S]:
        pass

    @abstractmethod
    def step(self, action: Action[A]) -> tuple[State[S], float, bool]:
        pass

    @abstractmethod
    def actions(self, state: State[S]) -> list[Action[A]]:
        pass


class Policy(ABC, Generic[S, A]):
    @abstractmethod
    def select(self, state: State[S], available_actions: list[Action[A]]) -> Action[A]:
        pass

    @abstractmethod
    def probabilities(
        self, state: State[S], available_actions: list[Action[A]]
    ) -> dict[Action[A], float]:
        pass


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
