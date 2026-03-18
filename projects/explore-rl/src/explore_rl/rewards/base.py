from abc import ABC, abstractmethod

from torch import Tensor


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, prompts: list[str], responses: list[str]) -> Tensor:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
