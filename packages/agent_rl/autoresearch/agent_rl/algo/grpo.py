from typing import Protocol, runtime_checkable

from autoresearch.agent_rl.types import Policy, Trajectory


@runtime_checkable
class GRPOTrainer(Protocol):
    def update(self, policy: Policy, trajectories: list[Trajectory]) -> dict[str, float]: ...


class UnconfiguredTrainer:
    def update(self, policy: Policy, trajectories: list[Trajectory]) -> dict[str, float]:
        raise NotImplementedError(
            "No GRPO backend configured. Implement the GRPOTrainer protocol "
            "(e.g. wrap SkyRL behind it) and install the 'train' extra."
        )
