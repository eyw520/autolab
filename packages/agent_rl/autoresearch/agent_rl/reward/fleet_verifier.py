from autoresearch.agent_rl.envs.fleet import FleetEnv
from autoresearch.agent_rl.envs.protocol import AgentEnv
from autoresearch.agent_rl.types import Trajectory


class FleetVerifierReward:
    def __init__(self, verifier_key: str, final_answer_key: str = "final_answer") -> None:
        self._verifier_key = verifier_key
        self._final_answer_key = final_answer_key

    def __call__(self, env: AgentEnv, trajectory: Trajectory) -> float:
        if not isinstance(env, FleetEnv):
            raise TypeError("FleetVerifierReward requires a FleetEnv produced by make_fleet_env_factory")
        raise NotImplementedError(
            f"FleetVerifierReward({self._verifier_key!r}) is a stub: connect it to the "
            "fleet-sdk verifier run surface (fleet.verifiers) for your task suite, returning its score."
        )
