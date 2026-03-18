from collections import deque
from dataclasses import dataclass, field
from typing import Any

from explore_rl.core.types import TrainingMetrics


@dataclass
class MetricsTracker:
    window_size: int = 100
    _history: dict[str, deque[float]] = field(default_factory=dict)
    _all_values: dict[str, list[float]] = field(default_factory=dict)
    _step: int = 0

    def log(self, metrics: TrainingMetrics) -> None:
        self._step += 1
        self._log_value("loss", metrics.loss)
        self._log_value("policy_loss", metrics.policy_loss)
        self._log_value("value_loss", metrics.value_loss)
        self._log_value("entropy", metrics.entropy)
        self._log_value("kl_divergence", metrics.kl_divergence)
        self._log_value("clip_fraction", metrics.clip_fraction)

        for key, value in metrics.extras.items():
            self._log_value(key, value)

    def _log_value(self, name: str, value: float) -> None:
        if name not in self._history:
            self._history[name] = deque(maxlen=self.window_size)
            self._all_values[name] = []

        self._history[name].append(value)
        self._all_values[name].append(value)

    def get_latest(self, name: str) -> float | None:
        if name not in self._history or len(self._history[name]) == 0:
            return None
        return self._history[name][-1]

    def get_average(self, name: str) -> float | None:
        if name not in self._history or len(self._history[name]) == 0:
            return None
        return sum(self._history[name]) / len(self._history[name])

    def get_all_averages(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in self._history:
            avg = self.get_average(name)
            if avg is not None:
                result[name] = avg
        return result

    def get_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"step": self._step}

        for name in self._history:
            latest = self.get_latest(name)
            avg = self.get_average(name)
            if latest is not None:
                summary[f"{name}_latest"] = latest
            if avg is not None:
                summary[f"{name}_avg"] = avg

        return summary

    @property
    def step(self) -> int:
        return self._step

    def reset(self) -> None:
        self._history.clear()
        self._all_values.clear()
        self._step = 0


class ConsoleLogger:
    def __init__(self, log_interval: int = 10, metrics_to_log: list[str] | None = None) -> None:
        self.log_interval = log_interval
        self.metrics_to_log = metrics_to_log or [
            "loss",
            "policy_loss",
            "value_loss",
            "kl_divergence",
        ]

    def log(self, tracker: MetricsTracker) -> None:
        if tracker.step % self.log_interval != 0:
            return

        parts = [f"Step {tracker.step}"]

        for name in self.metrics_to_log:
            avg = tracker.get_average(name)
            if avg is not None:
                parts.append(f"{name}: {avg:.4f}")

        print(" | ".join(parts))


class WandbLogger:
    def __init__(self, project: str, config: dict[str, Any] | None = None) -> None:
        try:
            import wandb  # type: ignore[import-not-found]

            self.wandb: Any = wandb
        except ImportError as e:
            raise ImportError("wandb package required for WandbLogger") from e

        self.wandb.init(project=project, config=config)

    def log(self, tracker: MetricsTracker) -> None:
        metrics = {name: tracker.get_latest(name) for name in tracker._history if tracker.get_latest(name) is not None}
        metrics["step"] = tracker.step
        self.wandb.log(metrics)

    def finish(self) -> None:
        self.wandb.finish()
