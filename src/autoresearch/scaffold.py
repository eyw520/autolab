import argparse
from pathlib import Path
import re


_GITIGNORE = """__pycache__/
*.py[cod]
.venv/
.ruff_cache/
.mypy_cache/
.pytest_cache/
.DS_Store

run.log
results.tsv
"""


def _interface_template(package: str) -> str:
    return f"""from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from autoresearch.interface import HarnessConfig, TrainingConfig


class {_class_prefix(package)}Harness:
    @property
    def config(self) -> HarnessConfig:
        return HarnessConfig(
            time_budget=300,
            seq_len=1024,
            primary_metric="val_loss",
            cache_dir="~/.cache/{package}",
        )

    def prepare(self) -> None:
        raise NotImplementedError("Download data / train tokenizer here.")

    def get_vocab_size(self) -> int:
        raise NotImplementedError

    def make_dataloader(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        raise NotImplementedError

    def evaluate(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, float]:
        raise NotImplementedError


class {_class_prefix(package)}Experiment:
    def get_training_config(self, device: torch.device) -> TrainingConfig:
        return TrainingConfig(total_batch_size=2**12, device_batch_size=8)

    def build_model(
        self,
        vocab_size: int,
        seq_len: int,
        device: torch.device,
    ) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> optim.Optimizer:
        raise NotImplementedError

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        progress: float,
        device: torch.device,
    ) -> float:
        raise NotImplementedError


harness = {_class_prefix(package)}Harness()
experiment = {_class_prefix(package)}Experiment()
"""


def _experiment_md(name: str, package: str) -> str:
    return f"""# {name}

Experiment for autoresearch. Implement the harness and experiment in
`src/{package}/interface.py`, then run from the repo root:

```bash
poetry run autoresearch experiments/exp-{name}
```

## Boundaries

- `src/{package}/experiment/` and `interface.py` are **modifiable** (model, optimizer, training).
- `src/{package}/harness/` is **fixed** (data, eval, constraints).

See `PROGRAM.md` at the repo root for the full experiment loop.
"""


def _class_prefix(package: str) -> str:
    return "".join(part.capitalize() for part in package.split("_"))


def create_experiment(name: str, experiments_dir: Path) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ValueError(f"Invalid experiment name '{name}': use lowercase letters, digits, and hyphens.")
    package = name.replace("-", "_")
    exp_dir = experiments_dir / f"exp-{name}"
    if exp_dir.exists():
        raise FileExistsError(f"{exp_dir} already exists.")

    pkg_dir = exp_dir / "src" / package
    pkg_dir.mkdir(parents=True)
    (exp_dir / ".gitignore").write_text(_GITIGNORE)
    (exp_dir / "EXPERIMENT.md").write_text(_experiment_md(name, package))
    (pkg_dir / "interface.py").write_text(_interface_template(package))
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new autoresearch experiment")
    parser.add_argument("name", type=str, help="Experiment name, e.g. 'my-task' (creates exp-my-task)")
    parser.add_argument(
        "--dir",
        type=str,
        default="experiments",
        help="Directory to create the experiment in (default: experiments)",
    )
    args = parser.parse_args()

    exp_dir = create_experiment(args.name, Path(args.dir))
    package = args.name.replace("-", "_")
    print(f"Created {exp_dir}")
    print("Next steps:")
    print(f"  cd {exp_dir} && git init && git add -A && git commit -m 'init'")
    print(f"  # implement src/{package}/interface.py")
    print(f"  poetry run autoresearch {exp_dir}")


if __name__ == "__main__":
    main()
