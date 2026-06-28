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
    prefix = _class_prefix(package)
    return f"""from typing import Any

from autoresearch.interface import Budget, HarnessSpec, RunContext


class {prefix}Harness:
    @property
    def spec(self) -> HarnessSpec:
        return HarnessSpec(
            primary_metric="metric",
            direction="min",
            budget=Budget(wall_clock_s=300),
            domain={{}},
        )

    def prepare(self) -> None:
        pass

    def evaluate(self, artifact: Any, ctx: RunContext) -> dict[str, float]:
        raise NotImplementedError


class {prefix}Experiment:
    def run(self, harness: {prefix}Harness, ctx: RunContext) -> Any:
        raise NotImplementedError


harness = {prefix}Harness()
experiment = {prefix}Experiment()
"""


def _experiment_md(name: str, package: str) -> str:
    return f"""# {name}

Experiment for autoresearch. Implement the experiment side of
`src/{package}/interface.py`, then run from the repo root:

```bash
poetry run autoresearch experiments/exp-{name}
```

## Contract

- `Harness.spec` declares the fixed `primary_metric`, `direction` (min/max), and `budget`.
- `Harness.evaluate(artifact, ctx)` scores the artifact your experiment produces.
- `Experiment.run(harness, ctx)` owns the training/optimization loop and returns the artifact.

Keep harness code (objective, eval, constraints) fixed; optimize the experiment side.
Reusable loop helpers live in `autoresearch.loops`. See `PROGRAM.md` for the full loop.
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
