import argparse
import importlib
from pathlib import Path
import sys

from autoresearch.interface import Experiment, Harness
from autoresearch.runner import run


def _resolve_module_name(project_path: Path, search_path: Path) -> str:
    packages = [
        d.name
        for d in search_path.iterdir()
        if d.is_dir() and not d.name.startswith((".", "__")) and any(d.glob("*.py"))
    ]
    if len(packages) == 1:
        return packages[0]
    name = project_path.name
    if name.startswith("exp-"):
        name = name[len("exp-") :]
    return name.replace("-", "_")


def load_project(project: str) -> tuple[Harness, Experiment]:
    project_path = Path(project)
    if project_path.exists():
        src_path = project_path / "src"
        search_path = src_path if src_path.exists() else project_path
        sys.path.insert(0, str(search_path))
        module_name = _resolve_module_name(project_path, search_path)
    else:
        module_name = project

    project_module = importlib.import_module(f"{module_name}.interface")

    try:
        harness = project_module.harness
        experiment = project_module.experiment
    except AttributeError as e:
        raise AttributeError(
            f"Project '{module_name}.interface' must define module-level 'harness' and 'experiment' objects ({e})"
        ) from e

    if not isinstance(harness, Harness):
        raise TypeError(
            f"'{module_name}.interface.harness' does not implement the Harness protocol "
            f"(missing: {_missing_members(harness, Harness)})"
        )
    if not isinstance(experiment, Experiment):
        raise TypeError(
            f"'{module_name}.interface.experiment' does not implement the Experiment protocol "
            f"(missing: {_missing_members(experiment, Experiment)})"
        )

    return harness, experiment


def _missing_members(obj: object, protocol: type) -> str:
    members = getattr(protocol, "__protocol_attrs__", None) or [
        name for name in dir(protocol) if not name.startswith("_")
    ]
    missing = [name for name in members if not hasattr(obj, name)]
    return ", ".join(sorted(missing)) or "none (signature mismatch)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an autoresearch experiment")
    parser.add_argument(
        "project",
        type=str,
        help="Path to project directory or module name (e.g., 'pretraining_gpt')",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run data preparation instead of training",
    )
    args = parser.parse_args()

    try:
        harness, experiment = load_project(args.project)
    except (ModuleNotFoundError, AttributeError, TypeError) as e:
        print(f"Error: Could not load project '{args.project}'")
        print(f"  {e}")
        sys.exit(1)

    label = Path(args.project).name

    if args.prepare:
        print(f"Preparing data for {label}...")
        harness.prepare()
        print("Done!")
    else:
        print(f"Running experiment: {label}")
        run(harness, experiment)


if __name__ == "__main__":
    main()
