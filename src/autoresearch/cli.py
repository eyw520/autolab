import argparse
import importlib
from pathlib import Path
import sys

from autoresearch.runner import run


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

    project_path = Path(args.project)
    if project_path.exists():
        src_path = project_path / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        else:
            sys.path.insert(0, str(project_path))
        module_name = project_path.name.replace("-", "_")
    else:
        module_name = args.project

    try:
        project_module = importlib.import_module(f"{module_name}.interface")
    except ModuleNotFoundError as e:
        print(f"Error: Could not import project '{module_name}'")
        print(f"  {e}")
        sys.exit(1)

    harness = project_module.harness
    experiment = project_module.experiment

    if args.prepare:
        print(f"Preparing data for {module_name}...")
        harness.prepare()
        print("Done!")
    else:
        print(f"Running experiment: {module_name}")
        run(harness, experiment)


if __name__ == "__main__":
    main()
