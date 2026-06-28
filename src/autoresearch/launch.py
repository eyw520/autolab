import argparse
import importlib.util
import subprocess
import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class Launcher(Protocol):
    def run(self, project: str, prepare: bool = False) -> int: ...


class LocalLauncher:
    def run(self, project: str, prepare: bool = False) -> int:
        command = [sys.executable, "-m", "autoresearch.cli", project]
        if prepare:
            command.append("--prepare")
        return subprocess.run(command, check=False).returncode


class CloudLauncher:
    def __init__(self, cluster: str = "autoresearch", accelerators: str | None = None, down: bool = True) -> None:
        self._cluster = cluster
        self._accelerators = accelerators
        self._down = down

    def run(self, project: str, prepare: bool = False) -> int:
        if importlib.util.find_spec("sky") is None:
            raise ImportError("The 'cloud' target requires SkyPilot. Install it with: pip install skypilot")
        raise NotImplementedError(
            "CloudLauncher is a stub. Build a sky.Task whose run command is "
            f"'autoresearch {project}'{' --prepare' if prepare else ''}, set its resources "
            f"(accelerators={self._accelerators!r}), and sky.launch it on cluster "
            f"{self._cluster!r} (down={self._down}). The experiment adapts to the remote "
            "device via get_device(); no experiment changes are needed."
        )


def resolve_launcher(target: str, accelerators: str | None = None) -> Launcher:
    if target == "local":
        return LocalLauncher()
    if target == "cloud":
        return CloudLauncher(accelerators=accelerators)
    raise ValueError(f"unknown target {target!r}: expected 'local' or 'cloud'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an autoresearch experiment locally or on the cloud")
    parser.add_argument("project", type=str, help="Path to the experiment directory")
    parser.add_argument("--target", choices=["local", "cloud"], default="local", help="Where to run (default: local)")
    parser.add_argument("--accelerators", type=str, default=None, help="Cloud accelerator spec, e.g. 'A100:1'")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation instead of training")
    args = parser.parse_args()

    print(f"Target: {args.target}")
    launcher = resolve_launcher(args.target, accelerators=args.accelerators)
    try:
        code = launcher.run(args.project, prepare=args.prepare)
    except (ImportError, NotImplementedError) as error:
        print(f"Error: {error}")
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
