## Overview

`autolab` is a workspace for autonomous ML research experiments. The core harness is
domain-agnostic: it runs supervised, reinforcement-learning, or any other experiment
that implements two small protocols, and an agent optimizes the experiment by keeping
changes that improve a declared metric and reverting those that don't.

Reference `PROGRAM.md` for the full setup and the autonomous experiment loop.

## Architecture

Three layers, by responsibility:

1. **Contract** — `src/autoresearch/`. The universal primitives (`interface.py`:
   `Budget`, `RunContext`, `HarnessSpec`, and the `Harness`/`Experiment` protocols),
   the domain-agnostic `runner.py` (seed -> run -> evaluate -> report), `util.py`, the
   `cli.py`/`scaffold.py`/`launch.py` entry points. Paradigm-agnostic: it knows nothing
   about gradients, rollouts, or tokens. Keep it dependency-light.

2. **Training protocols** — `packages/`. One paradigm per package, each a reusable
   training loop, sharing the `autoresearch` namespace:
   - `packages/supervised/` -> `autoresearch.supervised.loop` (gradient accumulation,
     divergence detection, timing).
   - `packages/agent_rl/` -> `autoresearch.agent_rl.*` (agent RL: see below).
   These are path-dependencies of the root project, installed editable.

3. **Experiments** — `experiments/exp-<name>/`. Each lives in its own directory
   (conventionally its own git repo) with `src/<package>/` (split into a fixed
   `harness/` and a modifiable `experiment/`),
   an `interface.py` exposing module-level `harness` and `experiment`, and an
   `EXPERIMENT.md`. Experiments run in the root environment (no per-experiment
   `pyproject.toml`); the runner discovers the package under `src/` automatically.

Current experiments: `exp-pretraining-gpt` (supervised LM), `exp-rl-control`
(reference RL on CartPole), `exp-fleet-agent-rl` and `exp-fleet-taskboard` (agent RL).

## Agent RL (`autoresearch.agent_rl`)

For training/evaluating agents in environments. Everything is protocol-typed so
backends swap without touching the loop:

- Protocols (`types.py`, `envs/protocol.py`, `reward/protocol.py`, `algo/grpo.py`):
  `AgentEnv` (`reset`/`step`/`close`), `Reward`, `Policy`, `TrainablePolicy`,
  `GRPOTrainer`.
- Loop + rollout: `agent_rl_loop` (`loop.py`), `RolloutEngine` (grouped, parallel).
- Local, dependency-light building blocks: `envs/taskboard.py` (a stateful,
  multi-step, Fleet-shaped env), the `verifiers/` toolkit (`StateSnapshot`,
  `diff().expect_only(...)` — Fleet's verifier idiom), and `TorchGRPOTrainer`
  (`algo/grpo_torch.py`, group-relative policy optimization, torch-only).
- Heavy backends are optional extras, lazily imported behind a single adapter each,
  never on the default path: `[fleet]` (`envs/fleet.py`, `reward/fleet_verifier.py`),
  `[anthropic]` (`policies/claude.py`), `[grpo]` (torch), `[train]` (vLLM/SkyRL).

Two orthogonal axes — do not conflate them:
- **Where it runs** (location): `autoresearch-launch --target {local,cloud}`. The
  experiment is unchanged; `get_device()` adapts to the machine. Local is the default.
- **What it talks to** (backend): local env/torch policy vs. Fleet env/LLM policy,
  selected via the optional extras above. Independent of location.

## Using it

- Scaffold: `poetry run autoresearch-new <name>` (creates `experiments/exp-<name>/`).
- Implement the two protocols in the experiment's `interface.py`.
- Run: `poetry run autoresearch experiments/exp-<name>`, or
  `poetry run autoresearch-launch --target {local,cloud} experiments/exp-<name>`.
- Iterate: change files under `experiment/`, keep the change if the primary metric
  improved in `spec.direction`, else revert. See `PROGRAM.md`.

## First Principles

1. Avoid including comments or docstrings unless absolutely necessary.
2. Never use emojis in your output code generation.
3. Do not generate `README.md` or `.md` files during code generation unless instructed to do so.
4. Avoid `__init__.py` files - use direct imports instead. `autoresearch` is a PEP 420
   namespace package shared across `src/` and `packages/*/`; adding an `__init__.py`
   anywhere in it breaks the namespace merge.
5. Keep heavy third-party dependencies behind optional extras and import them lazily
   inside the one adapter that needs them — never at module load on the default path.

## Utilities

Run from the root directory (globs cover `src/`, `packages/*/autoresearch`, and
`experiments/*/src/`):

- `make install` - `poetry install`
- `make lint` - Auto-fix lint and format
- `make check` - Lint + format check (no writes)
- `make typecheck` - Type check `src/` and `packages/*/autoresearch`
- `make test` - Run `pytest tests/`

## Configuration files

- `ruff.toml` - Ruff linting/formatting
- `mypy.ini` - Mypy (namespace-package aware)
- `pyrightconfig.json` - Pyright type checking
- `poetry.toml` - Poetry virtualenv settings
