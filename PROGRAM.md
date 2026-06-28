# autolab Experiments

How autonomous research experiments are designed, conducted, and run in autolab.

## Interaction paradigm: the agent is the optimizer

autolab treats research as autonomous hill-climbing against a fixed objective. Four
roles:

- **Researcher** (human) defines the *benchmark* — the objective, how it is scored, and
  the budget. Fixed.
- **Agent** (you) is the *optimizer* — proposes and applies changes to the *approach*,
  runs, reads the result, keeps or reverts.
- **Harness** is the immutable grader — produces one primary metric so attempts are
  comparable across the whole project.
- **Runner** executes one attempt deterministically and reports.

The loop: edit the approach -> run -> read `primary_metric` -> keep if it improved in
`spec.direction`, else revert -> repeat. The framework's job is to make "is it better?"
a grep, and to keep the thing being optimized cleanly separable from the thing it is
judged by — so iteration cannot quietly corrupt the benchmark.

## Architecture

Three layers, by responsibility (see `CLAUDE.md` for the short version):

```
autolab/
  src/autoresearch/         # Layer 1 - contract (paradigm-agnostic, dependency-light)
    interface.py            #   Budget, RunContext, HarnessSpec, Harness/Experiment protocols
    runner.py               #   seed -> run -> evaluate -> report
    cli.py                  #   `autoresearch` - discover + run an experiment
    scaffold.py             #   `autoresearch-new` - create an experiment
    launch.py               #   `autoresearch-launch` - run local or on cloud
    util.py                 #   get_device, seed_everything, is_better
  packages/                 # Layer 2 - reusable training protocols (share the autoresearch namespace)
    supervised/             #   autoresearch.supervised.loop
    agent_rl/               #   autoresearch.agent_rl.* (see Agent RL below)
  experiments/              # Layer 3 - concrete experiments
    exp-<name>/
      src/<package>/
        harness/            #   Fixed: objective, eval, constraints
        experiment/         #   Modifiable: the approach (model/policy/optimizer/algorithm)
        interface.py        #   Wires harness to experiment; exposes `harness` and `experiment`
      EXPERIMENT.md         #   Experiment-specific instructions
```

`autoresearch` is a PEP 420 namespace package shared across `src/` and `packages/*/`;
never add an `__init__.py`. Packages are editable path-dependencies of the root project.
Experiments run in the root environment (no per-experiment `pyproject.toml`); the runner
discovers the package under `src/` automatically, so the directory and package names need
not match.

## The contract: two protocols

Defined in `src/autoresearch/interface.py`. Intentionally minimal — the experiment owns
its own loop.

**Harness** (fixed — the benchmark):
- `spec` -> `HarnessSpec(primary_metric, direction, budget, domain)`
  - `direction` is `"min"` or `"max"` (e.g. `val_bpb` -> min, `eval_return` -> max)
  - `budget` is `Budget(wall_clock_s=..., steps=..., env_steps=...)`; any axis set acts as a stop
  - `domain` holds domain-specific config (e.g. `{"seq_len": 2048}`)
- `prepare()` — one-time setup (download data, etc.); may be a no-op
- `evaluate(artifact, ctx)` — score the produced artifact, returning `{metric: value}`

**Experiment** (modifiable — the approach):
- `run(harness, ctx)` — build the model/policy and optimizer, run the loop honoring
  `ctx.budget`, record telemetry via `ctx.record({...})`, and return the artifact.

Domain resources (a dataloader for an LM, an `AgentEnv` for RL) are NOT part of the
universal protocol — they live on the concrete harness and are called by the matching
experiment, which is co-designed with it.

**Reusable loop helpers** live in the layer-2 packages, not the runner. For supervised
training, `autoresearch.supervised.loop.supervised_loop(model, optimizer, dataloader,
step_fn, ctx, config)` provides gradient accumulation, timing, divergence detection, and
progress logging — opt in, don't reinvent.

## Agent RL (`autoresearch.agent_rl`)

For training/evaluating agents in environments. Everything is protocol-typed so backends
swap without touching the loop. An agent-RL experiment assembles four pieces:

- **`env_factory`** (`AgentEnv`: `reset`/`step`/`close`) — reuse
  `envs.taskboard.make_taskboard_env_factory`, or write your own.
- **`reward`** (`Reward`) — write a `verifier(env)` with the `verifiers` toolkit
  (`StateSnapshot`, `diff().expect_only(...)` — Fleet's verifier idiom) wrapped in
  `VerifierReward`.
- **`policy`** — a `TrainablePolicy` (torch net exposing `act` + `action_log_prob` +
  `parameters`) for learning, or a baseline `Policy` (`EchoPolicy`, `ClaudeAgentPolicy`).
- **`trainer`** (`GRPOTrainer`) — `algo.grpo_torch.TorchGRPOTrainer` (group-relative
  policy optimization, torch-only) locally; a distributed trainer later.

The harness wires `env_factory` + `reward` + `evaluate`; the experiment wires `policy` +
`trainer` and calls `agent_rl_loop`. `exp-fleet-taskboard` is the worked reference.

Heavy backends are optional extras, lazily imported behind one adapter each, never on the
default path: `[fleet]` (`envs/fleet.py`, `reward/fleet_verifier.py`), `[anthropic]`
(`policies/claude.py`), `[grpo]` (torch), `[train]` (vLLM/SkyRL).

Two orthogonal axes — do not conflate them:
- **Where it runs** (location): `autoresearch-launch --target {local,cloud}`. Same
  experiment; `get_device()` adapts to the machine. Local is the default.
- **What it talks to** (backend): local env/torch policy vs. Fleet env/LLM policy, chosen
  via the extras above. Independent of location.

## Designing an experiment

Start from the objective, not the model:

1. **Name the artifact and the metric.** What does `run()` produce, and what number judges
   it? Set `primary_metric` and `direction`.
2. **Pick the budget — the stop.** The same budget for every attempt is what makes runs
   comparable.
3. **Split fixed vs modifiable.** Objective, eval, environment dynamics, reward -> `harness/`.
   Architecture, optimizer, algorithm, hyperparameters -> `experiment/`. Get this wrong and
   you either can't improve anything or you optimize the grader.
4. **Co-design `evaluate` with `run`.** `evaluate` must score exactly the artifact `run`
   returns (RL reuses `RolloutEngine` over held-out eval seeds).

## Configuring a new experiment

1. `poetry run autoresearch-new <name>` -> `experiments/exp-<name>/src/<package>/interface.py` stub.
2. Fill `HarnessSpec`: `primary_metric`, `direction`, `budget`, `domain`.
3. Implement `prepare()` (often a no-op).
4. Implement `evaluate(artifact, ctx)` -> `{metric: value}`.
5. Implement `run(harness, ctx)`: build the approach, loop until `ctx.budget.exceeded(...)`,
   `ctx.record(...)` telemetry, return the artifact.
6. Leave the module-level `harness` / `experiment` objects in place.
7. For agent RL, assemble the four pieces above (most have ready local implementations).
8. `make check` and `make typecheck` (globs cover `experiments/*/src/`).

## Running an experiment

All commands run from the autolab root directory.

```bash
poetry run autoresearch --prepare experiments/exp-<name>           # one-time data setup
poetry run autoresearch experiments/exp-<name>                     # run here
poetry run autoresearch-launch --target local experiments/exp-<name>   # run here (explicit)
poetry run autoresearch-launch --target cloud experiments/exp-<name>   # run on a cloud VM (opt-in)
```

The runner seeds everything, builds `RunContext(device, budget, seed)`, calls
`experiment.run` then `harness.evaluate`, and prints a flat record (telemetry + metrics +
`total_seconds` + `peak_vram_mb` + `status: ok|crash`). A `RuntimeError` is reported as a
crash with a NaN metric — a broken attempt is a bad data point, not a halted loop.

Gates:

```bash
make check       # lint + format check (no writes)
make lint        # auto-fix lint and format
make typecheck   # mypy src/ and packages/*/autoresearch
make test        # pytest tests/
```

## Results and reporting

Each run persists to `<experiment>/runs/<run-id>/` (the CLI passes the experiment dir as
the output location automatically; `runs/` is gitignored):

- `result.json` — final metrics, status, seed, the full `spec`, git commit/branch, and
  timestamp. Every number is attributable to an exact config and code state.
- `metrics.jsonl` — one telemetry line per training iteration (the learning curve),
  emitted via `ctx.log_step(...)`. Use `record()` for one-off summary values,
  `log_step()` for per-iteration history.

Summarize and compare runs with the report CLI (sorts by `primary_metric` in
`spec.direction`; crashed/NaN runs last):

```bash
poetry run autoresearch-report experiments/exp-<name>           # leaderboard, best first
poetry run autoresearch-report experiments/exp-<name> --json    # machine-readable
```

## The autonomous loop

Experiments are conventionally their own git repos, so each attempt is a branch and
provenance is just git history.

1. Create a run branch in the experiment (`run/<tag>`).
2. Modify files in `experiment/` or `interface.py`.
3. Commit.
4. Run: `poetry run autoresearch experiments/exp-<name> > experiments/exp-<name>/run.log 2>&1`.
5. Read the result: `autoresearch-report experiments/exp-<name>`, or read
   `runs/<run-id>/result.json` (the per-run `metrics.jsonl` holds the learning curve).
6. Keep the change if the metric improved **in the direction declared by `spec.direction`**, else revert.
7. Repeat indefinitely.

## Project lifecycle

1. **Scaffold** — `autoresearch-new <name>`.
2. **Prepare** — implement `prepare()` (or no-op) and run `--prepare` once.
3. **Baseline** — implement the protocols, run once, record the starting metric.
4. **Iterate** — the autonomous loop; each attempt a branch, kept or reverted by the metric.
5. **Graduate (agent RL)** — prototype against the local env + torch policy + local verifier
   + `TorchGRPOTrainer` (fast, free, CI-able); then swap in real backends one at a time
   behind their extras, and flip `--target cloud` when compute demands it. De-risk the
   research locally; change only the backend to confront the real environment — never both
   at once.

## Available experiments

Each has its own `EXPERIMENT.md`:

- `exp-pretraining-gpt/` — GPT pretraining (minimize `val_bpb`, wall-clock budget)
- `exp-rl-control/` — reference RL on a dependency-free CartPole (maximize `eval_return`)
- `exp-fleet-agent-rl/` — agent RL on a local matching env; torch GRPO policy, with an
  `EchoPolicy`/Claude baseline
- `exp-fleet-taskboard/` — agent RL on a multi-step, verifier-graded task-board env
