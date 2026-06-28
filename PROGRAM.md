# autolab Experiments

This document describes how to run autonomous research experiments in autolab.

## Architecture

```
autolab/
  src/autoresearch/
    interface.py            # Universal primitives: Budget, RunContext, HarnessSpec, protocols
    runner.py               # Domain-agnostic: seed -> run -> evaluate -> report
    loops.py                # Opt-in loop helpers (e.g. supervised_loop)
    util.py                 # get_device, seed_everything, is_better
  experiments/
    exp-<name>/             # Each experiment is its own git repo
      src/<package>/
        harness/            # Fixed: objective, eval, constraints
        experiment/         # Modifiable: the approach (model/policy/optimizer/algorithm)
        interface.py        # Wires harness to experiment; exposes `harness` and `experiment`
      EXPERIMENT.md         # Experiment-specific instructions
```

Experiments follow the `exp-<name>` naming convention (e.g., `exp-pretraining-gpt`).
The framework is domain-agnostic: it runs supervised, RL, or any other experiment that
implements the two protocols. The training loop is not baked into the runner.

## Creating a New Experiment

Scaffold a new experiment (creates `experiments/exp-<name>/` with stub harness and experiment):

```bash
poetry run autoresearch-new <name>
```

Then implement the two protocols in `src/<name>/interface.py`.

## Running an Experiment

All commands run from the autolab root directory. The runner discovers the package
under the experiment's `src/` automatically, so the directory name and package name
need not match.

### Prepare data (one-time)

```bash
poetry run autoresearch --prepare experiments/exp-<name>
```

### Run training

```bash
poetry run autoresearch experiments/exp-<name>
```

### Lint and check

```bash
make check      # Lint all code
make lint       # Auto-fix lint issues
make typecheck  # Type check autoresearch
```

## Creating an Experiment Branch

Each experiment directory is its own git repo. To start an experiment run:

```bash
cd experiments/exp-<name>
git checkout -b run/<tag>   # e.g., run/mar20
```

## Experiment Interface

Every experiment implements two protocols defined in `src/autoresearch/interface.py`.
They are intentionally minimal — the experiment owns its own loop.

**Harness** (fixed for the experiment — the benchmark contract):
- `spec` → `HarnessSpec(primary_metric, direction, budget, domain)`
  - `direction` is `"min"` or `"max"` (e.g. `val_bpb` → min, `eval_return` → max)
  - `budget` is a `Budget(wall_clock_s=..., steps=..., env_steps=...)`; any axis set acts as a stop
  - `domain` holds domain-specific config (e.g. `{"seq_len": 2048}`)
- `prepare()` — one-time setup (download data, etc.); may be a no-op
- `evaluate(artifact, ctx)` — score the produced artifact, returning `{metric: value}`

**Experiment** (what you optimize — the approach):
- `run(harness, ctx)` — build the model/policy and optimizer, run the training/optimization
  loop honoring `ctx.budget`, and return the artifact. Record telemetry via `ctx.record({...})`.

Domain resources (a dataloader for LM, an environment for RL) are NOT part of the universal
protocol — they live on the concrete harness and are called by the matching experiment, which
is co-designed with it.

**Reusable loop helpers** live in `autoresearch.loops`. For ordinary supervised training,
`supervised_loop(model, optimizer, dataloader, step_fn, ctx, config)` provides gradient
accumulation, timing, divergence detection, and progress logging — opt in, don't reinvent.

## Available Experiments

Each experiment has its own `EXPERIMENT.md` with detailed instructions:

- `experiments/exp-pretraining-gpt/` — GPT language model pretraining (minimize `val_bpb`, wall-clock budget)
- `experiments/exp-rl-control/` — RL on a dependency-free CartPole (maximize `eval_return`, env-step budget)

## The Experiment Loop

See each experiment's `EXPERIMENT.md` for the full autonomous loop instructions. The general pattern:

1. Create run branch in the experiment's git repo (`run/<tag>`)
2. Modify files in `experiment/` or `interface.py`
3. Commit changes
4. Run: `poetry run autoresearch experiments/exp-<name> > experiments/exp-<name>/run.log 2>&1`
5. Check results: grep the log for the experiment's `primary_metric`
6. Keep if the primary metric improved **in the direction declared by `spec.direction`**, else revert
7. Repeat indefinitely

The goal is autonomous optimization: keep changes that improve the primary metric, discard those that don't.
