# Veritas Experiments

This document describes how to run autonomous research experiments in veritas.

## Architecture

```
veritas/
  src/autoresearch/         # Generic experiment runner
  experiments/
    <experiment-name>/      # Each experiment is its own git repo
      src/<package>/
        harness/            # Fixed: data, eval, constraints
        experiment/         # Modifiable: model, optimizer, training
        interface.py        # Wires harness to experiment
      EXPERIMENT.md         # Experiment-specific instructions
```

## Running an Experiment

All commands run from the veritas root directory.

### Prepare data (one-time)

```bash
poetry run autoresearch --prepare experiments/<experiment-name>
```

### Run training

```bash
poetry run autoresearch experiments/<experiment-name>
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
cd experiments/<experiment-name>
git checkout -b experiment/<tag>   # e.g., experiment/mar20
```

## Experiment Interface

Every experiment implements two protocols defined in `src/autoresearch/interface.py`:

**Harness** (fixed for the experiment):
- `config` — time budget, sequence length, primary metric
- `prepare()` — download data, train tokenizer
- `get_vocab_size()` — vocabulary size
- `make_dataloader()` — training/validation data
- `evaluate()` — compute metrics

**Experiment** (what you optimize):
- `get_training_config()` — batch sizes, learning rates
- `build_model()` — construct the model
- `build_optimizer()` — construct the optimizer
- `train_step()` — single training step with loss computation

## Available Experiments

Each experiment has its own `EXPERIMENT.md` with detailed instructions:

- `experiments/pretraining-gpt/` — GPT language model pretraining

## The Experiment Loop

See each experiment's `EXPERIMENT.md` for the full autonomous loop instructions. The general pattern:

1. Create experiment branch in the experiment's git repo
2. Modify files in `experiment/` or `interface.py`
3. Commit changes
4. Run: `poetry run autoresearch experiments/<name> > experiments/<name>/run.log 2>&1`
5. Check results: `grep "^val_bpb:" experiments/<name>/run.log`
6. Keep (if improved) or revert (if not)
7. Repeat indefinitely

The goal is autonomous optimization: keep changes that improve the primary metric, discard those that don't.
