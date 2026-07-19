SHELL := /bin/zsh

# Expanded by make, not the shell: experiments are local-only (gitignored), so
# a fresh checkout has none and a bare zsh glob would die with "no matches".
LINT_PATHS := src/ $(wildcard packages/*/autoresearch) $(wildcard experiments/*/src/)

.PHONY: install lint check typecheck test fmt hooks dev

install:
	poetry install

fmt: lint

hooks:
	git config core.hooksPath .githooks

dev: hooks install

lint:
	poetry run ruff check $(LINT_PATHS) --unsafe-fixes --fix
	poetry run ruff format $(LINT_PATHS)

check:
	poetry run ruff check $(LINT_PATHS)
	poetry run ruff format --check $(LINT_PATHS)

typecheck:
	poetry run mypy src/ packages/*/autoresearch

test:
	poetry run pytest tests/ -v
