SHELL := /bin/zsh

.PHONY: install lint check typecheck test fmt hooks dev

install:
	poetry install

fmt: lint

hooks:
	git config core.hooksPath .githooks

dev: hooks install

lint:
	poetry run ruff check src/ packages/*/autoresearch experiments/*/src/ --unsafe-fixes --fix
	poetry run ruff format src/ packages/*/autoresearch experiments/*/src/

check:
	poetry run ruff check src/ packages/*/autoresearch experiments/*/src/
	poetry run ruff format --check src/ packages/*/autoresearch experiments/*/src/

typecheck:
	poetry run mypy src/ packages/*/autoresearch

test:
	poetry run pytest tests/ -v
