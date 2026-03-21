SHELL := /bin/zsh

.PHONY: install lint check typecheck test

install:
	poetry install

lint:
	poetry run ruff check src/ experiments/*/src/ --unsafe-fixes --fix
	poetry run ruff format src/ experiments/*/src/

check:
	poetry run ruff check src/ experiments/*/src/
	poetry run ruff format --check src/ experiments/*/src/

typecheck:
	poetry run mypy src/

test:
	poetry run pytest tests/ -v
