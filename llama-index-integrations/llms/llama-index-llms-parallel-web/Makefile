GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:
	@echo "Commands:"
	@echo "  install             install all dependencies"
	@echo "  test                run tests"
	@echo "  lint                run linters"
	@echo "  watch-docs          run docs in watch mode"

install:
	pip install -e .

test:
	pytest tests/

lint:
	ruff check .
	ruff format . --check

format:
	ruff check . --fix
	ruff format .
