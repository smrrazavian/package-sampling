# Project configuration
PACKAGE_NAME=src
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(PYTHON) -m pip
PYTEST=$(PYTHON) -m pytest
BLACK=$(PYTHON) -m black
ISORT=$(PYTHON) -m isort
FLAKE8=$(PYTHON) -m flake8
MYPY=$(PYTHON) -m mypy
BUILD=$(PYTHON) -m build
PYLINT=$(PYTHON) -m pylint

# Create virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Install development dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# Run tests
.PHONY: test
test:
	$(PYTEST) tests/

# Run tests with coverage
.PHONY: coverage
coverage:
	$(PYTEST) --cov=$(PACKAGE_NAME) tests/

# Lint with Flake8
.PHONY: lint
lint:
	$(FLAKE8) $(PACKAGE_NAME) tests

# Format with Black
.PHONY: format
format:
	$(BLACK) $(PACKAGE_NAME) tests

# Sort imports
.PHONY: isort
isort:
	$(ISORT) $(PACKAGE_NAME) tests

# Static type checking with MyPy
.PHONY: typecheck
typecheck:
	$(MYPY) $(PACKAGE_NAME)

# Run all checks (lint, format, type checking)
.PHONY: check
check: lint format isort typecheck

# Build the package
.PHONY: build
build:
	$(BUILD)

# Clean cache and build artifacts
.PHONY: clean
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache dist build

# Run pylint
.PHONY: pylint
pylint:
	$(PYLINT) $(PACKAGE_NAME)

# Run everything before a commit
.PHONY: precommit
precommit: format isort lint typecheck test

# Default target
.DEFAULT_GOAL := help

# Show available commands
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv       - Create a virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make coverage   - Run tests with coverage"
	@echo "  make lint       - Lint with Flake8"
	@echo "  make format     - Format with Black"
	@echo "  make isort      - Sort imports"
	@echo "  make typecheck  - Static type checking with MyPy"
	@echo "  make check      - Run lint, format, and type checking"
	@echo "  make build      - Build the package"
	@echo "  make clean      - Clean cache and build artifacts"
	@echo "  make pylint     - Run Pylint"
	@echo "  make precommit  - Run all checks before committing"
