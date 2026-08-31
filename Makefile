PROJECT_NAME := yn40m-scheduler-solver
VERSION := 20260831

ifneq (,$(wildcard .local/.env))
$(info Using local .env file)
include .local/.env
endif

all: install clean

clean:
	rm -rf build dist $(PROJECT_NAME).egg-info
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache

install: clean
	pip install .

install-dev: clean
	pip install -e ".[dev]"

uninstall:
	pip uninstall $(PROJECT_NAME) -y

test:
	pytest tests/ -x -v

test-coverage:
	coverage run -m pytest tests -x -vs --import-mode=importlib --cov-report=html --cov-report=term-missing --cov-config=.coveragerc
	coverage report --show-missing

wheel: clean
	python -m build

version:
	bumpver update --patch

# Code quality targets
format:
	black src/yn40mss tests
	isort src/yn40mss tests

lint:
	flake8 src/yn40mss tests
	black --check src/yn40mss tests
	isort --check-only src/yn40mss tests

type-check:
	mypy src/yn40mss --ignore-missing-imports


# Build targets
build: clean
	python -m build

check-package:
	twine check dist/*

# Help target
help:
	@echo "Available targets:"
	@echo "  install      	- Install package"
	@echo "  install-dev  	- Install in development mode"
	@echo "  uninstall    	- Uninstall package"
	@echo "  test         	- Run full test suite"
	@echo "  test-coverage	- Run tests with HTML coverage report"
	@echo "  format       	- Format code with black and isort"
	@echo "  lint         	- Run linting checks"
	@echo "  type-check   	- Run type checking with mypy"
	@echo "  build        	- Build package"
	@echo "  wheel        	- Build wheel package"
	@echo "  deploy       	- Deploy package to PyPI"
	@echo "  clean        	- Clean build artifacts"
	@echo "  version     	- Bump patch version"
	@echo "  help         	- Show this help message"

.PHONY: all clean install install-dev uninstall test test-coverage wheel deploy version format lint type-check build help
