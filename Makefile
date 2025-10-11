.PHONY: test test-unit test-integration test-system test-all test-slow test-coverage test-parallel test-ci test-failed dev ci help lint format clean

##@ Testing
test:  ## Run all tests
	uv run pytest

test-unit:  ## Run only unit tests
	uv run pytest -m "unit and not slow" -v

test-integration:  ## Run only integration tests
	uv run pytest -m "integration and not slow" -v

test-system:  ## Run only system tests
	uv run pytest -m "system and not slow" -v

test-all:  ## Run all tests except slow ones
	uv run pytest -m "not slow" -v

test-slow:  ## Run only slow tests
	uv run pytest -m "slow" -v

test-coverage:  ## Run tests with coverage report
	uv run pytest --cov=app --cov-report=term --cov-report=html:coverage_html --cov-report=xml:coverage.xml

test-parallel:  ## Run tests in parallel
	uv run pytest -n auto

test-ci:  ## Run tests in CI mode
	uv run pytest --junitxml=junit.xml --cov-report=xml -q

test-failed:  ## Run failed tests first
	uv run pytest --failed-first

##@ Development
dev: test-unit test-integration  ## Run development test suite (unit + integration)

ci: test-ci  ## Run CI test suite

##@ Code Quality
lint:  ## Run linting
	uv run ruff check .

format:  ## Format code
	uv run ruff format .

##@ Utility
clean:  ## Clean up temporary files
	rm -rf .coverage coverage_html junit.xml bandit-results.json .ruff_cache .pytest_cache .mypy_cache ai_assistant.egg-info

##@ Help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)