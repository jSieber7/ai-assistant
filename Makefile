# =============================================================================
# AI Assistant Makefile
# =============================================================================
# Provides convenient commands for development, testing, and deployment

# =============================================================================
# Configuration Variables
# =============================================================================

# Project Configuration
PROJECT_NAME := ai-assistant
PYTHON_VERSION := 3.12.3

# Docker Configuration
DOCKER_COMPOSE := docker compose
DEV_PROFILE := dev
PROD_PROFILE := production

# Port Configuration
APP_PORT := 8000
REDIS_PORT := 6379
SEARXNG_PORT := 8080
PROMETHEUS_PORT := 9090
GRAFANA_PORT := 3000

# Container Names
APP_CONTAINER := ai-assistant
APP_DEV_CONTAINER := ai-assistant-dev
REDIS_CONTAINER := ai-assistant-redis
SEARXNG_CONTAINER := ai-assistant-searxng
FIRECRAWL_CONTAINER := firecrawl
MILVUS_CONTAINER := milvus-standalone

# Test Configuration
TEST_RESULTS_DIR := test-results
COVERAGE_DIR := $(TEST_RESULTS_DIR)/htmlcov
COVERAGE_THRESHOLD := 80

# Command Options
PYTEST_OPTIONS := --verbose --strict-markers --strict-config --disable-warnings
PYTEST_UNIT_OPTIONS := --cov=app --cov-report=html:$(COVERAGE_DIR) --cov-report=xml:$(TEST_RESULTS_DIR)/coverage.xml --cov-report=term

# =============================================================================
# Default Target
# =============================================================================

.PHONY: help install dev test lint format clean production production-down production-logs firecrawl firecrawl-down firecrawl-logs health-check docs shutdown-everything nuke quality-check ci-test test-results-dir dev-jupyter links links-int

.DEFAULT_GOAL := help

# =============================================================================
# Help System
# =============================================================================

help: ## Show this help message
	@echo "AI Assistant Development Commands"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install dependencies"
	@echo "  install-prod     Install production dependencies only"
	@echo "  setup-dev        Set up development environment"
	@echo ""
	@echo "Development:"
	@echo "  dev              Run development server"
	@echo "  dev-docker       Run development with Docker (includes all tool dockers)"
	@echo "  dev-jupyter      Start Jupyter notebook for interactive development"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  quality-check    Run all code quality checks"
	@echo "  lint             Run linting"
	@echo "  format           Format code"
	@echo "  type-check       Run type checking"
	@echo "  security-check   Run security checks"
	@echo ""
	@echo "Docker Services:"
	@echo "  docker           Start all Docker services"
	@echo "  docker-down      Stop all Docker services"
	@echo "  docker-logs      Show Docker logs"
	@echo "  docker-restart   Restart Docker services"
	@echo ""
	@echo "Production:"
	@echo "  production       Start production environment"
	@echo "  production-down  Stop production environment"
	@echo "  production-logs  Show production logs"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up temporary files"
	@echo "  clean-docker     Clean up Docker resources"
	@echo "  clean-all        Clean up everything"
	@echo "  health-check     Run comprehensive health check"
	@echo "  docs             Generate documentation"
	@echo ""
	@echo "Emergency:"
	@echo "  shutdown-everything  Shut down all Docker services"
	@echo "  nuke                 Nuclear option - complete Docker reset"
	@echo ""
	@echo "Links:"
	@echo "  links            Show all accessible service URLs"
	@echo "  links-int        Show internal service URLs only"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation and Setup
# =============================================================================

install: ## Install dependencies
	@echo "Installing dependencies..."
	uv sync --dev

install-prod: ## Install production dependencies only
	@echo "Installing production dependencies..."
	uv sync --no-dev

setup-dev: ## Set up development environment
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from example"; \
		echo "Please edit .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi
	$(MAKE) install

# =============================================================================
# Development
# =============================================================================

dev: ## Run development server
	@echo "Starting development server..."
	uv run docker/run_dockers.py up -d

dev-docker: ## Run development with Docker (includes all tool dockers)
	@echo "Starting development environment with Docker and all tools..."
	uv run docker/run_dockers.py up --service all --dev
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Development environment with all tools is ready!"
	@echo "App: http://localhost:$(APP_PORT)"
	@echo "Firecrawl API: http://localhost:3002"

dev-quick: ## Quick development setup (install + start services)
	@echo "Quick development setup..."
	$(MAKE) setup-dev
	uv run docker/run_dockers.py up --service all --dev
	@echo "Development environment is ready!"
	@echo "App: http://localhost:$(APP_PORT)"

# =============================================================================
# Testing
# =============================================================================

test-results-dir: ## Create test-results directory
	@mkdir -p $(TEST_RESULTS_DIR)

test: test-results-dir ## Run all tests
	@echo "Running all tests..."
	uv run pytest $(PYTEST_OPTIONS) --junitxml=$(TEST_RESULTS_DIR)/junit.xml

test-unit: test-results-dir ## Run unit tests only
	@echo "Running unit tests..."
	uv run pytest tests/unit/ $(PYTEST_OPTIONS) --junitxml=$(TEST_RESULTS_DIR)/junit-unit.xml

test-integration: test-results-dir ## Run integration tests
	@echo "Running integration tests..."
	uv run pytest tests/integration/ $(PYTEST_OPTIONS) --junitxml=$(TEST_RESULTS_DIR)/junit-integration.xml

test-coverage: test-results-dir ## Run tests with coverage report
	@echo "Running tests with coverage..."
	uv run pytest $(PYTEST_OPTIONS) $(PYTEST_UNIT_OPTIONS) --junitxml=$(TEST_RESULTS_DIR)/junit-coverage.xml
	@echo "Coverage report generated: $(COVERAGE_DIR)/index.html"

# Multi-Speaker System Tests
test-multi-speaker: test-results-dir ## Run multi-speaker system tests
	@echo "Running multi-speaker system tests..."
	uv run pytest tests/unit/test_debate_system.py tests/unit/test_dynamic_selector.py tests/unit/test_personality_system.py tests/unit/test_collaborative_checker.py tests/unit/test_learning_system.py tests/unit/test_master_checker.py tests/unit/test_context_sharing.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-multi-speaker.xml

test-multi-speaker-unit: test-results-dir ## Run multi-speaker system unit tests
	@echo "Running multi-speaker system unit tests..."
	uv run pytest tests/unit/test_debate_system.py tests/unit/test_dynamic_selector.py tests/unit/test_personality_system.py tests/unit/test_collaborative_checker.py tests/unit/test_learning_system.py tests/unit/test_master_checker.py tests/unit/test_context_sharing.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-multi-speaker-unit.xml

test-multi-speaker-integration: test-results-dir ## Run multi-speaker system integration tests
	@echo "Running multi-speaker system integration tests..."
	uv run pytest tests/integration/test_multi_speaker_system.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-multi-speaker-integration.xml

test-multi-speaker-live: test-results-dir ## Run live multi-speaker system tests (requires services)
	@echo "Running live multi-speaker system tests..."
	@echo "Make sure all required services are running"
	uv run pytest tests/integration/test_multi_speaker_system.py::TestMultiSpeakerSystemLive -v -s --junitxml=$(TEST_RESULTS_DIR)/junit-multi-speaker-live.xml

test-all-multi-speaker: test-results-dir ## Run all multi-speaker system tests
	@echo "Running all multi-speaker system tests..."
	uv run pytest tests/unit/test_debate_system.py tests/unit/test_dynamic_selector.py tests/unit/test_personality_system.py tests/unit/test_collaborative_checker.py tests/unit/test_learning_system.py tests/unit/test_master_checker.py tests/unit/test_context_sharing.py tests/integration/test_multi_speaker_system.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-all-multi-speaker.xml

# Firecrawl Tests
test-firecrawl: test-results-dir ## Run Firecrawl Docker tests
	@echo "Running Firecrawl Docker tests..."
	uv run pytest tests/unit/test_firecrawl_docker_mode.py tests/integration/test_firecrawl_docker.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-firecrawl.xml

test-firecrawl-unit: test-results-dir ## Run Firecrawl unit tests only
	@echo "Running Firecrawl unit tests..."
	uv run pytest tests/unit/test_firecrawl_docker_mode.py tests/unit/test_firecrawl_scraping.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-firecrawl-unit.xml

test-firecrawl-integration: test-results-dir ## Run Firecrawl integration tests
	@echo "Running Firecrawl integration tests..."
	uv run pytest tests/integration/test_firecrawl_docker.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-firecrawl-integration.xml

test-firecrawl-live: test-results-dir ## Run live Firecrawl tests (requires Docker services)
	@echo "Running live Firecrawl tests..."
	@echo "Make sure Firecrawl Docker services are running: make firecrawl"
	uv run pytest tests/integration/test_firecrawl_docker.py::TestFirecrawlDockerLive -v -s --junitxml=$(TEST_RESULTS_DIR)/junit-firecrawl-live.xml

# Deep Search Tests
test-deep-search: test-results-dir ## Run Deep Search agent tests
	@echo "Running Deep Search agent tests..."
	uv run pytest tests/unit/test_deep_search_agent.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-deep-search.xml

test-deep-search-integration: test-results-dir ## Run Deep Search integration tests (requires Docker services)
	@echo "Running Deep Search integration tests..."
	@echo "Make sure all services are running: make milvus"
	uv run pytest tests/integration/test_deep_search_integration.py -v -s --junitxml=$(TEST_RESULTS_DIR)/junit-deep-search-integration.xml

test-deep-search-live: test-results-dir ## Run live Deep Search tests (requires all services)
	@echo "Running live Deep Search tests..."
	@echo "Make sure all services are running: make milvus"
	uv run pytest tests/integration/test_deep_search_integration.py::test_deep_search_full_workflow -v -s --junitxml=$(TEST_RESULTS_DIR)/junit-deep-search-live.xml

test-all-deep-search: test-results-dir ## Run all Deep Search tests
	@echo "Running all Deep Search tests..."
	uv run pytest tests/unit/test_deep_search_agent.py tests/integration/test_deep_search_integration.py -v --junitxml=$(TEST_RESULTS_DIR)/junit-all-deep-search.xml

test-multi-speaker-coverage: test-results-dir ## Run multi-speaker system tests with coverage report
	@echo "Running multi-speaker system tests with coverage..."
	uv run pytest tests/unit/test_debate_system.py tests/unit/test_dynamic_selector.py tests/unit/test_personality_system.py tests/unit/test_collaborative_checker.py tests/unit/test_learning_system.py tests/unit/test_master_checker.py tests/unit/test_context_sharing.py --cov=app.core.agents --cov-report=html:$(TEST_RESULTS_DIR)/htmlcov-multi-speaker --cov-report=xml:$(TEST_RESULTS_DIR)/multi-speaker-coverage.xml --cov-report=term --junitxml=$(TEST_RESULTS_DIR)/junit-multi-speaker-coverage.xml

# =============================================================================
# Code Quality
# =============================================================================

quality-check: ## Run all code quality checks (lint, type-check, security-check, test-coverage)
	@echo "Running comprehensive quality checks..."
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test-coverage
	@echo ""
	@echo "=== Quality Check Summary ==="
	@echo "Linting report: $(TEST_RESULTS_DIR)/ruff-report.xml"
	@echo "Type checking report: $(TEST_RESULTS_DIR)/mypy-report.xml"
	@echo "Security report: $(TEST_RESULTS_DIR)/bandit-report.json"
	@echo "Coverage report: $(TEST_RESULTS_DIR)/coverage.xml"
	@echo "HTML coverage report: $(COVERAGE_DIR)/index.html"

quality-check-multi-speaker: ## Run quality checks for multi-speaker system
	@echo "Running multi-speaker system quality checks..."
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test-multi-speaker-coverage
	@echo ""
	@echo "=== Multi-Speaker System Quality Check Summary ==="
	@echo "Linting report: $(TEST_RESULTS_DIR)/ruff-report.xml"
	@echo "Type checking report: $(TEST_RESULTS_DIR)/mypy-report.xml"
	@echo "Security report: $(TEST_RESULTS_DIR)/bandit-report.json"
	@echo "Multi-speaker coverage report: $(TEST_RESULTS_DIR)/multi-speaker-coverage.xml"
	@echo "HTML multi-speaker coverage report: $(TEST_RESULTS_DIR)/htmlcov-multi-speaker/index.html"

ci-test: ## Run CI pipeline (test, lint, type-check, security-check)
	@echo "Running CI pipeline..."
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	@echo ""
	@echo "=== CI Test Summary ==="
	@echo "Test report: $(TEST_RESULTS_DIR)/junit.xml"
	@echo "Linting report: $(TEST_RESULTS_DIR)/ruff-report.xml"
	@echo "Type checking report: $(TEST_RESULTS_DIR)/mypy-report.xml"
	@echo "Security report: $(TEST_RESULTS_DIR)/bandit-report.json"

ci-test-multi-speaker: ## Run CI pipeline for multi-speaker system
	@echo "Running multi-speaker system CI pipeline..."
	$(MAKE) test-multi-speaker
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	@echo ""
	@echo "=== Multi-Speaker System CI Test Summary ==="
	@echo "Test report: $(TEST_RESULTS_DIR)/junit-multi-speaker.xml"
	@echo "Linting report: $(TEST_RESULTS_DIR)/ruff-report.xml"
	@echo "Type checking report: $(TEST_RESULTS_DIR)/mypy-report.xml"
	@echo "Security report: $(TEST_RESULTS_DIR)/bandit-report.json"

lint: test-results-dir ## Run linting
	@echo "Running linting..."
	@uv run ruff check app/ tests/ --output-format=junit > $(TEST_RESULTS_DIR)/ruff-report.xml || true
	@uv run ruff check app/ tests/ || echo "Linting found issues (see above)"

format: ## Format code
	@echo "Formatting code..."
	uv run ruff format app/ tests/

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	@uv run ruff format --check app/ tests/ || echo "Code formatting issues found (run 'make format' to fix)"

type-check: test-results-dir ## Run type checking
	@echo "Running type checking..."
	@uv run mypy app/ --junit-xml $(TEST_RESULTS_DIR)/mypy-report.xml || true
	@uv run mypy app/ || echo "Type checking found issues (see above)"

security-check: test-results-dir ## Run security checks
	@echo "Running security checks..."
	@uv run bandit -r app/ -f json -o $(TEST_RESULTS_DIR)/bandit-report.json || true
	@uv run bandit -r app/ -f txt -o $(TEST_RESULTS_DIR)/bandit-report.txt || echo "Security check found issues (see above)"

# =============================================================================
# Docker Commands
# =============================================================================

docker: ## Start all Docker services
	@echo "Starting all Docker services..."
	uv run docker/run_dockers.py up --service all --dev

docker-down: ## Stop all Docker services
	@echo "Stopping all Docker services..."
	uv run docker/run_dockers.py down --service all --dev

docker-logs: ## Show Docker logs
	@echo "Showing Docker logs..."
	uv run docker/run_dockers.py logs --service all --dev

docker-restart: ## Restart Docker services
	@echo "Restarting Docker services..."
	uv run docker/run_dockers.py down --service all --dev
	uv run docker/run_dockers.py up --service all --dev

docker-status: ## Show Docker service status
	@echo "Docker service status:"
	uv run docker/run_dockers.py status --service all --dev

# =============================================================================
# Milvus Docker Commands
# =============================================================================

milvus: ## Start Milvus Docker services
	@echo "Starting Milvus Docker services..."
	uv run docker/run_dockers.py up --service milvus
	@echo "Waiting for services to be ready..."
	@sleep 30
	$(MAKE) milvus-health

milvus-down: ## Stop Milvus Docker services
	@echo "Stopping Milvus Docker services..."
	uv run docker/run_dockers.py down --service milvus

milvus-logs: ## Show Milvus Docker logs
	@echo "Showing Milvus Docker logs..."
	uv run docker/run_dockers.py logs --service milvus

milvus-restart: ## Restart Milvus Docker services
	@echo "Restarting Milvus Docker services..."
	uv run docker/run_dockers.py down --service milvus
	uv run docker/run_dockers.py up --service milvus

milvus-status: ## Show Milvus service status
	@echo "Milvus service status:"
	uv run docker/run_dockers.py status --service milvus

milvus-health: ## Check Milvus health
	@echo "Checking Milvus health..."
	@echo "Checking Etcd..."
	cd docker && $(DOCKER_COMPOSE) exec milvus-etcd etcdctl endpoint health || echo "Etcd not healthy"
	@echo "Checking MinIO..."
	cd docker && $(DOCKER_COMPOSE) exec milvus-minio curl -f http://localhost:9000/minio/health/live || echo "MinIO not healthy"
	@echo "Checking Milvus..."
	cd docker && $(DOCKER_COMPOSE) exec milvus-standalone curl -f http://localhost:9091/healthz || echo "Milvus not healthy"

# =============================================================================
# Firecrawl Docker Commands
# =============================================================================

firecrawl: ## Start Firecrawl Docker services
	@echo "Starting Firecrawl Docker services..."
	uv run docker/run_dockers.py up --service firecrawl
	@echo "Waiting for services to be ready..."
	@sleep 10
	$(MAKE) firecrawl-health

firecrawl-down: ## Stop Firecrawl Docker services
	@echo "Stopping Firecrawl Docker services..."
	uv run docker/run_dockers.py down --service firecrawl

firecrawl-logs: ## Show Firecrawl Docker logs
	@echo "Showing Firecrawl Docker logs..."
	uv run docker/run_dockers.py logs --service firecrawl

firecrawl-restart: ## Restart Firecrawl Docker services
	@echo "Restarting Firecrawl Docker services..."
	uv run docker/run_dockers.py down --service firecrawl
	uv run docker/run_dockers.py up --service firecrawl

firecrawl-dev: ## Start development with Firecrawl
	@echo "Starting development environment with Firecrawl..."
	uv run docker/run_dockers.py up --service firecrawl --dev

firecrawl-production: ## Start production with Firecrawl
	@echo "Starting production environment with Firecrawl..."
	uv run docker/run_dockers.py up --service firecrawl

firecrawl-status: ## Show Firecrawl service status
	@echo "Firecrawl service status:"
	uv run docker/run_dockers.py status --service firecrawl

firecrawl-health: ## Check Firecrawl health
	@echo "Checking Firecrawl health..."
	python utility/firecrawl_health_check.py --verbose || true

firecrawl-test: ## Test Firecrawl with example URL
	@echo "Testing Firecrawl functionality..."
	python utility/firecrawl_health_check.py --test-url https://httpbin.org/html --verbose

# =============================================================================
# Migration Commands
# =============================================================================

migrate-to-docker: ## Migrate from API to Docker mode
	@echo "Migrating to Firecrawl Docker mode..."
	@echo "1. Starting Firecrawl Docker services..."
	uv run docker/run_dockers.py up --service firecrawl
	@echo "2. Waiting for services to be ready..."
	@sleep 15
	@echo "3. Verifying installation..."
	$(MAKE) firecrawl-health
	@echo "4. Testing functionality..."
	$(MAKE) firecrawl-test
	@echo ""
	@echo "Migration complete! Update your .env file:"
	@echo "FIRECRAWL_DEPLOYMENT_MODE=docker"
	@echo "FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002"

migrate-to-api: ## Migrate from Docker to API mode
	@echo "Migrating to Firecrawl API mode..."
	@echo "1. Stopping Firecrawl Docker services..."
	uv run docker/run_dockers.py down --service firecrawl
	@echo ""
	@echo "Migration complete! Update your .env file:"
	@echo "FIRECRAWL_DEPLOYMENT_MODE=api"
	@echo "FIRECRAWL_API_KEY=your-api-key-here"
	@echo "FIRECRAWL_BASE_URL=https://api.firecrawl.dev"

# =============================================================================
# Health Checks
# =============================================================================

health-check: ## Run comprehensive health check
	@echo "Running comprehensive health check..."
	@echo "=== Docker Services ==="
	uv run docker/run_dockers.py status --service all --dev || echo "Docker services not running"
	@echo ""
	@echo "=== Firecrawl Services ==="
	uv run docker/run_dockers.py status --service firecrawl || echo "Firecrawl services not running"
	@echo ""
	@echo "=== Firecrawl Health ==="
	$(MAKE) firecrawl-health || echo "Firecrawl health check failed"

status-all: ## Show status of all services
	@echo "=== All Service Status ==="
	@echo "Docker Services:"
	uv run docker/run_dockers.py status --service all --dev || echo "No Docker services running"
	@echo ""
	@echo "Application Containers:"
	@docker ps --filter "name=$(PROJECT_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No application containers running"
	@echo ""
	@echo "Port Usage:"
	@echo "App: $(APP_PORT)"
	@echo "Redis: $(REDIS_PORT)"
	@echo "SearXNG: $(SEARXNG_PORT)"
	@echo "Prometheus: $(PROMETHEUS_PORT)"
	@echo "Grafana: $(GRAFANA_PORT)"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	uv run mkdocs serve

docs-build: ## Build documentation
	@echo "Building documentation..."
	uv run mkdocs build

docs-deploy: ## Deploy documentation
	@echo "Deploying documentation..."
	uv run mkdocs gh-deploy

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-docker: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	uv run docker/run_dockers.py down --service all --dev
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker ## Clean up everything

# =============================================================================
# EMERGENCY SHUTDOWN COMMANDS
# =============================================================================

shutdown-everything: ## SHUT DOWN ALL DOCKER SERVICES ACROSS ALL PROFILES
	@echo "🚨 EMERGENCY SHUTDOWN INITIATED - STOPPING ALL SERVICES 🚨"
	@echo "Stopping all Docker services across all profiles..."
	@echo "Phase 1: Stopping dev services..."
	uv run docker/run_dockers.py down --service all --dev || true
	@echo "Phase 2: Stopping prod services..."
	uv run docker/run_dockers.py down --service all || true
	@echo "Phase 3: Stopping any remaining services..."
	cd docker && $(DOCKER_COMPOSE) down --remove-orphans || true
	@echo "Phase 4: Force stopping any stubborn containers from this project..."
	cd docker && $(DOCKER_COMPOSE) ps -q | xargs -r docker stop 2>/dev/null || true
	@echo "✅ ALL SERVICES SHUT DOWN SUCCESSFULLY!"

nuke: ## NUCLEAR OPTION - COMPLETE DOCKER SYSTEM RESET
	@echo "☢️  NUCLEAR OPTION INITIATED - COMPLETE SYSTEM RESET ☢️"
	@echo "This will remove ALL Docker data - containers, networks, volumes, images!"
	@read -p "Are you absolutely sure? Type 'NUKE' to confirm: " confirm && [ "$$confirm" = "NUKE" ] || exit 1
	@echo "Phase 1: Stopping ALL containers..."
	docker stop $$(docker ps -aq) 2>/dev/null || true
	@echo "Phase 2: Removing ALL containers..."
	docker rm $$(docker ps -aq) 2>/dev/null || true
	@echo "Phase 3: Removing ALL networks..."
	docker network rm $$(docker network ls -q) 2>/dev/null || true
	@echo "Phase 4: Removing ALL volumes (THIS WILL DELETE ALL DATA)..."
	docker volume rm $$(docker volume ls -q) 2>/dev/null || true
	@echo "Phase 5: Removing ALL images..."
	docker rmi $$(docker images -q) 2>/dev/null || true
	@echo "Phase 6: System cleanup..."
	docker system prune -af --volumes || true
	@echo "☢️  COMPLETE SYSTEM RESET FINISHED - ALL DOCKER DATA REMOVED!"

# =============================================================================
# Production
# =============================================================================

production: ## Start production environment
	@echo "Starting production environment..."
	uv run docker/run_dockers.py up --service all


production-down: ## Stop production environment
	@echo "Stopping production environment..."
	uv run docker/run_dockers.py down --service all

production-logs: ## Show production logs
	@echo "Showing production logs..."
	uv run docker/run_dockers.py logs --service all

# =============================================================================
# Utilities
# =============================================================================

shell: ## Open shell in application container
	@echo "Opening shell in application container..."
	cd docker && $(DOCKER_COMPOSE) exec $(APP_CONTAINER) bash

shell-dev: ## Open shell in development container
	@echo "Opening shell in development container..."
	cd docker && $(DOCKER_COMPOSE) exec $(APP_DEV_CONTAINER) bash

firecrawl-shell: ## Open shell in Firecrawl API container
	@echo "Opening shell in Firecrawl API container..."
	cd docker && $(DOCKER_COMPOSE) --profile $(PROD_PROFILE) exec firecrawl-api bash

logs-app: ## Show application logs
	@echo "Showing application logs..."
	cd docker && $(DOCKER_COMPOSE) logs -f $(APP_CONTAINER) || cd docker && $(DOCKER_COMPOSE) logs -f $(APP_DEV_CONTAINER)

logs-all: ## Show all service logs
	@echo "Showing all service logs..."
	uv run docker/run_dockers.py logs --service all --dev

backup-firecrawl: ## Backup Firecrawl data
	@echo "Backing up Firecrawl data..."
	mkdir -p backups
	cd docker && $(DOCKER_COMPOSE) --profile $(PROD_PROFILE) exec firecrawl-postgres pg_dump -U firecrawl firecrawl > backups/firecrawl_$(shell date +%Y%m%d_%H%M%S).sql

# =============================================================================
# Database Commands
# =============================================================================

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -m alembic upgrade head

db-reset: ## Reset database (destructive)
	@echo "Resetting database..."
	@read -p "This will delete all data. Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -m alembic downgrade base
	$(MAKE) db-migrate

db-seed: ## Seed database with sample data
	@echo "Seeding database with sample data..."
	$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python utility/seed_database.py

# =============================================================================
# Version and Release
# =============================================================================

version: ## Show version information
	@echo "$(PROJECT_NAME) Version:"
	@python3 -c "import app; print(getattr(app, '__version__', 'unknown'))"
	@echo ""
	@echo "Docker Images:"
	@$(DOCKER_COMPOSE) images || echo "Docker services not running"

bump-patch: ## Bump patch version
	@echo "Bumping patch version..."
	python utility/bump_version.py patch

bump-minor: ## Bump minor version
	@echo "Bumping minor version..."
	python utility/bump_version.py minor

bump-major: ## Bump major version
	@echo "Bumping major version..."
	python utility/bump_version.py major

# =============================================================================
# Development Utilities
# =============================================================================

install-tools: ## Install additional development tools
	@echo "Installing additional development tools..."
	uv add --dev pre-commit
	uv add --dev commitizen
	@echo "Tools installed. Consider running 'pre-commit install' to set up git hooks."

pre-commit: ## Run pre-commit checks
	@echo "Running pre-commit checks..."
	uv run pre-commit run --all-files

watch-test: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	uv run pytest tests/ --watch

profile: ## Run application with profiling
	@echo "Running application with profiling..."
	uv run python -m cProfile -o profile.stats -m uvicorn app.main:app --host 0.0.0.0 --port $(APP_PORT) --reload
	@echo "Profile saved to profile.stats"
	@echo "View with: python -m pstats profile.stats"

# =============================================================================
# Jupyter Development
# =============================================================================

jupyter: ## Start Jupyter notebook for interactive development
	@echo "Starting Jupyter notebook for interactive development..."
	@echo "Checking if development environment is running..."
	@if ! cd docker && $(DOCKER_COMPOSE) --env-file ../.env ps app --format "table {{.Names}}" | grep -q "my-stack-app"; then \
		echo "Development environment not running. Please start it first with:"; \
		echo "  uv run docker/run_dockers.py app dev up"; \
		exit 1; \
	fi
	@echo "Stopping any existing Jupyter instances..."
	@cd docker && $(DOCKER_COMPOSE) --env-file ../.env exec app pkill -f jupyter || echo "No existing Jupyter instances found"
	@echo "Starting Jupyter notebook in development container..."
	@cd docker && $(DOCKER_COMPOSE) --env-file ../.env exec -d app uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
	@echo "Waiting for Jupyter to start..."
	@sleep 5
	@echo "Jupyter is ready!"
	@echo "Access it at: http://localhost:8888/"
	@echo "No token required - access is open for development"
	@echo "To stop Jupyter, run: make jupyter-stop"

jupyter-logs: ## Show Jupyter logs
	@echo "Showing Jupyter logs..."
	@cd docker && $(DOCKER_COMPOSE) --env-file ../.env logs -f app | grep jupyter || cd docker && $(DOCKER_COMPOSE) --env-file ../.env logs -f app

jupyter-stop: ## Stop Jupyter notebook
	@echo "Stopping Jupyter notebook..."
	@cd docker && $(DOCKER_COMPOSE) --env-file ../.env exec app pkill -f jupyter || echo "Jupyter not running"

# =============================================================================
# Service Links
# =============================================================================

links: ## Show all accessible service URLs
	@echo "=========================================="
	@echo "🔗 AI Assistant Service URLs"
	@echo "=========================================="
	@echo ""
	@echo "📱 Main Applications:"
	@echo "  App (API):           http://localhost:$(APP_PORT) or http://app.localhost"
	@echo "  Frontend (Dev):      http://frontend.localhost"
	@echo ""
	@echo "🔍 Search & Crawling:"
	@echo "  SearXNG:             http://searxng.localhost"
	@echo "  Firecrawl API:       http://firecrawl.localhost or http://localhost:3002"
	@echo ""
	@echo "📊 Monitoring & Management:"
	@echo "  Traefik Dashboard:   http://localhost:8080 or http://traefik.localhost"
	@echo "  Jupyter Notebook:    http://localhost:8888"
	@echo "  Prometheus:          http://localhost:$(PROMETHEUS_PORT)"
	@echo "  Grafana:             http://localhost:$(GRAFANA_PORT)"
	@echo ""
	@echo "💡 Tips:"
	@echo "  - Use localhost URLs for direct port access"
	@echo "  - Use subdomain URLs (app.localhost) for Traefik routing"
	@echo "  - Some services may require authentication"
	@echo "  - Make sure services are running before accessing URLs"
	@echo ""

links-int: ## Show internal Docker network service URLs (not accessible from outside)
	@echo "=========================================="
	@echo "🔗 AI Assistant Internal Docker Network URLs"
	@echo "=========================================="
	@echo ""
	@echo "🗄️  Database Services:"
	@echo "  Redis:               my-stack-redis:6379"
	@echo "  Supabase DB:         my-stack-supabase-db:5432"
	@echo "  Firecrawl PostgreSQL: nuq-postgres:5432"
	@echo ""
	@echo "🔍 Search & Crawling:"
	@echo "  SearXNG:             my-stack-searxng:8080"
	@echo "  Firecrawl API:       firecrawl-api:3002"
	@echo "  Playwright Service:  playwright-service:3000"
	@echo ""
	@echo "📊 Logging & Analytics:"
	@echo "  Supabase Vector:     my-stack-supabase-vector:9001"
	@echo ""
	@echo "💡 Tips:"
	@echo "  - These URLs are only accessible within the Docker network"
	@echo "  - Use these URLs for inter-service communication"
	@echo "  - Services must be running before accessing these URLs"
	@echo "  - Container names may vary based on COMPOSE_PROJECT_NAME"
	@echo ""