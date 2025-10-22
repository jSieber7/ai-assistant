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
CHAINLIT_PORT := 8001
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

.PHONY: help install dev test lint format clean production production-down production-logs firecrawl firecrawl-down firecrawl-logs health-check docs shutdown-everything nuke quality-check ci-test test-results-dir dev-jupyter

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
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(APP_PORT) --reload

dev-docker: ## Run development with Docker (includes all tool dockers)
	@echo "Starting development environment with Docker and all tools..."
	@echo "Starting main development services..."
	$(DOCKER_COMPOSE) --env-file .env --profile $(DEV_PROFILE) up -d
	@echo "Starting Firecrawl services..."
	$(DOCKER_COMPOSE) -f docker-compose.firecrawl.yml --env-file .env.firecrawl up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Development environment with all tools is ready!"
	@echo "App: http://localhost:$(APP_PORT)"
	@echo "Firecrawl API: http://localhost:3002"

dev-quick: ## Quick development setup (install + start services)
	@echo "Quick development setup..."
	$(MAKE) setup-dev
	$(MAKE) dev-docker
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
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all Docker services
	@echo "Stopping all Docker services..."
	$(DOCKER_COMPOSE) down -v

docker-logs: ## Show Docker logs
	@echo "Showing Docker logs..."
	$(DOCKER_COMPOSE) logs -f

docker-restart: ## Restart Docker services
	@echo "Restarting Docker services..."
	$(DOCKER_COMPOSE) restart

docker-status: ## Show Docker service status
	@echo "Docker service status:"
	$(DOCKER_COMPOSE) ps

# =============================================================================
# Milvus Docker Commands
# =============================================================================

milvus: ## Start Milvus Docker services
	@echo "Starting Milvus Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) up -d
	@echo "Waiting for services to be ready..."
	@sleep 30
	$(MAKE) milvus-health

milvus-down: ## Stop Milvus Docker services
	@echo "Stopping Milvus Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) down -v

milvus-logs: ## Show Milvus Docker logs
	@echo "Showing Milvus Docker logs..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) logs -f

milvus-restart: ## Restart Milvus Docker services
	@echo "Restarting Milvus Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) restart

milvus-status: ## Show Milvus service status
	@echo "Milvus service status:"
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) ps

milvus-health: ## Check Milvus health
	@echo "Checking Milvus health..."
	@echo "Checking Etcd..."
	$(DOCKER_COMPOSE) exec milvus-etcd etcdctl endpoint health || echo "Etcd not healthy"
	@echo "Checking MinIO..."
	$(DOCKER_COMPOSE) exec milvus-minio curl -f http://localhost:9000/minio/health/live || echo "MinIO not healthy"
	@echo "Checking Milvus..."
	$(DOCKER_COMPOSE) exec milvus-standalone curl -f http://localhost:9091/healthz || echo "Milvus not healthy"

# =============================================================================
# Firecrawl Docker Commands
# =============================================================================

firecrawl: ## Start Firecrawl Docker services
	@echo "Starting Firecrawl Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	$(MAKE) firecrawl-health

firecrawl-down: ## Stop Firecrawl Docker services
	@echo "Stopping Firecrawl Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) down -v

firecrawl-logs: ## Show Firecrawl Docker logs
	@echo "Showing Firecrawl Docker logs..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) logs -f

firecrawl-restart: ## Restart Firecrawl Docker services
	@echo "Restarting Firecrawl Docker services..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) restart

firecrawl-dev: ## Start development with Firecrawl
	@echo "Starting development environment with Firecrawl..."
	$(DOCKER_COMPOSE) --profile $(DEV_PROFILE) up -d

firecrawl-production: ## Start production with Firecrawl
	@echo "Starting production environment with Firecrawl..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) up -d

firecrawl-status: ## Show Firecrawl service status
	@echo "Firecrawl service status:"
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) ps

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
	$(MAKE) firecrawl
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
	$(MAKE) firecrawl-down
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
	$(DOCKER_COMPOSE) ps || echo "Docker services not running"
	@echo ""
	@echo "=== Firecrawl Services ==="
	$(MAKE) firecrawl-status || echo "Firecrawl services not running"
	@echo ""
	@echo "=== Firecrawl Health ==="
	$(MAKE) firecrawl-health || echo "Firecrawl health check failed"

status-all: ## Show status of all services
	@echo "=== All Service Status ==="
	@echo "Docker Services:"
	$(DOCKER_COMPOSE) ps || echo "No Docker services running"
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
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker ## Clean up everything

# =============================================================================
# EMERGENCY SHUTDOWN COMMANDS
# =============================================================================

shutdown-everything: ## SHUT DOWN ALL DOCKER SERVICES ACROSS ALL PROFILES
	@echo "🚨 EMERGENCY SHUTDOWN INITIATED - STOPPING ALL SERVICES 🚨"
	@echo "Stopping all Docker services across all profiles..."
	@echo "Phase 1: Stopping main profiles..."
	$(DOCKER_COMPOSE) --profile $(DEV_PROFILE) down --remove-orphans || true
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) down --remove-orphans || true
	@echo "Phase 3: Stopping any remaining services..."
	$(DOCKER_COMPOSE) down --remove-orphans || true
	@echo "Phase 4: Force stopping any stubborn containers from this project..."
	$(DOCKER_COMPOSE) ps -q | xargs -r docker stop 2>/dev/null || true
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
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) up -d


production-down: ## Stop production environment
	@echo "Stopping production environment..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) down -v

production-logs: ## Show production logs
	@echo "Showing production logs..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) logs -f

# =============================================================================
# Utilities
# =============================================================================

shell: ## Open shell in application container
	@echo "Opening shell in application container..."
	$(DOCKER_COMPOSE) exec $(APP_CONTAINER) bash

shell-dev: ## Open shell in development container
	@echo "Opening shell in development container..."
	$(DOCKER_COMPOSE) exec $(APP_DEV_CONTAINER) bash

firecrawl-shell: ## Open shell in Firecrawl API container
	@echo "Opening shell in Firecrawl API container..."
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) exec firecrawl-api bash

logs-app: ## Show application logs
	@echo "Showing application logs..."
	$(DOCKER_COMPOSE) logs -f $(APP_CONTAINER) || $(DOCKER_COMPOSE) logs -f $(APP_DEV_CONTAINER)

logs-all: ## Show all service logs
	@echo "Showing all service logs..."
	$(DOCKER_COMPOSE) logs -f

backup-firecrawl: ## Backup Firecrawl data
	@echo "Backing up Firecrawl data..."
	mkdir -p backups
	$(DOCKER_COMPOSE) --profile $(PROD_PROFILE) exec firecrawl-postgres pg_dump -U firecrawl firecrawl > backups/firecrawl_$(shell date +%Y%m%d_%H%M%S).sql

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

dev-jupyter: ## Start Jupyter notebook for interactive development
	@echo "Starting Jupyter notebook for interactive development..."
	@echo "Checking if development environment is running..."
	@if ! $(DOCKER_COMPOSE) ps --filter "name=ai-assistant-dev" --format "table {{.Names}}" | grep -q "ai-assistant-dev"; then \
		echo "Development environment not running. Starting it first..."; \
		$(DOCKER_COMPOSE) --env-file .env --profile $(DEV_PROFILE) up -d; \
		echo "Waiting for services to be ready..."; \
		sleep 10; \
	fi
	@echo "Stopping any existing Jupyter instances..."
	@$(DOCKER_COMPOSE) exec ai-assistant-dev pkill -f jupyter || echo "No existing Jupyter instances found"
	@echo "Starting Jupyter notebook in development container..."
	@$(DOCKER_COMPOSE) exec -d ai-assistant-dev uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	@echo "Waiting for Jupyter to start..."
	@sleep 5
	@echo "Getting Jupyter access token..."
	@$(eval TOKEN := $(shell $(DOCKER_COMPOSE) exec ai-assistant-dev uv run jupyter notebook list | grep ":8888/" | grep -o 'token=[a-zA-Z0-9]*' | cut -d'=' -f2))
	@echo "Jupyter is ready!"
	@echo "Access it at: http://localhost:8888/?token=$(TOKEN)"
	@echo "To get the token again, run: docker exec ai-assistant-dev uv run jupyter notebook list"
	@echo "To stop Jupyter, run: docker exec ai-assistant-dev pkill -f jupyter"

jupyter-logs: ## Show Jupyter logs
	@echo "Showing Jupyter logs..."
	$(DOCKER_COMPOSE) logs -f ai-assistant-dev | grep jupyter || $(DOCKER_COMPOSE) logs -f ai-assistant-dev

jupyter-stop: ## Stop Jupyter notebook
	@echo "Stopping Jupyter notebook..."
	$(DOCKER_COMPOSE) exec ai-assistant-dev pkill -f jupyter || echo "Jupyter not running"