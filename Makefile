# =============================================================================
# AI Assistant Makefile
# =============================================================================
# Provides convenient commands for development, testing, and deployment

.PHONY: help install dev test lint format clean docker docker-down docker-logs firecrawl firecrawl-down firecrawl-logs health-check docs shutdown-everything nuke

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "AI Assistant Development Commands"
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
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	$(MAKE) install

# =============================================================================
# Development
# =============================================================================

dev: ## Run development server
	@echo "Starting development server..."
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-docker: ## Run development with Docker
	@echo "Starting development environment with Docker..."
	docker compose --profile dev up

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "Running all tests..."
	uv run pytest --junitxml=test-results/junit.xml

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	uv run  pytest tests/unit/ --junitxml=test-results/junit-unit.xml

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	uv run pytest tests/integration/ --junitxml=test-results/junit-integration.xml

test-firecrawl: ## Run Firecrawl Docker tests
	@echo "Running Firecrawl Docker tests..."
	uv run pytest tests/unit/test_firecrawl_docker_mode.py tests/integration/test_firecrawl_docker.py -v --junitxml=test-results/junit-firecrawl.xml

test-firecrawl-unit: ## Run Firecrawl unit tests only
	@echo "Running Firecrawl unit tests..."
	uv run pytest tests/unit/test_firecrawl_docker_mode.py tests/unit/test_firecrawl_scraping.py -v --junitxml=test-results/junit-firecrawl-unit.xml

test-firecrawl-integration: ## Run Firecrawl integration tests
	@echo "Running Firecrawl integration tests..."
	uv run pytest tests/integration/test_firecrawl_docker.py -v --junitxml=test-results/junit-firecrawl-integration.xml

test-firecrawl-live: ## Run live Firecrawl tests (requires Docker services)
	@echo "Running live Firecrawl tests..."
	@echo "Make sure Firecrawl Docker services are running: make firecrawl"
	uv run pytest tests/integration/test_firecrawl_docker.py::TestFirecrawlDockerLive -v -s --junitxml=test-results/junit-firecrawl-live.xml

test-deep-search: ## Run Deep Search agent tests
	@echo "Running Deep Search agent tests..."
	uv run pytest tests/unit/test_deep_search_agent.py -v --junitxml=test-results/junit-deep-search.xml

test-deep-search-integration: ## Run Deep Search integration tests (requires Docker services)
	@echo "Running Deep Search integration tests..."
	@echo "Make sure all services are running: make milvus"
	uv run pytest tests/integration/test_deep_search_integration.py -v -s --junitxml=test-results/junit-deep-search-integration.xml

test-deep-search-live: ## Run live Deep Search tests (requires all services)
	@echo "Running live Deep Search tests..."
	@echo "Make sure all services are running: make milvus"
	uv run pytest tests/integration/test_deep_search_integration.py::test_deep_search_full_workflow -v -s --junitxml=test-results/junit-deep-search-live.xml

test-all-deep-search: ## Run all Deep Search tests
	@echo "Running all Deep Search tests..."
	uv run pytest tests/unit/test_deep_search_agent.py tests/integration/test_deep_search_integration.py -v --junitxml=test-results/junit-all-deep-search.xml

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	uv run pytest --cov=app --cov-report=html:test-results/htmlcov --cov-report=xml:test-results/coverage.xml --cov-report=term --junitxml=test-results/junit-coverage.xml

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting
	@echo "Running linting..."
	uv run ruff check app/ tests/ --output-format=junit > test-results/ruff-report.xml
	uv run ruff check app/ tests/

format: ## Format code
	@echo "Formatting code..."
	uv run ruff format app/ tests/

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	uv run ruff format --check app/ tests/

type-check: ## Run type checking
	@echo "Running type checking..."
	uv run mypy app/ --junit-xml test-results/mypy-report.xml
	uv run mypy app/

security-check: ## Run security checks
	@echo "Running security checks..."
	uv run bandit -r app/ -f json -o test-results/bandit-report.json
	uv run bandit -r app/ -f txt -o test-results/bandit-report.txt

# =============================================================================
# Docker Commands
# =============================================================================

docker: ## Start all Docker services
	@echo "Starting all Docker services..."
	docker compose up -d

docker-down: ## Stop all Docker services
	@echo "Stopping all Docker services..."
	docker compose down -v

docker-logs: ## Show Docker logs
	@echo "Showing Docker logs..."
	docker compose logs -f

docker-restart: ## Restart Docker services
	@echo "Restarting Docker services..."
	docker compose restart

# =============================================================================
# Milvus Docker Commands
# =============================================================================

milvus: ## Start Milvus Docker services
	@echo "Starting Milvus Docker services..."
	docker compose --profile milvus up -d
	@echo "Waiting for services to be ready..."
	@sleep 30
	$(MAKE) milvus-health

milvus-down: ## Stop Milvus Docker services
	@echo "Stopping Milvus Docker services..."
	docker compose --profile milvus down -v

milvus-logs: ## Show Milvus Docker logs
	@echo "Showing Milvus Docker logs..."
	docker compose --profile milvus logs -f

milvus-restart: ## Restart Milvus Docker services
	@echo "Restarting Milvus Docker services..."
	docker compose --profile milvus restart

milvus-status: ## Show Milvus service status
	@echo "Milvus service status:"
	docker compose --profile milvus ps

milvus-health: ## Check Milvus health
	@echo "Checking Milvus health..."
	@echo "Checking Etcd..."
	docker compose exec milvus-etcd etcdctl endpoint health || echo "Etcd not healthy"
	@echo "Checking MinIO..."
	docker compose exec milvus-minio curl -f http://localhost:9000/minio/health/live || echo "MinIO not healthy"
	@echo "Checking Milvus..."
	docker compose exec milvus-standalone curl -f http://localhost:9091/healthz || echo "Milvus not healthy"

# =============================================================================
# Firecrawl Docker Commands
# =============================================================================

firecrawl: ## Start Firecrawl Docker services
	@echo "Starting Firecrawl Docker services..."
	docker compose --profile firecrawl up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	$(MAKE) firecrawl-health

firecrawl-down: ## Stop Firecrawl Docker services
	@echo "Stopping Firecrawl Docker services..."
	docker compose --profile firecrawl down -v

firecrawl-logs: ## Show Firecrawl Docker logs
	@echo "Showing Firecrawl Docker logs..."
	docker compose --profile firecrawl logs -f

firecrawl-restart: ## Restart Firecrawl Docker services
	@echo "Restarting Firecrawl Docker services..."
	docker compose --profile firecrawl restart

firecrawl-dev: ## Start development with Firecrawl
	@echo "Starting development environment with Firecrawl..."
	docker compose --profile dev --profile firecrawl up -d

firecrawl-prod: ## Start production with Firecrawl
	@echo "Starting production environment with Firecrawl..."
	docker compose --profile production --profile firecrawl up -d

firecrawl-status: ## Show Firecrawl service status
	@echo "Firecrawl service status:"
	docker compose --profile firecrawl ps

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
	docker compose ps || echo "Docker services not running"
	@echo ""
	@echo "=== Firecrawl Services ==="
	$(MAKE) firecrawl-status || echo "Firecrawl services not running"
	@echo ""
	@echo "=== Firecrawl Health ==="
	$(MAKE) firecrawl-health || echo "Firecrawl health check failed"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	mkdocs serve

docs-build: ## Build documentation
	@echo "Building documentation..."
	mkdocs build

docs-deploy: ## Deploy documentation
	@echo "Deploying documentation..."
	mkdocs gh-deploy

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
	docker compose down -v --remove-orphans
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
	docker compose --profile dev down --remove-orphans || true
	docker compose --profile production down --remove-orphans || true
	docker compose --profile firecrawl down --remove-orphans || true
	@echo "Phase 2: Stopping optional profiles..."
	docker compose --profile monitoring down --remove-orphans || true
	docker compose --profile mongodb down --remove-orphans || true
	docker compose --profile postgres down --remove-orphans || true
	docker compose --profile jina-reranker down --remove-orphans || true
	docker compose --profile milvus down --remove-orphans || true
	@echo "Phase 3: Stopping any remaining services..."
	docker compose down --remove-orphans || true
	@echo "Phase 4: Force stopping any stubborn containers..."
	docker stop $$(docker ps -q) 2>/dev/null || true
# 	@echo "Phase 5: Removing all containers..."
# 	docker rm $$(docker ps -aq) 2>/dev/null || true
# 	@echo "Phase 6: Removing all networks..."
# 	docker network prune -f || true
# 	@echo "Phase 7: Cleaning up system resources..."
# 	docker system prune -af || true
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

prod: ## Start production environment
	@echo "Starting production environment..."
	docker compose --profile production up -d

prod-down: ## Stop production environment
	@echo "Stopping production environment..."
	docker compose --profile production down -v

prod-logs: ## Show production logs
	@echo "Showing production logs..."
	docker compose --profile production logs -f

# =============================================================================
# Utilities
# =============================================================================

shell: ## Open shell in application container
	@echo "Opening shell in application container..."
	docker compose exec ai-assistant bash

firecrawl-shell: ## Open shell in Firecrawl API container
	@echo "Opening shell in Firecrawl API container..."
	docker compose --profile firecrawl exec firecrawl-api bash

backup-firecrawl: ## Backup Firecrawl data
	@echo "Backing up Firecrawl data..."
	mkdir -p backups
	docker compose --profile firecrawl exec firecrawl-postgres pg_dump -U firecrawl firecrawl > backups/firecrawl_$(shell date +%Y%m%d_%H%M%S).sql

# =============================================================================
# Version and Release
# =============================================================================

version: ## Show version information
	@echo "AI Assistant Version:"
	@python -c "import app; print(getattr(app, '__version__', 'unknown'))"
	@echo ""
	@echo "Docker Images:"
	@docker compose images || echo "Docker services not running"

bump-patch: ## Bump patch version
	@echo "Bumping patch version..."
	python utility/bump_version.py patch

bump-minor: ## Bump minor version
	@echo "Bumping minor version..."
	python utility/bump_version.py minor

bump-major: ## Bump major version
	@echo "Bumping major version..."
	python utility/bump_version.py major