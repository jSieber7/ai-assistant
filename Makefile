# =============================================================================
# AI Assistant Makefile
# =============================================================================
# Provides convenient commands for development, testing, and deployment

.PHONY: help install dev test lint format clean docker docker-down docker-logs firecrawl firecrawl-down firecrawl-logs health-check docs

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
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-docker: ## Run development with Docker
	@echo "Starting development environment with Docker..."
	docker compose --profile dev up

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "Running all tests..."
	pytest

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/integration/

test-firecrawl: ## Run Firecrawl Docker tests
	@echo "Running Firecrawl Docker tests..."
	pytest tests/integration/test_firecrawl_docker.py -v

test-firecrawl-live: ## Run live Firecrawl tests (requires Docker services)
	@echo "Running live Firecrawl tests..."
	@echo "Make sure Firecrawl Docker services are running: make firecrawl"
	pytest tests/integration/test_firecrawl_docker.py::TestFirecrawlDockerLive -v -s

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest --cov=app --cov-report=html --cov-report=term

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting
	@echo "Running linting..."
	ruff check app/ tests/

format: ## Format code
	@echo "Formatting code..."
	ruff format app/ tests/

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	ruff format --check app/ tests/

type-check: ## Run type checking
	@echo "Running type checking..."
	mypy app/

security-check: ## Run security checks
	@echo "Running security checks..."
	bandit -r app/

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
# Firecrawl Docker Commands
# =============================================================================

firecrawl: ## Start Firecrawl Docker services
	@echo "Starting Firecrawl Docker services..."
	docker compose -f docker-configs/firecrawl/docker-compose.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	$(MAKE) firecrawl-health

firecrawl-down: ## Stop Firecrawl Docker services
	@echo "Stopping Firecrawl Docker services..."
	docker compose -f docker-configs/firecrawl/docker-compose.yml down -v

firecrawl-logs: ## Show Firecrawl Docker logs
	@echo "Showing Firecrawl Docker logs..."
	docker compose -f docker-configs/firecrawl/docker-compose.yml logs -f

firecrawl-restart: ## Restart Firecrawl Docker services
	@echo "Restarting Firecrawl Docker services..."
	docker compose -f docker-configs/firecrawl/docker-compose.yml restart

firecrawl-status: ## Show Firecrawl service status
	@echo "Firecrawl service status:"
	docker compose -f docker-configs/firecrawl/docker-compose.yml ps

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
	docker compose -f docker-configs/firecrawl/docker-compose.yml exec firecrawl-api bash

backup-firecrawl: ## Backup Firecrawl data
	@echo "Backing up Firecrawl data..."
	mkdir -p backups
	docker compose -f docker-configs/firecrawl/docker-compose.yml exec firecrawl-postgres pg_dump -U firecrawl firecrawl > backups/firecrawl_$(shell date +%Y%m%d_%H%M%S).sql

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