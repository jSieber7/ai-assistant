# =============================================================================
# Docker Services Management Makefile
# =============================================================================
# Standard Docker Compose approach for managing complex container setup
# Replaces docker/run_dockers.py with native Docker Compose functionality

# =============================================================================
# Configuration Variables
# =============================================================================

# Environment
ENVIRONMENT ?= dev
COMPOSE_PROJECT_NAME ?= my-stack

# Docker Compose files
BASE_COMPOSE_FILE := docker-compose.yml
DEV_COMPOSE_FILE := docker-compose.dev.yml
PROD_COMPOSE_FILE := docker-compose.prod.yml
DEV_ENV_FILE := .env.dev
PROD_ENV_FILE := .env.prod

# Determine which files to use based on environment
ifeq ($(ENVIRONMENT),dev)
    COMPOSE_FILES := -f $(BASE_COMPOSE_FILE) -f $(DEV_COMPOSE_FILE)
    ENV_FILE := --env-file $(DEV_ENV_FILE)
    PROJECT_NAME := $(COMPOSE_PROJECT_NAME)-dev
else ifeq ($(ENVIRONMENT),prod)
    COMPOSE_FILES := -f $(BASE_COMPOSE_FILE) -f $(PROD_COMPOSE_FILE)
    ENV_FILE := --env-file $(PROD_ENV_FILE)
    PROJECT_NAME := $(COMPOSE_PROJECT_NAME)-prod
else
    $(error Invalid environment. Use ENVIRONMENT=dev or ENVIRONMENT=prod)
endif

# Docker Compose command
DOCKER_COMPOSE := docker compose $(COMPOSE_FILES) $(ENV_FILE) -p $(PROJECT_NAME)

# =============================================================================
# Default Target
# =============================================================================

.PHONY: help up down logs status reset test build clean links health-check jupyter
.DEFAULT_GOAL := help

# =============================================================================
# Help System
# =============================================================================

help: ## Show this help message
	@echo "Docker Services Management Commands"
	@echo ""
	@echo "Usage: make [command] ENVIRONMENT=[dev|prod] SERVICE=[service_name]"
	@echo ""
	@echo "Environment Variables:"
	@echo "  ENVIRONMENT     Set environment (dev or prod, default: dev)"
	@echo "  SERVICE         Target specific service (optional, default: all)"
	@echo ""
	@echo "Service Management:"
	@echo "  up              Start services (default: all services)"
	@echo "  down            Stop services (default: all services)"
	@echo "  restart         Restart services (default: all services)"
	@echo "  logs            Show service logs (default: all services)"
	@echo "  status          Show service status (default: all services)"
	@echo "  build           Build services (default: all services)"
	@echo "  reset           Reset service data (default: all services)"
	@echo "  test            Test service configuration (default: all services)"
	@echo ""
	@echo "Individual Services:"
	@echo "  app             Manage app service only"
	@echo "  frontend        Manage frontend service only"
	@echo "  supabase        Manage supabase service only"
	@echo "  searxng         Manage searxng service only"
	@echo "  firecrawl        Manage firecrawl service only"
	@echo "  traefik         Manage traefik service only"
	@echo "  redis           Manage redis service only"
	@echo ""
	@echo "Utilities:"
	@echo "  links           Show all service URLs"
	@echo "  health-check    Run comprehensive health check"
	@echo "  clean           Clean up Docker resources"
	@echo "  jupyter         Launch Jupyter notebook in app container"
	@echo ""
	@echo "Examples:"
	@echo "  make up                          # Start all services in dev mode"
	@echo "  make up ENVIRONMENT=prod           # Start all services in prod mode"
	@echo "  make up SERVICE=app               # Start app service only"
	@echo "  make logs SERVICE=app              # Show app service logs"
	@echo "  make down ENVIRONMENT=prod         # Stop all services in prod mode"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Service Management Commands
# =============================================================================

up: ## Start services (default: all services)
	@echo "Starting services in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Starting $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) up -d $(SERVICE); \
	else \
		echo "Starting all services..."; \
		$(DOCKER_COMPOSE) up -d; \
		$(MAKE) links; \
	fi

down: ## Stop services (default: all services)
	@echo "Stopping services in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Stopping $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) stop $(SERVICE); \
	else \
		echo "Stopping all services..."; \
		$(DOCKER_COMPOSE) down; \
	fi

restart: down up ## Restart services (default: all services)

logs: ## Show service logs (default: all services, use -f for follow)
	@echo "Showing logs for services in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Showing logs for $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) logs -f $(SERVICE); \
	else \
		echo "Showing logs for all services..."; \
		$(DOCKER_COMPOSE) logs -f; \
	fi

status: ## Show service status (default: all services)
	@echo "Checking service status in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Status for $(SERVICE) service:"; \
		$(DOCKER_COMPOSE) ps $(SERVICE); \
	else \
		echo "Status for all services:"; \
		$(DOCKER_COMPOSE) ps; \
	fi

build: ## Build services (default: all services)
	@echo "Building services in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Building $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) build $(SERVICE); \
	else \
		echo "Building all services..."; \
		$(DOCKER_COMPOSE) build; \
	fi

reset: ## Reset service data (default: all services) - DANGEROUS
	@echo "⚠️  WARNING: This will delete all service data!"
	@read -p "Are you sure you want to reset all data? Type 'RESET' to confirm: " confirm && [ "$$confirm" = "RESET" ] || exit 1
	@echo "Resetting services in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Resetting $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) down -v $(SERVICE); \
	else \
		echo "Resetting all services..."; \
		$(DOCKER_COMPOSE) down -v; \
	fi

test: ## Test service configuration (default: all services)
	@echo "Testing service configuration in $(ENVIRONMENT) mode..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Testing $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) config $(SERVICE); \
	else \
		echo "Testing all services..."; \
		$(DOCKER_COMPOSE) config; \
	fi

# =============================================================================
# Individual Service Commands
# =============================================================================

app: ## Manage app service only
	@$(MAKE) up SERVICE=app ENVIRONMENT=$(ENVIRONMENT)

frontend: ## Manage frontend service only
	@$(MAKE) up SERVICE=frontend ENVIRONMENT=$(ENVIRONMENT)

supabase: ## Manage supabase service only
	@$(MAKE) up SERVICE=supabase-db ENVIRONMENT=$(ENVIRONMENT)

searxng: ## Manage searxng service only
	@$(MAKE) up SERVICE=searxng ENVIRONMENT=$(ENVIRONMENT)

firecrawl: ## Manage firecrawl service only
	@$(MAKE) up SERVICE=api ENVIRONMENT=$(ENVIRONMENT)

traefik: ## Manage traefik service only
	@$(MAKE) up SERVICE=traefik ENVIRONMENT=$(ENVIRONMENT)

redis: ## Manage redis service only
	@$(MAKE) up SERVICE=redis ENVIRONMENT=$(ENVIRONMENT)

# =============================================================================
# Utility Commands
# =============================================================================

links: ## Show all service URLs
	@echo "=========================================="
	@echo "🔗 Service URLs for $(ENVIRONMENT) Environment"
	@echo "=========================================="
	@if [ "$(ENVIRONMENT)" = "dev" ]; then \
		echo "📱 Main Applications:"; \
		echo "  App (API):           http://app.localhost or http://localhost:8000"; \
		echo "  Frontend (Dev):      http://frontend.localhost or http://localhost:3000"; \
		echo "  Jupyter Notebook:    http://localhost:8889 (run 'make jupyter' to start)"; \
		echo ""; \
		echo "🔍 Search & Crawling:"; \
		echo "  SearXNG:             http://searxng.localhost"; \
		echo "  Firecrawl API:       http://firecrawl.localhost or http://localhost:3002"; \
		echo ""; \
		echo "📊 Monitoring & Management:"; \
		echo "  Traefik Dashboard:   http://traefik.localhost or http://localhost:8080"; \
		echo "  Redis:               redis://localhost:6379"; \
		echo "  Supabase DB:         postgresql://postgres:password@localhost:5433/postgres"; \
		echo "  Supabase DB:         postgresql://postgres:password@localhost:5433/postgres"; \
	else \
		echo "📱 Main Applications:"; \
		echo "  App (API):           http://app.yourdomain.com"; \
		echo "  Frontend (Prod):     http://frontend.yourdomain.com"; \
		echo ""; \
		echo "🔍 Search & Crawling:"; \
		echo "  SearXNG:             http://searxng.yourdomain.com"; \
		echo "  Firecrawl API:       http://firecrawl.yourdomain.com"; \
		echo ""; \
		echo "📊 Monitoring & Management:"; \
		echo "  Traefik Dashboard:   http://traefik.yourdomain.com"; \
		echo "  Redis:               Internal only"; \
		echo "  Supabase DB:         Internal only"; \
	fi
	@echo ""
	@echo "💡 Tips:"
	@echo "  - Use subdomain URLs for Traefik routing"
	@echo "  - Services must be running before accessing URLs"
	@echo "  - Some services may require authentication"
	@echo ""

health-check: ## Run comprehensive health check
	@echo "Running comprehensive health check..."
	@echo "=== Service Status ==="
	@$(MAKE) status
	@echo ""
	@echo "=== URL Health Check ==="
	@if [ "$(ENVIRONMENT)" = "dev" ]; then \
		echo "Checking app.localhost..."; \
		curl -f -s http://app.localhost/health > /dev/null && echo "✅ App: Healthy" || echo "❌ App: Unhealthy"; \
		echo "Checking frontend.localhost..."; \
		curl -f -s http://frontend.localhost > /dev/null && echo "✅ Frontend: Healthy" || echo "❌ Frontend: Unhealthy"; \
		echo "Checking traefik.localhost..."; \
		curl -f -s http://traefik.localhost > /dev/null && echo "✅ Traefik: Healthy" || echo "❌ Traefik: Unhealthy"; \
	else \
		echo "Checking app.yourdomain.com..."; \
		curl -f -s http://app.yourdomain.com/health > /dev/null && echo "✅ App: Healthy" || echo "❌ App: Unhealthy"; \
		echo "Checking frontend.yourdomain.com..."; \
		curl -f -s http://frontend.yourdomain.com > /dev/null && echo "✅ Frontend: Healthy" || echo "❌ Frontend: Unhealthy"; \
		echo "Checking traefik.yourdomain.com..."; \
		curl -f -s http://traefik.yourdomain.com > /dev/null && echo "✅ Traefik: Healthy" || echo "❌ Traefik: Unhealthy"; \
	fi

clean: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	@$(DOCKER_COMPOSE) down -v --remove-orphans || true
	@docker system prune -f
	@docker volume prune -f
	@echo "✅ Cleanup complete"

# =============================================================================
# Environment Switching
# =============================================================================

dev: ## Switch to development environment
	@echo "Switching to development environment..."
	@$(MAKE) up ENVIRONMENT=dev

prod: ## Switch to production environment
	@echo "Switching to production environment..."
	@$(MAKE) up ENVIRONMENT=prod

# =============================================================================
# Advanced Commands
# =============================================================================

shell: ## Open shell in service container
	@if [ -z "$(SERVICE)" ]; then \
		echo "Usage: make shell SERVICE=[service_name]"; \
		exit 1; \
	fi
	@echo "Opening shell in $(SERVICE) service..."
	@$(DOCKER_COMPOSE) exec $(SERVICE) bash

exec: ## Execute command in service container
	@if [ -z "$(SERVICE)" ] || [ -z "$(CMD)" ]; then \
		echo "Usage: make exec SERVICE=[service_name] CMD=[command]"; \
		exit 1; \
	fi
	@echo "Executing '$(CMD)' in $(SERVICE) service..."
	@$(DOCKER_COMPOSE) exec $(SERVICE) $(CMD)

pull: ## Pull latest images for services
	@echo "Pulling latest images for $(ENVIRONMENT) environment..."
	@if [ -n "$(SERVICE)" ]; then \
		echo "Pulling $(SERVICE) service..."; \
		$(DOCKER_COMPOSE) pull $(SERVICE); \
	else \
		echo "Pulling all services..."; \
		$(DOCKER_COMPOSE) pull; \
	fi

# =============================================================================
# =============================================================================
# Jupyter Command (Development Only)
# =============================================================================

jupyter: ## Launch Jupyter notebook in app container (dev environment only)
	@if [ "$(ENVIRONMENT)" != "dev" ]; then \
		echo "❌ Jupyter is only available in development environment."; \
		echo "Please use: make jupyter ENVIRONMENT=dev"; \
		exit 1; \
	fi
	@echo "🚀 Starting Jupyter notebook in development environment..."
	@echo "📋 Checking if app service is running..."
	@$(DOCKER_COMPOSE) ps app | grep -q "Up" || $(MAKE) up SERVICE=app ENVIRONMENT=dev
	@echo "🔧 Launching Jupyter notebook..."
	@echo "📝 Jupyter will be available at: http://localhost:8889"
	@echo "💡 Note: No token or password required for access"
	@$(DOCKER_COMPOSE) exec -d app /bin/bash -c "cd /app && .venv/bin/python -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
	@echo "✅ Jupyter notebook started successfully!"
	@echo "🌐 Access it at: http://localhost:8889"
# =============================================================================
# Migration Helpers
# =============================================================================

migrate-from-python: ## Migrate from Python script to standard approach
	@echo "Migrating from Python script to standard Docker Compose approach..."
	@echo ""
	@echo "✅ Migration complete! The following commands replace the Python script:"
	@echo ""
	@echo "Old: uv run docker/run_dockers.py up --service all --dev"
	@echo "New: make up ENVIRONMENT=dev"
	@echo ""
	@echo "Old: uv run docker/run_dockers.py up --service app --dev"
	@echo "New: make up SERVICE=app ENVIRONMENT=dev"
	@echo ""
	@echo "Old: uv run docker/run_dockers.py logs --service all --dev"
	@echo "New: make logs ENVIRONMENT=dev"
	@echo ""
	@echo "Old: uv run docker/run_dockers.py status --service all --dev"
	@echo "New: make status ENVIRONMENT=dev"
	@echo ""
	@echo "See 'make help' for all available commands."