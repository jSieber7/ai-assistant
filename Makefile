# Makefile for AI Assistant Docker operations

.PHONY: help build up down logs clean test dev prod

# Default target
help:
	@echo "AI Assistant Docker Commands:"
	@echo ""
	@echo "  build     Build all Docker images"
	@echo "  up        Start all services in production mode"
	@echo "  dev       Start services in development mode with hot reload"
	@echo "  tools     Start services with development tools"
	@echo "  monitor   Start services with monitoring tools"
	@echo "  down      Stop all services"
	@echo "  logs      Show logs for all services"
	@echo "  logs-app  Show logs for AI Assistant only"
	@echo "  clean     Remove containers, images, and volumes"
	@echo "  test      Run tests in Docker"
	@echo "  shell     Open shell in AI Assistant container"
	@echo "  db        Start with PostgreSQL database"
	@echo ""
	@echo "Examples:"
	@echo "  make dev          # Start in development mode"
	@echo "  make logs-app     # View application logs"
	@echo "  make shell        # Access container shell"

# Build all images
build:
	@echo "Building Docker images..."
	docker compose build

# Start all services (production)
up:
	@echo "Starting all services..."
	docker compose up -d
	@echo "Services started. Access:"
	@echo "  AI Assistant: http://localhost:8000"
	@echo "  SearXNG: http://localhost:8080"
	@echo "  Redis: localhost:6379"

# Start in development mode
dev:
	@echo "Starting in development mode..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up
	@echo "Development mode started with hot reload"

# Start with development tools
tools:
	@echo "Starting with development tools..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tools up
	@echo "Development tools started:"
	@echo "  Redis Commander: http://localhost:8081"

# Start with monitoring
monitor:
	@echo "Starting with monitoring tools..."
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile monitoring up
	@echo "Monitoring tools started:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

# Start with PostgreSQL
db:
	@echo "Starting with PostgreSQL..."
	docker compose --profile postgres up -d
	@echo "PostgreSQL started on localhost:5432"

# Stop all services
down:
	@echo "Stopping all services..."
	docker compose down

# Show logs
logs:
	docker compose logs -f

# Show application logs
logs-app:
	docker compose logs -f ai-assistant

# Clean up
clean:
	@echo "Cleaning up Docker resources..."
	docker compose down -v --rmi all
	docker system prune -f

# Run tests
test:
	@echo "Running tests in Docker..."
	docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit

# Open shell in container
shell:
	docker compose exec ai-assistant bash

# Check status
status:
	@echo "Service status:"
	docker compose ps

# Backup Redis
backup-redis:
	@echo "Backing up Redis data..."
	mkdir -p backups
	docker compose exec redis redis-cli BGSAVE
	docker cp ai-assistant-redis:/data/dump.rdb ./backups/redis-$(shell date +%Y%m%d-%H%M%S).rdb

# Restore Redis
restore-redis:
	@echo "Available backups:"
	@ls -la backups/
	@echo "Run: docker cp ./backups/FILENAME ai-assistant-redis:/data/dump.rdb && docker compose restart redis"

# Quick setup for new users
setup:
	@echo "Setting up AI Assistant..."
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp .env.docker .env; \
		echo "Please edit .env and add your API key"; \
	else \
		echo ".env already exists"; \
	fi
	@echo "Setup complete! Edit .env and run 'make up' to start"