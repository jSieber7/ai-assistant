# Makefile for AI Assistant Docker operations

.PHONY: help build up down logs clean test dev dev-basic tools monitor mongodb db shell status backup-redis restore-redis setup

# Default target
help:
	@echo "AI Assistant Docker Commands:"
	@echo ""
	@echo "  build     Build all Docker images"
	@echo "  up        Start all services in production mode (full stack)"
	@echo "  dev       Start full development environment with all features"
	@echo "  dev-basic Start minimal development environment (core services only)"
	@echo "  tools     Start services with development tools"
	@echo "  monitor   Start services with monitoring tools"
	@echo "  mongodb   Start with MongoDB for multi-writer system"
	@echo "  db        Start with PostgreSQL database"
	@echo "  down      Stop all services"
	@echo "  logs      Show logs for all services"
	@echo "  logs-app  Show logs for AI Assistant only"
	@echo "  clean     Remove containers, images, and volumes"
	@echo "  test      Run tests in Docker"
	@echo "  shell     Open shell in AI Assistant container"
	@echo "  status    Check service status"
	@echo "  setup     Quick setup for new users"
	@echo ""
	@echo "Examples:"
	@echo "  make dev              # Start full development environment (recommended)"
	@echo "  make dev-basic        # Start minimal development environment"
	@echo "  make up               # Start full production environment"
	@echo "  make up monitor       # Production with monitoring"
	@echo "  make logs-app         # View application logs"
	@echo "  make shell            # Access container shell"

# Build all images
build:
	@echo "Building Docker images..."
	docker compose build

# Start all services (production)
up:
	@echo "Starting all services in production mode..."
	docker compose --profile mongodb up -d
	@echo "Services started. Access:"
	@echo "  AI Assistant: http://localhost"
	@echo "  Traefik Dashboard: http://localhost:8080"
	@echo "  SearXNG: http://localhost/search"
	@echo "  MongoDB: localhost:27017"
	@echo "  Mongo Express: http://localhost:8082"

# Start in development mode (full)
dev:
	@echo "Starting full development environment..."
	docker compose --profile dev --profile mongodb up -d
	@echo "Full development mode started with hot reload"
	@echo "Access:"
	@echo "  AI Assistant: http://localhost:8000"
	@echo "  Redis Commander: http://localhost:8081"
	@echo "  MongoDB: localhost:27017"
	@echo "  Mongo Express: http://localhost:8082"
	@echo "  SearXNG: http://localhost:8000/search"

# Start minimal development mode
dev-basic:
	@echo "Starting minimal development environment..."
	docker compose --profile dev up -d
	@echo "Basic development mode started with hot reload"
	@echo "Access:"
	@echo "  AI Assistant: http://localhost:8000"
	@echo "  Redis Commander: http://localhost:8081"

# Start with development tools
tools:
	@echo "Starting with development tools..."
	docker compose --profile dev up -d
	@echo "Development tools started:"
	@echo "  Redis Commander: http://localhost:8081"

# Start with monitoring
monitor:
	@echo "Starting with monitoring tools..."
	docker compose --profile monitoring up -d
	@echo "Monitoring tools started:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

# Start with MongoDB
mongodb:
	@echo "Starting with MongoDB..."
	docker compose --profile mongodb up -d
	@echo "MongoDB started:"
	@echo "  MongoDB: localhost:27017"
	@echo "  Mongo Express: http://localhost:8082"

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
	docker compose --profile dev run --rm ai-assistant .venv/bin/python -m pytest tests/ -v

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
		cp .env.example .env; \
		echo "Please edit .env and add your API keys"; \
	else \
		echo ".env already exists"; \
	fi
	@echo "Setup complete! Edit .env and run 'make dev' to start"