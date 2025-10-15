# Simplified Docker Configuration

This document describes the simplified Docker configuration for the AI Assistant project.

## Overview

We've consolidated multiple Docker Compose files and environment configurations into a more manageable structure:

### Files

- `docker-compose.yml` - Single compose file with profiles for different environments
- `.env` - Development environment configuration
- `.env.example` - Template for environment variables
- `Dockerfile` - Multi-stage build supporting dev and production

## Usage

### Development

```bash
# Start development environment
docker compose --profile dev up

# With optional services
docker compose --profile dev --profile mongodb up
docker compose --profile dev --profile monitoring up
```

### Production

```bash
# Start production environment
docker compose up

# With optional services
docker compose --profile mongodb up
docker compose --profile monitoring up
```

## Profiles

### Core Services
Always running in production:
- `ai-assistant` - Main application
- `redis` - Caching
- `searxng` - Web search (with custom configuration from docker-configs/searxng/)
- `traefik` - Reverse proxy (production only)

### Development Profile (`--profile dev`)
- `ai-assistant-dev` - Development version with hot reload
- `debug-tools` - Python container for debugging
- `redis-commander` - Redis GUI

### Optional Profiles
- `mongodb` - MongoDB for multi-writer system
- `postgres` - PostgreSQL database
- `monitoring` - Prometheus and Grafana

## Environment Variables

The `.env` file contains all necessary configuration. For production, override values as needed:

```bash
# Production overrides
docker compose up -e ENVIRONMENT=production -e DEBUG=false
```

## Migration from Old Structure

### Old Files (can be removed)
- `.env.docker`
- `.env.minimal`
- `.env.multi-writer.template`
- `.env.test`
- `.env.traefik`
- `docker-compose.dev.yml`
- `docker-compose.multi-writer.yml`
- `docker-compose.test.yml`

### New Structure
- Single `.env.example` template
- Single `docker-compose.yml` with profiles
- Environment-specific overrides in compose file

## Examples

### Basic Development
```bash
# Copy environment template
cp .env.example .env

# Start development
docker compose --profile dev up
```

### With Multi-Writer System
```bash
# Enable multi-writer in .env
MULTI_WRITER_ENABLED=true
MULTI_WRITER_FIRECRAWL_API_KEY=your-key-here

# Start with MongoDB
docker compose --profile dev --profile mongodb up
```

### Production with Monitoring
```bash
# Set production values in .env or override
docker compose up -e ENVIRONMENT=production --profile monitoring
```

## Port Mappings

Development (direct access):
- Application: 8000
- Redis Commander: 8081
- MongoDB: 27017
- Mongo Express: 8082
- Prometheus: 9090
- Grafana: 3000

Production (through Traefik):
- Application: 80
- Traefik Dashboard: 8080
- SearXNG: 80/search