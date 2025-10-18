# Docker Integration

This document provides comprehensive information about the Docker integration for the AI Assistant project.

## Overview

The AI Assistant project includes full Docker support with pre-configured services for:
- Main AI Assistant application
- Redis for caching
- SearXNG for web search capabilities
- Optional PostgreSQL database

## Quick Start

### Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of RAM available
- Sufficient disk space for Docker images and volumes

### Basic Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-assistant
```

2. Copy the Docker environment template:
```bash
cp .env.docker .env
```

3. Edit the `.env` file to configure your API keys:
```bash
nano .env
```
Make sure to set at least:
- `OPENAI_COMPATIBLE_API_KEY` (or your preferred LLM provider)
- `SECRET_KEY` (generate a long random string)

4. Start all services:
```bash
docker-compose up -d
```

5. Check the status:
```bash
docker-compose ps
```

6. Access the application:
- AI Assistant: http://localhost (through Traefik)
- Gradio Interface: http://localhost/gradio
- Traefik Dashboard: http://localhost:8080
- SearXNG Search: http://localhost/search
- Redis: localhost:6379 (internal only)

### Stopping the Services

```bash
docker-compose down
```

## Services

### AI Assistant Application

The main application service that provides the AI assistant API.

- **Container Name**: `ai-assistant`
- **Port**: 8000
- **Health Check**: HTTP GET to `/`
- **Environment Variables**: See `.env.docker` for complete list

### Redis

Redis is used for caching and session management.

- **Container Name**: `ai-assistant-redis`
- **Image**: `redis:7-alpine`
- **Port**: 6379
- **Persistence**: Data is stored in a Docker volume
- **Health Check**: Redis PING command

### SearXNG

SearXNG provides privacy-focused web search capabilities.

- **Container Name**: `ai-assistant-searxng`
- **Image**: `searxng/searxng:latest`
- **Port**: 8080
- **Configuration**: Custom settings in `searxng/settings.yml`
- **Health Check**: HTTP GET to `/`

### PostgreSQL (Optional)

PostgreSQL database for future use (disabled by default).

- **Container Name**: `ai-assistant-postgres`
- **Image**: `postgres:15-alpine`
- **Port**: 5432
- **Profile**: `postgres` (must be explicitly enabled)

## Configuration

### Environment Variables

Key environment variables for Docker deployment:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Host to bind to | `0.0.0.0` |
| `PORT` | Port to bind to | `8000` |
| `ENVIRONMENT` | Environment mode | `production` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `SEARXNG_URL` | SearXNG URL | `http://searxng:8080` |
| `SECRET_KEY` | Application secret key | Must be set |
| `BASE_URL` | Base URL for the application | `http://localhost` |
| `BEHIND_PROXY` | Whether running behind a proxy | `true` |

### Traefik Configuration

The project includes Traefik as a reverse proxy with configuration in `config/docker/traefik.yml`. Key features:
- Single entry point for all services
- Automatic service discovery
- Path-based routing
- Development dashboard
- Basic metrics and logging

### Customizing SearXNG

To customize SearXNG settings:

1. Edit `searxng/settings.yml`
2. Restart the service:
```bash
docker-compose restart searxng
```

### Adding PostgreSQL

To enable PostgreSQL:

1. Start with the postgres profile:
```bash
docker-compose --profile postgres up -d
```

2. Update your `.env` file to use PostgreSQL:
```bash
POSTGRES_URL=postgresql://postgres:password@postgres:5432/langchain_agent_hub
```

## Development

### Development Mode

For development with hot reload:

1. Use the development Docker Compose file:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

2. Or run locally with Docker services:
```bash
# Start only Redis and SearXNG
docker-compose up -d redis searxng

# Run the application locally
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Viewing Logs

View logs for all services:
```bash
docker-compose logs -f
```

View logs for a specific service:
```bash
docker-compose logs -f ai-assistant
docker-compose logs -f redis
docker-compose logs -f searxng
```

### Debugging

#### Accessing Container Shells

```bash
# AI Assistant container
docker-compose exec ai-assistant bash

# Redis container
docker-compose exec redis sh

# SearXNG container
docker-compose exec searxng sh
```

#### Checking Redis

```bash
docker-compose exec redis redis-cli
> INFO
> KEYS *
```

#### Rebuilding Images

After making changes to the application:

```bash
# Rebuild without cache
docker-compose build --no-cache

# Rebuild specific service
docker-compose build ai-assistant
```

## Production Considerations

### Security

1. **Change Default Secrets**: Always change the default `SECRET_KEY` and Redis password
2. **Network Security**: Consider using custom networks for production
3. **Resource Limits**: Set appropriate resource limits in production

### Resource Limits

Example production resource limits:

```yaml
services:
  ai-assistant:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Backup and Recovery

#### Redis Backup

```bash
# Create backup
docker-compose exec redis redis-cli BGSAVE
docker cp ai-assistant-redis:/data/dump.rdb ./redis-backup.rdb

# Restore backup
docker cp ./redis-backup.rdb ai-assistant-redis:/data/dump.rdb
docker-compose restart redis
```

#### PostgreSQL Backup (if enabled)

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres langchain_agent_hub > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres langchain_agent_hub < backup.sql
```

## Troubleshooting

### Common Issues

#### Port Conflicts

If ports are already in use, modify the `docker-compose.yml`:

```yaml
services:
  ai-assistant:
    ports:
      - "8001:8000"  # Use different host port
```

#### Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./logs
```

#### Memory Issues

If services are running out of memory:

1. Check resource usage:
```bash
docker stats
```

2. Add swap space or increase available RAM

#### Service Won't Start

1. Check logs:
```bash
docker-compose logs service-name
```

2. Verify configuration:
```bash
docker-compose config
```

3. Check if all required environment variables are set

### Health Checks

All services include health checks. To check health status:

```bash
docker-compose ps
```

### Performance Optimization

1. **Redis Optimization**: Configure Redis memory limits in production
2. **Application Scaling**: Consider using Docker Swarm or Kubernetes for scaling
3. **Caching**: Ensure Redis is properly configured for your workload

## Advanced Usage

### Custom Networks

For more complex setups, you can define custom networks:

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
```

### External Services

To connect to external services instead of Docker ones:

```yaml
services:
  ai-assistant:
    environment:
      REDIS_URL: redis://external-redis-host:6379/0
      SEARXNG_URL: https://external-searxng-instance.com
```

### Multi-Stage Builds

For production, consider using multi-stage builds to reduce image size:

```dockerfile
# Build stage
FROM python:3.12-slim as builder
# ... build steps ...

# Runtime stage
FROM python:3.12-slim as runtime
# ... copy only what's needed ...
```

## Migration from Local Development

To migrate from local development to Docker:

1. **Export Data**: Export any local data you need to keep
2. **Update Configuration**: Use `.env.docker` as a template
3. **Test Migration**: Test in a staging environment first
4. **Backup**: Create backups before migration
5. **Gradual Migration**: Consider migrating service by service

## Contributing

When contributing to the Docker setup:

1. Test changes with both `docker-compose up` and `docker-compose --profile postgres up`
2. Update documentation for any new environment variables
3. Include health checks for new services
4. Follow Docker best practices for security and performance