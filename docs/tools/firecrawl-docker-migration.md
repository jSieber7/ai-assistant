# Firecrawl Docker Migration Guide

This guide explains how to migrate from the external Firecrawl API to a self-hosted Docker deployment.

## Overview

The migration provides several benefits:
- **Cost Control**: No per-request API costs
- **Privacy**: All scraping stays within your infrastructure
- **Performance**: Local network access reduces latency
- **Customization**: Full control over scraping configuration
- **Reliability**: No external service dependencies

## Architecture

### Before (External API)
```
AI Assistant → Firecrawl API (External)
```

### After (Docker Self-Hosted)
```
AI Assistant → Firecrawl API (Local Docker)
                 ↓
           Redis + PostgreSQL + Playwright
```

## Migration Steps

### 1. Update Environment Configuration

Update your `.env` file to use Docker mode:

```bash
# Firecrawl Deployment Mode
FIRECRAWL_DEPLOYMENT_MODE=docker

# Docker Configuration
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
FIRECRAWL_BULL_AUTH_KEY=change_me_firecrawl

# Keep external API as fallback (optional)
FIRECRAWL_API_KEY=your-firecrawl-api-key-here
FIRECRAWL_BASE_URL=https://api.firecrawl.dev
FIRECRAWL_ENABLE_FALLBACK=true
```

### 2. Enable Firecrawl Docker Services

Uncomment the Firecrawl include in your main `docker-compose.yml`:

```yaml
# Firecrawl Self-Hosted Services
include:
  - config/docker/firecrawl/docker-compose.yml
```

### 3. Start Services

Start all services including Firecrawl:

```bash
# Start main services
docker compose up -d

# Start Firecrawl services
docker compose -f config/docker/firecrawl/docker-compose.yml up -d
```

Or start everything at once:

```bash
docker compose --profile firecrawl up -d
```

### 4. Verify Installation

Run the health check utility:

```bash
python utility/firecrawl_health_check.py
```

Expected output:
```
==================================================
FIRECRAWL HEALTH CHECK RESULTS
==================================================
Overall Status: HEALTHY
API URL: http://firecrawl-api:3002

API Health:
  Status: healthy
  Status Code: 200

API Version:
  Version: 1.0.0

Dependencies:
  Redis: expected_failure
    Redis is not HTTP accessible (expected)
  Postgres: expected_failure
    PostgreSQL is not HTTP accessible (expected)
  Playwright: healthy
    Status Code: 200

Scraping Test:
  Status: success
  Title: httpbin.org
  Content Length: 1234 characters
  Formats: markdown, raw
==================================================
```

## Configuration Options

### Deployment Modes

#### API Mode (Default)
```bash
FIRECRAWL_DEPLOYMENT_MODE=api
FIRECRAWL_API_KEY=your-api-key
FIRECRAWL_BASE_URL=https://api.firecrawl.dev
```

#### Docker Mode
```bash
FIRECRAWL_DEPLOYMENT_MODE=docker
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
```

#### Hybrid Mode (Docker with Fallback)
```bash
FIRECRAWL_DEPLOYMENT_MODE=docker
FIRECRAWL_ENABLE_FALLBACK=true
FIRECRAWL_API_KEY=your-api-key  # For fallback
```

### Docker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRECRAWL_DOCKER_URL` | `http://firecrawl-api:3002` | Local Firecrawl API URL |
| `FIRECRAWL_BULL_AUTH_KEY` | `change_me_firecrawl` | Queue admin authentication |
| `FIRECRAWL_ENABLE_FALLBACK` | `true` | Enable fallback to external API |
| `FIRECRAWL_FALLBACK_TIMEOUT` | `10` | Health check timeout (seconds) |

### Service Configuration

#### Redis
- **Purpose**: Queue management and caching
- **Default**: `redis:7.2-alpine`
- **Persistence**: Enabled with volume mount

#### PostgreSQL
- **Purpose**: Data storage for crawling results
- **Default**: `postgres:15-alpine`
- **Credentials**: Configurable via environment

#### Playwright
- **Purpose**: Browser automation for JavaScript rendering
- **Default**: `ghcr.io/firecrawl/firecrawl-playwright:latest`
- **Port**: 3000

## Monitoring and Health Checks

### Built-in Health Checks

All Firecrawl services include health checks:

```bash
# Check individual services
docker compose ps firecrawl-api
docker compose ps firecrawl-worker
docker compose ps firecrawl-playwright

# View logs
docker compose logs firecrawl-api
docker compose logs firecrawl-worker
```

### Health Check Utility

Use the provided health check script:

```bash
# Basic check
python utility/firecrawl_health_check.py

# Custom URL
python utility/firecrawl_health_check.py --url http://localhost:3002

# Verbose output
python utility/firecrawl_health_check.py --verbose

# Custom test URL
python utility/firecrawl_health_check.py --test-url https://example.com
```

### Monitoring Integration

Add to your monitoring system:

```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'firecrawl'
    static_configs:
      - targets: ['firecrawl-api:3002']
    metrics_path: '/metrics'
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker compose logs firecrawl-api

# Check dependencies
docker compose ps firecrawl-redis firecrawl-postgres firecrawl-playwright
```

#### Scraping Fails
```bash
# Test API directly
curl -X POST http://localhost:3002/v1/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org/html"}'

# Check Playwright service
curl http://localhost:3000/health
```

#### Performance Issues
```bash
# Check resource usage
docker stats firecrawl-api firecrawl-worker

# Scale workers
docker compose up -d --scale firecrawl-worker=3
```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Docker Firecrawl instance is unhealthy` | Firecrawl API not responding | Check service logs and dependencies |
| `Fallback to external API disabled` | Fallback not configured | Enable fallback or fix Docker instance |
| `Connection refused` | Service not running | Start Firecrawl services |
| `Timeout during scraping` | Slow website or network | Increase timeout values |

## Performance Optimization

### Resource Allocation

Adjust resource limits in `docker-compose.yml`:

```yaml
services:
  firecrawl-api:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Scaling Workers

```bash
# Scale workers horizontally
docker compose up -d --scale firecrawl-worker=3

# Scale in production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale firecrawl-worker=5
```

### Caching Configuration

```bash
# Redis optimization
FIRECRAWL_REDIS_MAX_CONNECTIONS=20
FIRECRAWL_REDIS_CONNECTION_TIMEOUT=5

# Scraping optimization
FIRECRAWL_MAX_CONCURRENT_SCRAPES=10
FIRECRAWL_SCRAPE_TIMEOUT=120
```

## Security Considerations

### Network Security

- Firecrawl services are internal-only (no external exposure)
- API accessible only within Docker network
- Optional authentication via `BULL_AUTH_KEY`

### Data Privacy

- All scraping data stays within your infrastructure
- No external API calls when using Docker mode
- Configurable data retention policies

### Access Control

```bash
# Enable authentication
USE_DB_AUTHENTICATION=true

# Set admin credentials
BULL_AUTH_KEY=your_secure_key_here

# Restrict API access
FIRECRAWL_RATE_LIMIT_MAX_REQUESTS=100
FIRECRAWL_RATE_LIMIT_WINDOW_MS=60000
```

## Backup and Recovery

### Data Backup

```bash
# Backup PostgreSQL
docker compose exec firecrawl-postgres pg_dump -U firecrawl firecrawl > backup.sql

# Backup Redis
docker compose exec firecrawl-redis redis-cli BGSAVE
```

### Service Recovery

```bash
# Restart services
docker compose restart firecrawl-api firecrawl-worker

# Full recovery
docker compose down
docker compose up -d
```

## Migration Validation

### Test Scraping

```python
# Test script
import asyncio
from app.core.tools.firecrawl_tool import FirecrawlTool

async def test_migration():
    tool = FirecrawlTool()
    result = await tool.execute(
        url="https://httpbin.org/html",
        formats=["markdown"]
    )
    print(f"Success: {result['title']}")
    print(f"Content length: {len(result['content'])}")

asyncio.run(test_migration())
```

### Performance Comparison

```bash
# API mode timing
time python -c "test_firecrawl_api()"

# Docker mode timing  
time python -c "test_firecrawl_docker()"
```

## Rollback Plan

If you need to rollback to the external API:

1. Update environment:
   ```bash
   FIRECRAWL_DEPLOYMENT_MODE=api
   ```

2. Stop Docker services:
   ```bash
   docker compose -f config/docker/firecrawl/docker-compose.yml down
   ```

3. Restart main application:
   ```bash
   docker compose restart ai-assistant
   ```

## Additional Resources

- [Firecrawl Documentation](https://docs.firecrawl.dev)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Health Check Utility](../../utility/firecrawl_health_check.py)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)