# Firecrawl Docker-Only Configuration

This document explains the Docker-only configuration for Firecrawl in the AI Assistant system.

## Overview

Firecrawl has been configured to work exclusively with Docker containers, providing complete control over your web scraping infrastructure and data.

## Architecture

```
AI Assistant → Firecrawl API (Docker)
                 ↓
           Redis + PostgreSQL + Playwright (Docker)
```

## Benefits of Docker-Only Mode

1. **Data Privacy**: All scraping data stays within your infrastructure
2. **Cost Control**: No per-request API costs or external dependencies
3. **Performance**: Local network access reduces latency
4. **Customization**: Full control over scraping configuration
5. **Security**: No external API calls or data exposure

## Required Services

Firecrawl requires the following Docker services to be running:

- **firecrawl-api**: Main API service (port 3002)
- **firecrawl-worker**: Background processing service
- **firecrawl-redis**: Queue management and caching
- **firecrawl-postgres**: Data storage for crawling results
- **firecrawl-playwright**: Browser automation for JavaScript rendering

## Configuration

### Environment Variables

```bash
# Required - Docker deployment mode
FIRECRAWL_DEPLOYMENT_MODE=docker

# Required - Docker service URL
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002

# Required - Queue authentication
FIRECRAWL_BULL_AUTH_KEY=change_me_firecrawl

# Optional - Health check timeout
FIRECRAWL_HEALTH_CHECK_TIMEOUT=10

# Scraping configuration
FIRECRAWL_SCRAPING_ENABLED=true
FIRECRAWL_MAX_CONCURRENT_SCRAPES=5
FIRECRAWL_SCRAPE_TIMEOUT=60
FIRECRAWL_FORMATS=["markdown", "raw"]
FIRECRAWL_WAIT_FOR=2000
FIRECRAWL_SCREENSHOT=false
FIRECRAWL_EXTRACT_IMAGES=false
FIRECRAWL_EXTRACT_LINKS=true
```

### Docker Compose

The Firecrawl services are defined in the main `docker-compose.yml` file under the `firecrawl` profile:

```bash
# Start all Firecrawl services
docker compose --profile firecrawl up -d

# Check service status
docker compose --profile firecrawl ps

# View logs
docker compose --profile firecrawl logs
```

## Usage

### Basic Scraping

```python
from app.core.tools.firecrawl_tool import FirecrawlTool

# Create tool instance (no API key needed)
tool = FirecrawlTool()

# Scrape a URL
result = await tool.execute(url="https://example.com")

# Access the content
print(f"Title: {result['title']}")
print(f"Content: {result['content']}")
```

### Batch Scraping

```python
# Scrape multiple URLs
urls = ["https://example1.com", "https://example2.com"]
results = await tool.batch_scrape(urls=urls)

for result in results:
    if result.get("success", False):
        print(f"Scraped {result['url']}: {result['title']}")
    else:
        print(f"Failed to scrape {result['url']}: {result['error']}")
```

## Health Checks

### Manual Health Check

```bash
# Check API health
curl http://localhost:3002/health

# Check all services
docker compose --profile firecrawl ps
```

### Health Check Utility

```bash
# Run the health check script
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

Dependencies:
  Redis: healthy
  Postgres: healthy
  Playwright: healthy

Scraping Test:
  Status: success
  Title: httpbin.org
  Content Length: 1234 characters
==================================================
```

## Troubleshooting

### Common Issues

1. **Services Not Running**
   ```bash
   # Start services
   docker compose --profile firecrawl up -d
   
   # Check status
   docker compose --profile firecrawl ps
   ```

2. **Unhealthy Services**
   ```bash
   # Check logs
   docker compose --profile firecrawl logs firecrawl-api
   docker compose --profile firecrawl logs firecrawl-worker
   
   # Restart services
   docker compose --profile firecrawl restart
   ```

3. **Connection Issues**
   ```bash
   # Check network connectivity
   docker compose exec ai-assistant ping firecrawl-api
   
   # Verify port accessibility
   docker compose exec ai-assistant curl http://firecrawl-api:3002/health
   ```

4. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats --filter "name=firecrawl"
   
   # Scale workers
   docker compose --profile firecrawl up -d --scale firecrawl-worker=3
   ```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Docker Firecrawl instance is unhealthy` | Firecrawl API not responding | Check service logs and restart services |
| `Connection refused` | Services not running | Start Firecrawl services |
| `Timeout during scraping` | Slow website or network | Increase timeout values |
| `All Firecrawl services must be running` | Missing dependencies | Start all required services |

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
docker compose --profile firecrawl up -d --scale firecrawl-worker=3

# Scale in production
docker compose --profile firecrawl up -d --scale firecrawl-worker=5
```

### Optimization Settings

```bash
# Increase concurrent scrapes
FIRECRAWL_MAX_CONCURRENT_SCRAPES=10

# Adjust timeout for slow websites
FIRECRAWL_SCRAPE_TIMEOUT=120

# Optimize wait time
FIRECRAWL_WAIT_FOR=5000
```

## Security Considerations

### Network Security

- Firecrawl services are internal-only (no external exposure)
- API accessible only within Docker network
- Authentication via `BULL_AUTH_KEY`

### Data Privacy

- All scraping data stays within your infrastructure
- No external API calls
- Configurable data retention policies

### Access Control

```bash
# Enable authentication
USE_DB_AUTHENTICATION=true

# Set secure authentication key
BULL_AUTH_KEY=your_secure_key_here

# Rate limiting
FIRECRAWL_RATE_LIMIT_MAX_REQUESTS=100
FIRECRAWL_RATE_LIMIT_WINDOW_MS=60000
```

## Monitoring

### Built-in Metrics

Firecrawl provides metrics at `/metrics` endpoint:

```bash
# Access metrics
curl http://localhost:3002/metrics
```

### Prometheus Integration

Add to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'firecrawl'
    static_configs:
      - targets: ['firecrawl-api:3002']
    metrics_path: '/metrics'
```

## Migration from API Mode

If you were previously using the external API:

1. Update your `.env` file:
   ```bash
   FIRECRAWL_DEPLOYMENT_MODE=docker
   FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
   ```

2. Remove API key configuration:
   ```bash
   # Remove these lines
   # FIRECRAWL_API_KEY=your-api-key
   # FIRECRAWL_BASE_URL=https://api.firecrawl.dev
   ```

3. Start Docker services:
   ```bash
   docker compose --profile firecrawl up -d
   ```

4. Test the configuration:
   ```bash
   python utility/firecrawl_health_check.py
   ```

## Additional Resources

- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Health Check Utility](../../utility/firecrawl_health_check.py)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)
- [Performance Optimization Guide](../deployment/monitoring.md)