# SearXNG Setup for AI Assistant

This document explains the SearXNG search configuration for both development and production environments.

## Overview

SearXNG is a privacy-respecting metasearch engine that provides search capabilities to the AI Assistant. It's configured to work seamlessly with both development and production Traefik setups.

## Configuration Files

### 1. Docker Compose Service
- **Service Name**: `searxng`
- **Image**: `searxng/searxng:latest`
- **Container**: `ai-assistant-searxng`
- **Shared between dev and production environments**

### 2. Configuration Files
- **`docker-configs/searxng/settings.yml`**: Main SearXNG configuration
- **`docker-configs/searxng/limiter.toml`**: Rate limiting and bot protection
- **`docker-configs/searxng/plugins.yml`**: Available plugins

## Environment Variables

### Core Variables
- `SEARXNG_SECRET_KEY`: Secret key for SearXNG security (required)
- `SEARXNG_BASE_URL`: Base URL for SearXNG (default: `http://searxng:8080/`)

### Customization Variables
- `SEARXNG_INSTANCE_NAME`: Display name (default: "AI Assistant Search")
- `SEARXNG_AUTOCOMPLETE`: Autocomplete provider (default: "google")
- `SEARXNG_SEARCH_LANGUAGE`: Default search language (default: "auto")
- `SEARXNG_THEME`: UI theme (default: "simple")
- `SEARXNG_UI_DEFAULT_LOCALE`: UI language (default: "en")
- `SEARXNG_DEBUG`: Enable debug mode (default: "false")

## Access URLs

### Development Environment
- **Search Interface**: `http://localhost:8000/search`
- **Direct Access**: `http://localhost:8000/search` (routed through Traefik-dev)

### Production Environment
- **Search Interface**: `http://localhost/search`
- **Direct Access**: `http://localhost/search` (routed through Traefik with security)

## Routing Configuration

### Development Routing
- **Router**: `searxng` (basic)
- **Middleware**: `searxng-stripprefix` only
- **Entry Points**: `web` (port 8000 via Traefik-dev)
- **Priority**: 10

### Production Routing
- **Router**: `searxng-prod` (enhanced security)
- **Middleware**: `searxng-stripprefix, security-headers, rate-limit`
- **Entry Points**: `web` (port 80 via Traefik)
- **Priority**: 10

## Search Engines Configuration

SearXNG is configured with the following search engines:
- **Google**: Primary search with mobile UI disabled
- **Bing**: Secondary search provider
- **DuckDuckGo**: Privacy-focused search
- **Wikipedia**: Encyclopedia results
- **Brave Search**: Privacy-focused search
- **Qwant**: European search provider

### Engine Settings
- **Timeout**: 3-4 seconds per engine
- **Safe Search**: Level 1 (moderate)
- **Output Formats**: HTML, JSON, CSV
- **Image Proxy**: Enabled for privacy

## Security Features

### Production Only
- **Rate Limiting**: Applied via Traefik middleware
- **Security Headers**: HSTS, CSP, XSS protection
- **Bot Protection**: Via limiter.toml configuration

### Bot Detection (limiter.toml)
- **Allowed Networks**: Docker networks and localhost
- **Link Local Filtering**: Disabled for Docker compatibility
- **IP Lists**: Configurable allow/block lists

## Dependencies

### Required Services
- **Redis**: For caching and session storage
- **Traefik**: For reverse proxy routing

### Docker Network
- **Network**: `ai-assistant-network`
- **Internal Communication**: `http://searxng:8080/`

## Usage Examples

### Basic Search
```bash
# Development
curl "http://localhost:8000/search?q=test"

# Production
curl "http://localhost/search?q=test"
```

### API Access
```bash
# JSON format
curl "http://localhost:8000/search?q=test&format=json"

# CSV format
curl "http://localhost:8000/search?q=test&format=csv"
```

## Troubleshooting

### Common Issues

1. **SearXNG not accessible**
   - Check if Redis is running: `docker compose ps redis`
   - Verify SearXNG logs: `docker compose logs searxng`
   - Check Traefik logs: `docker compose logs traefik` or `docker compose logs traefik-dev`

2. **Search results not loading**
   - Verify search engine timeouts in settings.yml
   - Check network connectivity from container
   - Review SearXNG logs for engine-specific errors

3. **Configuration not updating**
   - Restart SearXNG after config changes: `docker compose restart searxng`
   - Verify environment variables in .env file
   - Check settings.yml syntax

### Health Checks
```bash
# Check SearXNG health
curl -f http://localhost:8000/search/health

# Check configuration
docker compose exec searxng curl -f http://localhost:8080/
```

### Debug Mode
Enable debug logging by setting in `.env`:
```bash
SEARXNG_DEBUG=true
```

Then restart SearXNG:
```bash
docker compose restart searxng
```

## Maintenance

### Updates
- **Image**: Update `searxng/searxng:latest` in docker-compose.yml
- **Config**: Review settings.yml for new features
- **Engines**: Periodically check search engine availability

### Performance
- **Monitor**: Check response times via Traefik dashboard
- **Optimize**: Adjust engine timeouts based on network conditions
- **Scale**: Consider multiple SearXNG instances for high load

## Security Considerations

1. **Secret Key**: Always set a strong `SEARXNG_SECRET_KEY`
2. **Rate Limiting**: Production environment has built-in limits
3. **Privacy**: Image proxy enabled for all images
4. **Network**: SearXNG only accessible through Traefik
5. **Updates**: Regularly update SearXNG image for security patches