# Traefik Integration Guide

This guide explains how to use the AI Assistant application with Traefik reverse proxy integration.

## Overview

The application now uses Traefik as a reverse proxy to provide:
- Single entry point for all services
- Automatic service discovery
- Path-based routing
- Development dashboard
- Better security through controlled access

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Browser  │───▶│   Traefik :80   │───▶│ AI Assistant    │
│                 │    │   Dashboard:8080│    │ (Port 8000)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ├───▶ Gradio UI (/gradio)
                              ├───▶ SearXNG (/search)
                              └───▶ API (/v1/*)
```

## Service URLs

With Traefik integration, all services are accessible through the following URLs:

- **AI Assistant API**: `http://localhost/`
- **Gradio Interface**: `http://localhost/gradio/` (note the trailing slash)
- **OpenAI-Compatible API**: `http://localhost/v1/`
- **SearXNG Search**: `http://localhost/search/`
- **Traefik Dashboard**: `http://localhost:8080/`

## Quick Start

1. **Start the application**:
   ```bash
   docker-compose up -d
   ```

2. **Access services**:
   - Main application: http://localhost
   - Gradio UI: http://localhost/gradio
   - Traefik dashboard: http://localhost:8080

3. **Check service status**:
   ```bash
   docker-compose ps
   ```

## Configuration Files

### docker-compose.yml
The main configuration file now includes:
- Traefik service with dashboard enabled
- Updated service labels for routing
- Removed direct port exposures for internal services

### docker-configs/traefik.yml
Traefik configuration file located in the docker-configs directory with:
- Development-friendly settings
- Dashboard enabled (insecure)
- Docker provider configuration
- Basic metrics enabled
- Entry points for web traffic (port 80) and dashboard (port 8080)
- Prometheus metrics enabled
- Access logging enabled

## Service Routing

### AI Assistant
- **Router**: `ai-assistant`
- **Rule**: `PathPrefix(`/`)`
- **Priority**: 1 (highest)
- **Port**: 8000

### Gradio UI
- **Router**: `gradio`
- **Rule**: `PathPrefix(`/gradio`)`
- **Priority**: 2
- **Port**: 8000
- **Middleware**: Strips `/gradio` prefix

### SearXNG
- **Router**: `searxng`
- **Rule**: `PathPrefix(`/search`)`
- **Priority**: 10
- **Port**: 8080
- **Middleware**: Strips `/search` prefix

### API Endpoints
- **Router**: `ai-assistant-api`
- **Rule**: `PathPrefix(`/v1`)`
- **Priority**: 3
- **Port**: 8000

## Security Features

### Development Setup
- Basic security headers
- Forwarded headers for proxy information
- Internal services not exposed to host

### Production Considerations
For production deployment, you should:
1. Disable insecure dashboard: `TRAEFIK_API_INSECURE=false`
2. Add SSL/TLS termination
3. Implement authentication
4. Add rate limiting
5. Enable comprehensive security headers

## Monitoring

### Traefik Dashboard
Access at `http://localhost:8080` provides:
- Service health status
- Request metrics
- Router configuration
- Real-time monitoring

### Metrics
Prometheus metrics are enabled and available at:
- `http://localhost:8080/metrics`

## Troubleshooting

### Common Issues

1. **Services not accessible**:
   ```bash
   docker-compose logs traefik
   ```

2. **Port conflicts**:
   - Ensure ports 80 and 8080 are available
   - Check for other services using these ports

3. **Service discovery issues**:
   ```bash
   docker-compose restart traefik
   ```

4. **Routing problems**:
   - Check service labels in docker-compose.yml
   - Verify Traefik dashboard for router status

### Debug Commands

```bash
# Check all services
docker-compose ps

# View Traefik logs
docker-compose logs -f traefik

# Check service health
curl http://localhost/

# View Traefik configuration
curl http://localhost:8080/api/http/routers
```

## Migration from Previous Setup

If you're migrating from the previous direct port setup:

1. **URL Changes**:
   - Old: `http://localhost:8001` → New: `http://localhost`
   - Old: `http://localhost:8080` (SearXNG) → New: `http://localhost/search`
   - Old: `http://localhost:6379` (Redis) → Internal only

2. **Configuration Changes**:
   - No need to expose internal ports
   - All requests go through Traefik
   - Better security through controlled access

3. **Development Workflow**:
   - Same `docker-compose up` command
   - Enhanced monitoring through dashboard
   - Easier service management

## Environment Variables

Key environment variables for the setup:

```bash
# Traefik settings
TRAEFIK_API_INSECURE=true
TRAEFIK_LOG_LEVEL=INFO
TRAEFIK_DASHBOARD=true

# Application settings
BASE_URL=http://localhost
BEHIND_PROXY=true
```

## Next Steps

1. **Test all services** to ensure they work correctly
2. **Monitor the dashboard** to understand traffic patterns
3. **Consider SSL setup** for production use
4. **Add authentication** for the dashboard in production
5. **Implement rate limiting** for API endpoints

## Support

If you encounter issues:
1. Check the Traefik dashboard at `http://localhost:8080`
2. Review service logs with `docker-compose logs`
3. Verify all services are running with `docker-compose ps`
4. Check this documentation for common solutions