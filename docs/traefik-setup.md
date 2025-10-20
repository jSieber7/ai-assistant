# Traefik Setup for AI Assistant

This document explains the Traefik configuration for both development and production environments.

## Overview

The AI Assistant uses separate Traefik configurations for development and production environments to provide optimal experiences for each use case.

## Development Environment

### Services
- **traefik-dev**: Development-specific reverse proxy
- **ai-assistant-dev**: Main application with hot reload
- **redis-commander**: Redis GUI accessible via Traefik

### Access Points
- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Chainlit Interface**: http://localhost:8000/chainlit
- **Traefik Dashboard**: http://localhost:8080/dashboard
- **Redis Commander**: http://localhost:8000/redis
- **SearXNG Search**: http://localhost:8000/search

### Features
- Insecure API dashboard for easy debugging
- Debug logging enabled
- Direct port mapping (8000:80) for main application
- All services routed through Traefik at port 8000

## Production Environment

### Services
- **traefik**: Production-ready reverse proxy with security features
- **ai-assistant**: Production application container
- **searxng**: Search service with security middleware

### Access Points
- **Main Application**: http://localhost
- **API Documentation**: http://localhost/docs
- **Chainlit Interface**: http://localhost/chainlit
- **Traefik Dashboard**: http://localhost:8080/dashboard (admin/admin)
- **SearXNG Search**: http://localhost/search

### Security Features
- Disabled insecure API (dashboard requires authentication)
- Security headers middleware
- Rate limiting (100 requests/minute, burst 200)
- SSL/TLS ready (ports 443 and websecure entrypoint configured)
- Compression enabled
- Warning-level logging for reduced noise

## Usage

### Development
```bash
# Start development environment
make dev

# Start minimal development environment
make dev-basic

# Start with development tools
make tools
```

### Production
```bash
# Start production environment
make up
```

## Configuration Files

- **docker-compose.yml**: Main service definitions
- **config/docker/traefik.yml**: Base Traefik configuration
- **config/docker/traefik-prod.yml**: Production-specific security settings

## Security Middleware (Production Only)

### security-headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security (31536000 seconds)
- Referrer-Policy: strict-origin-when-cross-origin
- Content-Security-Policy: restrictive default policy

### rate-limit
- Average: 100 requests per minute
- Burst: 200 requests

### compression
- Excludes streaming content types
- Minimum response body size: 1024 bytes

## SSL Configuration

Production is SSL-ready with:
- TLS 1.2+ minimum version
- Modern cipher suites
- HSTS preloading
- Secure curve preferences

To enable SSL, add SSL certificates and update the Traefik configuration accordingly.

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 80, 443, and 8080 are available for production
2. **Development port**: Ensure port 8000 is available for development
3. **Authentication**: Production dashboard uses admin/admin credentials
4. **Service discovery**: All services must be on the ai-assistant-network

### Checking Configuration
```bash
# Validate Docker compose configuration
docker compose config --quiet

# Check running services
docker compose ps

# View Traefik logs
docker compose logs traefik  # production
docker compose logs traefik-dev  # development
```

## Migration from Previous Setup

- Development no longer uses direct port mapping for services
- All services now route through Traefik for consistency
- Production has enhanced security middleware
- Dashboard access is now at /dashboard endpoint in both environments