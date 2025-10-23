# Frontend Integration with Traefik

This guide explains how to integrate your React frontend with Traefik and addresses common development concerns.

## Overview

The setup provides two modes:
- **Development Mode**: Hot-reloading with Vite dev server
- **Production Mode**: Optimized static files served by Nginx

## Configuration Files

### Docker Compose Service

The frontend service in `docker/docker-compose.yml` handles both development and production:

```yaml
frontend:
  build:
    context: ../
    dockerfile: docker/frontend/Dockerfile.${BUILD_MODE:-prod}
  container_name: ${COMPOSE_PROJECT_NAME:-my-stack}-frontend
  restart: unless-stopped
  environment:
    - VITE_API_BASE_URL=http://fastapi.${BASE_DOMAIN:-localhost}
  volumes:
    # Mount the local frontend directory for hot-reloading in development
    - ../../frontend:/app
    - /app/node_modules  # Prevent overwriting node_modules
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.frontend.rule=Host(`frontend.${BASE_DOMAIN:-localhost}`,`${BASE_DOMAIN:-localhost}`)"
    - "traefik.http.services.frontend.loadbalancer.server.port=5173"
```

### Environment Variables

Create a `.env` file in the `docker/` directory:

```bash
# Development mode with hot-reloading
BUILD_MODE=dev

# Production mode (optimized builds)
# BUILD_MODE=prod

# Base domain for services
BASE_DOMAIN=localhost
```

## Access Points

Once running, your services will be available at:

- **Frontend**: http://frontend.localhost (or http://localhost)
- **Backend API**: http://fastapi.localhost
- **Traefik Dashboard**: http://traefik.localhost:8080

## Development Workflow

### Starting Development Environment

1. Set environment variables:
   ```bash
   cd docker
   cp .env.example .env
   # Edit .env to set BUILD_MODE=dev
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Access your frontend at http://frontend.localhost

### Hot-Reloading

In development mode:
- Changes to your frontend code will automatically reload the browser
- The Vite dev server runs inside the container with hot-reloading enabled
- File changes are synced through the volume mount

### API Integration

The frontend automatically connects to the backend via the `VITE_API_BASE_URL` environment variable:
- Development: http://fastapi.localhost
- Production: http://fastapi.localhost

## Production Deployment

### Building for Production

1. Set environment variables:
   ```bash
   # In docker/.env
   BUILD_MODE=prod
   ```

2. Build and start services:
   ```bash
   docker-compose up -d --build
   ```

3. The production build:
   - Uses Nginx to serve optimized static files
   - Enables gzip compression
   - Sets appropriate caching headers
   - Handles client-side routing

## CORS Configuration

The backend includes CORS middleware configured for:
- `http://frontend.localhost`
- `http://localhost`

Traefik also adds CORS headers to ensure proper cross-origin requests.

## Development Speed Considerations

### Initial Setup Overhead

- **Container startup time**: ~30-60 seconds initially
- **Build time**: ~2-5 minutes for first build
- **Subsequent starts**: ~10-20 seconds with cached layers

### Development Benefits

1. **Consistent Environment**: Same configuration as production
2. **Isolated Dependencies**: No conflicts with local tools
3. **Easy Team Onboarding**: One command to start entire stack
4. **Hot-Reloading**: Maintained in development mode

### Performance Optimizations

1. **Volume Mounts**: Only mount necessary directories
2. **Node Modules Cache**: Separate volume to prevent rebuilds
3. **Docker Build Cache**: Reuses unchanged layers
4. **Parallel Development**: Frontend and backend can be developed independently

### Tips for Faster Development

1. **Use SSD Storage**: Improves I/O performance for volume mounts
2. **Allocate Enough Resources**: Ensure Docker has sufficient CPU/memory
3. **Selective Volume Mounts**: In production-like development, mount only specific files
4. **Use .dockerignore**: Exclude unnecessary files from build context

## Troubleshooting

### Frontend Not Loading

1. Check if the frontend container is running:
   ```bash
   docker-compose ps frontend
   ```

2. Check container logs:
   ```bash
   docker-compose logs frontend
   ```

3. Verify Traefik routing:
   - Access http://traefik.localhost:8080
   - Check if frontend service is registered

### API Connection Issues

1. Verify the API URL is correctly set:
   ```bash
   docker-compose exec frontend env | grep VITE_API_BASE_URL
   ```

2. Check backend container is running and healthy:
   ```bash
   docker-compose ps fast-api-app
   ```

3. Test API connectivity:
   ```bash
   curl http://fastapi.localhost/health
   ```

### Hot-Reloading Not Working

1. Verify volume mounts are correct:
   ```bash
   docker-compose exec frontend ls -la /app
   ```

2. Check Vite dev server logs:
   ```bash
   docker-compose logs frontend
   ```

3. Ensure file permissions are correct:
   ```bash
   docker-compose exec frontend ls -la /app/src/
   ```

## Best Practices

1. **Development Mode**: Use `BUILD_MODE=dev` for active development
2. **Production Testing**: Test production builds before deployment
3. **Environment Variables**: Use environment-specific configurations
4. **Health Checks**: All services include health checks for reliability
5. **Logging**: Use structured logging for better debugging
6. **Security**: Keep production and development configurations separate

## Next Steps

1. Customize the Nginx configuration for your specific needs
2. Add SSL/TLS certificates for production
3. Implement CI/CD pipelines for automated deployments
4. Set up monitoring and alerting for your services
5. Configure backup strategies for your data