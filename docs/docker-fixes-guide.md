# Docker Fixes Guide - Chainlit and Firecrawl

This guide documents the fixes applied to resolve Docker issues with chainlit and firecrawl services in the development environment.

## Issues Fixed

### 1. Firecrawl Service URL Mismatch

**Problem**: The firecrawl service in `docker-compose.yml` was named `firecrawl` but the configuration in `app/core/config.py` was trying to connect to `firecrawl-api:3002`.

**Solution**: Updated all references from `firecrawl-api` to `firecrawl`:
- `app/core/config.py` - Changed `docker_url` from `http://firecrawl-api:3002` to `http://firecrawl:3002`
- `app/core/secure_settings.py` - Updated secure settings to match
- `utility/firecrawl_health_check.py` - Updated default URL
- All test files updated for consistency

### 2. Chainlit Service Configuration

**Problem**: Chainlit service was properly configured but needed verification.

**Solution**: Verified that:
- Chainlit service is correctly configured in `docker-compose.yml`
- Middleware configuration in `config/docker/middlewares.yml` includes proper WebSocket and CORS settings
- Chainlit app initialization is correct

## Testing the Fixes

### Quick Test Commands

1. **Test All Services** (Recommended):
   ```bash
   # Run from within the ai-assistant-dev container
   python utility/test_docker_services.py
   ```

2. **Test Firecrawl Only**:
   ```bash
   # Run from within the ai-assistant-dev container
   python utility/test_firecrawl_connection.py
   ```

3. **Test Chainlit Only**:
   ```bash
   # Run from within the ai-assistant-dev container
   python utility/test_chainlit_connection.py
   ```

### Manual Testing

1. **Start theDevelopment Environment**:
   ```bash
   docker compose --profile dev up -d
   ```

2. **Check Service Status**:
   ```bash
   docker compose --profile dev ps
   ```

3. **Access Services in Browser**:
   - AI Assistant: http://localhost:8000
   - Chainlit Chat: http://localhost:8000/chat
   - Traefik Dashboard: http://localhost:8080
   - SearXNG Search: http://localhost:8000/search
   - Redis Commander: http://localhost:8000/redis

### Running Tests Inside Container

To run the test scripts from within the Docker container:

```bash
# Enter the ai-assistant-dev container
docker compose --profile dev exec ai-assistant-dev bash

# Run the comprehensive test
python utility/test_docker_services.py

# Or run individual tests
python utility/test_firecrawl_connection.py
python utility/test_chainlit_connection.py
```

## Troubleshooting

### If Firecrawl Still Fails

1. Check firecrawl container logs:
   ```bash
   docker compose --profile dev logs firecrawl
   ```

2. Verify firecrawl is healthy:
   ```bash
   docker compose --profile dev exec firecrawl curl -f http://localhost:3002/health
   ```

3. Restart firecrawl service:
   ```bash
   docker compose --profile dev restart firecrawl
   ```

### If Chainlit Still Fails

1. Check chainlit container logs:
   ```bash
   docker compose --profile dev logs chainlit
   ```

2. Verify chainlit is running:
   ```bash
   docker compose --profile dev exec chainlit curl -f http://localhost:8001/
   ```

3. Restart chainlit service:
   ```bash
   docker compose --profile dev restart chainlit
   ```

### Network Connectivity Issues

1. Check if services are on the same network:
   ```bash
   docker network ls
   docker network inspect ai-assistant_ai-assistant-network
   ```

2. Test connectivity between containers:
   ```bash
   docker compose --profile dev exec ai-assistant-dev ping firecrawl
   docker compose --profile dev exec ai-assistant-dev ping chainlit
   ```

## Configuration Files Modified

1. `app/core/config.py` - Fixed firecrawl docker_url
2. `app/core/secure_settings.py` - Updated secure settings for firecrawl
3. `utility/firecrawl_health_check.py` - Updated default URL
4. All test files in `tests/` directory - Updated URLs for consistency

## New Test Scripts Created

1. `utility/test_docker_services.py` - Comprehensive test for all services
2. `utility/test_firecrawl_connection.py` - Specific firecrawl tests
3. `utility/test_chainlit_connection.py` - Specific chainlit tests

## Best Practices

1. Always run the comprehensive test script after making changes:
   ```bash
   python utility/test_docker_services.py
   ```

2. Check container logs if services fail to start:
   ```bash
   docker compose --profile dev logs [service-name]
   ```

3. Use the health check utilities to verify service health:
   ```bash
   python utility/firecrawl_health_check.py
   python utility/healthcheck_dev.py
   ```

4. Restart services after configuration changes:
   ```bash
   docker compose --profile dev restart