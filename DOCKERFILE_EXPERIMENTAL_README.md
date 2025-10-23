# Experimental Dockerfile

This document describes the experimental Dockerfile (`Dockerfile.experimental`) that combines the functionality of both the production (`Dockerfile`) and development (`Dockerfile.dev`) Dockerfiles into a single, configurable Dockerfile.

## Purpose

The experimental Dockerfile is designed to simplify the Docker configuration by:
1. Using a single Dockerfile for both production and development environments
2. Controlling the build behavior through environment variables
3. Installing uv via pip instead of using the uv-astral image
4. Providing a consistent base for both environments

## Key Differences from Original Dockerfiles

1. **UV Installation**: Installs uv via pip (`pip install --no-cache-dir uv==0.9.3`) instead of using the uv-astral image
2. **Single File**: Combines both production and development configurations in one file
3. **Environment Variable Control**: Uses `BUILD_MODE` environment variable to control the build behavior
4. **Unified Entrypoint**: Uses a single entrypoint script that handles both modes

## Usage

### Building for Production

```bash
docker build -f Dockerfile.experimental --build-arg BUILD_MODE=production -t ai-assistant:production .
```

Or using docker-compose:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.experimental
      args:
        BUILD_MODE: production
```

### Building for Development

```bash
docker build -f Dockerfile.experimental --build-arg BUILD_MODE=development -t ai-assistant:development .
```

Or using docker-compose:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.experimental
      args:
        BUILD_MODE: development
```

### Running the Container

For production:

```bash
docker run -d -p 8000:8000 -e BUILD_MODE=production ai-assistant:production
```

For development:

```bash
docker run -d -p 8000:8000 -e BUILD_MODE=development ai-assistant:development
```

## Environment Variables

- `BUILD_MODE`: Controls the build mode (production or development)
  - `production`: Installs production dependencies, includes Chrome/ChromeDriver, runs with workers
  - `development`: Installs development dependencies, includes Playwright deps, runs with hot reload

## Behavior Differences

### Production Mode
- Installs production dependencies only (no dev dependencies)
- Includes Chrome and ChromeDriver for Selenium WebDriver
- Runs with a single worker for stability
- Includes health checks
- Optimized for production deployment

### Development Mode
- Installs development dependencies
- Includes Playwright system dependencies
- Runs with hot reload enabled
- Exposes additional port 8888 for potential development tools
- Optimized for development workflow

## Testing

A test script (`test_experimental_docker.sh`) is provided to verify that the Dockerfile works correctly in both modes:

```bash
./test_experimental_docker.sh
```

This script will:
1. Build the Docker image in production mode
2. Start a container and verify it runs correctly
3. Build the Docker image in development mode
4. Start a container and verify it runs correctly

## Migration Path

To migrate from the current Dockerfiles to the experimental one:

1. Replace references to `Dockerfile` with `Dockerfile.experimental` in your docker-compose files
2. Add the `BUILD_MODE` build argument to your docker-compose configuration
3. Set the `BUILD_MODE` environment variable when running containers

## Future Considerations

This experimental Dockerfile serves as a proof of concept for a unified Docker configuration. If successful, it could replace the separate `Dockerfile` and `Dockerfile.dev` files, simplifying the project structure and maintenance.

## Troubleshooting

If you encounter issues:
1. Check that the `BUILD_MODE` environment variable is set correctly
2. Verify that all required files are present in the build context
3. Run the test script to identify any issues with the Dockerfile
4. Check the container logs for detailed error messages