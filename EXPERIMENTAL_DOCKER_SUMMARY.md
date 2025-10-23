# Experimental Docker Setup Summary

This document provides a summary of the experimental Docker setup that combines the functionality of both the production and development Dockerfiles into a single, configurable solution.

## Files Created

### 1. Dockerfile.experimental
The main experimental Dockerfile that replaces both `Dockerfile` and `Dockerfile.dev`.

**Key Features:**
- Single Dockerfile for both production and development modes
- Installs uv via pip instead of using uv-astral image
- Uses `BUILD_MODE` build argument to control behavior
- Multi-stage build with builder and production stages
- Conditional installation of dependencies based on mode

**Usage:**
```bash
# Build for production
docker build -f Dockerfile.experimental --build-arg BUILD_MODE=production -t ai-assistant:prod .

# Build for development
docker build -f Dockerfile.experimental --build-arg BUILD_MODE=development -t ai-assistant:dev .
```

### 2. entrypoint.sh
A shell script that handles the startup logic for both production and development modes.

**Features:**
- Checks the `BUILD_MODE` environment variable
- Runs the appropriate startup command based on mode
- Enables hot reload in development mode
- Uses single worker in production mode

### 3. docker-compose.experimental.yml
An example docker-compose file that demonstrates how to use the experimental Dockerfile.

**Features:**
- Uses `BUILD_MODE` build argument
- Includes separate service definitions for production and development
- Uses profiles to switch between modes
- Includes commented examples for additional services (postgres, redis)

**Usage:**
```bash
# Production mode
docker-compose -f docker-compose.experimental.yml --profile production up

# Development mode
docker-compose -f docker-compose.experimental.yml --profile development up
```

### 4. run_experimental.sh
A convenience script for building and running the experimental Dockerfile.

**Features:**
- Simple command-line interface
- Validates input parameters
- Handles container lifecycle (stop, remove, run)
- Provides helpful output and instructions

**Usage:**
```bash
# Run in production mode
./run_experimental.sh production 8000

# Run in development mode
./run_experimental.sh development 8000
```

### 5. test_experimental_docker.sh
A test script to verify that the experimental Dockerfile works correctly in both modes.

**Features:**
- Builds and tests production mode
- Builds and tests development mode
- Verifies container startup
- Performs basic health checks

**Usage:**
```bash
./test_experimental_docker.sh
```

### 6. DOCKERFILE_EXPERIMENTAL_README.md
Detailed documentation for the experimental Dockerfile.

**Content:**
- Purpose and goals
- Usage instructions
- Environment variables
- Behavior differences between modes
- Migration path
- Troubleshooting guide

## How It Works

1. **Build Process:**
   - The `BUILD_MODE` build argument is passed during the docker build
   - This controls which dependencies are installed and how the container is configured
   - The same Dockerfile produces different results based on the build argument

2. **Runtime Process:**
   - The `BUILD_MODE` environment variable is passed to the container
   - The `entrypoint.sh` script checks this variable and runs the appropriate command
   - Production mode runs with workers and no hot reload
   - Development mode runs with hot reload enabled

3. **Port Configuration:**
   - Port 8000 is always exposed for the main application
   - Port 8888 is also exposed for potential development tools (unused in production)

## Migration Path

To migrate from the current Docker setup to the experimental one:

1. Replace references to `Dockerfile` with `Dockerfile.experimental`
2. Add the `BUILD_MODE` build argument to your docker-compose configuration
3. Set the `BUILD_MODE` environment variable when running containers
4. Update CI/CD pipelines to use the new build arguments

## Benefits

1. **Simplified Maintenance:** Single Dockerfile instead of two
2. **Consistent Base:** Same base image for both environments
3. **Flexible Configuration:** Easy to switch between modes
4. **Reduced Duplication:** Shared logic between production and development
5. **UV via pip:** As requested, installs uv via pip instead of using uv-astral image

## Next Steps

1. Test the experimental Dockerfile thoroughly
2. Verify that all functionality works in both modes
3. Update documentation and scripts as needed
4. Consider replacing the existing Dockerfiles if the experiment is successful
5. Update CI/CD pipelines to use the new Dockerfile

## Notes

- The experimental Dockerfile is designed to be a drop-in replacement for the existing Dockerfiles
- All existing functionality should be preserved
- The main difference is how uv is installed (via pip) and the unified configuration approach
- The experimental setup maintains the security and performance best practices from the original Dockerfiles