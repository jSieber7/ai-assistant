#!/bin/bash

# =============================================================================
# Experimental Docker Run Script
# =============================================================================
# This script provides an easy way to run the experimental Dockerfile
# in either production or development mode

set -e

# Default mode
MODE=${1:-production}
PORT=${2:-8000}

# Function to display usage
usage() {
    echo "Usage: $0 [production|development] [port]"
    echo ""
    echo "Examples:"
    echo "  $0 production 8000    # Run in production mode on port 8000"
    echo "  $0 development 8000   # Run in development mode on port 8000"
    echo "  $0 dev 8000           # Short form for development"
    echo ""
    echo "Default: production mode on port 8000"
    exit 1
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Normalize mode argument
case "$MODE" in
    prod|production)
        MODE="production"
        ;;
    dev|development)
        MODE="development"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Use 'production' or 'development'."
        usage
        ;;
esac

# Validate port
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: Invalid port '$PORT'. Port must be between 1 and 65535."
    exit 1
fi

echo "======================================"
echo "AI Assistant - Experimental Docker"
echo "======================================"
echo "Mode: $MODE"
echo "Port: $PORT"
echo "======================================"

# Build the Docker image
echo "Building Docker image in $MODE mode..."
docker build -f Dockerfile.experimental --build-arg BUILD_MODE=$MODE -t ai-assistant:$MODE .

# Stop and remove any existing container
echo "Stopping any existing container..."
docker stop ai-assistant-$MODE 2>/dev/null || true
docker rm ai-assistant-$MODE 2>/dev/null || true

# Run the new container
echo "Starting container in $MODE mode..."
docker run -d \
    --name ai-assistant-$MODE \
    -e BUILD_MODE=$MODE \
    -p $PORT:8000 \
    -p 8888:8888 \
    ai-assistant:$MODE

echo "Container started successfully!"
echo ""
echo "Application is running at: http://localhost:$PORT"
echo ""
echo "To view logs: docker logs -f ai-assistant-$MODE"
echo "To stop: docker stop ai-assistant-$MODE"
echo ""
if [ "$MODE" = "development" ]; then
    echo "Development mode features:"
    echo "- Hot reload enabled"
    echo "- Source code mounted (if using docker-compose)"
    echo "- Additional port 8888 available"
else
    echo "Production mode features:"
    echo "- Optimized for production"
    echo "- Health checks enabled"
    echo "- Single worker process"
fi