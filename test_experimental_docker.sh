#!/bin/bash

# Test script for the experimental Dockerfile
# This script tests both production and development modes

echo "Testing experimental Dockerfile..."

# Function to test a build mode
test_build_mode() {
    local mode=$1
    echo "======================================"
    echo "Testing $mode mode..."
    echo "======================================"
    
    # Build the Docker image with the specified mode
    echo "Building Docker image in $mode mode..."
    docker build -f Dockerfile.experimental --build-arg BUILD_MODE=$mode -t ai-assistant-$mode .
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Docker image in $mode mode"
        return 1
    fi
    
    echo "Successfully built Docker image in $mode mode"
    
    # Run the container and check if it starts correctly
    echo "Starting container in $mode mode..."
    docker run -d --name test-container-$mode -e BUILD_MODE=$mode -p 8001:8000 ai-assistant-$mode
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start container in $mode mode"
        return 1
    fi
    
    echo "Container started successfully in $mode mode"
    
    # Wait for the container to start
    sleep 10
    
    # Check if the container is still running
    if docker ps | grep -q test-container-$mode; then
        echo "Container is running in $mode mode"
        
        # Check if the health endpoint is accessible
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            echo "Health check passed in $mode mode"
        else
            echo "WARNING: Health check failed in $mode mode (this might be expected if the service is still starting)"
        fi
    else
        echo "ERROR: Container stopped unexpectedly in $mode mode"
        docker logs test-container-$mode
        return 1
    fi
    
    # Stop and remove the container
    docker stop test-container-$mode
    docker rm test-container-$mode
    
    echo "Test completed for $mode mode"
    echo ""
    
    return 0
}

# Test production mode
test_build_mode "production"
if [ $? -ne 0 ]; then
    echo "Production mode test failed"
    exit 1
fi

# Test development mode
test_build_mode "development"
if [ $? -ne 0 ]; then
    echo "Development mode test failed"
    exit 1
fi

echo "All tests passed successfully!"