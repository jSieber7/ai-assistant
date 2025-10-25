#!/bin/bash
set -e

echo "🐳 Testing Docker builds..."

# Function to test a build mode
test_build() {
    local mode=$1
    echo ""
    echo "🔧 Testing $mode build..."
    
    # Build and start
    echo "Building $mode container..."
    BUILD_MODE=$mode docker-compose up --build -d
    
    # Wait for container to be ready
    echo "Waiting for container to be ready..."
    sleep 30
    
    # Check if container is running
    if docker ps | grep -q "my-stack-app"; then
        echo "✅ Container is running"
    else
        echo "❌ Container failed to start"
        BUILD_MODE=$mode docker-compose logs app
        exit 1
    fi
    
    # Check health endpoint
    echo "Checking health endpoint..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        BUILD_MODE=$mode docker-compose logs app
        exit 1
    fi
    
    # Check environment
    echo "Checking environment variables..."
    if BUILD_MODE=$mode docker-compose exec app env | grep -q "BUILD_MODE=$mode"; then
        echo "✅ Environment is correct"
    else
        echo "❌ Environment is incorrect"
        exit 1
    fi
    
    # Check Python version
    echo "Checking Python version..."
    if BUILD_MODE=$mode docker-compose exec app python --version | grep -q "3.12"; then
        echo "✅ Python version is correct"
    else
        echo "❌ Python version is incorrect"
        exit 1
    fi
    
    # Mode-specific tests
    if [ "$mode" = "dev" ]; then
        echo "Checking development dependencies..."
        if BUILD_MODE=$mode docker-compose exec app pip list | grep -q "pytest"; then
            echo "✅ Development dependencies present"
        else
            echo "❌ Development dependencies missing"
            exit 1
        fi
    elif [ "$mode" = "prod" ]; then
        echo "Checking production dependencies..."
        if ! BUILD_MODE=$mode docker-compose exec app pip list | grep -q "pytest"; then
            echo "✅ Development dependencies correctly excluded"
        else
            echo "❌ Development dependencies present in production"
            exit 1
        fi
    fi
    
    # Stop container
    echo "Stopping $mode container..."
    BUILD_MODE=$mode docker-compose down
    
    echo "✅ $mode build test passed!"
}

# Test both builds
test_build "dev"
test_build "prod"

echo ""
echo "🎉 All tests passed! Both dev and prod builds are working correctly."