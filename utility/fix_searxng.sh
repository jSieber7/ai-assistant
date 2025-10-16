#!/bin/bash
# Script to restart services with new SearXNG configuration and test connectivity

echo "🔧 Restarting services with new SearXNG configuration..."
echo "=================================================="

# Stop existing containers
echo "Stopping existing containers..."
docker compose down

# Start services with new configuration
echo "Starting services with new configuration..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check if SearXNG container is running
echo "Checking SearXNG container status..."
docker ps | grep searxng

# Run the test script
echo ""
echo "🧪 Running SearXNG connectivity test..."
echo "======================================"
docker compose exec -T ai-assistant python utility/test_searxng_connection.py

echo ""
echo "✅ Fix complete! If the test passed, your SearXNG should now be working."
echo "If the test failed, check the error messages above for troubleshooting."