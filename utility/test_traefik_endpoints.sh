#!/bin/bash

# =============================================================================
# Traefik Endpoints Test Script
# =============================================================================
# This script tests all Traefik-routed endpoints to verify they're working

echo "Testing Traefik endpoints..."
echo "=================================="

# Function to test an endpoint
test_endpoint() {
    local url="$1"
    local name="$2"
    local expected_code="${3:-200}"
    
    echo -n "Testing $name: $url ... "
    
    # Use curl with timeout and only get status code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
    
    if [ "$status_code" = "$expected_code" ]; then
        echo "✅ OK (Status: $status_code)"
        return 0
    else
        echo "❌ FAILED (Status: $status_code, Expected: $expected_code)"
        return 1
    fi
}

# Track overall success
failed=0

# Test Traefik dashboard
test_endpoint "http://traefik.localhost:8881/dashboard/" "Traefik Dashboard" || failed=1

# Test Frontend
test_endpoint "http://frontend.localhost:8880" "Frontend" || failed=1

# Test Firecrawl API
test_endpoint "http://firecrawl.localhost:8880" "Firecrawl API" || failed=1

# Test SearXNG
test_endpoint "http://searxng.localhost:8880" "SearXNG" || failed=1

# Test Supabase
test_endpoint "http://supabase.localhost:8880" "Supabase" || failed=1

# Test main Traefik proxy
test_endpoint "http://localhost:8880" "Main Traefik Proxy" || failed=1

echo "=================================="

if [ $failed -eq 0 ]; then
    echo "🎉 All endpoints are working correctly!"
    exit 0
else
    echo "💥 Some endpoints failed. Check the logs for more details."
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Make sure Docker containers are running: docker ps"
    echo "2. Check Traefik logs: docker logs my-stack-traefik"
    echo "3. Verify .localhost domains are in /etc/hosts: cat /etc/hosts | grep localhost"
    exit 1
fi