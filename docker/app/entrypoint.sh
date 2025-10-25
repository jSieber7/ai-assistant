#!/bin/bash
# =============================================================================
# FastAPI Entrypoint Script
# =============================================================================
# This script handles both production and development modes based on the BUILD_MODE environment variable.

set -e

echo "Starting FastAPI application in ${BUILD_MODE:-production} mode..."

# Default to production mode if not set
MODE=${BUILD_MODE:-production}

if [ "$MODE" = "development" ]; then
    echo "Running with hot reload for development..."
    exec /opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running for production..."
    exec /opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
fi
