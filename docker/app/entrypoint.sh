#!/bin/bash
set -e

echo "Starting FastAPI application in ${BUILD_MODE:-production} mode..."

# Default to production mode if not set
MODE=${BUILD_MODE:-production}

# Change to app directory
cd /app

# Run application
if [ "$MODE" = "development" ]; then
    echo "Running with hot reload for development..."
    exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running for production..."
    exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
fi