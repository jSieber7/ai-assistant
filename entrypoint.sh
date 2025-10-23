#!/bin/sh
# Check if we're in development mode
if [ "${BUILD_MODE}" = "development" ]; then
    echo "Starting in development mode with hot reload..."
    uv run python utility/startup_dev.py && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Starting in production mode..."
    python utility/startup_dev.py && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
fi