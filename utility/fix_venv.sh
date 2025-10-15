#!/bin/bash

# Script to fix the virtual environment issue

# Kill any running uvicorn processes that might be using the old .venv
echo "Attempting to stop any running uvicorn processes..."
pkill -f uvicorn 2>/dev/null || true

# Wait a moment for processes to stop
sleep 2

# Try to remove the old .venv directory
echo "Attempting to remove the old .venv directory..."
if rm -rf .venv 2>/dev/null; then
    echo "Successfully removed old .venv directory"
else
    echo "Could not remove .venv directory, it might be in use"
    echo "You may need to manually stop any processes using it"
fi

# Rename the working virtual environment
if [ -d ".venv_new" ]; then
    echo "Renaming .venv_new to .venv..."
    mv .venv_new .venv
    echo "Virtual environment fixed!"
    echo "You can now use 'uv run' again"
else
    echo "Error: .venv_new directory not found"
    exit 1
fi