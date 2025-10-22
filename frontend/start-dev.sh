#!/bin/bash

# Script to start the AI Assistant backend and frontend in development mode

echo "Starting AI Assistant Development Environment..."
echo "=============================================="

# Check if the backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is already running on http://localhost:8000"
else
    echo "❌ Backend is not running on http://localhost:8000"
    echo ""
    echo "Please start the backend first by running:"
    echo "  cd .. && python -m app.main"
    echo ""
    echo "Or using Docker:"
    echo "  docker-compose up"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "🚀 Starting React frontend development server..."
echo "The frontend will be available at: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the development server"
echo ""

# Start the frontend
npm run dev