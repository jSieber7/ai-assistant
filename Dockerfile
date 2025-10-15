# =============================================================================
# AI Assistant Dockerfile
# =============================================================================
# Multi-stage build for development and production environments

# =============================================================================
# Base Stage
# =============================================================================
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    UV_CACHE_DIR=/app/.uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Set work directory
WORKDIR /app

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy application code first (needed for version import)
COPY app/ ./app/

# Install dependencies with dev tools
RUN python -m venv .venv && .venv/bin/pip install --upgrade pip setuptools wheel && uv sync --frozen --reinstall

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application with reload
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production Stage
# =============================================================================
FROM base as production

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy application code first (needed for version import)
COPY app/ ./app/

# Install dependencies without dev tools
RUN python -m venv .venv && .venv/bin/pip install --upgrade pip setuptools wheel && uv sync --frozen --no-dev --reinstall

# Copy the rest of the application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]