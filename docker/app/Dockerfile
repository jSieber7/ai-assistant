# =============================================================================
# AI Assistant Experimental Dockerfile
# =============================================================================
# Single Dockerfile with environment variable controls for production/development modes
# Installs uv via pip instead of using uv-astral image

# =============================================================================
# Builder Stage
# =============================================================================
FROM python:3.12.3-slim AS builder

# Set environment variables for builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (as requested)
RUN pip install --no-cache-dir uv==0.9.3

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Copy application code first (needed for version import)
COPY app/ ./app/

# Create virtual environment and install dependencies
# Environment variable controls whether to install dev dependencies
ARG BUILD_MODE=production
ENV BUILD_MODE=${BUILD_MODE}
RUN if [ "$BUILD_MODE" = "development" ]; then \
        uv venv /opt/venv && \
        uv sync --frozen --group dev --no-cache; \
    else \
        uv venv /opt/venv && \
        uv sync --frozen --no-dev --no-cache; \
    fi

# Install Playwright browsers
# RUN uv run -m playwright install chromium firefox webkit
# RUN uv run playwright install-deps

# =============================================================================
# Production Stage
# =============================================================================
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    UV_CACHE_DIR=/tmp/.uv-cache

# Install basic runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv via pip (as requested)
RUN pip install --no-cache-dir uv==0.9.3

# Install Chrome and ChromeDriver only in production mode
ARG BUILD_MODE=production

# Create non-root user with proper permissions
RUN groupadd -r app && \
    useradd -r -g app --home-dir /app --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code with proper ownership
COPY --chown=app:app app/ ./app/
COPY --chown=app:app pyproject.toml ./
# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/.uv-cache && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports (both 8000 and 8888 to cover both scenarios)
EXPOSE 8000 8888

# Health check (will be used in production mode)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy entrypoint script that handles both production and development modes
COPY --chown=app:app docker/app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Run the application with the appropriate command
CMD ["/app/entrypoint.sh"]