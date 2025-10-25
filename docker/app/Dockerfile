# =============================================================================
# AI Assistant Experimental Dockerfile
# =============================================================================
# Single Dockerfile with environment variable controls for production/development modes
# Installs uv via pip instead of using uv-astral image

# =============================================================================
# Builder Stage
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

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

# Set uv environment variables to create venv at /opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Create virtual environment and install dependencies
# Environment variable controls whether to install dev dependencies
ADD . /app

ARG BUILD_MODE=production
ENV BUILD_MODE=${BUILD_MODE}
RUN if [ "$BUILD_MODE" = "development" ]; then \
        uv sync --frozen --group dev --no-cache; \
    else \
        uv sync --frozen --no-dev --no-cache; \
    fi

# Install Playwright browsers
# RUN uv run -m playwright install chromium firefox webkit
# RUN uv run playwright install-deps

# =============================================================================
# =============================================================================
# Production Stage
# =============================================================================
FROM ubuntu:noble AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    UV_CACHE_DIR=/tmp/.uv-cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install basic runtime dependencies and Python
RUN apt-get update -qy && apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    curl \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    python3.12 \
    libpython3.12 \
    libpcre3 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy uv from uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# Install Chrome and ChromeDriver only in production mode
ARG BUILD_MODE=production

# Create non-root user with proper permissions
RUN groupadd -r app && \
    useradd -r -g app --home-dir /app --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage with proper ownership
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# Fix python symlink to point to system Python
RUN rm /opt/venv/bin/python && \
    rm /opt/venv/bin/python3 && \
    rm /opt/venv/bin/python3.12 && \
    ln -s /usr/bin/python3.12 /opt/venv/bin/python && \
    ln -s /usr/bin/python3.12 /opt/venv/bin/python3 && \
    ln -s /usr/bin/python3.12 /opt/venv/bin/python3.12 && \
    chown app:app /opt/venv/bin/python /opt/venv/bin/python3 /opt/venv/bin/python3.12

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