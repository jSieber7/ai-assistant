# =============================================================================
# AI Assistant Production Dockerfile
# =============================================================================
# Multi-stage build optimized for production with security and performance best practices

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

# Install uv
RUN pip install --no-cache-dir uv==0.9.3

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy application code first (needed for version import)
COPY app/ ./app/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv && \
    uv sync --frozen --no-dev --no-cache

# Install Playwright browsers
RUN /opt/venv/bin/playwright install chromium firefox webkit

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

# Install runtime dependencies including Selenium WebDriver and Playwright
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Chrome for Selenium WebDriver
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install ChromeDriver
RUN CHROMEDRIVER_VERSION=$(curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE) \
    && wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip \
    && unzip /tmp/chromedriver.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm /tmp/chromedriver.zip

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
COPY --chown=app:app utility/ ./utility/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/.uv-cache && \
    chown -R app:app /app && \
    chmod +x /app/utility/startup_dev.py

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check with proper timeout and retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with production settings
CMD ["sh", "-c", "python utility/startup_dev.py && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"]