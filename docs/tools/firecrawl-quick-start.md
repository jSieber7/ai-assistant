# Firecrawl Quick Start Guide

This guide provides a quick and intuitive way to get started with Firecrawl in your AI Assistant project.

## What is Firecrawl?

Firecrawl is a powerful web scraping tool that can:
- Extract clean, structured content from web pages
- Handle JavaScript-rendered pages
- Provide content in multiple formats (Markdown, HTML, raw text)
- Extract links and images
- Work with both self-hosted Docker instances and external API

## Deployment Modes

Firecrawl supports two deployment modes:

### 1. API Mode (Default)
- Uses the external Firecrawl API service
- Requires an API key
- Easy to set up
- Usage-based pricing

### 2. Docker Mode
- Uses a self-hosted Firecrawl instance
- No API costs
- More control over data
- Requires Docker setup

## Quick Start

### 1. Choose Your Deployment Mode

Set the deployment mode in your `.env` file:

```bash
# For API mode (default)
FIRECRAWL_DEPLOYMENT_MODE=api
FIRECRAWL_API_KEY=your_api_key_here

# For Docker mode
FIRECRAWL_DEPLOYMENT_MODE=docker
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
```

### 2. Start Firecrawl Services

#### For Docker Mode:
```bash
# Start Firecrawl Docker services
make firecrawl

# Check if services are running
make firecrawl-status

# View logs
make firecrawl-logs
```

#### For API Mode:
No additional services needed. Just ensure your API key is valid.

### 3. Use Firecrawl in Your Application

#### Basic Usage:
```python
from app.core.tools.firecrawl_tool import FirecrawlTool

# Create a tool instance
tool = FirecrawlTool()

# Scrape a URL
result = await tool.execute(url="https://example.com")

# Access the content
print(f"Title: {result['title']}")
print(f"Content: {result['content']}")
```

#### Advanced Options:
```python
# Scrape with custom options
result = await tool.execute(
    url="https://example.com",
    formats=["markdown", "html"],
    wait_for=5000,  # Wait 5 seconds for page load
    screenshot=True,  # Take a screenshot
    extract_images=True,  # Extract images
    extract_links=True  # Extract links
)
```

#### Batch Scraping:
```python
# Scrape multiple URLs
urls = ["https://example1.com", "https://example2.com", "https://example3.com"]
results = await tool.batch_scrape(urls=urls)

for result in results:
    if result.get("success", False):
        print(f"Scraped {result['url']}: {result['title']}")
    else:
        print(f"Failed to scrape {result['url']}: {result['error']}")
```

## Configuration Options

### Basic Configuration:
```bash
# Deployment mode (api or docker)
FIRECRAWL_DEPLOYMENT_MODE=docker

# API mode settings
FIRECRAWL_API_KEY=your_api_key
FIRECRAWL_BASE_URL=https://api.firecrawl.dev

# Docker mode settings
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002

# Fallback settings
FIRECRAWL_ENABLE_FALLBACK=true
FIRECRAWL_FALLBACK_TIMEOUT=10
```

### Scraping Settings:
```bash
# Enable/disable scraping
FIRECRAWL_SCRAPING_ENABLED=true

# Default formats
FIRECRAWL_FORMATS=["markdown", "raw"]

# Default wait time (ms)
FIRECRAWL_WAIT_FOR=2000

# Screenshot settings
FIRECRAWL_SCREENSHOT=false

# Content extraction
FIRECRAWL_EXTRACT_IMAGES=false
FIRECRAWL_EXTRACT_LINKS=true

# Timeout settings
FIRECRAWL_SCRAPE_TIMEOUT=60
FIRECRAWL_MAX_CONCURRENT_SCRAPES=5
```

## Fallback Mechanism

Firecrawl includes a built-in fallback mechanism:

- If Docker mode is enabled but the Docker instance is unhealthy, it will automatically fall back to the external API
- This ensures high availability and reliability
- You can disable fallback by setting `FIRECRAWL_ENABLE_FALLBACK=false`

## Testing

### Run Tests:
```bash
# Run all Firecrawl tests
make test-firecrawl

# Run unit tests only
make test-firecrawl-unit

# Run integration tests
make test-firecrawl-integration

# Run live tests (requires running services)
make test-firecrawl-live
```

### Quick Test:
```bash
# Run a quick test to verify configuration
python utility/test_firecrawl_docker.py
```

## Troubleshooting

### Common Issues:

1. **API Key Not Found**
   - Ensure `FIRECRAWL_API_KEY` is set in your `.env` file
   - Verify the API key is valid

2. **Docker Services Not Running**
   - Run `make firecrawl-status` to check service status
   - Start services with `make firecrawl`

3. **Connection Timeout**
   - Increase `FIRECRAWL_SCRAPE_TIMEOUT` in your configuration
   - Check network connectivity

4. **Fallback Not Working**
   - Ensure `FIRECRAWL_ENABLE_FALLBACK=true`
   - Verify API key is configured for fallback

### Getting Help:

1. Check logs with `make firecrawl-logs`
2. Run the test script: `python utility/test_firecrawl_docker.py`
3. Review the configuration documentation

## Best Practices

1. **Use Docker Mode for:**
   - High-volume scraping
   - Sensitive data
   - Cost control

2. **Use API Mode for:**
   - Quick prototyping
   - Low-volume scraping
   - Simpler setup

3. **Enable Fallback:**
   - For production systems
   - When high availability is required

4. **Monitor Performance:**
   - Use the built-in metrics
   - Monitor error rates
   - Adjust timeout and concurrency settings as needed

## Next Steps

- Explore the [Firecrawl API documentation](https://docs.firecrawl.dev)
- Try the [examples](examples/firecrawl_examples.py)
- Configure advanced [options](docs/tools/firecrawl-scraping.md)
- Set up [monitoring](docs/deployment/monitoring.md)