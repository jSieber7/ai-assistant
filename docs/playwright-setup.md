# Playwright Setup and Usage

This document explains how to set up and use Playwright for direct browser automation in the AI Assistant project.

## Overview

Playwright has been integrated as a direct dependency, giving you full control over browser automation without relying on external services like Firecrawl. This allows for more flexible and powerful web scraping and automation tasks.

## Installation

Playwright is now included as a dependency in `pyproject.toml`. The Docker images have been updated with all necessary system dependencies and browser binaries.

## Configuration

Add the following to your `.env` file to enable Playwright:

```env
# Enable Playwright system
PLAYWRIGHT_ENABLED=true

# Browser settings
PLAYWRIGHT_HEADLESS=true
PLAYWRIGHT_BROWSER_TYPE=chromium
PLAYWRIGHT_TIMEOUT=30

# Viewport settings
PLAYWRIGHT_VIEWPORT_WIDTH=1280
PLAYWRIGHT_VIEWPORT_HEIGHT=720

# User agent
PLAYWRIGHT_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

## Available Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `PLAYWRIGHT_ENABLED` | `false` | Enable/disable Playwright system |
| `PLAYWRIGHT_HEADLESS` | `true` | Run browser in headless mode |
| `PLAYWRIGHT_BROWSER_TYPE` | `chromium` | Browser to use (chromium, firefox, webkit) |
| `PLAYWRIGHT_TIMEOUT` | `30` | Default timeout in seconds |
| `PLAYWRIGHT_VIEWPORT_WIDTH` | `1280` | Browser viewport width |
| `PLAYWRIGHT_VIEWPORT_HEIGHT` | `720` | Browser viewport height |
| `PLAYWRIGHT_USER_AGENT` | Default | Custom user agent string |

## Usage Examples

### Direct Tool Usage

```python
from app.core.tools.playwright_tool import PlaywrightTool

# Create tool instance
tool = PlaywrightTool(headless=True, browser_type="chromium")

# Simple page scraping
result = await tool.execute(
    url="https://example.com",
    extract_text=True,
    wait_for=3000
)

print(f"Title: {result['title']}")
print(f"Content: {result['content']}")
```

### With Actions

```python
# Define actions to perform
actions = [
    {"type": "wait", "duration": 1000},
    {"type": "scroll", "direction": "down", "distance": 500},
    {"type": "click", "selector": "button.submit"},
    {"type": "wait_for_selector", "selector": ".result", "timeout": 10000},
]

# Execute with actions
result = await tool.execute(
    url="https://example.com",
    actions=actions,
    screenshot=True
)
```

### Available Actions

| Action Type | Parameters | Description |
|-------------|------------|-------------|
| `click` | `selector` | Click on an element |
| `type` | `selector`, `text` | Type text into an input field |
| `wait` | `duration` | Wait for specified milliseconds |
| `wait_for_selector` | `selector`, `timeout` | Wait for element to appear |
| `scroll` | `direction`, `distance` | Scroll the page |
| `screenshot` | `path` (optional) | Take a screenshot |

### Through Tool Registry

```python
from app.core.tools.registry import tool_registry

# Get the Playwright tool
playwright_tool = tool_registry.get_tool("playwright_automation")

# Use it
result = await playwright_tool.execute(
    url="https://example.com",
    extract_text=True
)
```

### Through Agent System

```python
from app.core.agents.registry import agent_registry

# Get default agent
agent = agent_registry.get_default_agent()

# Send a message that will trigger Playwright usage
result = await agent.process_message(
    "Navigate to https://example.com and extract the main heading text"
)
```

## Running the Example

A complete example is provided in `examples/playwright_example.py`. To run it:

```bash
# For local development
python examples/playwright_example.py

# In Docker
docker compose --profile dev run --rm ai-assistant-dev python examples/playwright_example.py
```

## Docker Setup

The Docker images have been pre-configured with:

1. All required system dependencies for Playwright
2. Browser binaries (Chromium, Firefox, WebKit)
3. Proper display server setup for headless operation

### Building with Playwright Support

```bash
# Build the images
docker compose build

# Or just build the development image
docker compose -f docker-compose.yml build ai-assistant-dev
```

## Comparison with Firecrawl

| Feature | Playwright Tool | Firecrawl Tool |
|---------|----------------|----------------|
| Direct browser control | ✅ | ❌ |
| Complex interactions | ✅ | ✅ |
| JavaScript execution | ✅ | ✅ |
| Screenshots | ✅ | ✅ |
| Custom selectors | ✅ | ✅ |
| No external dependencies | ✅ | ❌ |
| Simplicity | ❌ | ✅ |
| Built-in error handling | ✅ | ✅ |

## Best Practices

1. **Use headless mode** for production deployments
2. **Set appropriate timeouts** to avoid hanging
3. **Clean up resources** by calling `await tool.cleanup()`
4. **Handle errors gracefully** with try/except blocks
5. **Use specific selectors** for reliable element targeting
6. **Add waits** for dynamic content to load

## Troubleshooting

### Browser Not Found

If you get browser not found errors:

1. Ensure Playwright browsers are installed:
   ```bash
   uv run playwright install
   ```

2. For Docker, rebuild the image to ensure browsers are installed.

### Timeout Issues

Increase the timeout setting:
```env
PLAYWRIGHT_TIMEOUT=60
```

### Memory Issues

1. Clean up browser resources after use:
   ```python
   await tool.cleanup()
   ```

2. Consider using smaller viewports for memory-constrained environments.

## Advanced Usage

### Custom Browser Context

You can extend the PlaywrightTool to create custom browser contexts with specific settings:

```python
class CustomPlaywrightTool(PlaywrightTool):
    async def _ensure_browser(self):
        await super()._ensure_browser()
        
        # Create custom context with additional settings
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Custom User Agent",
            ignore_https_errors=True,
            # Add more context options as needed
        )
```

### Concurrent Execution

For concurrent page operations, create separate browser instances or use multiple contexts:

```python
tools = [PlaywrightTool() for _ in range(5)]
tasks = [tool.execute(url=url) for tool, url in zip(tools, urls)]
results = await asyncio.gather(*tasks)

# Clean up all tools
for tool in tools:
    await tool.cleanup()