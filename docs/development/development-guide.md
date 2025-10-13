# Development Guide

This guide provides comprehensive instructions for developing with the LLM Tool System Foundation, including creating new tools, testing, and deployment.

## Development Workflow

### Setting Up Development Environment
```bash
# Install development dependencies
uv sync --dev

# Run code quality checks
uv run black .
uv run ruff check .
uv run mypy app/

# Run tests
python run_tests.py --unit
python run_tests.py --integration
```

### Creating New Tools

#### 1. Implement BaseTool Interface
```python
from app.core.tools import BaseTool

class NewTool(BaseTool):
    @property
    def name(self) -> str:
        return "new_tool"
    
    @property
    def description(self) -> str:
        return "Description of the new tool"
    
    async def execute(self, **kwargs) -> Any:
        # Tool implementation
        return {"result": "success"}
```

#### 2. Register the Tool
```python
# In app/core/tools/__init__.py or tool initialization module
from .new_tool import NewTool
tool_registry.register(NewTool())
```

#### 3. Write Tests
```python
@pytest.mark.asyncio
async def test_new_tool():
    tool = NewTool()
    result = await tool.execute(param="value")
    assert result["result"] == "success"
```

### Code Standards
- Use type hints throughout
- Follow Black code formatting
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes

## Testing

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --coverage

# Run with verbose output
python run_tests.py --verbose
```

### Test Structure
```
tests/
├── test_main.py              # Unit tests
├── test_integration.py       # Integration tests
├── test_routes.py            # API route tests
├── test_monitoring.py        # Monitoring tests
└── test_caching/             # Caching system tests
    ├── test_base.py
    ├── test_caching_system.py
    └── test_memory_cache.py
```

### Example Test
```python
@pytest.mark.asyncio
async def test_tool_execution():
    from app.core.tools.examples import CalculatorTool
    
    tool = CalculatorTool()
    result = await tool.execute(expression="10 + 15")
    assert result == 25
```

## Deployment

### Production Deployment
```bash
# Install production dependencies
uv sync

# Start production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN uv sync --dev

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration
```bash
# Production environment
ENVIRONMENT=production
DEBUG=false
RELOAD=false

# Enable Redis for production caching
REDIS_CACHE_ENABLED=true
REDIS_URL=redis://redis-server:6379

# Monitoring and logging
MONITORING_ENABLED=true
LOG_LEVEL=INFO
```

## Examples & Best Practices

### Tool Integration Example
```python
# Complete tool integration example
from app.core.tools import BaseTool, tool_registry
from app.core.agents import ToolAgent

class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information"
    
    async def execute(self, query: str, max_results: int = 5) -> List[Dict]:
        # Implementation using SearX or other search engine
        return search_results

# Register and use
tool_registry.register(WebSearchTool())
agent = ToolAgent(llm, tool_registry)
response = await agent.process_message("Current news about AI")
```

### Caching Best Practices
```python
# Use caching for expensive operations
async def get_user_data(user_id: str):
    cache_key = f"user:{user_id}:profile"
    cached_data = await cache_get(cache_key)
    
    if cached_data:
        return cached_data
    
    # Expensive operation
    user_data = await database.get_user(user_id)
    await cache_set(cache_key, user_data, ttl=3600)
    return user_data
```

### Error Handling
```python
try:
    result = await tool.execute_with_timeout(**parameters)
    if result.success:
        return result.data
    else:
        logger.error(f"Tool failed: {result.error}")
        return fallback_response
except ToolTimeoutError:
    logger.warning("Tool execution timed out")
    return timeout_response
```

## Monitoring & Observability

### Key Metrics
- Tool execution times and success rates
- Cache hit/miss ratios
- Agent response times
- System resource usage

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

### Logging Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Configuration

### Environment Variables
```bash
# Core Settings
OPENROUTER_API_KEY=your_openrouter_api_key
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
HOST=127.0.0.1
PORT=8000

# Tool System
TOOL_CALLING_ENABLED=true
MAX_TOOLS_PER_QUERY=3
TOOL_TIMEOUT_SECONDS=30

# Caching
MEMORY_CACHE_ENABLED=true
REDIS_CACHE_ENABLED=false
CACHE_COMPRESSION_ENABLED=true

# Monitoring
MONITORING_ENABLED=true
METRICS_COLLECTION_ENABLED=true
```

### Tool-Specific Configuration
```python
from app.core.tools.config import tool_settings

# Access tool configuration
print(f"Calculator enabled: {tool_settings.calculator_tool_enabled}")
print(f"Default timeout: {tool_settings.tool_timeout_seconds}")
```

## Related Documentation

- [Development Setup](setup.md)
- [Contributing Guidelines](contributing.md)
- [API Endpoints](../api/endpoints.md)
- [Architecture Overview](../architecture/overview.md)