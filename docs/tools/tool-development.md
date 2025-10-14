# Tool Development Guide

This guide explains how to develop custom tools for the AI Assistant System.

## Overview

Tools are modular components that extend the capabilities of the AI assistant. Each tool provides specific functionality that can be invoked by agents.

## Tool Structure

A basic tool consists of:

1. **Tool Class**: Implements the tool logic
2. **Parameters Schema**: Defines the expected input parameters
3. **Metadata**: Describes the tool's purpose and usage

## Creating a Basic Tool

### Step 1: Create the Tool Class

```python
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
from typing import Optional

class MyToolParameters(BaseModel):
    query: str
    limit: Optional[int] = 10

class MyTool(BaseTool):
    name = "my_tool"
    description = "A custom tool that processes queries"
    parameters_schema = MyToolParameters
    
    async def execute(self, parameters: MyToolParameters) -> ToolResult:
        # Tool implementation goes here
        result = f"Processed query: {parameters.query}"
        return ToolResult(
            success=True,
            data=result,
            metadata={"limit": parameters.limit}
        )
```

### Step 2: Register the Tool

```python
from app.core.tools.registry import tool_registry

# Register the tool
tool_registry.register(MyTool)
```

## Advanced Tool Features

### Async Operations

Tools can perform asynchronous operations:

```python
import httpx

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a location"
    
    async def execute(self, parameters: WeatherParameters) -> ToolResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/weather/{parameters.location}"
            )
            weather_data = response.json()
            
        return ToolResult(
            success=True,
            data=weather_data
        )
```

### Error Handling

Implement proper error handling:

```python
class APIError(Exception):
    pass

class APITool(BaseTool):
    async def execute(self, parameters: APIToolParameters) -> ToolResult:
        try:
            # API call
            response = await self._make_api_call(parameters)
            return ToolResult(success=True, data=response)
            
        except APIError as e:
            return ToolResult(
                success=False,
                error=str(e),
                error_code="API_ERROR"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error="Unexpected error occurred",
                error_code="UNKNOWN_ERROR"
            )
```

### Caching

Add caching to improve performance:

```python
from app.core.caching.integration.tool_cache import cached_tool

@cached_tool(ttl=3600)
class CachedSearchTool(BaseTool):
    name = "cached_search"
    description = "Search with caching"
    
    async def execute(self, parameters: SearchParameters) -> ToolResult:
        # This result will be cached for 1 hour
        results = await self._perform_search(parameters.query)
        return ToolResult(success=True, data=results)
```

## Tool Configuration

Tools can be configured using environment variables:

```python
import os
from app.core.config import get_settings

settings = get_settings()

class ConfigurableTool(BaseTool):
    name = "configurable_tool"
    description = "A tool with configuration"
    
    def __init__(self):
        self.api_key = os.getenv("TOOL_API_KEY")
        self.timeout = settings.TOOL_TIMEOUT
    
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        # Use configuration values
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # ...
```

## Testing Tools

Write unit tests for your tools:

```python
import pytest
from app.core.tools.my_tool import MyTool, MyToolParameters

@pytest.mark.asyncio
async def test_my_tool():
    tool = MyTool()
    parameters = MyToolParameters(query="test query")
    
    result = await tool.execute(parameters)
    
    assert result.success is True
    assert "test query" in result.data
```

## Best Practices

1. **Input Validation**: Use Pydantic models for parameter validation
2. **Error Handling**: Return structured error information
3. **Logging**: Add appropriate logging for debugging
4. **Documentation**: Include clear descriptions and examples
5. **Testing**: Write comprehensive unit and integration tests
6. **Caching**: Cache results when appropriate
7. **Timeouts**: Implement timeouts for external API calls

## Tool Examples

### File Operations Tool

```python
import aiofiles
from pathlib import Path

class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read contents of a file"
    
    async def execute(self, parameters: FileParameters) -> ToolResult:
        try:
            async with aiofiles.open(parameters.filepath, 'r') as f:
                content = await f.read()
            
            return ToolResult(success=True, data=content)
            
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="File not found",
                error_code="FILE_NOT_FOUND"
            )
```

### Database Query Tool

```python
import asyncpg

class DatabaseQueryTool(BaseTool):
    name = "db_query"
    description = "Execute database queries"
    
    def __init__(self):
        self.connection_pool = None
    
    async def execute(self, parameters: DBQueryParameters) -> ToolResult:
        if not self.connection_pool:
            self.connection_pool = await asyncpg.create_pool(
                parameters.connection_string
            )
        
        async with self.connection_pool.acquire() as conn:
            result = await conn.fetch(parameters.query)
            
        return ToolResult(
            success=True,
            data=[dict(row) for row in result]
        )
```

## Contributing Tools

To contribute your tools to the project:

1. Create a new file in `app/core/tools/custom/`
2. Implement your tool following the guidelines above
3. Add tests in `tests/unit/tools/`
4. Update the documentation
5. Submit a pull request