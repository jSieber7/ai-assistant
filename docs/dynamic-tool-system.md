# Dynamic Tool System

This document describes the dynamic tool system improvements made to the AI Assistant project.

## Overview

The dynamic tool system provides a more flexible and extensible way to execute tools based on context and requirements, rather than using hardcoded tool references in agents.

## Key Components

### 1. Dynamic Tool Executor (`app/core/tools/dynamic_executor.py`)

The `DynamicToolExecutor` is the central component that:

- Analyzes task requirements and context
- Selects appropriate tools from the registry
- Executes tools with proper parameter mapping
- Handles errors gracefully
- Tracks execution history and statistics

#### Key Features:

- **Task-based execution**: Tools are selected based on task types (SEARCH, SCRAPE, RERANK, etc.)
- **Context-aware selection**: Tools are chosen based on keywords, categories, and context
- **Parameter mapping**: Automatically maps request parameters to tool-specific parameters
- **Error handling**: Graceful handling of tool failures with detailed error reporting
- **Execution tracking**: Maintains history and statistics for monitoring and debugging

#### Usage Example:

```python
from app.core.tools.dynamic_executor import DynamicToolExecutor, TaskRequest, TaskType

# Create executor
executor = DynamicToolExecutor(tool_registry)

# Create task request
request = TaskRequest(
    task_type=TaskType.SEARCH,
    query="Find information about Python",
    context={"results_count": 10},
    max_tools=3
)

# Execute task
result = await executor.execute_task(request)

if result.success:
    print(f"Task completed with {len(result.tool_results)} tools")
    print(f"Result: {result.data}")
else:
    print(f"Task failed: {result.error}")
```

### 2. Standalone Scraper (`app/core/tools/standalone_scraper.py`)

The `StandaloneScraper` provides a simple interface for web scraping that can be used independently from any agent or workflow.

#### Key Features:

- **Independent operation**: Can be used without any agent setup
- **Multiple scraping modes**: Single URL, batch scraping, content extraction
- **Flexible configuration**: Support for various formats and options
- **Error handling**: Comprehensive error handling and reporting
- **Convenience functions**: Quick scrape functionality for simple use cases

#### Usage Example:

```python
from app.core.tools.standalone_scraper import StandaloneScraper, quick_scrape

# Using the class
scraper = StandaloneScraper()
await scraper.initialize()

# Scrape a single URL
result = await scraper.scrape_url("https://example.com")
if result.success:
    print(f"Title: {result.data['title']}")
    print(f"Content: {result.data['content'][:100]}...")

# Extract only content
content = await scraper.extract_content_only("https://example.com")
if content:
    print(f"Content: {content[:100]}...")

# Quick scrape convenience function
data = await quick_scrape("https://example.com")
if data:
    print(f"Quick result: {data['title']}")

await scraper.cleanup()
```

### 3. Refactored Deep Search Agent

The `DeepSearchAgent` has been refactored to use the dynamic tool executor instead of hardcoded tool references.

#### Changes Made:

- Replaced direct tool references with dynamic executor
- Modified search and scrape workflow to use task-based execution
- Improved error handling and flexibility
- Maintained backward compatibility with existing interfaces

## Benefits

### 1. Improved Flexibility

- Tools can be added or modified without changing agent code
- Dynamic tool selection based on context and requirements
- Support for multiple tools with similar functionality

### 2. Better Error Handling

- Graceful degradation when tools fail
- Detailed error reporting and logging
- Fallback mechanisms for critical operations

### 3. Enhanced Testability

- Tools can be easily mocked for testing
- Independent testing of tool selection logic
- Comprehensive test coverage for all components

### 4. Monitoring and Debugging

- Execution history tracking
- Performance statistics
- Tool usage analytics

## Migration Guide

### For Existing Code

If you're using the old tool system, here's how to migrate:

#### Old Way:

```python
# Direct tool usage
search_tool = tool_registry.get_tool("searxng_search")
result = await search_tool.execute(query="test")
```

#### New Way:

```python
# Dynamic tool execution
executor = DynamicToolExecutor(tool_registry)
request = TaskRequest(
    task_type=TaskType.SEARCH,
    query="test",
    context={}
)
result = await executor.execute_task(request)
```

### For New Development

When creating new agents or workflows:

1. Use the `DynamicToolExecutor` for tool execution
2. Define task types and contexts appropriately
3. Handle both successful and failed executions
4. Use the `StandaloneScraper` for independent scraping needs

## Testing

The dynamic tool system includes comprehensive tests:

- Unit tests for `DynamicToolExecutor`
- Unit tests for `StandaloneScraper`
- Integration tests for the refactored `DeepSearchAgent`
- Mock tools for testing various scenarios

### Running Tests:

```bash
# Run dynamic executor tests
uv run pytest tests/unit/test_dynamic_executor.py -v

# Run standalone scraper tests
uv run pytest tests/unit/test_standalone_scraper.py -v

# Run all tool-related tests
uv run pytest tests/unit/test_*tool*.py -v
```

## Configuration

The dynamic tool system uses the existing configuration system. No additional configuration is required.

## Performance Considerations

- Tool selection is optimized for performance with caching where appropriate
- Execution history is limited to prevent memory leaks
- Concurrent tool execution is supported where applicable

## Future Enhancements

Potential future improvements include:

1. **Tool chaining**: Automatic chaining of tools for complex workflows
2. **Tool priority system**: Prioritization of tools based on performance or quality
3. **Dynamic tool loading**: Runtime loading of new tools without restart
4. **Tool versioning**: Support for multiple versions of the same tool
5. **AI-powered tool selection**: Using ML models for optimal tool selection

## Troubleshooting

### Common Issues

1. **No suitable tools found**: Check tool keywords and categories
2. **Parameter validation failures**: Ensure required parameters are provided
3. **Tool execution timeouts**: Adjust tool timeout settings
4. **Memory usage**: Monitor execution history size

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("app.core.tools.dynamic_executor").setLevel(logging.DEBUG)
```

## API Reference

### DynamicToolExecutor

#### Methods:

- `execute_task(request: TaskRequest) -> TaskResult`: Execute a task
- `get_execution_history() -> List[TaskResult]`: Get execution history
- `clear_execution_history()`: Clear execution history
- `get_stats() -> Dict[str, Any]`: Get execution statistics

### TaskRequest

#### Fields:

- `task_type: TaskType`: Type of task to execute
- `query: str`: Query or description of the task
- `context: Dict[str, Any]`: Additional context information
- `parameters: Optional[Dict[str, Any]]`: Tool parameters
- `required_tools: Optional[List[str]]`: Specific tools to use
- `excluded_tools: Optional[List[str]]`: Tools to exclude
- `max_tools: int`: Maximum number of tools to use

### TaskResult

#### Fields:

- `success: bool`: Whether the task was successful
- `data: Any`: Result data
- `tool_results: List[ToolResult]`: Individual tool results
- `execution_time: float`: Total execution time
- `task_type: TaskType`: Type of task that was executed
- `error: Optional[str]`: Error message if failed
- `metadata: Dict[str, Any]`: Additional metadata

### StandaloneScraper

#### Methods:

- `initialize() -> bool`: Initialize the scraper
- `scrape_url(url, **kwargs) -> ToolResult`: Scrape a single URL
- `scrape_multiple_urls(urls, **kwargs) -> List[ToolResult]`: Scrape multiple URLs
- `extract_content_only(url, min_length=100) -> Optional[str]`: Extract only content
- `extract_links_from_url(url) -> List[str]`: Extract links
- `extract_images_from_url(url) -> List[Dict]`: Extract images
- `get_page_metadata(url) -> Dict[str, Any]`: Get page metadata
- `cleanup()`: Clean up resources

## Conclusion

The dynamic tool system provides a more flexible, maintainable, and extensible way to execute tools in the AI Assistant project. It improves upon the previous hardcoded approach by adding context-aware selection, better error handling, and comprehensive monitoring capabilities.