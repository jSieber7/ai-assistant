# Tool System Design

This document describes the tool system architecture for the AI Assistant, including how tools are integrated, managed, and executed by the agent.

## Tool System Overview

The tool system provides a modular way to extend the AI assistant's capabilities beyond its built-in knowledge. Tools are self-contained modules that can be called by the agent to perform specific tasks.

## Architecture

### Tool Interface

All tools implement a standard interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Expected parameters for the tool."""
        return {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    def should_use(self, query: str) -> bool:
        """Determine if this tool should be used for the given query."""
        # Default implementation based on keyword matching
        keywords = getattr(self, 'keywords', [])
        return any(keyword in query.lower() for keyword in keywords)
```

### Tool Registry

Tools are registered in a central registry for discovery and management:

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}
    
    def register(self, tool: BaseTool):
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """List all available tools."""
        return list(self._tools.values())
    
    def find_relevant_tools(self, query: str) -> List[BaseTool]:
        """Find tools relevant to the given query."""
        return [tool for tool in self._tools.values() if tool.should_use(query)]
```

## Built-in Tools

### Web Search Tool (SearX Integration)

**Purpose**: Search the web for current information

**Implementation**:
```python
class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information, news, and real-time data"
    
    @property
    def keywords(self) -> List[str]:
        return ["current", "latest", "news", "search", "find", "look up"]
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "int", "default": 5, "description": "Maximum results to return"}
        }
    
    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Execute web search using SearX."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SEARX_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": "en",
                    "time_range": "day"  # Recent results
                }
            )
            results = response.json()
            return self._format_results(results["results"][:max_results])
```

### RAG Tool (Knowledge Base)

**Purpose**: Search internal documents and knowledge base

**Implementation**:
```python
class RAGTool(BaseTool):
    @property
    def name(self) -> str:
        return "knowledge_search"
    
    @property
    def description(self) -> str:
        return "Search internal documents and knowledge base for specific information"
    
    @property
    def keywords(self) -> List[str]:
        return ["document", "knowledge", "internal", "file", "pdf", "doc"]
    
    async def execute(self, query: str, document_ids: List[str] = None) -> Dict[str, Any]:
        """Search knowledge base using vector similarity."""
        # Convert query to embedding
        embedding = await self._get_embedding(query)
        
        # Search vector database
        results = await self._vector_db.similarity_search(
            embedding, 
            k=5, 
            document_ids=document_ids
        )
        
        return self._format_rag_results(results)
```

### Calculator Tool

**Purpose**: Perform mathematical calculations

**Implementation**:
```python
class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations and conversions"
    
    @property
    def keywords(self) -> List[str]:
        return ["calculate", "math", "equation", "convert", "sum", "multiply"]
    
    async def execute(self, expression: str) -> float:
        """Evaluate mathematical expression."""
        try:
            # Safe evaluation of mathematical expressions
            result = eval(expression, {"__builtins__": {}}, math.__dict__)
            return result
        except Exception as e:
            raise ToolError(f"Failed to evaluate expression: {e}")
```

## Tool Integration

### Agent-Tool Communication

The agent uses a structured format to communicate with tools:

```python
class ToolCall:
    def __init__(self, tool_name: str, parameters: Dict[str, Any]):
        self.tool_name = tool_name
        self.parameters = parameters
    
    async def execute(self, registry: ToolRegistry) -> ToolResult:
        tool = registry.get_tool(self.tool_name)
        if not tool:
            raise ToolError(f"Tool not found: {self.tool_name}")
        
        try:
            result = await tool.execute(**self.parameters)
            return ToolResult(success=True, data=result, tool_name=self.tool_name)
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.tool_name)
```

### Tool Selection Strategy

The agent uses a multi-factor approach to select tools:

```python
class ToolSelector:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    async def select_tools(self, query: str, context: Dict[str, Any]) -> List[ToolCall]:
        # Step 1: Keyword-based matching
        relevant_tools = self.registry.find_relevant_tools(query)
        
        # Step 2: Context-aware filtering
        filtered_tools = await self._filter_by_context(relevant_tools, context)
        
        # Step 3: Priority scoring
        scored_tools = self._score_tools(filtered_tools, query)
        
        # Step 4: Select top tools (limit to avoid overuse)
        selected_tools = scored_tools[:3]  # Max 3 tools per query
        
        return [ToolCall(tool.name, self._extract_parameters(tool, query)) 
                for tool in selected_tools]
```

## Error Handling

### Tool Error Types

```python
class ToolError(Exception):
    """Base class for tool errors."""
    pass

class ToolTimeoutError(ToolError):
    """Tool execution timed out."""
    pass

class ToolConfigurationError(ToolError):
    """Tool is misconfigured."""
    pass

class ToolExecutionError(ToolError):
    """Tool execution failed."""
    pass
```

### Graceful Error Handling

```python
async def execute_tool_with_fallback(tool_call: ToolCall, registry: ToolRegistry) -> ToolResult:
    try:
        return await tool_call.execute(registry)
    except ToolTimeoutError:
        logger.warning(f"Tool {tool_call.tool_name} timed out")
        return ToolResult(
            success=False,
            error="Tool timed out",
            tool_name=tool_call.tool_name,
            fallback_used=True
        )
    except ToolExecutionError as e:
        logger.error(f"Tool {tool_call.tool_name} failed: {e}")
        return ToolResult(
            success=False,
            error=str(e),
            tool_name=tool_call.tool_name
        )
```

## Performance Optimization

### Caching Strategy

```python
class CachedTool(BaseTool):
    """Tool wrapper with caching capabilities."""
    
    def __init__(self, tool: BaseTool, cache_ttl: int = 300):
        self._tool = tool
        self._cache = {}
        self._cache_ttl = cache_ttl
    
    async def execute(self, **kwargs) -> Any:
        # Create cache key from parameters
        cache_key = self._create_cache_key(kwargs)
        
        # Check cache
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            if time.time() - cached_result["timestamp"] < self._cache_ttl:
                return cached_result["data"]
        
        # Execute tool and cache result
        result = await self._tool.execute(**kwargs)
        self._cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
```

### Parallel Execution

```python
async def execute_tools_parallel(tool_calls: List[ToolCall], registry: ToolRegistry) -> List[ToolResult]:
    """Execute multiple tools in parallel."""
    tasks = []
    for tool_call in tool_calls:
        task = asyncio.create_task(
            execute_tool_with_fallback(tool_call, registry)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

## Security Considerations

### Input Validation

```python
def validate_tool_parameters(tool: BaseTool, parameters: Dict[str, Any]) -> bool:
    """Validate tool parameters against expected schema."""
    expected_params = tool.parameters
    
    for param_name, param_schema in expected_params.items():
        if param_name not in parameters:
            if "default" not in param_schema:
                return False
        
        param_value = parameters.get(param_name, param_schema.get("default"))
        
        # Type validation
        expected_type = param_schema.get("type")
        if expected_type and not isinstance(param_value, expected_type):
            return False
    
    return True
```

### Rate Limiting

```python
class RateLimitedTool(BaseTool):
    """Tool wrapper with rate limiting."""
    
    def __init__(self, tool: BaseTool, calls_per_minute: int = 60):
        self._tool = tool
        self._rate_limiter = RateLimiter(calls_per_minute)
    
    async def execute(self, **kwargs) -> Any:
        await self._rate_limiter.acquire()
        return await self._tool.execute(**kwargs)
```

## Monitoring and Metrics

### Tool Usage Tracking

```python
class InstrumentedTool(BaseTool):
    """Tool wrapper with instrumentation."""
    
    def __init__(self, tool: BaseTool):
        self._tool = tool
        self._metrics = ToolMetrics()
    
    async def execute(self, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = await self._tool.execute(**kwargs)
            self._metrics.record_success(
                self._tool.name,
                time.time() - start_time
            )
            return result
        except Exception as e:
            self._metrics.record_failure(self._tool.name, str(e))
            raise
```

### Metrics Collection

```python
class ToolMetrics:
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0
    
    def record_success(self, tool_name: str, duration: float):
        self.success_count += 1
        self.total_time += duration
        # Export to monitoring system
        export_metric(f"tool.{tool_name}.success", 1)
        export_metric(f"tool.{tool_name}.duration", duration)
    
    def record_failure(self, tool_name: str, error: str):
        self.failure_count += 1
        export_metric(f"tool.{tool_name}.failure", 1)
        export_metric(f"tool.{tool_name}.error", error)
```

## Creating Custom Tools

### Tool Development Guide

1. **Implement BaseTool Interface**:
```python
class CustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "custom_tool"
    
    @property
    def description(self) -> str:
        return "Description of what this tool does"
    
    async def execute(self, **kwargs) -> Any:
        # Tool implementation
        return {"result": "success"}
```

2. **Register the Tool**:
```python
# In tool initialization
registry.register(CustomTool())
```

3. **Test the Tool**:
```python
@pytest.mark.asyncio
async def test_custom_tool():
    tool = CustomTool()
    result = await tool.execute(param1="value1")
    assert result["result"] == "success"
```

## Configuration

### Tool Configuration File

```yaml
tools:
  web_search:
    enabled: true
    searx_url: "http://localhost:8080"
    timeout: 30
    max_results: 5
  
  knowledge_search:
    enabled: true
    vector_db_url: "postgresql://localhost:5432/rag_db"
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  
  calculator:
    enabled: true
    precision: 10
```

### Environment-Based Configuration

```python
class ToolConfig:
    def __init__(self):
        self.web_search_enabled = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
        self.searx_url = os.getenv("SEARX_URL")
        self.rag_enabled = os.getenv("RAG_ENABLED", "false").lower() == "true"
        self.vector_db_url = os.getenv("VECTOR_DB_URL")
```

## Related Documentation

- [System Architecture Overview](overview.md)
- [Agent Workflow](workflow.md)
- [API Endpoints Reference](../api/endpoints.md)
- [SearX Integration](../tools/searx.md)
- [RAG System](../tools/rag.md)

This tool system provides a robust, extensible foundation for adding capabilities to the AI assistant while maintaining security, performance, and reliability.