# LLM Tool System Foundation

A comprehensive, extensible tool system foundation for AI assistants built with FastAPI and LangChain. This project provides a robust framework for integrating tool-calling capabilities into LLM applications, featuring advanced caching, monitoring, and agent orchestration.

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- UV package manager
- OpenRouter API key (for cloud models)
- Ollama server (optional, for local models)

### Installation
```bash
# Clone the repository
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant

# Set up environment
cp .env.template .env
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --dev

# Configure your OpenRouter API key in .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

### Running the System
```bash
# Start the development server
uvicorn app.main:app --reload

# Access the API documentation
# http://localhost:8000/docs
```

### Basic Usage
```python
import httpx

# Test the API with OpenRouter (cloud)
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What's 15 * 25?"}]
    }
)

## 🦙 Ollama Integration

The system now supports local Ollama models alongside cloud-based OpenRouter models, providing flexibility for different use cases:

### Setting Up Ollama

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

3. **Pull Models**:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

4. **Configure in .env**:
   ```bash
   PREFERRED_PROVIDER=ollama
   OLLAMA_ENABLED=true
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_DEFAULT_MODEL=llama2
   ```

### Using Ollama Models

```python
# Use specific Ollama model
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama:llama2",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)

# List available Ollama models
response = httpx.get("http://localhost:8000/v1/providers/ollama/models")
models = response.json()

# Check provider status
response = httpx.get("http://localhost:8000/v1/providers")
providers = response.json()
```

### Provider Features

- **Automatic Fallback**: Falls back to other providers if preferred fails
- **Health Monitoring**: Continuous health checks for all providers
- **Model Discovery**: Automatic detection of available models
- **Mixed Usage**: Use both cloud and local models in the same application

For detailed setup instructions, see [Ollama Integration Guide](docs/ollama-integration.md).

print(response.json())

# Test the API with Ollama (local)
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama:llama2",
        "messages": [{"role": "user", "content": "What's 15 * 25?"}]
    }
)
print(response.json())

# List available models
response = httpx.get("http://localhost:8000/v1/models")
print(response.json())
```

## 📋 System Overview

The LLM Tool System Foundation is built on a modular architecture that enables sophisticated tool-calling capabilities for AI assistants:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │◄──►│   Tool Agent     │◄──►│   Tool Registry │
│ (OpenAI-compat) │    │  (Orchestrator)  │    │  (Management)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Caching Layer │    │   Monitoring     │    │   Tool Storage │
│  (Multi-layer)  │    │   & Metrics      │    │   & Discovery   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features
- **🔧 Extensible Tool System**: Dynamic tool registration and discovery
- **🤖 Intelligent Agent Orchestration**: Context-aware tool selection and execution
- **⚡ Advanced Caching**: Multi-layer caching with compression and batching
- **📊 Comprehensive Monitoring**: Real-time metrics and health checks
- **🔒 Security-First Design**: Input validation and access control
- **🔄 LangChain Integration**: Seamless compatibility with LangChain ecosystem

## 🏗️ Architecture

### Core Components

#### Tool System ([`app/core/tools/`](app/core/tools/))
- **[`BaseTool`](app/core/tools/base.py:57)**: Abstract base class for all tools
- **[`ToolRegistry`](app/core/tools/registry.py:15)**: Central tool management and discovery
- **[`ToolResult`](app/core/tools/base.py:27)**: Standardized tool execution results
- **Example Tools**: Calculator, Time, Echo tools for testing

#### Agent System ([`app/core/agents/`](app/core/agents/))
- **[`ToolAgent`](app/core/agents/tool_agent.py)**: Orchestrates LLM-tool interactions
- **[`AgentRegistry`](app/core/agents/registry.py)**: Manages agent instances and strategies
- **[`LangChainIntegration`](app/core/agents/langchain_integration.py)**: LangChain compatibility layer

#### Caching System ([`app/core/caching/`](app/core/caching/))
- **[`CachingSystem`](app/core/caching/base.py)**: Multi-layer caching architecture
- **Memory Cache**: Fast in-memory caching ([`MemoryCache`](app/core/caching/layers/memory.py))
- **Redis Cache**: Distributed caching support ([`RedisCache`](app/core/caching/layers/redis_cache.py))
- **Compression**: Data compression for large responses ([`Compressor`](app/core/caching/compression/compressor.py))
- **Batching**: Batch processing for efficiency ([`BatchProcessor`](app/core/caching/batching/batch_processor.py))

#### Monitoring System ([`app/core/monitoring/`](app/core/monitoring/))
- **[`MetricsCollector`](app/core/monitoring/metrics.py)**: Performance metrics collection
- **[`HealthMonitor`](app/core/monitoring/health.py)**: System health monitoring
- **[`MonitoringMiddleware`](app/core/monitoring/middleware.py)**: Request/response monitoring

### API Layer ([`app/api/`](app/api/))
- **[`Tool Routes`](app/api/tool_routes.py)**: Tool management and execution endpoints
- **[`Agent Routes`](app/api/agent_routes.py)**: Agent orchestration endpoints
- **[`Monitoring Routes`](app/api/monitoring_routes.py)**: Metrics and health endpoints
- **OpenAI-Compatible API**: Standard chat completions endpoint

## 🔧 Core Components

### Tool System

#### Creating Custom Tools
```python
from app.core.tools import BaseTool, ToolResult

class CustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "custom_tool"
    
    @property
    def description(self) -> str:
        return "A custom tool example"
    
    @property
    def keywords(self) -> List[str]:
        return ["custom", "example", "demo"]
    
    async def execute(self, input_data: str) -> str:
        return f"Processed: {input_data}"

# Register the tool
from app.core.tools.registry import tool_registry
tool_registry.register(CustomTool())
```

#### Tool Registry Operations
```python
# Get all available tools
tools = tool_registry.list_tools()

# Find relevant tools for a query
relevant_tools = tool_registry.find_relevant_tools("What's the time in New York?")

# Enable/disable tools dynamically
tool_registry.enable_tool("calculator")
tool_registry.disable_tool("web_search")
```

### Agent System

#### Using the Tool Agent
```python
from app.core.agents.tool_agent import ToolAgent
from app.core.tools.registry import tool_registry

# Create agent instance
agent = ToolAgent(llm_instance, tool_registry)

# Process message with tool calling
response = await agent.process_message(
    "What's the current time in London and calculate 15% of 200?",
    context={"user_id": "123"}
)
```

### Caching System

#### Using the Caching Layer
```python
from app.core.caching import CachingSystem, cache_set, cache_get

# Initialize caching system
caching_system = CachingSystem({
    'memory_cache_enabled': True,
    'redis_cache_enabled': False,
    'compression_enabled': True
})

await caching_system.initialize()

# Set and get cached values
await caching_system.set('user:123:profile', user_data, ttl=3600)
cached_data = await caching_system.get('user:123:profile')

# Convenience functions
await cache_set('query:456:result', result_data, ttl=300)
result = await cache_get('query:456:result')
```

## 🌐 API Reference

### Core Endpoints

#### Chat Completions (OpenAI-Compatible)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "anthropic/claude-3.5-sonnet",
  "messages": [
    {"role": "user", "content": "What's 15 * 25?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations"
      }
    }
  ]
}
```

#### Tool Management
```http
GET /api/v1/tools
# List all registered tools

POST /api/v1/tools/execute
# Execute a specific tool

GET /api/v1/tools/{tool_name}/stats
# Get tool usage statistics
```

#### Monitoring Endpoints
```http
GET /health
# System health check

GET /metrics
# Performance metrics

GET /api/v1/monitoring/stats
# Detailed system statistics
```

### Complete API Documentation
See the full API documentation at `http://localhost:8000/docs` when the server is running.

## ⚙️ Configuration

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

## 🛠️ Development Guide

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

1. **Implement BaseTool Interface**
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

2. **Register the Tool**
```python
# In app/core/tools/__init__.py or tool initialization module
from .new_tool import NewTool
tool_registry.register(NewTool())
```

3. **Write Tests**
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

## 🧪 Testing

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

## 🚀 Deployment

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

## 📚 Examples & Best Practices

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

## 🔍 Monitoring & Observability

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

### Code Review Guidelines
- Ensure all tests pass
- Follow code style guidelines
- Update documentation as needed
- Include appropriate test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the excellent LLM orchestration framework
- **FastAPI** for the high-performance API framework
- **OpenRouter** for LLM API access
- **UV** for fast Python package management

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/jSieber7/ai_assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jSieber7/ai_assistant/discussions)

---

**Note**: This is a foundational system designed for extensibility. Refer to the specific component documentation for detailed implementation guides:

- [Tool System Design](docs/architecture/tools.md)
- [Agent Workflow](docs/architecture/workflow.md)
- [API Endpoints](docs/api/endpoints.md)
- [Development Setup](docs/development/setup.md)
