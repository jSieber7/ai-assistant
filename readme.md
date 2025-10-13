# LLM Tool System Foundation

A comprehensive, extensible tool system foundation for AI assistants built with FastAPI and LangChain. This project provides a robust framework for integrating tool-calling capabilities into LLM applications, featuring advanced caching, monitoring, and agent orchestration.

## 🚀 Quick Start

### Prerequisites
- UV package manager
- API key for any OpenAI-compatible provider (OpenRouter, OpenAI, Together AI, etc.)
- Ollama server (optional, for local models)

### Installation
```bash
# Clone the repository
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant

# Set up environment
cp .env.template .env
uv venv .venv
uv sync

# Configure your API key in .env (choose one option)

# Option 1: Generic OpenAI-compatible provider (recommended)
echo "OPENAI_COMPATIBLE_API_KEY=your_key_here" >> .env
echo "OPENAI_COMPATIBLE_BASE_URL=https://your-provider.com/api/v1" >> .env

# Option 2: OpenRouter (backward compatible)
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

### Running the System
```bash
# Start the development server
uv run uvicorn app.main:app --reload
```

### Provider Features
- **Automatic Fallback**: Falls back to other providers if preferred fails
- **Health Monitoring**: Continuous health checks for all providers
- **Model Discovery**: Automatic detection of available models
- **Mixed Usage**: Use both cloud and local models in the same application

For detailed setup instructions, see [Ollama Integration Guide](docs/ollama-integration.md).

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

## 📚 Documentation

### [OpenAI-Compatible Provider Refactoring](docs/openai-compatible-provider-refactoring.md)
- Migration guide from OpenRouter to generic provider
- Supported providers and configuration options
- Backward compatibility information

### [Architecture Overview](docs/architecture/overview.md)
- System design and components
- Agent workflow and tool orchestration
- Integration patterns and extensibility

### [Core Components](docs/architecture/core-components.md)
- Detailed component documentation
- Tool system and agent orchestration
- Caching and monitoring systems

### [API Reference](docs/api/endpoints.md)
- OpenAI-compatible endpoints
- Request/response formats
- Authentication and error handling
- Tool management endpoints

### [Development Guide](docs/development/development-guide.md)
- Creating custom tools
- Testing and deployment
- Best practices and examples

### [Quick Setup](docs/development/setup.md)
- Installation and configuration
- Environment setup
- Getting started quickly

## 🌐 API Usage

### Basic Chat Completion
```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What's 15 * 25?"}]
    }
)
print(response.json())
```

### Tool Calling
```python
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What's the current time?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time"
                }
            }
        ]
    }
)
```

## ⚙️ Configuration

### Environment Variables

#### Option 1: Generic OpenAI-Compatible Provider (Recommended)
```bash
# Core Settings
OPENAI_COMPATIBLE_ENABLED=true
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.com/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=your-preferred-model
PREFERRED_PROVIDER=openai_compatible

# Optional: Custom headers and settings
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"X-Custom-Header": "value"}
OPENAI_COMPATIBLE_TIMEOUT=30
OPENAI_COMPATIBLE_MAX_RETRIES=3
```

#### Option 2: OpenRouter (Backward Compatible)
```bash
# Core Settings
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
PREFERRED_PROVIDER=openrouter
```

#### Option 3: Ollama (Local Models)
```bash
# Ollama Settings
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
OLLAMA_SETTINGS_DEFAULT_MODEL=llama2
```

#### General Settings
```bash
# Server Configuration
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
```

### Supported Providers

The generic OpenAI-compatible provider works with:

- **OpenRouter**: `https://openrouter.ai/api/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **Together AI**: `https://api.together.xyz/v1`
- **Azure OpenAI**: `https://your-resource.openai.azure.com/`
- **Any OpenAI-compatible API**: Custom endpoints

For detailed migration instructions, see [OpenAI-Compatible Provider Refactoring](docs/openai-compatible-provider-refactoring.md).

For complete configuration details, see [Development Guide](docs/development/development-guide.md).

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --coverage
```

## 🚀 Deployment

### Production Deployment
```bash
uv sync
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN uv sync
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

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