# AI Assistant System Documentation

Welcome to the comprehensive documentation for the LLM Tool System Foundation! This project provides a production-ready, OpenAI-compatible API interface for LLM agents with advanced tool-calling capabilities, multi-provider support, and extensible architecture.

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone and setup
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.docker .env

# Configure your API key
echo "OPENAI_COMPATIBLE_API_KEY=your_key_here" >> .env
echo "SECRET_KEY=your_secret_key_here" >> .env

# Start all services
docker-compose up -d

# Access the application
open http://localhost:8000
```

### Option 2: Local Development
```bash
# Clone and setup
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.template .env

# Setup environment
uv venv .venv
uv sync

# Configure API key
echo "OPENAI_COMPATIBLE_API_KEY=your_key_here" >> .env

# Start development server
uv run uvicorn app.main:app --reload
```

### First Steps
1. **Get an API key** from any OpenAI-compatible provider:
   - [OpenRouter](https://openrouter.ai) (recommended for variety)
   - [OpenAI](https://platform.openai.com)
   - [Together AI](https://together.ai)
2. **Configure your environment** with the API key
3. **Visit the interactive API docs** at `http://localhost:8000/docs`
4. **Try a simple chat completion** to verify setup

## 📚 Documentation Sections

### 🏗️ [Architecture](architecture/overview.md)
- System design and components
- Agent workflow and tool orchestration
- Integration patterns and extensibility
- [Core Components](architecture/core-components.md)
- [Tool System Design](architecture/tools.md)
- [Agent Workflow](architecture/workflow.md)

### 🔌 [API Reference](api/endpoints.md)
- OpenAI-compatible endpoints
- Request/response formats
- Authentication and error handling
- Tool management endpoints
- Streaming and batch processing

### 🛠️ [Development](development/setup.md)
- [Setup Guide](development/setup.md)
- [Development Guide](development/development-guide.md)
- [Contributing Guidelines](development/contributing.md)
- [Testing Strategy](development/testing.md)
- [Security Practices](development/security-and-api-key-handling.md)
- [Branch Protection](development/branch-protection.md)

### 🔧 [Tools & Integrations](tools/searx.md)
- [SearXNG Search Integration](tools/searx.md)
- [RAG Knowledge Base](tools/rag.md)
- Tool development framework
- Custom tool examples

### 🐳 [Deployment](docker-integration.md)
- [Docker Integration Guide](docker-integration.md)
- [Docker Testing](docker-testing.md)
- [Ollama Integration](ollama-integration.md)
- Production deployment patterns

### 🔄 [Provider Configuration](openai-compatible-provider-refactoring.md)
- [OpenAI-Compatible Provider Refactoring](openai-compatible-provider-refactoring.md)
- Multi-provider setup
- Migration guides
- Backward compatibility

## ✨ Key Features

### 🌐 OpenAI API Compatibility
Full compatibility with the OpenAI API specification, allowing seamless integration with various LLM frontends and tools.

### 🤖 Intelligent Tool-Calling Agents
Extensible architecture for adding new tools and capabilities to the AI assistant with context-aware selection.

### ⚡ Real-time Streaming
Support for streaming responses for interactive chat experiences with minimal latency.

### 🔍 Multi-Provider Support
Unified interface for OpenAI, OpenRouter, Together AI, Azure OpenAI, and custom providers with automatic fallback.

### 🗄️ Advanced Caching System
Multi-layer caching with compression, batching, and intelligent cache invalidation strategies.

### 📊 Comprehensive Monitoring
Built-in Prometheus metrics, health checks, and performance monitoring with Grafana dashboards.

### 🐳 Container-Ready
Complete Docker support with docker-compose configurations for development, testing, and production.

### 🔒 Security-First Design
Input validation, API key security, dependency scanning, and secure development practices.

### 🧪 Comprehensive Testing
Robust test suite with unit tests, integration tests, system tests, and security scanning.

## 🔧 Technology Stack

- **Backend**: FastAPI with Python 3.12
- **LLM Integration**: LangChain with OpenRouter
- **Dependency Management**: UV
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions with security scanning
- **Documentation**: MkDocs with Material theme

## Development Status

**Current Version**: 0.3.2


### Implemented Features
- ✅ OpenAI-compatible API endpoints
- ✅ OpenRouter integration for multiple LLM providers
- ✅ Streaming response support
- ✅ Comprehensive test suite
- ✅ GitHub Actions CI/CD pipeline
- ✅ Security scanning and code quality checks

### Planned Features
- 🔄 Ability to use local as well as more cloud based LLMs
- 🔄 SearX web search integration
- 🔄 RAG knowledge base system
- 🔄 Additional tool integrations
- 🔄 Docker containerization
- 🔄 Advanced agent capabilities

## Contributing

Contributions will be welcomed soon! Please see our [Contributing Guide](development/contributing.md) for details on how to get involved.

### Getting Help
- **Documentation**: This site contains comprehensive documentation
- **Issues**: Check existing issues or create new ones on GitHub

## Project Metrics

- **Test Coverage**: Comprehensive unit and integration tests
- **Code Quality**: Enforced with ruff, black, and mypy
- **Security**: Regular scanning with bandit and pip-audit
- **Performance**: Optimized for low-latency responses

## Security

Security measures include

* No hardcoded API keys or secrets
* Comprehensive security scanning in CI/CD
* Regular dependency vulnerability checks
* Secure development practices

## License

This project is open source with an MIT license.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance APIs
- Powered by [LangChain](https://www.langchain.com/) for LLM orchestration
- Integrated with [OpenRouter](https://openrouter.ai/) for model access
- Documented with [MkDocs](https://www.mkdocs.org/) and [Material](https://squidfunk.github.io/mkdocs-material/)