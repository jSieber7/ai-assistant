# AI Assistant System Documentation

Welcome to the comprehensive documentation for the AI Assistant System! This project provides a production-ready, OpenAI-compatible API interface for LLM agents with advanced tool-calling capabilities, multi-provider support, and extensible architecture.

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
open http://localhost  # Main application through Traefik
open http://localhost:8080  # Traefik dashboard
open http://localhost/gradio  # Gradio interface
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
   - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
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
- [Caching Architecture](architecture/caching.md)

### 🔌 [API Reference](api/endpoints.md)
- OpenAI-compatible endpoints
- Request/response formats
- Authentication and error handling
- Tool management endpoints
- Streaming and batch processing
- [OpenAI Compatibility](api/openai-compatibility.md)
- [Tool Management](api/tool-management.md)

### 🛠️ [Development](development/setup.md)
- [Setup Guide](development/setup.md)
- [Development Guide](development/development-guide.md)
- [Contributing Guidelines](development/contributing.md)
- [Testing Strategy](development/testing.md)
- [Security Practices](development/security-and-api-key-handling.md)
- [Branch Protection](development/branch-protection.md)

### 🔧 [Tools & Integrations](tools/overview.md)
- [Tool System Overview](tools/overview.md)
- [SearXNG Search Integration](tools/searx.md)
- [Firecrawl Web Scraping](tools/firecrawl-quick-start.md)
- [Jina Reranker](tools/jina-reranker.md)
- [RAG Knowledge Base](tools/rag.md)
- [Custom Tool Development](tools/tool-development.md)

### 🐳 [Deployment](deployment/production.md)
- [Production Deployment](deployment/production.md)
- [Docker Integration Guide](docker-integration.md)
- [Docker Testing](docker-testing.md)
- [Docker Simplified](docker-simplified.md)
- [Monitoring Setup](deployment/monitoring.md)
- [Traefik Integration](deployment/traefik-integration.md)

### 🔄 [Provider Configuration](providers/multi-provider.md)
- [Multi-Provider Setup](providers/multi-provider.md)
- [Provider Comparison](providers/comparison.md)
- [OpenAI-Compatible Provider Refactoring](openai-compatible-provider-refactoring.md)
- [Ollama Integration](ollama-integration.md)
- Migration guides and backward compatibility

## ✨ Key Features

### 🌐 OpenAI API Compatibility
Full compatibility with the OpenAI API specification, allowing seamless integration with various LLM frontends and tools.

### 🤖 Intelligent Tool-Calling Agents
Extensible architecture for adding new tools and capabilities to the AI assistant with context-aware selection and execution.

### ⚡ Real-time Streaming
Support for streaming responses for interactive chat experiences with minimal latency.

### 🔍 Multi-Provider Support
Unified interface for OpenAI, OpenRouter, Together AI, Azure OpenAI, Ollama, and custom providers with automatic fallback.

### 🗄️ Advanced Caching System
Multi-layer caching with Redis, compression, batching, and intelligent cache invalidation strategies.

### 📊 Comprehensive Monitoring
Built-in Prometheus metrics, health checks, and performance monitoring with Grafana dashboards.

### 🔧 Extensible Tool System
Dynamic tool registration and execution framework with built-in tools for web scraping, search, calculations, and more.

### 🐳 Container-Ready
Complete Docker support with docker-compose configurations for development, testing, and production.

### 🔒 Security-First Design
Input validation, API key security with SecretStr, dependency scanning, and secure development practices.

### 🧪 Comprehensive Testing
Robust test suite with unit tests, integration tests, system tests, and security scanning.

### 📝 Multi-Writer System
Advanced content generation pipeline with multiple AI writers and checkers for high-quality content creation.

### 🌐 Privacy-Focused Search
Integrated SearXNG for privacy-focused web search capabilities without tracking.

## 🔧 Technology Stack

- **Backend**: FastAPI with Python 3.12
- **LLM Integration**: LangChain with multi-provider support
- **Tool System**: Custom extensible tool framework with dynamic execution
- **Caching**: Redis with multi-layer caching and compression
- **Web Interface**: Gradio for configuration and testing
- **Search Integration**: SearXNG for privacy-focused web search
- **Web Scraping**: Firecrawl for advanced content extraction
- **Reranking**: Jina Reranker for improved search results
- **Monitoring**: Prometheus metrics with Grafana dashboards
- **Containerization**: Docker and Docker Compose with Traefik
- **Dependency Management**: UV for fast package management
- **Testing**: pytest with comprehensive coverage
- **CI/CD**: GitHub Actions with security scanning
- **Documentation**: MkDocs with Material theme

## Development Status

**Current Version**: 0.4.0

### ✅ Implemented Features
- 🔧 **Extensible Tool System**: Dynamic tool registration and discovery
- 🤖 **Intelligent Agent Orchestration**: Context-aware tool selection and execution
- ⚡ **Advanced Caching**: Multi-layer caching with compression and batching
- 📊 **Comprehensive Monitoring**: Real-time metrics and health checks
- 🔒 **Security-First Design**: Input validation and access control
- 🔄 **LangChain Integration**: Seamless compatibility with LangChain ecosystem
- 🌐 **Multi-Provider Support**: OpenAI, OpenRouter, Together AI, Azure OpenAI, Ollama, and custom providers
- 🐳 **Docker Support**: Complete containerization with docker-compose
- 🔍 **SearXNG Integration**: Privacy-focused web search capabilities
- 🔥 **Firecrawl Integration**: Advanced web scraping with content extraction
- 📈 **Jina Reranker**: Improved search result reranking
- 📊 **Prometheus Metrics**: Built-in monitoring and alerting
- 🖥️ **Gradio Interface**: Web-based UI for configuration and testing
- 🔄 **OpenAI Compatibility**: Full API compatibility with OpenAI specification
- 🚀 **Real-time Streaming**: Streaming responses for interactive chat experiences
- 🔧 **Tool Development Framework**: Easy creation and integration of custom tools
- 📝 **Multi-Writer System**: Advanced content generation with AI collaboration
- 🌐 **Traefik Integration**: Advanced reverse proxy and load balancing

### 🎯 Development Roadmap
- 🔄 **Advanced Agent Capabilities**: Multi-agent systems and complex workflows
- 🔄 **RAG Knowledge Base**: Vector-based document retrieval and knowledge management
- 🔄 **Production Monitoring**: Enhanced observability and alerting
- 🔄 **Performance Optimization**: Additional caching layers and batching strategies
- 🔄 **Enhanced Security**: Advanced authentication and authorization

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get involved.

### Getting Help
- **Documentation**: This site contains comprehensive documentation
- **Issues**: Check existing issues or create new ones on GitHub
- **Discussions**: Join our GitHub Discussions for community support

## Project Metrics

- **Test Coverage**: Comprehensive unit and integration tests
- **Code Quality**: Enforced with ruff, black, and mypy
- **Security**: Regular scanning with bandit and pip-audit
- **Performance**: Optimized for low-latency responses
- **Docker Support**: Multi-container deployment with health checks

## Security

Security measures include:

* No hardcoded API keys or secrets - all stored in environment variables
* Comprehensive security scanning in CI/CD pipeline
* Regular dependency vulnerability checks
* Secure development practices with input validation
* SecretStr type for sensitive configuration values
* Rate limiting and request validation capabilities

## License

This project is open source with an MIT license.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance APIs
- Powered by [LangChain](https://www.langchain.com/) for LLM orchestration
- Integrated with multiple providers for model access
- Enhanced by [SearXNG](https://searxng.org/) for privacy-focused search
- Powered by [Firecrawl](https://firecrawl.dev/) for web scraping
- Improved by [Jina AI](https://jina.ai/) for reranking capabilities
- Documented with [MkDocs](https://www.mkdocs.org/) and [Material](https://squidfunk.github.io/mkdocs-material/)