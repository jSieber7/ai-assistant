# AI Agent Workflow Documentation

Welcome to the AI Assistant project documentation! This project provides an OpenAI-compatible API interface for LLM agents with tool-calling capabilities.

## 🚀 Quick Start

### Prerequisites
- UV package manager
- OpenRouter API key

### Installation
```bash
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.template .env

uv venv .venv
uv sync
uv run uvicorn app.main:app --reload # For the current build in development
```

### First Steps
1. Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai)
2. Add it to your `.env` file: `OPENROUTER_API_KEY=your_key_here`
3. Visit the API documentation at `http://localhost:8000/docs`

## 📖 Documentation Sections

### [Architecture](architecture/overview.md)
- System design and components
- Agent workflow and tool orchestration
- Integration patterns and extensibility

### [API Reference](api/endpoints.md)
- OpenAI-compatible endpoints
- Request/response formats
- Authentication and error handling

### [Development](development/setup.md)
- Setup instructions and environment configuration
- Contributing guidelines
- Testing and code quality standards

### [Tools](tools/searx.md)
- Tool integrations and extensions
- SearX web search integration
- RAG knowledge base system

## 🎯 Key Features

### OpenAI API Compatibility
Full compatibility with the OpenAI API specification, allowing integration with various LLM frontends and tools.

### Tool-Calling Agents
Extensible architecture for adding new tools and capabilities to the AI assistant.

### Real-time Streaming
Support for streaming responses for interactive chat experiences.

### Comprehensive Testing
Robust test suite with unit tests, integration tests, and security scanning.

## 🔧 Technology Stack

- **Backend**: FastAPI with Python 3.12
- **LLM Integration**: LangChain with OpenRouter
- **Dependency Management**: UV
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions with security scanning
- **Documentation**: MkDocs with Material theme

## 🛠️ Development Status

**Current Version**: 0.0.2

### Implemented Features
- ✅ OpenAI-compatible API endpoints
- ✅ OpenRouter integration for multiple LLM providers
- ✅ Streaming response support
- ✅ Comprehensive test suite
- ✅ GitHub Actions CI/CD pipeline
- ✅ Security scanning and code quality checks

### Planned Features
- 🔄 SearX web search integration
- 🔄 RAG knowledge base system
- 🔄 Additional tool integrations
- 🔄 Docker containerization
- 🔄 Advanced agent capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get involved.

### Getting Help
- **Documentation**: This site contains comprehensive documentation
- **Issues**: Check existing issues or create new ones on GitHub
- **Discussions**: Join the conversation in GitHub Discussions

## 📊 Project Metrics

- **Test Coverage**: Comprehensive unit and integration tests
- **Code Quality**: Enforced with ruff, black, and mypy
- **Security**: Regular scanning with bandit and pip-audit
- **Performance**: Optimized for low-latency responses

## 🔒 Security

This project takes security seriously:
- No hardcoded API keys or secrets
- Comprehensive security scanning in CI/CD
- Regular dependency vulnerability checks
- Secure development practices

## 📄 License

This project is open source. Please see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance APIs
- Powered by [LangChain](https://www.langchain.com/) for LLM orchestration
- Integrated with [OpenRouter](https://openrouter.ai/) for model access
- Documented with [MkDocs](https://www.mkdocs.org/) and [Material](https://squidfunk.github.io/mkdocs-material/)

---

*Last updated: {{ git.date }}*
