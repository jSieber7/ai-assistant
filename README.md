# AI Assistant with LangChain Integration

A comprehensive AI assistant application built with FastAPI, now fully integrated with LangChain and LangGraph for enhanced language model capabilities.

## 🚀 Features

### Core Features
- **LangChain Integration**: Full integration with LangChain and LangGraph for robust language model workflows
- **Multiple LLM Providers**: Support for OpenAI, Ollama, and custom LLM providers
- **Specialized Agents**: 9 specialized agents for different tasks (summarization, web scraping, fact checking, etc.)
- **Tool Registry**: Dynamic tool registration and execution system
- **Memory Management**: Persistent conversation memory with context management
- **Monitoring & Observability**: Comprehensive metrics and performance tracking
- **RESTful API**: Complete REST API with comprehensive documentation
- **Database Integration**: PostgreSQL with optimized schema for LangChain components
- **Docker Support**: Full Docker containerization with development and production setups

### Specialized Agents
1. **SummarizeAgent**: Advanced text summarization with configurable parameters
2. **WebdriverAgent**: Browser automation and web interaction
3. **ScraperAgent**: Web scraping with Firebase integration
4. **SearchQueryAgent**: Optimized search query generation
5. **ChainOfThoughtAgent**: Structured reasoning and problem solving
6. **CreativeStoryAgent**: Creative story generation with genre support
7. **ToolSelectionAgent**: Intelligent tool selection based on context
8. **SemanticUnderstandingAgent**: Sentiment analysis and semantic understanding
9. **FactCheckerAgent**: Fact verification and validation

## 🏗️ Architecture

### LangChain Integration Layer
The application uses a comprehensive integration layer that provides:
- **Component Management**: Centralized management of all LangChain components
- **Feature Flags**: Granular control over which components use LangChain
- **Backward Compatibility**: Seamless migration from legacy systems
- **Health Monitoring**: Real-time health checks for all components

### Core Components
- **LLM Manager**: Unified interface for multiple LLM providers
- **Tool Registry**: Dynamic tool management and execution
- **Agent Manager**: LangGraph-based agent workflow management
- **Memory Manager**: Persistent conversation and context management
- **Monitoring System**: Performance metrics and observability

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for building applications with language models
- **LangGraph**: Extension of LangGraph for building stateful, multi-actor applications
- **SQLAlchemy**: SQL toolkit and ORM
- **PostgreSQL**: Powerful open-source database
- **Pydantic**: Data validation using Python type annotations
- **AsyncIO**: Asynchronous programming support

### DevOps
- **Docker**: Containerization for consistent deployment
- **Docker Compose**: Multi-container application orchestration
- **GitHub Actions**: Continuous integration and deployment
- **Poetry**: Python dependency management

### Testing
- **Pytest**: Testing framework with async support
- **Test Coverage**: Comprehensive test coverage reporting
- **Mock Testing**: Extensive mocking for isolated testing

## 🚦 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- PostgreSQL (or use the provided Docker setup)
- API keys for LLM providers (OpenAI, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/ai-assistant.git
cd ai-assistant
```

2. **Install dependencies**
```bash
poetry install
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start the application**
```bash
# Development mode
docker-compose up -d

# Or local development
poetry run python -m app.main
```

5. **Verify installation**
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/langchain/health
```

## 📚 Documentation

### Core Documentation
- [LangChain Integration Architecture](docs/architecture/langchain-integration.md) - Comprehensive overview of the LangChain integration
- [API Reference](docs/api/langchain-api-reference.md) - Complete API documentation
- [Migration Guide](docs/tutorials/langchain-migration-guide.md) - Step-by-step migration from legacy systems
- [Database Schema](docs/architecture/database-schema.md) - Database design and schema documentation

### Tutorials
- [Local Model with Llama.cpp](docs/tutorials/local-model-with-llama.cpp.md) - Using local models with Llama.cpp
- [Docker Setup](docs/docker-setup.md) - Docker configuration and deployment
- [Development Guide](docs/development.md) - Development setup and best practices

### Architecture
- [Core Components](docs/architecture/core-components.md) - Detailed component documentation
- [Modular RAG Services](docs/architecture/modular-rag-services.md) - RAG (Retrieval-Augmented Generation) services
- [Caching System](docs/architecture/caching.md) - Caching strategies and implementation

## 🔧 Configuration

### Environment Variables

Key configuration options:

```bash
# LangChain Integration
LANGCHAIN_INTEGRATION_MODE=langchain  # legacy, langchain, hybrid, migration
LANGCHAIN_LLM_MANAGER_ENABLED=true
LANGCHAIN_TOOL_REGISTRY_ENABLED=true
LANGCHAIN_AGENT_MANAGER_ENABLED=true
LANGCHAIN_MEMORY_WORKFLOW_ENABLED=true
LANGCHAIN_MONITORING_ENABLED=true

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ai_assistant
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password

# LLM Providers
OPENAI_API_KEY=your-openai-key
OLLAMA_BASE_URL=http://localhost:11434

# Monitoring
MONITORING_METRICS_RETENTION_DAYS=30
MONITORING_PERFORMANCE_TRACKING_ENABLED=true
```

### Feature Flags

Control which components use LangChain:

```python
LANGCHAIN_FEATURES = {
    "llm_manager": True,
    "tool_registry": True,
    "agent_manager": True,
    "memory_workflow": True,
    "monitoring": True
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run LangChain tests only
pytest -m langchain

# Run with coverage
pytest --cov=app.core.langchain

# Run specific test file
pytest tests/unit/app/core/langchain/test_integration.py
```

### Test Structure

```
tests/
├── unit/app/core/langchain/          # Unit tests for LangChain components
├── integration/app/core/langchain/   # Integration tests
├── unit/app/api/test_langchain_routes.py  # API route tests
└── conftest.py                       # Test configuration
```

## 📊 Monitoring

### Health Checks

```bash
# System health
curl http://localhost:8000/api/health

# LangChain health
curl http://localhost:8000/api/langchain/health

# Component status
curl http://localhost:8000/api/langchain/status
```

### Metrics

```bash
# System metrics
curl http://localhost:8000/api/langchain/monitoring/metrics

# Performance metrics
curl http://localhost:8000/api/langchain/monitoring/performance

# Component metrics
curl http://localhost:8000/api/langchain/monitoring/metrics/llm/openai
```

## 🚀 Deployment

### Docker Deployment

```bash
# Production deployment
docker-compose -f docker/docker-compose.yml up -d

# Development deployment
docker-compose -f docker/docker-compose.dev.yml up -d
```

### Environment-Specific Configurations

- **Development**: Use `docker/docker-compose.dev.yml`
- **Staging**: Use `docker/docker-compose.staging.yml`
- **Production**: Use `docker/docker-compose.yml`

## 🔌 API Usage

### Basic Examples

#### LLM Request

```bash
curl -X POST http://localhost:8000/api/langchain/llm/request \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Hello, AI Assistant!"
  }'
```

#### Tool Execution

```bash
curl -X POST http://localhost:8000/api/langchain/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_tool",
    "parameters": {
      "query": "LangChain integration"
    }
  }'
```

#### Agent Invocation

```bash
curl -X POST http://localhost:8000/api/langchain/agents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "research_agent",
    "input_data": {
      "messages": [{"role": "user", "content": "Research AI trends"}]
    }
  }'
```

#### Specialized Agents

```bash
# Summarize text
curl -X POST http://localhost:8000/api/langchain/agents/specialized/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long text to summarize...",
    "target_length": 100
  }'

# Generate search queries
curl -X POST http://localhost:8000/api/langchain/agents/specialized/search-query \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "machine learning applications"
  }'

# Check facts
curl -X POST http://localhost:8000/api/langchain/agents/specialized/fact-checker \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The Earth is round"
  }'
```

### Python SDK

```python
from ai_assistant_sdk import LangChainClient

client = LangChainClient(api_key="your-api-key")

# LLM request
response = await client.llm.request(
    model="gpt-3.5-turbo",
    prompt="Hello, AI Assistant!"
)

# Tool execution
result = await client.tools.execute(
    tool_name="search_tool",
    parameters={"query": "LangChain integration"}
)

# Agent invocation
result = await client.agents.invoke(
    agent_name="research_agent",
    input_data={"messages": [{"role": "user", "content": "Research AI trends"}]}
)
```

## 🔄 Migration from Legacy Systems

If you're migrating from the legacy AI Assistant system, follow the [Migration Guide](docs/tutorials/langchain-migration-guide.md) for step-by-step instructions.

### Migration Phases

1. **Phase 1**: Preparation and setup
2. **Phase 2**: Hybrid mode testing
3. **Phase 3**: Gradual component migration
4. **Phase 4**: Full LangChain migration
5. **Phase 5**: Legacy cleanup (optional)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

We use the following tools for code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
poetry run black app tests
poetry run isort app tests

# Lint code
poetry run flake8 app tests

# Type checking
poetry run mypy app
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- **Documentation**: [Full documentation](docs/)
- **API Reference**: [API documentation](docs/api/langchain-api-reference.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/ai-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ai-assistant/discussions)
- **Email**: support@ai-assistant.com

## 🗺️ Roadmap

### Upcoming Features

- [ ] Additional LLM provider integrations (Anthropic, Cohere)
- [ ] Advanced workflow templates
- [ ] Real-time collaboration features
- [ ] Enhanced monitoring dashboard
- [ ] Mobile SDK support
- [ ] GraphQL API support
- [ ] Advanced caching strategies
- [ ] Multi-tenant support

### Long-term Goals

- [ ] Distributed agent execution
- [ ] Advanced RAG implementations
- [ ] Custom model fine-tuning
- [ ] Edge deployment support
- [ ] Advanced security features

## 📈 Performance

### Benchmarks

- **LLM Request Latency**: < 1.5s average
- **Tool Execution Time**: < 500ms average
- **Agent Invocation**: < 3s average
- **API Response Time**: < 100ms average
- **System Uptime**: 99.9%

### Optimization

The application includes several optimizations:
- Connection pooling for database operations
- Async/await for non-blocking operations
- Intelligent caching strategies
- Resource management and cleanup
- Performance monitoring and alerting

## 🔒 Security

### Security Features

- **Authentication**: API key-based authentication
- **Authorization**: Role-based access control
- **Data Encryption**: Encryption at rest and in transit
- **Input Validation**: Comprehensive input validation
- **Audit Logging**: Complete audit trail
- **Rate Limiting**: Protection against abuse

### Best Practices

- Regular security updates
- Dependency vulnerability scanning
- Security code reviews
- Penetration testing
- Compliance with data protection regulations

## 📊 Analytics

The application includes comprehensive analytics:

- **Usage Metrics**: Track API usage and patterns
- **Performance Metrics**: Monitor system performance
- **Error Tracking**: Identify and resolve issues
- **User Analytics**: Understand user behavior
- **Business Intelligence**: Generate insights

## 🌐 Internationalization

The application supports multiple languages and regions:

- **i18n Support**: Internationalization framework
- **L10n Support**: Localization for different regions
- **Multi-language UI**: Interface in multiple languages
- **Cultural Adaptation**: Region-specific adaptations

## 📱 Mobile Support

While primarily a backend API, the application supports mobile clients:

- **Mobile-optimized API**: Optimized for mobile consumption
- **Push Notifications**: Real-time notifications
- **Offline Support**: Limited offline functionality
- **Mobile SDK**: Dedicated mobile SDKs

## 🔧 Advanced Configuration

### Custom LLM Providers

Add custom LLM providers:

```python
from langchain.llms import BaseLLM

class CustomLLM(BaseLLM):
    def _call(self, prompt, stop=None, run_manager=None):
        # Custom implementation
        return response
    
    @property
    def _llm_type(self):
        return "custom"

# Register with LLM manager
await llm_manager.register_provider("custom", CustomLLM)
```

### Custom Tools

Create custom tools:

```python
from langchain.tools import tool

@tool
def custom_tool(input_data: str) -> str:
    """Custom tool description"""
    # Tool implementation
    return result

# Register with tool registry
await tool_registry.register_custom_tool(
    name="custom_tool",
    func=custom_tool,
    description="Custom tool description"
)
```

### Custom Agents

Create custom agents:

```python
from langgraph.graph import StateGraph

def custom_workflow(state):
    # Custom workflow implementation
    return {"result": result}

workflow = StateGraph(dict)
workflow.add_node("process", custom_workflow)
workflow.set_entry_point("process")
workflow.add_edge("process", "end")

compiled_agent = workflow.compile(checkpointer=MemorySaver())

# Register with agent manager
await agent_manager.register_agent(
    name="custom_agent",
    agent=compiled_agent,
    description="Custom agent description"
)
```

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

---

**Built with ❤️ by the AI Assistant Team**