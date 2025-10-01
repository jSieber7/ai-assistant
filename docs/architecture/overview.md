# System Architecture Overview

This document describes the high-level architecture of the AI Assistant project, including its components, data flow, and design principles.

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Apps   │◄──►│   FastAPI API    │◄──►│   LLM Agents    │
│ (OpenWebUI,     │    │  (OpenAI-compat) │    │   (LangChain)   │
│  Chat Apps)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Tool System    │◄──►│  External APIs  │
│   (Optional)    │    │  (Extensible)    │    │ (OpenRouter,    │
│                 │    │                  │    │   SearX, RAG)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. API Layer (FastAPI)
- **Purpose**: Provide OpenAI-compatible interface
- **Technology**: FastAPI with Pydantic models<br><br>
<u>**Features**</u> 
  - OpenAPI documentation
  - Streaming responses
  - CORS support
  - Authentication (planned)

### 2. LLM Agent Layer (LangChain)
- **Purpose**: Orchestrate LLM interactions and tool calling
- **Technology**: LangChain with custom agents<br><br>
<u>**Features**</u>
  - Multi-model support via OpenRouter
  - Tool calling capabilities
  - Conversation memory
  - Response streaming

### 3. Tool System (Extensible)
- **Purpose**: Extend AI capabilities with external tools
- **Technology**: Modular tool architecture<br><br>
<u>**Current Tools**</u>
  - Web search (SearX integration - planned)
  - Knowledge base (RAG system - planned)
  - Custom tool development framework

### 4. Data Layer
- **Purpose**: Manage application data and vector storage
- **Technology**: PostgreSQL with pgvector (planned)<br><br>
<u>**Features**</u>
  - Vector embeddings for RAG
  - Document storage and retrieval
  - Conversation history (planned)

## Data Flow

### Standard Chat Flow
1. **Request Reception**: Client sends chat request to `/v1/chat/completions`
2. **Message Processing**: Convert OpenAI format to LangChain messages
3. **Agent Execution**: LangChain agent processes request with available tools
4. **Tool Execution**: If needed, tools are called to gather information
5. **Response Generation**: LLM generates response based on context
6. **Response Formatting**: Convert LangChain response to OpenAI format
7. **Streaming**: Send response chunks back to client

### Tool Calling Flow
1. **Tool Detection**: Agent determines if tools are needed
2. **Tool Selection**: Choose appropriate tool based on query
3. **Tool Execution**: Run tool with parameters
4. **Result Integration**: Combine tool results with conversation context
5. **Response Generation**: Generate final response with tool insights

## Design Principles

### 1. OpenAI Compatibility
- Full compliance with OpenAI API specification
- Support for both streaming and non-streaming responses
- Standard error codes and response formats

### 2. Extensibility
- Modular tool system for adding new capabilities
- Plugin architecture for custom integrations
- Configuration-driven behavior

### 3. Security First
- No hardcoded API keys or secrets
- Environment-based configuration
- Input validation and sanitization
- Regular security scanning

### 4. Performance
- Async/await for non-blocking operations
- Connection pooling for external APIs
- Caching strategies for frequent operations
- Efficient vector search algorithms

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.12)
- **LLM Orchestration**: LangChain
- **Vector Database**: PostgreSQL + pgvector (planned)
- **API Client**: HTTPX for async requests

### Development Tools
- **Package Manager**: UV
- **Testing**: pytest with coverage
- **Code Quality**: ruff, black, mypy
- **Security**: bandit, pip-audit

### Infrastructure
- **CI/CD**: GitHub Actions
- **Documentation**: MkDocs + Material theme
- **Containerization**: Docker (planned)

## Configuration Management

### Environment-Based Configuration
```python
# app/core/config.py
class Settings(BaseSettings):
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"
    # ... other settings
```

### Security Considerations
- API keys stored as `SecretStr`
- Environment variables for sensitive data
- Validation of all configuration values
- Secure defaults for production

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- External session storage (planned)
- Load balancer compatibility

### Performance Optimization
- Connection pooling for database and API calls
- Caching layer for frequent queries
- Async processing for I/O operations

### Monitoring and Observability
- Structured logging
- Performance metrics collection
- Health check endpoints
- Error tracking and alerting

## Deployment Architecture

### Development Environment
- Local execution with hot reload
- Mock external services for testing
- Detailed logging and debugging

### Production Environment
- Containerized deployment
- Environment-specific configuration
- Health monitoring and auto-recovery
- Scalable infrastructure

## Integration Patterns

### External API Integration
- Async HTTP clients with retry logic
- Circuit breaker pattern for resilience
- Rate limiting and backoff strategies

### Tool Integration
- Standardized tool interface
- Error handling and fallbacks
- Performance monitoring

## Future Architecture Evolution

### Phase 1: Core Stability
- [x] Basic OpenAI-compatible API
- [x] LangChain integration
- [ ] Tool system foundation

### Phase 2: Advanced Features
- [ ] SearX web search integration
- [ ] RAG knowledge base
- [ ] Advanced tool capabilities

### Phase 3: Production Ready
- [ ] Docker containerization
- [ ] Advanced monitoring
- [ ] High availability setup

## Related Documentation

- [API Endpoints Reference](../api/endpoints.md)
- [Development Setup Guide](../development/setup.md)
- [Tool System Design](tools.md)
- [Agent Workflow](workflow.md)

## Decision Log

### Technology Choices
- **FastAPI**: Chosen for performance, async support, and automatic OpenAPI docs
- **LangChain**: Industry standard for LLM orchestration with extensive tooling
- **PostgreSQL + pgvector**: Robust, scalable vector database solution
- **UV**: Fast, modern Python package manager with excellent dependency resolution

### Architecture Decisions
- **OpenAI Compatibility**: Ensures wide compatibility with existing tools
- **Modular Tool System**: Allows incremental feature development
- **Async-First Design**: Optimal for I/O-heavy LLM operations
- **Security-First Approach**: Protects sensitive API keys and user data

This architecture provides a solid foundation for building a powerful, extensible AI assistant while maintaining security, performance, and developer experience.