# Configuration Guide

This comprehensive guide covers all configuration options available for the AI Assistant System.

## Environment Variables

The system can be configured using environment variables. Refer to the `.env.template` file for a complete list of available options.

## Core Settings

### LLM Provider Configuration

The system supports multiple LLM providers with automatic fallback capabilities.

#### OpenAI-Compatible Provider (Recommended)
```bash
# Core OpenAI-Compatible Provider Settings
OPENAI_COMPATIBLE_ENABLED=true
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
OPENAI_COMPATIBLE_PROVIDER_NAME=OpenRouter  # Auto-detected if not set

# Optional Settings
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"X-Custom-Header": "value"}
OPENAI_COMPATIBLE_TIMEOUT=30
OPENAI_COMPATIBLE_MAX_RETRIES=3
```

#### OpenAI
```bash
OPENAI_COMPATIBLE_API_KEY=sk-your-openai-key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=gpt-4-turbo
```

#### OpenRouter (Backward Compatible)
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
PREFERRED_PROVIDER=openrouter
```

#### Together AI
```bash
OPENAI_COMPATIBLE_API_KEY=your_together_key
OPENAI_COMPATIBLE_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

#### Azure OpenAI
```bash
OPENAI_COMPATIBLE_API_KEY=your_azure_key
OPENAI_COMPATIBLE_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"api-key": "your_azure_key"}
```

#### Ollama (Local Models)
```bash
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
OLLAMA_SETTINGS_DEFAULT_MODEL=llama3.1:8b
```

#### Provider Selection
```bash
# Preferred provider (openai_compatible, openrouter, ollama, auto)
PREFERRED_PROVIDER=openai_compatible

# Enable automatic fallback if preferred provider fails
ENABLE_PROVIDER_FALLBACK=true
```

### Application Settings

```bash
# Application host and port
HOST=0.0.0.0
PORT=8000

# Environment mode (development, production, testing)
ENVIRONMENT=production

# Security
SECRET_KEY=your_long_random_secret_key_here
BEHIND_PROXY=true

# CORS settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://localhost:7860"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
STRUCTURED_LOGGING=true
```

### Caching Configuration

#### Redis Cache
```bash
REDIS_CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
CACHE_COMPRESSION=true
CACHE_BATCH_SIZE=10
REDIS_CACHE_MAX_CONNECTIONS=10
```

#### Memory Cache
```bash
MEMORY_CACHE_ENABLED=true
MEMORY_CACHE_SIZE=1000
MEMORY_CACHE_TTL=300
```

### Tool System Configuration

```bash
# Enable/disable tool system
TOOL_CALLING_ENABLED=true

# Tool execution settings
TOOL_EXECUTION_TIMEOUT=30
MAX_TOOLS_PER_QUERY=3
MAX_TOOL_ITERATIONS=10

# Individual tool settings
CALCULATOR_TOOL_ENABLED=true
TIME_TOOL_ENABLED=true
SEARXNG_TOOL_ENABLED=true
FIRECRAWL_TOOL_ENABLED=false  # Requires Docker setup
JINA_RERANKER_TOOL_ENABLED=false  # Requires Docker setup
```

### Agent System Configuration

```bash
# Enable/disable agent system
ENABLE_AGENT_SYSTEM=true

# Default agent
DEFAULT_AGENT=tool_agent

# Agent execution timeout (seconds)
AGENT_EXECUTION_TIMEOUT=60
```

### Search Integration

```bash
# SearXNG integration
SEARXNG_URL=http://localhost:8080
SEARXNG_TIMEOUT=10
ENABLE_WEB_SEARCH=true

# Firecrawl integration
FIRECRAWL_DEPLOYMENT_MODE=docker  # docker only
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
FIRECRAWL_BULL_AUTH_KEY=change_me_firecrawl
FIRECRAWL_SCRAPING_ENABLED=true
FIRECRAWL_SCRAPE_TIMEOUT=60

# Jina Reranker
JINA_RERANKER_ENABLED=false
JINA_RERANKER_URL=http://jina-reranker:8000
JINA_RERANKER_MODEL=jina-reranker-v1-base-en
```

### Monitoring Configuration

#### Prometheus Metrics
```bash
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics
METRICS_COLLECTION_ENABLED=true
```

#### Health Checks
```bash
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

### Web Interface

The application now uses Chainlit for the web interface instead of Gradio.

```bash
# Chainlit interface is enabled by default
# Access at http://localhost:8000/chainlit
```

### Multi-Writer System (Advanced)

```bash
# Enable multi-writer system (disabled by default)
MULTI_WRITER_ENABLED=false

# Required for multi-writer
MULTI_WRITER_FIRECRAWL_API_KEY=your-firecrawl-api-key
MULTI_WRITER_MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MULTI_WRITER_MONGODB_DATABASE_NAME=multi_writer_system

# Optional settings
MULTI_WRITER_QUALITY_THRESHOLD=70.0
MULTI_WRITER_MAX_ITERATIONS=2
MULTI_WRITER_MAX_CONCURRENT_WORKFLOWS=5
MULTI_WRITER_WORKFLOW_TIMEOUT=600
```

## Configuration Files

### Docker Configuration

For Docker deployment, copy and customize the `.env.docker` file:

```bash
cp .env.docker .env
# Edit .env with your specific configuration
```

### Local Development

For local development, copy and customize the `.env.template` file:

```bash
cp .env.template .env
# Edit .env with your specific configuration
```

### Environment File Templates

The project includes several environment templates:

- `.env.docker` - Docker deployment configuration
- `.env.template` - Local development template
- `.env.example` - Example configuration with all options

## Advanced Configuration

### Logging Configuration

```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Log format
LOG_FORMAT=json

# Log file location
LOG_FILE=./logs/app.log

# Structured logging
STRUCTURED_LOGGING=true
```

### Performance Tuning

```bash
# Connection pooling
HTTP_POOL_SIZE=100
HTTP_POOL_TIMEOUT=30
HTTP_POOL_MAX_CONNECTIONS=1000

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Batch processing
BATCH_SIZE=50
BATCH_TIMEOUT=5
```

### Security Configuration

```bash
# API security
API_KEY_VALIDATION=true
REQUEST_VALIDATION=true
INPUT_SANITIZATION=true

# CORS security
CORS_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST", "PUT", "DELETE"]

# Encryption
ENCRYPTION_ENABLED=false
ENCRYPTION_KEY=your_encryption_key_here
```

## Docker-Specific Configuration

### Docker Environment

When using Docker, some services have specific configuration:

```bash
# Service URLs (Docker networking)
REDIS_URL=redis://redis:6379/0
SEARXNG_URL=http://searxng:8080
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
JINA_RERANKER_URL=http://jina-reranker:8000

# Traefik configuration
TRAEFIK_ENABLED=true
TRAEFIK_DASHBOARD_PORT=8080
```

### Docker Compose Profiles

```bash
# Enable specific profiles
docker-compose --profile monitoring up -d
docker-compose --profile debug up -d
docker-compose --profile postgres up -d
```

## Environment-Specific Configuration

### Development Environment

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
```

### Production Environment

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
RELOAD=false
BEHIND_PROXY=true
```

### Testing Environment

```bash
ENVIRONMENT=testing
LOG_LEVEL=WARNING
CACHE_ENABLED=false
```

## Troubleshooting Configuration

### Common Issues

1. **API Key Not Working**
   - Verify the API key is correct and active
   - Check the base URL matches the provider
   - Ensure the API key has the required permissions

2. **Provider Connection Issues**
   - Check network connectivity to the provider
   - Verify the base URL and port
   - Check if any firewall is blocking the connection

3. **Cache Issues**
   - Verify Redis is running and accessible
   - Check Redis connection string format
   - Ensure Redis has sufficient memory

4. **Tool System Not Working**
   - Verify `ENABLE_TOOL_SYSTEM=true`
   - Check tool execution logs
   - Ensure tools are properly registered

## Configuration Validation

Use these commands to validate your configuration:

```bash
# Check application health
curl http://localhost:8000/health

# Check available tools
curl http://localhost:8000/v1/tools

# Check provider status
curl http://localhost:8000/v1/providers

# Check available models
curl http://localhost:8000/v1/models

# Test configuration
python -c "from app.core.config import settings; print(settings.dict())"

# Check Docker services
docker-compose ps
```

### Configuration Testing

```bash
# Run configuration tests
python -c "
from app.core.config import settings
from app.core.llm_providers import get_provider_manager

print('Configuration loaded successfully')
print(f'Preferred provider: {settings.PREFERRED_PROVIDER}')
print(f'Tool calling enabled: {settings.TOOL_CALLING_ENABLED}')

# Test provider
provider_manager = get_provider_manager()
print(f'Available providers: {list(provider_manager.providers.keys())}')
"
```

## Best Practices

1. **Use Environment Variables**: Never hardcode secrets in configuration files
2. **Separate Environments**: Use different configurations for development, testing, and production
3. **Regular Security Audits**: Regularly review and update security settings
4. **Monitoring**: Enable comprehensive monitoring for production deployments
5. **Backup Configuration**: Keep backups of working configurations
6. **Documentation**: Document any custom configuration changes

## Further Reading

- [Architecture Overview](architecture/overview.md) - System design and components
- [Development Setup](development/setup.md) - Development environment configuration
- [Production Deployment](deployment/production.md) - Production deployment
- [API Reference](api/endpoints.md) - API endpoint documentation
- [Multi-Provider Setup](providers/multi-provider.md) - Configure multiple LLM providers
- [Tool Configuration](tools/overview.md) - Configure individual tools
- [Secure Settings Guide](secure-settings-guide.md) - Security best practices