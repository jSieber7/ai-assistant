# Configuration Guide

This comprehensive guide covers all configuration options available for the AI Assistant System.

## Environment Variables

The system can be configured using environment variables. Refer to the `.env.template` file for a complete list of available options.

## Core Settings

### LLM Provider Configuration

The system supports multiple LLM providers with automatic fallback capabilities.

#### OpenAI-Compatible Provider (Recommended)
```bash
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

#### OpenAI
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

#### OpenRouter
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

#### Ollama (Local Models)
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2
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

# Environment mode (development, production)
ENVIRONMENT=production

# Security
SECRET_KEY=your_long_random_secret_key_here
BEHIND_PROXY=true

# CORS settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### Caching Configuration

#### Redis Cache
```bash
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
CACHE_COMPRESSION=true
CACHE_BATCH_SIZE=10
```

#### Memory Cache
```bash
MEMORY_CACHE_SIZE=1000
MEMORY_CACHE_TTL=300
```

### Tool System Configuration

```bash
# Enable/disable tool system
ENABLE_TOOL_SYSTEM=true

# Tool execution timeout (seconds)
TOOL_EXECUTION_TIMEOUT=30

# Maximum number of tool iterations
MAX_TOOL_ITERATIONS=10
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

# Web search enabled
ENABLE_WEB_SEARCH=true
```

### Monitoring Configuration

#### Prometheus Metrics
```bash
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics
```

#### Health Checks
```bash
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

### Gradio Interface

```bash
# Enable Gradio interface
GRADIO_ENABLED=true

# Gradio configuration
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_SHARE=false
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

## Advanced Configuration

### Multi-Writer System

For distributed deployments, configure the multi-writer system:

```bash
# Enable multi-writer mode
MULTI_WRITER_ENABLED=true

# Writer configuration
WRITER_ID=writer-1
WRITER_CLUSTER_SIZE=3

# Load balancer configuration
LOAD_BALANCER_STRATEGY=round_robin
```

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

## Security Configuration

### API Security

```bash
# API key validation
API_KEY_VALIDATION=true

# Request validation
REQUEST_VALIDATION=true

# Input sanitization
INPUT_SANITIZATION=true

# CORS security
CORS_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST", "PUT", "DELETE"]
```

### Encryption

```bash
# Enable encryption for sensitive data
ENCRYPTION_ENABLED=true
ENCRYPTION_KEY=your_encryption_key_here
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

### Configuration Validation

Use these commands to validate your configuration:

```bash
# Check application health
curl http://localhost:8000/health

# Check available tools
curl http://localhost:8000/v1/tools

# Check provider status
curl http://localhost:8000/v1/providers

# Test configuration
python -c "from app.core.config import settings; print(settings.dict())"
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
- [Deployment Guide](deployment/production.md) - Production deployment
- [API Reference](api/endpoints.md) - API endpoint documentation