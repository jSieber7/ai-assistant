# Configuration Guide

This guide covers the configuration options available for the AI Assistant System.

## Environment Variables

The system can be configured using environment variables. Refer to the `.env.template` file for a complete list of available options.

## Core Settings

### LLM Provider Configuration

The system supports multiple LLM providers. Configure your preferred provider using the appropriate environment variables.

#### OpenAI
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

#### Ollama
```
OLLAMA_BASE_URL=http://localhost:11434
```

### Caching Configuration

#### Redis Cache
```
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
```

### Monitoring Configuration

#### Prometheus Metrics
```
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

## Advanced Configuration

For more advanced configuration options, see the [Architecture](architecture/overview.md) and [Development](development/setup.md) sections.