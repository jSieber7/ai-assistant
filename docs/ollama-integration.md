# Ollama Integration Guide

This guide covers how to set up and use Ollama models as an alternative to cloud-based providers in the AI Assistant system.

## Overview

The AI Assistant now supports multiple LLM providers through a flexible provider system. Ollama models can be used alongside OpenRouter models, with automatic fallback and health monitoring capabilities.

## Architecture

### Multi-Provider System

The system uses a provider registry pattern that supports:

- **OpenRouter Provider**: Cloud-based models (Claude, GPT-4, etc.)
- **Ollama Provider**: Local models running on Ollama server
- **Automatic Fallback**: Falls back to other providers if the preferred one fails
- **Health Monitoring**: Continuous health checks for all providers
- **Model Discovery**: Automatic detection of available models

### Provider Resolution

Models can be specified in several ways:

1. **Provider-prefixed**: `ollama:llama2` or `openrouter:anthropic/claude-3.5-sonnet`
2. **Default provider**: `llama2` (uses configured default provider)
3. **Auto-resolution**: System automatically finds the model across providers

## Installation and Setup

### Prerequisites

1. **Ollama Server**: Install and run Ollama locally
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   ```

2. **Pull Models**: Download desired models
   ```bash
   # Example models
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

3. **Install Dependencies**: Ensure langchain-community is installed
   ```bash
   pip install langchain-community
   ```

### Configuration

Update your `.env` file with Ollama settings:

```bash
# =============================================================================
# LLM Provider Configuration
# =============================================================================

# Preferred provider (openrouter, ollama, or auto)
PREFERRED_PROVIDER=ollama

# Enable fallback to other providers if preferred fails
ENABLE_FALLBACK=true

# =============================================================================
# Ollama Configuration
# =============================================================================

# Enable Ollama provider
OLLAMA_ENABLED=true

# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434

# Default model for Ollama
OLLAMA_DEFAULT_MODEL=llama2

# Connection settings
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=3

# Model settings
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=
OLLAMA_STREAMING=true

# Health check settings
OLLAMA_HEALTH_CHECK_INTERVAL=60
OLLAMA_AUTO_HEALTH_CHECK=true
```

## Usage Examples

### Basic Chat Completions

```python
import httpx

# Use Ollama model with provider prefix
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama:llama2",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)

# Use default model (configured in settings)
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "llama2",  # Will be resolved to Ollama if it's the default
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

### Model Discovery

```python
# List all available models from all providers
response = httpx.get("http://localhost:8000/v1/models")
models = response.json()

# List models from specific provider
response = httpx.get("http://localhost:8000/v1/providers/ollama/models")
ollama_models = response.json()

# List provider status
response = httpx.get("http://localhost:8000/v1/providers")
providers = response.json()
```

### Health Checks

```python
# Check all providers' health
response = httpx.post("http://localhost:8000/v1/providers/health-check")
health_status = response.json()
```

## Configuration Options

### Provider Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `PREFERRED_PROVIDER` | Default provider to use | `openrouter` |
| `ENABLE_FALLBACK` | Enable automatic fallback | `true` |

### Ollama Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `OLLAMA_ENABLED` | Enable Ollama provider | `true` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_DEFAULT_MODEL` | Default Ollama model | `llama2` |
| `OLLAMA_TIMEOUT` | Request timeout in seconds | `30` |
| `OLLAMA_TEMPERATURE` | Default temperature | `0.7` |
| `OLLAMA_STREAMING` | Enable streaming | `true` |
| `OLLAMA_HEALTH_CHECK_INTERVAL` | Health check interval | `60` |

## Supported Models

### Popular Ollama Models

- **Llama 2**: `llama2`
- **Code Llama**: `codellama`
- **Mistral**: `mistral`
- **Mixtral**: `mixtral`
- **Qwen**: `qwen`
- **Phi-2**: `phi`

### Model Capabilities

| Model | Context Length | Tool Support | Streaming |
|-------|----------------|--------------|-----------|
| All Ollama models | Varies | ❌ | ✅ |

Note: Ollama models currently don't support function calling/tool use, but this may change in future versions.

## API Endpoints

### Models

- `GET /v1/models` - List all available models
- `GET /v1/providers/{provider}/models` - List models for specific provider

### Providers

- `GET /v1/providers` - List all providers and their status
- `POST /v1/providers/health-check` - Perform health check on all providers

### Chat Completions

- `POST /v1/chat/completions` - Standard chat completions with provider support

## Error Handling

### Common Errors

1. **Ollama Server Not Running**
   ```
   Error: Connection refused to Ollama server
   Solution: Start Ollama server with `ollama serve`
   ```

2. **Model Not Found**
   ```
   Error: Model 'llama2' not found in Ollama
   Solution: Pull the model with `ollama pull llama2`
   ```

3. **Provider Not Configured**
   ```
   Error: Provider 'ollama' not configured
   Solution: Check OLLAMA_ENABLED and OLLAMA_BASE_URL settings
   ```

### Fallback Behavior

When `ENABLE_FALLBACK=true`, the system will:

1. Try the preferred provider first
2. If it fails, try other configured providers
3. Return the first successful response
4. Log the fallback attempt

## Performance Considerations

### Ollama vs Cloud Models

| Aspect | Ollama | Cloud Models |
|--------|--------|--------------|
| Latency | Low (local) | Variable (network) |
| Cost | Free (hardware) | Pay-per-use |
| Privacy | Full control | Third-party |
| Scalability | Limited by hardware | Unlimited |
| Model Quality | Good to excellent | State-of-the-art |

### Optimization Tips

1. **Model Selection**: Choose smaller models for faster responses
2. **Hardware**: Ensure sufficient RAM for model sizes
3. **Batching**: Use streaming for long responses
4. **Caching**: Enable caching for repeated queries

## Troubleshooting

### Health Check Issues

```python
# Check provider health manually
response = httpx.get("http://localhost:8000/v1/providers")
print(response.json())

# Check Ollama server directly
response = httpx.get("http://localhost:11434/api/tags")
print(response.json())
```

### Model Loading Issues

```bash
# Check available models in Ollama
ollama list

# Pull missing models
ollama pull <model_name>

# Check model details
ollama show <model_name>
```

### Configuration Issues

```python
# Verify configuration
from app.core.config import settings
from app.core.llm_providers import provider_registry

print(f"Preferred provider: {settings.preferred_provider}")
print(f"Ollama enabled: {settings.ollama_settings.enabled}")
print(f"Available providers: {[p.name for p in provider_registry.list_providers()]}")
```

## Advanced Usage

### Custom Provider Configuration

```python
from app.core.llm_providers import OllamaProvider, provider_registry

# Create custom Ollama provider
custom_ollama = OllamaProvider(
    base_url="http://custom-server:11434"
)

# Register provider
provider_registry.register_provider(custom_ollama)

# Set as default
provider_registry.set_default_provider(ProviderType.OLLAMA)
```

### Model-Specific Configuration

```python
# Use different settings for specific models
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama:codellama",
        "temperature": 0.1,  # Lower temperature for code
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": "Write a Python function"}]
    }
)
```

## Migration Guide

### From OpenRouter-Only

1. **Install Ollama**: Set up Ollama server and models
2. **Update Configuration**: Add Ollama settings to `.env`
3. **Test Integration**: Use provider-prefixed model names
4. **Enable Fallback**: Set `ENABLE_FALLBACK=true` for smooth transition
5. **Update Client Code**: Gradually migrate to Ollama models

### Client Code Changes

```python
# Before
model = "anthropic/claude-3.5-sonnet"

# After (explicit)
model = "openrouter:anthropic/claude-3.5-sonnet"

# After (using Ollama)
model = "ollama:llama2"

# After (let system decide)
model = "llama2"  # Uses default provider
```

## Best Practices

1. **Start Small**: Begin with smaller models like Llama 2 7B
2. **Monitor Resources**: Keep track of CPU/RAM usage
3. **Use Fallback**: Always enable fallback for production
4. **Test Thoroughly**: Test all model combinations before deployment
5. **Document Setup**: Keep configuration documented for team members

## Future Enhancements

- **GPU Support**: Enhanced GPU acceleration for Ollama
- **Model Management**: Automatic model downloading and updates
- **Load Balancing**: Distribute requests across multiple Ollama instances
- **Fine-tuning**: Support for custom fine-tuned models
- **Tool Calling**: Native tool calling support in Ollama models

## Support

For issues with Ollama integration:

1. Check [Ollama Documentation](https://github.com/ollama/ollama)
2. Review system logs for error details
3. Test Ollama server independently
4. Check GitHub issues for known problems
5. Report issues with detailed configuration information