# OpenAI-Compatible Provider Refactoring

## Overview

This document describes the refactoring of the OpenRouter-specific implementation to a generic OpenAI-compatible API provider. This change allows the system to work with any OpenAI-compatible API endpoint while maintaining full backward compatibility with existing OpenRouter configurations.

## Architecture Changes

### 1. Provider System Update

#### Before
- `ProviderType.OPENROUTER` - Hardcoded to OpenRouter
- `OpenRouterProvider` - Specific to OpenRouter API
- Fixed model list and configuration

#### After
- `ProviderType.OPENAI_COMPATIBLE` - Generic for any OpenAI-compatible API
- `OpenAICompatibleProvider` - Works with any OpenAI-compatible endpoint
- `OpenRouterProvider` - Backward compatibility wrapper
- Dynamic model discovery and flexible configuration

### 2. New Provider Classes

#### OpenAICompatibleProvider
The new generic provider class that can work with any OpenAI-compatible API:

```python
class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://openrouter.ai/api/v1",
        provider_name: str = None,
        custom_headers: Dict[str, str] = None
    )
```

**Key Features:**
- Automatic provider name detection from base URL
- Dynamic model discovery via `/models` endpoint
- Fallback model list for endpoints without model listing
- Custom headers support
- Flexible authentication (Bearer token, API key)
- Health checking via standard endpoints

#### OpenRouterProvider (Backward Compatibility)
Maintains existing behavior while using the new generic implementation:

```python
class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter LLM provider - backward compatibility wrapper"""
```

### 3. Configuration System

#### New OpenAISettings Class
```python
class OpenAISettings(BaseSettings):
    enabled: bool = True
    api_key: Optional[SecretStr] = None
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"
    provider_name: Optional[str] = None
    custom_headers: Dict[str, str] = {}
    timeout: int = 30
    max_retries: int = 3
```

#### Environment Variables
- **New (Recommended):** `OPENAI_COMPATIBLE_*` prefix
- **Backward Compatibility:** `OPENROUTER_*` variables still work

## Migration Guide

### For New Users (Recommended)

Use the new OpenAI-compatible configuration:

```bash
# Enable the provider
OPENAI_COMPATIBLE_ENABLED=true

# Configure API endpoint
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.com/api/v1

# Optional: Specify provider name (auto-detected if not set)
OPENAI_COMPATIBLE_PROVIDER_NAME=YourProvider

# Optional: Add custom headers
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"X-Custom-Header": "value"}
```

### For Existing OpenRouter Users

No changes required! Existing configurations continue to work:

```bash
# Existing configuration still works
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PREFERRED_PROVIDER=openrouter
```

### Migrating to Generic Configuration

To migrate from OpenRouter-specific to generic configuration:

1. **Update environment variables:**
   ```bash
   # From:
   OPENROUTER_API_KEY=your_key
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   
   # To:
   OPENAI_COMPATIBLE_API_KEY=your_key
   OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
   ```

2. **Update preferred provider:**
   ```bash
   # From:
   PREFERRED_PROVIDER=openrouter
   
   # To:
   PREFERRED_PROVIDER=openai_compatible
   ```

## Supported Providers

The generic provider works with any OpenAI-compatible API, including:

### OpenRouter
```bash
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_API_KEY=your_openrouter_key
```

### OpenAI
```bash
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_API_KEY=your_openai_key
```

### Together AI
```bash
OPENAI_COMPATIBLE_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPATIBLE_API_KEY=your_together_key
```

### Azure OpenAI
```bash
OPENAI_COMPATIBLE_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_COMPATIBLE_API_KEY=your_azure_key
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"api-key": "your_azure_key"}
```

### Custom Providers
Any provider that implements the OpenAI API specification:

```bash
OPENAI_COMPATIBLE_BASE_URL=https://your-custom-provider.com/v1
OPENAI_COMPATIBLE_API_KEY=your_api_key
OPENAI_COMPATIBLE_PROVIDER_NAME=Custom Provider
```

## API Changes

### Model Endpoint (`/v1/models`)
- Now supports models from any OpenAI-compatible provider
- Maintains backward compatibility for OpenRouter model naming
- Automatically handles different model response formats

### Provider Endpoint (`/v1/providers`)
- Shows both `openai_compatible` and `ollama` providers
- Displays actual provider names (e.g., "OpenRouter", "OpenAI", "Custom")

### Model Resolution
Enhanced model resolution supports:
- Provider-prefixed: `openai_compatible:model-name`
- Bare model names (uses default provider)
- Automatic provider detection

## Testing

### New Test Cases
Added comprehensive tests for the new provider:

- `TestOpenAICompatibleProvider` - Tests generic functionality
- `TestOpenRouterBackwardCompatibility` - Ensures backward compatibility
- Model listing with different API response formats
- Health checking with various endpoints
- Custom headers and authentication

### Running Tests
```bash
# Run all tests
pytest

# Run only provider tests
pytest tests/unit/test_ollama_integration.py::TestOpenAICompatibleProvider

# Run backward compatibility tests
pytest tests/unit/test_ollama_integration.py::TestOpenRouterBackwardCompatibility
```

## Implementation Details

### Provider Detection
The system automatically detects provider types from base URLs:

```python
def _detect_provider_name(self, base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return "OpenRouter"
    elif "api.openai.com" in base_url:
        return "OpenAI"
    elif "together.ai" in base_url:
        return "Together AI"
    elif "azure.com" in base_url:
        return "Azure OpenAI"
    else:
        return "OpenAI Compatible"
```

### Model Discovery
The provider attempts to discover models dynamically:

1. **Primary:** Fetch from `/models` endpoint
2. **Fallback:** Use predefined fallback models
3. **Error Handling:** Graceful degradation when API is unavailable

### Backward Compatibility
- Existing `OPENROUTER_*` environment variables continue to work
- Old `PREFERRED_PROVIDER=openrouter` setting is mapped to new provider
- OpenRouter-specific model naming is preserved in API responses
- Existing code using `ProviderType.OPENROUTER` continues to work

## Benefits

### For Users
- **Flexibility:** Use any OpenAI-compatible provider
- **Easy Migration:** No breaking changes for existing users
- **Better Performance:** Dynamic model discovery
- **Enhanced Features:** Custom headers, timeout configuration, retries

### For Developers
- **Extensible:** Easy to add new providers
- **Maintainable:** Single implementation for multiple providers
- **Testable:** Comprehensive test coverage
- **Clean Architecture:** Separation of concerns

## Troubleshooting

### Common Issues

#### Provider Not Recognized
```bash
# Solution: Manually specify provider name
OPENAI_COMPATIBLE_PROVIDER_NAME=YourProvider
```

#### Model Listing Fails
```bash
# The system will use fallback models automatically
# Check API key and base URL configuration
```

#### Authentication Issues
```bash
# For providers requiring custom headers
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"X-API-Key": "your_key"}
```

### Debug Logging
Enable debug logging to troubleshoot provider issues:

```bash
export LOG_LEVEL=DEBUG
```

## Future Enhancements

Planned improvements to the generic provider system:

1. **Automatic Model Capability Detection** - Detect streaming/tool support
2. **Provider-Specific Optimizations** - Tailored implementations per provider
3. **Load Balancing** - Distribute requests across multiple providers
4. **Cost Tracking** - Monitor API usage and costs
5. **Rate Limiting** - Handle provider-specific rate limits

## Conclusion

This refactoring provides a solid foundation for supporting multiple OpenAI-compatible providers while maintaining full backward compatibility. The new architecture is flexible, extensible, and ready for future enhancements.

Users can continue using their existing OpenRouter configurations without any changes, while new users can take advantage of the generic provider to work with any OpenAI-compatible API endpoint.