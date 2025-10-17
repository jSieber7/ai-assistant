# Secure Settings Guide

This guide explains how to use the new secure settings system in the AI Assistant application, which replaces the need for .env file configuration with a user-friendly Gradio interface.

## Overview

The secure settings system provides:
- **Encrypted Storage**: All sensitive data (API keys, secrets) are encrypted and stored locally
- **Web-based Configuration**: Configure all settings through the Gradio interface
- **API Key Validation**: Test API keys before saving them
- **Import/Export**: Backup and restore settings
- **Security**: No API keys are exposed in plain text files

## Accessing Settings

1. Start the AI Assistant application
2. Open the Gradio interface (usually at `http://localhost:7860`)
3. Click on the **🔐 Settings** tab

## Configuration Sections

### 🤖 LLM Providers

Configure your language model providers:

#### OpenAI-Compatible Provider
- **Enable**: Toggle the provider on/off
- **API Key**: Enter your API key (e.g., OpenRouter, Anthropic, OpenAI)
- **Base URL**: API endpoint URL
- **Default Model**: Default model to use
- **Provider Name**: Optional custom provider name
- **Timeout & Retries**: Connection settings
- **Validate API Key**: Test your API key before saving

#### Ollama (Local Models)
- **Enable**: Toggle Ollama on/off
- **Base URL**: Ollama server URL (default: `http://localhost:11434`)
- **Default Model**: Default local model
- **Temperature**: Model creativity (0.0-2.0)
- **Max Tokens**: Response length limit
- **Streaming**: Enable streaming responses

### 🔌 External Services

Configure external services:

#### Firecrawl (Web Scraping)
- **Enable**: Toggle Firecrawl on/off
- **Docker URL**: Firecrawl API URL
- **Bull Auth Key**: Authentication key
- **Scraping Settings**: Concurrency and timeout limits

#### Jina AI Reranker
- **Enable**: Toggle reranker on/off
- **API Key**: Your Jina API key
- **URL**: Reranker service URL
- **Model**: Reranker model to use
- **Cache Settings**: Performance tuning options
- **Validate API Key**: Test your API key

#### SearXNG (Search Engine)
- **Secret Key**: SearXNG secret key
- **URL**: SearXNG service URL

### ⚙️ System Configuration

Basic system settings:

#### Feature Toggles
- **Tool System**: Enable/disable tool usage
- **Agent System**: Enable/disable agent functionality
- **Provider Fallback**: Allow fallback to other providers
- **Debug Mode**: Enable debug logging

#### Server Configuration
- **Preferred Provider**: Default LLM provider
- **Environment**: Development/Production mode
- **Host & Port**: Server binding settings
- **Secret Key**: Application secret key

### 📝 Multi-Writer System

Configure the multi-writer/checker system:
- **Enable**: Toggle multi-writer functionality
- **MongoDB Connection**: Database connection string
- **Database Name**: Target database name

### 📥 Import/Export

Backup and restore your settings:
- **Export Settings**: Download current configuration as JSON
- **Include Secrets**: Option to include API keys in export (use carefully!)
- **Import Settings**: Restore settings from JSON file

## Security Features

### Encryption
- All sensitive data is encrypted using Fernet symmetric encryption
- Keys are derived from machine-specific and user-specific data
- Settings are stored in `~/.ai_assistant/secure_settings.enc`

### API Key Protection
- API keys are never displayed in plain text after entry
- Validation occurs without exposing the full key
- Import/export with secrets is optional and clearly marked

## Migration from .env Files

If you're migrating from .env files:

1. Open the Settings tab in the Gradio interface
2. Navigate to each configuration section
3. Enter your API keys and settings
4. Use the validation buttons to test your configurations
5. Save each section
6. Optionally export your settings as a backup

## Troubleshooting

### Settings Not Loading
- Check that the `~/.ai_assistant` directory exists and is writable
- Ensure the cryptography package is installed (included in requirements)
- Check application logs for encryption errors

### API Key Validation Fails
- Verify the API key is correct and active
- Check network connectivity to the provider
- Ensure the base URL is correct for the provider

### Provider Not Working
- Make sure the provider is enabled in settings
- Check that the API key is validated and saved
- Verify the model name is correct for the provider
- Check system logs for error messages

## File Locations

- **Secure Settings**: `~/.ai_assistant/secure_settings.enc`
- **Logs**: Check application logs for detailed error information
- **Backups**: Use the Export feature to create backup files

## Best Practices

1. **Regular Backups**: Export your settings periodically, especially before major changes
2. **API Key Security**: Never share exported settings that include secrets
3. **Validation**: Always validate API keys after entering them
4. **Testing**: Use the Query Testing tab to verify your configuration works
5. **Monitoring**: Check the System Information tab for provider status

## Advanced Usage

### Programmatic Access

You can access settings programmatically:

```python
from app.core.secure_settings import secure_settings

# Get a setting
api_key = secure_settings.get_setting("llm_providers", "openai_compatible", {}).get("api_key")

# Update a setting
secure_settings.set_setting("system_config", "debug", True)

# Get all settings (masked for security)
all_settings = secure_settings.get_all_settings()
```

### Custom Configuration

The secure settings system can be extended to handle additional configuration categories. See the `SecureSettingsManager` class in `app/core/secure_settings.py` for details.