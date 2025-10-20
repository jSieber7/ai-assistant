# Chainlit Interface

This document describes the Chainlit interface for the AI Assistant application. Chainlit provides a modern, conversational UI for interacting with AI models.

## Overview

The Chainlit interface replaces the Gradio interface with a more conversational, chat-based experience. It provides:

1. Interactive chat interface with message history
2. Dynamic provider and model selection
3. Support for adding new providers
4. Command system for configuration and help

## Running the Chainlit Interface

### Method 1: Using Make

```bash
make chainlit
```

### Method 2: Direct Command

```bash
uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8001
```

The interface will be available at `http://localhost:8001`

### Method 3: Docker with Traefik

#### Development
```bash
# Start all services with Chainlit
make dev-docker-chainlit

# Or using docker-compose directly
docker compose --profile dev up -d chainlit
```

#### Production
```bash
# Start production with Chainlit
make prod-chainlit

# Or using docker-compose directly
docker compose --profile production up -d chainlit-prod
```

When using Docker with Traefik, the Chainlit interface will be available at:
- Development: `http://localhost/chat`
- Production: `http://your-domain/chat`

## Features

### Provider Selection

When you start the Chainlit interface, you'll be prompted to select a provider:

1. Choose from existing configured providers
2. Select "Add Provider" to configure a new provider

### Model Selection

After selecting a provider, you'll see a list of available models for that provider. Select a model to continue.

### Chat Interface

Once a provider and model are selected, you can:

- Send messages to the AI assistant
- View conversation history
- Use commands to manage settings

### Commands

The Chainlit interface supports several commands:

- `/settings` - View current configuration
- `/reset` - Reset configuration and select new provider
- `/help` - Show help information

### Adding New Providers

To add a new provider:

1. Select "Add Provider" from the provider list
2. Enter the provider name
3. Select the provider type (openai_compatible, ollama)
4. Provide API key (if required)
5. Enter base URL (optional)

## Configuration

The Chainlit interface is configured via `.chainlit/config.toml`. Key settings include:

- `name` - Application name
- `layout` - UI layout (wide or default)
- `cache` - Enable/disable caching

## Architecture

The Chainlit interface is implemented in:

- `app/ui/chainlit_app.py` - Main Chainlit application
- `chainlit_app.py` - Entry point for running the interface
- `.chainlit/config.toml` - Configuration file

## Testing

To test the Chainlit interface:

```bash
uv run python test_chainlit_basic.py
```

This will verify that all required modules can be imported and the configuration is correct.

## Comparison with Gradio (Deprecated)

| Feature | Chainlit | Gradio (Deprecated) |
|---------|----------|--------|
| UI Style | Conversational chat | Form-based |
| Message History | Built-in | Limited |
| Real-time Updates | Yes | Limited |
| Mobile Support | Excellent | Good |
| Customization | High | Medium |

## Docker Integration

The Chainlit interface is fully integrated with Docker and Traefik for both development and production environments.

### Development Configuration

When running with `make dev-docker-chainlit`:
- Chainlit runs in a separate container with hot reload
- Traefik routes `/chat` to the Chainlit service
- Logs are shared with the host system
- Source code is mounted for live updates

### Production Configuration

When running with `make prod-chainlit`:
- Chainlit runs in an optimized production container
- Traefik applies security middleware (rate limiting, compression)
- Health checks ensure service reliability
- Logs are collected in persistent volumes

### Environment Variables

The Chainlit service uses the same environment variables as the main application:
- `ENVIRONMENT` - Set to `development` or `production`
- `HOST` - Bind address (default: 0.0.0.0)
- `PORT` - Port for Chainlit (default: 8001)
- `REDIS_URL` - Redis connection string
- `SEARXNG_URL` - SearXNG service URL

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change the port using the `--port` flag
   - Stop other services using the same port
   - When using Docker, check if port 8001 is exposed

2. **Provider connection errors**
   - Check API keys in environment variables
   - Verify network connectivity
   - Ensure provider URLs are correct

3. **Model list empty**
   - Provider may not be configured correctly
   - Check authentication credentials
   - Verify provider API endpoints

4. **Docker Issues**
   - Ensure the container builds successfully: `docker compose build chainlit`
   - Check container logs: `docker compose logs chainlit`
   - Verify health check status: `docker compose ps`

### Debug Mode

To run Chainlit in debug mode:

```bash
uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8001 --debug
```

## Future Enhancements

Potential improvements to the Chainlit interface:

1. File upload support
2. Streaming responses
3. Conversation export/import
4. Custom themes
5. Plugin system for extensions