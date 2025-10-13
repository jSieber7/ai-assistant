# API Endpoints Reference

This document describes the FastAPI endpoints available in the AI Assistant application.

## Overview

The AI Assistant provides an OpenAI-compatible API interface that can be used with various LLM frontends. All endpoints are designed to be compatible with the OpenAI API specification.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local development. For production deployments, consider adding API key authentication.

## Available Endpoints

### GET /health
**Health check endpoint**

Returns the current status of the service.

**Response:**
```json
{
  "status": "healthy",
  "service": "langchain-agent-hub"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### GET /v1/models
**List available models**

Returns a list of available models. Currently supports the LangChain agent hub model.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "langchain-agent-hub",
      "object": "model",
      "created": 1677610602,
      "owned_by": "langchain-agent-hub",
      "permission": [],
      "root": "langchain-agent-hub",
      "parent": null
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/v1/models
```

### Chat Completions (OpenAI-Compatible)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "anthropic/claude-3.5-sonnet",
  "messages": [
    {"role": "user", "content": "What's 15 * 25?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations"
      }
    }
  ]
}
```

**Parameters:**
- `messages` (array): Array of message objects with role and content
- `model` (string, optional): Model to use (default: "anthropic/claude-3.5-sonnet")
- `stream` (boolean): Whether to stream the response (default: false)
- `temperature` (number): Sampling temperature (default: 0.7)
- `max_tokens` (number, optional): Maximum tokens to generate
- `tools` (array, optional): Array of tool definitions for tool calling

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "model": "anthropic/claude-3.5-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

**Streaming Response:**
The streaming response uses Server-Sent Events (SSE) format with `text/plain` content type.

**Example (non-streaming):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "model": "anthropic/claude-3.5-sonnet"
  }'
```

### Tool Management
```http
GET /api/v1/tools
# List all registered tools

POST /api/v1/tools/execute
# Execute a specific tool

GET /api/v1/tools/{tool_name}/stats
# Get tool usage statistics
```

### Monitoring Endpoints
```http
GET /health
# System health check

GET /metrics
# Performance metrics

GET /api/v1/monitoring/stats
# Detailed system statistics
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (malformed request)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

**Error Response Example:**
```json
{
  "detail": "OPENROUTER_API_KEY is not set in the environment"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

## CORS Support

The API supports CORS (Cross-Origin Resource Sharing) for web applications. The following origins are allowed by default:

- `http://localhost:3000`
- `http://localhost:8080`

## Testing the API

You can test the API using the provided test suite:

```bash
# Run unit tests
python run_tests.py --unit

# Run integration tests (starts a test server)
python run_tests.py --integration
```

## OpenAPI Documentation

The API includes auto-generated OpenAPI documentation available at:

```
http://localhost:8000/docs
```

This interactive documentation allows you to test endpoints directly from the browser.

### Complete API Documentation
See the full API documentation at `http://localhost:8000/docs` when the server is running.

## Related Documentation

- [Core Components](../architecture/core-components.md)
- [Development Guide](../development/development-guide.md)
- [Architecture Overview](../architecture/overview.md)