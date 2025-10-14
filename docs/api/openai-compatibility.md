# OpenAI API Compatibility

This document details the OpenAI API compatibility layer of the AI Assistant system, including supported endpoints, request/response formats, and any extensions.

## 🎯 Overview

The AI Assistant provides **full compatibility** with the OpenAI API specification, allowing you to use it as a drop-in replacement for OpenAI's API in existing applications. This includes support for:

- Chat completions (streaming and non-streaming)
- Tool/function calling
- Model listing
- Error handling and response formats
- Standard HTTP status codes

## 📡 Supported Endpoints

### Chat Completions

**Endpoint**: `POST /v1/chat/completions`

Fully compatible with OpenAI's chat completions API with additional tool-calling capabilities.

#### Basic Request
```json
{
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "What is the capital of France?"
    }
  ]
}
```

#### Tool Calling Request
```json
{
  "model": "claude-3.5-sonnet",
  "messages": [
    {
      "role": "user",
      "content": "What is 15 * 24?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {
              "type": "string",
              "description": "Mathematical expression to evaluate"
            }
          },
          "required": ["expression"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

#### Streaming Request
```json
{
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Write a short story about AI."
    }
  ],
  "stream": true
}
```

### Models List

**Endpoint**: `GET /v1/models`

Returns available models from configured providers.

```bash
curl http://localhost:8000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai"
    },
    {
      "id": "claude-3.5-sonnet", 
      "object": "model",
      "created": 1677610602,
      "owned_by": "anthropic"
    }
  ]
}
```

## 🔄 Request/Response Format

### Standard Chat Completion

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model ID to use |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `temperature` | number | No | Sampling temperature (0-2) |
| `top_p` | number | No | Nucleus sampling parameter |
| `stream` | boolean | No | Enable streaming responses |
| `stop` | string/array | No | Stop sequences |
| `presence_penalty` | number | No | Presence penalty (-2 to 2) |
| `frequency_penalty` | number | No | Frequency penalty (-2 to 2) |
| `tools` | array | No | Available tools for the model |
| `tool_choice` | string/object | No | Tool usage strategy |

#### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  }
}
```

### Tool Calling Response

#### Tool Call Response
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "claude-3.5-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "calculator",
              "arguments": "{\"expression\": \"15 * 24\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 15,
    "total_tokens": 60,
    "tools_used": 1
  }
}
```

#### Tool Result Response
```json
{
  "id": "chatcmpl-124",
  "object": "chat.completion", 
  "created": 1677652290,
  "model": "claude-3.5-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "15 * 24 = 360. The result is 360."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 18,
    "total_tokens": 80,
    "tools_used": 1
  }
}
```

### Streaming Response

#### Server-Sent Events Format
```
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4-turbo", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4-turbo", "choices": [{"index": 0, "delta": {"content": "The"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4-turbo", "choices": [{"index": 0, "delta": {"content": " capital"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4-turbo", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

## 🛠️ Extended Features

### Enhanced Usage Information

The API provides extended usage information beyond the OpenAI standard:

```json
{
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 18,
    "total_tokens": 63,
    "tools_used": 1,
    "cache_hits": 2,
    "response_time_ms": 1250,
    "provider": "openrouter",
    "model_used": "anthropic/claude-3.5-sonnet"
  }
}
```

### Tool Management

Additional endpoints for tool management:

#### List Available Tools
```bash
GET /v1/tools
```

#### Get Tool Details
```bash
GET /v1/tools/{tool_name}
```

#### Execute Tool Directly
```bash
POST /v1/tools/{tool_name}/execute
```

### System Information

#### Health Check
```bash
GET /health
```

Extended health information:
```json
{
  "status": "healthy",
  "version": "0.3.2",
  "uptime_seconds": 3600,
  "providers": {
    "openrouter": {
      "status": "connected",
      "models_available": 15,
      "last_check": "2024-01-15T10:30:00Z"
    }
  },
  "tools": {
    "calculator": {"enabled": true, "status": "healthy"},
    "web_search": {"enabled": true, "status": "healthy"}
  }
}
```

## 🔧 Client Library Examples

### Python (OpenAI Library)

```python
from openai import OpenAI

# Point to your AI Assistant instance
client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# Simple chat
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)

# Tool calling
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "What is 15 * 24?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations"
        }
    }]
)

print(response.choices[0].message.tool_calls)
```

### JavaScript (OpenAI Library)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8000/v1',
  dangerouslyAllowBrowser: true // Only for development
});

async function chat() {
  const response = await client.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: [
      { role: 'user', content: 'Hello!' }
    ]
  });
  
  console.log(response.choices[0].message.content);
}

async function toolCalling() {
  const response = await client.chat.completions.create({
    model: 'claude-3.5-sonnet',
    messages: [
      { role: 'user', 'content': 'What is 15 * 24?' }
    ],
    tools: [{
      type: 'function',
      function: {
        name: 'calculator',
        description: 'Perform mathematical calculations'
      }
    }]
  });
  
  console.log(response.choices[0].message.tool_calls);
}
```

### cURL

```bash
# Simple chat
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Tool calling
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "claude-3.5-sonnet",
    "messages": [
      {"role": "user", "content": "What is 15 * 24?"}
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
  }'

# Streaming
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {"role": "user", "content": "Write a story"}
    ],
    "stream": true
  }'
```

## 🚨 Error Handling

### Standard HTTP Status Codes

| Status Code | Meaning | When Used |
|-------------|---------|-----------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Endpoint/model not found |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Provider down |

### Error Response Format

```json
{
  "error": {
    "message": "Invalid model: 'invalid-model'",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Common Error Scenarios

#### Invalid API Key
```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "invalid_request_error", 
    "code": "invalid_api_key"
  }
}
```

#### Model Not Available
```json
{
  "error": {
    "message": "Model 'gpt-5' not found",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

#### Tool Execution Error
```json
{
  "error": {
    "message": "Tool 'calculator' execution failed: Invalid expression",
    "type": "tool_execution_error",
    "tool": "calculator",
    "code": "execution_failed"
  }
}
```

## 🔄 Migration from OpenAI

### Step 1: Update Base URL
```python
# Before
client = OpenAI(api_key="sk-...")

# After  
client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)
```

### Step 2: Update Model Names
```python
# OpenAI models work as-is
model = "gpt-4-turbo"

# For other providers, use provider-prefixed names
model = "anthropic/claude-3.5-sonnet"  # OpenRouter
model = "meta-llama/llama-3.1-70b-instruct"  # OpenRouter
```

### Step 3: Enable Tools (Optional)
```python
# Add tool capabilities to existing requests
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations"
        }
    }]
)
```

### Step 4: Handle Enhanced Responses
```python
# Access extended usage information
usage = response.usage
print(f"Tools used: {usage.tools_used}")
print(f"Response time: {usage.response_time_ms}ms")
```

## 📊 Compatibility Notes

### Fully Compatible
- ✅ Chat completions API
- ✅ Streaming responses
- ✅ Tool/function calling
- ✅ Error response format
- ✅ Usage tracking
- ✅ Model listing

### Extended Features
- 🔄 Enhanced usage information (tools used, timing)
- 🔄 Tool management endpoints
- 🔄 Health check endpoints
- 🔄 Multi-provider support
- 🔄 Advanced caching

### Limitations
- ❌ Embeddings API (not implemented)
- ❌ Fine-tuning API (not implemented)
- ❌ Image generation (not implemented)
- ❌ Audio processing (not implemented)

## 🧪 Testing Compatibility

### OpenAI SDK Test
```python
from openai import OpenAI

def test_openai_compatibility():
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Test basic chat
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10
    )
    
    assert response.choices[0].message.content
    assert response.usage.total_tokens > 0
    
    print("✅ OpenAI compatibility test passed")

test_openai_compatibility()
```

This comprehensive compatibility layer ensures you can migrate existing OpenAI-based applications with minimal changes while gaining access to advanced features like multi-provider support and enhanced tool capabilities.