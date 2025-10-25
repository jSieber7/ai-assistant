# LangChain API Reference

This document provides a comprehensive reference for all LangChain API endpoints in the AI Assistant application.

## Base URL

All LangChain API endpoints are prefixed with `/api/langchain/`.

## Authentication

All API endpoints require authentication. Include your API key in the request header:

```
Authorization: Bearer <your-api-key>
```

## Response Format

All responses follow a consistent format:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2023-01-01T00:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": { ... }
  },
  "timestamp": "2023-01-01T00:00:00Z"
}
```

## System Endpoints

### Health Check

Check the health of the LangChain system.

**Endpoint:** `GET /api/langchain/health`

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "components": {
      "llm_manager": {"status": "healthy"},
      "tool_registry": {"status": "healthy"},
      "agent_manager": {"status": "healthy"},
      "memory_manager": {"status": "healthy"},
      "monitoring": {"status": "healthy"}
    }
  }
}
```

### System Status

Get detailed system status and configuration.

**Endpoint:** `GET /api/langchain/status`

**Response:**
```json
{
  "success": true,
  "data": {
    "integration_mode": "langchain",
    "components": {
      "llm_manager": {"enabled": true, "status": "healthy"},
      "tool_registry": {"enabled": true, "status": "healthy"},
      "agent_manager": {"enabled": true, "status": "healthy"},
      "memory_manager": {"enabled": true, "status": "healthy"},
      "monitoring": {"enabled": true, "status": "healthy"}
    },
    "version": "1.0.0"
  }
}
```

## LLM Endpoints

### List LLM Providers

Get a list of available LLM providers and their models.

**Endpoint:** `GET /api/langchain/llm/providers`

**Response:**
```json
{
  "success": true,
  "data": {
    "providers": [
      {
        "name": "openai",
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "status": "available"
      },
      {
        "name": "ollama",
        "models": ["llama2", "mistral"],
        "status": "available"
      }
    ]
  }
}
```

### LLM Request

Send a request to an LLM.

**Endpoint:** `POST /api/langchain/llm/request`

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "prompt": "Hello, world!",
  "temperature": 0.7,
  "max_tokens": 100,
  "system_prompt": "You are a helpful assistant."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Hello! How can I help you today?",
    "model": "gpt-3.5-turbo",
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 9,
      "total_tokens": 19
    },
    "duration": 1.5
  }
}
```

### Batch LLM Request

Send multiple requests to an LLM in parallel.

**Endpoint:** `POST /api/langchain/llm/batch`

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "prompts": [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?"
  ],
  "temperature": 0.7,
  "max_tokens": 50
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "responses": [
      {"response": "The capital of France is Paris.", "usage": {...}},
      {"response": "The capital of Germany is Berlin.", "usage": {...}},
      {"response": "The capital of Italy is Rome.", "usage": {...}}
    ],
    "total_duration": 2.1
  }
}
```

## Tool Endpoints

### List Tools

Get a list of available tools.

**Endpoint:** `GET /api/langchain/tools`

**Response:**
```json
{
  "success": true,
  "data": {
    "tools": [
      {
        "name": "search_tool",
        "description": "Search the web for information",
        "parameters": {
          "query": {"type": "string", "required": true},
          "max_results": {"type": "integer", "default": 10}
        }
      },
      {
        "name": "scraper_tool",
        "description": "Scrape web pages",
        "parameters": {
          "url": {"type": "string", "required": true},
          "selector": {"type": "string", "required": false}
        }
      }
    ]
  }
}
```

### Execute Tool

Execute a tool with the given parameters.

**Endpoint:** `POST /api/langchain/tools/execute`

**Request Body:**
```json
{
  "tool_name": "search_tool",
  "parameters": {
    "query": "machine learning",
    "max_results": 5
  },
  "timeout": 30
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "result": {
      "success": true,
      "data": [
        {"title": "Introduction to Machine Learning", "url": "..."},
        {"title": "Machine Learning Algorithms", "url": "..."}
      ]
    },
    "execution_time": 2.5,
    "tool_name": "search_tool"
  }
}
```

### Register Custom Tool

Register a custom tool.

**Endpoint:** `POST /api/langchain/tools/register`

**Request Body:**
```json
{
  "name": "custom_calculator",
  "description": "Perform mathematical calculations",
  "function": "def calculator(expression: str) -> float: return eval(expression)",
  "parameters": {
    "expression": {"type": "string", "required": true}
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "tool_name": "custom_calculator",
    "status": "registered"
  }
}
```

## Agent Endpoints

### List Agents

Get a list of available agents.

**Endpoint:** `GET /api/langchain/agents`

**Response:**
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "name": "research_agent",
        "description": "Research agent for information gathering",
        "type": "workflow",
        "status": "available"
      },
      {
        "name": "writer_agent",
        "description": "Writing assistant agent",
        "type": "react",
        "status": "available"
      }
    ]
  }
}
```

### Invoke Agent

Invoke an agent with input data.

**Endpoint:** `POST /api/langchain/agents/invoke`

**Request Body:**
```json
{
  "agent_name": "research_agent",
  "input_data": {
    "messages": [
      {"role": "user", "content": "Research machine learning applications"}
    ]
  },
  "config": {
    "thread_id": "research_session_123",
    "timeout": 60
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "result": "Research completed successfully",
    "messages": [
      {"role": "user", "content": "Research machine learning applications"},
      {"role": "assistant", "content": "I'll research machine learning applications for you..."}
    ],
    "thread_id": "research_session_123",
    "execution_time": 5.2
  }
}
```

### Register Custom Agent

Register a custom agent.

**Endpoint:** `POST /api/langchain/agents/register`

**Request Body:**
```json
{
  "name": "custom_agent",
  "description": "Custom agent for specific tasks",
  "workflow": "def custom_workflow(state): return {...}",
  "config": {
    "timeout": 30,
    "checkpoint": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agent_name": "custom_agent",
    "status": "registered"
  }
}
```

## Memory Endpoints

### List Conversations

Get a list of conversations.

**Endpoint:** `GET /api/langchain/memory/conversations`

**Query Parameters:**
- `limit`: Maximum number of conversations to return (default: 50)
- `offset`: Offset for pagination (default: 0)
- `agent_name`: Filter by agent name (optional)

**Response:**
```json
{
  "success": true,
  "data": {
    "conversations": [
      {
        "conversation_id": "conv_123",
        "agent_name": "research_agent",
        "message_count": 5,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:05:00Z"
      }
    ],
    "total_count": 1
  }
}
```

### Get Conversation Messages

Get messages for a specific conversation.

**Endpoint:** `GET /api/langchain/memory/conversations/{conversation_id}/messages`

**Query Parameters:**
- `limit`: Maximum number of messages to return (default: 100)
- `offset`: Offset for pagination (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "messages": [
      {
        "message_id": "msg_123",
        "role": "user",
        "content": "Hello, world!",
        "timestamp": "2023-01-01T00:00:00Z"
      },
      {
        "message_id": "msg_124",
        "role": "assistant",
        "content": "Hi there! How can I help you?",
        "timestamp": "2023-01-01T00:00:01Z"
      }
    ],
    "total_count": 2
  }
}
```

### Create Conversation

Create a new conversation.

**Endpoint:** `POST /api/langchain/memory/conversations`

**Request Body:**
```json
{
  "conversation_id": "new_conversation_123",
  "agent_name": "research_agent",
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "conversation_id": "new_conversation_123",
    "agent_name": "research_agent",
    "created_at": "2023-01-01T00:00:00Z"
  }
}
```

### Add Message

Add a message to a conversation.

**Endpoint:** `POST /api/langchain/memory/conversations/{conversation_id}/messages`

**Request Body:**
```json
{
  "role": "user",
  "content": "What is machine learning?",
  "metadata": {
    "source": "web_interface"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message_id": "msg_125",
    "conversation_id": "new_conversation_123",
    "role": "user",
    "content": "What is machine learning?",
    "timestamp": "2023-01-01T00:01:00Z"
  }
}
```

## Monitoring Endpoints

### Get Metrics

Get system metrics.

**Endpoint:** `GET /api/langchain/monitoring/metrics`

**Query Parameters:**
- `component_type`: Filter by component type (optional)
- `component_name`: Filter by component name (optional)
- `time_range`: Time range for metrics (default: 1h)

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "llm_requests": {
        "count": 100,
        "success_rate": 0.95,
        "avg_duration": 1.5,
        "error_count": 5
      },
      "tool_executions": {
        "count": 50,
        "success_rate": 0.98,
        "avg_duration": 0.5,
        "error_count": 1
      },
      "agent_invocations": {
        "count": 25,
        "success_rate": 0.92,
        "avg_duration": 2.0,
        "error_count": 2
      }
    },
    "time_range": "1h"
  }
}
```

### Get Component Metrics

Get metrics for a specific component.

**Endpoint:** `GET /api/langchain/monitoring/metrics/{component_type}/{component_name}`

**Response:**
```json
{
  "success": true,
  "data": {
    "component_type": "llm",
    "component_name": "openai",
    "metrics": {
      "request_count": 50,
      "success_rate": 0.98,
      "avg_duration": 1.2,
      "error_count": 1,
      "last_updated": "2023-01-01T00:00:00Z"
    }
  }
}
```

### Get Performance Metrics

Get performance metrics.

**Endpoint:** `GET /api/langchain/monitoring/performance`

**Response:**
```json
{
  "success": true,
  "data": {
    "system_performance": {
      "avg_response_time": 1.5,
      "throughput": 10.0,
      "error_rate": 0.02,
      "uptime": 0.999
    },
    "component_performance": {
      "llm_manager": {"avg_response_time": 1.2},
      "tool_registry": {"avg_response_time": 0.5},
      "agent_manager": {"avg_response_time": 2.0},
      "memory_manager": {"avg_response_time": 0.3},
      "monitoring": {"avg_response_time": 0.1}
    }
  }
}
```

## Specialized Agent Endpoints

### Summarize Agent

Summarize text using the SummarizeAgent.

**Endpoint:** `POST /api/langchain/agents/specialized/summarize`

**Request Body:**
```json
{
  "text": "This is a long text that needs to be summarized...",
  "target_length": 100,
  "style": "informal"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": "This is a summary of the text...",
    "word_count": 20,
    "original_length": 500,
    "compression_ratio": 0.04
  }
}
```

### Search Query Agent

Generate optimized search queries using the SearchQueryAgent.

**Endpoint:** `POST /api/langchain/agents/specialized/search-query`

**Request Body:**
```json
{
  "topic": "machine learning applications",
  "context": "research for academic paper",
  "count": 3
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "queries": [
      "machine learning applications in healthcare",
      "real-world machine learning implementations",
      "industrial machine learning use cases"
    ],
    "primary_query": "machine learning applications",
    "keywords": ["machine", "learning", "applications", "implementation"]
  }
}
```

### Creative Story Agent

Generate creative stories using the CreativeStoryAgent.

**Endpoint:** `POST /api/langchain/agents/specialized/creative-story`

**Request Body:**
```json
{
  "prompt": "A magical forest with talking animals",
  "genre": "fantasy",
  "length": "short",
  "style": "whimsical"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "story": "Once upon a time, in a magical forest...",
    "genre": "fantasy",
    "word_count": 250,
    "characters": ["wise owl", "playful fox", "ancient tree"]
  }
}
```

### Fact Checker Agent

Check facts using the FactCheckerAgent.

**Endpoint:** `POST /api/langchain/agents/specialized/fact-checker`

**Request Body:**
```json
{
  "claim": "The Earth is round",
  "sources": ["https://nasa.gov/earth"],
  "strictness": "medium"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "verdict": "true",
    "confidence": 0.99,
    "sources": [
      {"url": "https://nasa.gov/earth", "reliability": "high"}
    ],
    "evidence": ["Scientific consensus confirms Earth's spherical shape"],
    "explanation": "The claim is accurate according to scientific evidence"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Invalid request format or parameters |
| `UNAUTHORIZED` | Authentication failed |
| `FORBIDDEN` | Access denied |
| `NOT_FOUND` | Resource not found |
| `COMPONENT_DISABLED` | Component is disabled |
| `TIMEOUT` | Request timed out |
| `RATE_LIMITED` | Too many requests |
| `INTERNAL_ERROR` | Internal server error |
| `LLM_ERROR` | LLM provider error |
| `TOOL_ERROR` | Tool execution error |
| `AGENT_ERROR` | Agent execution error |
| `MEMORY_ERROR` | Memory operation error |

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **LLM endpoints**: 100 requests per minute
- **Tool endpoints**: 200 requests per minute
- **Agent endpoints**: 50 requests per minute
- **Memory endpoints**: 300 requests per minute
- **Monitoring endpoints**: 60 requests per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## SDKs and Libraries

### Python SDK

```python
from ai_assistant_sdk import LangChainClient

client = LangChainClient(api_key="your-api-key")

# LLM request
response = client.llm.request(
    model="gpt-3.5-turbo",
    prompt="Hello, world!"
)

# Tool execution
result = client.tools.execute(
    tool_name="search_tool",
    parameters={"query": "machine learning"}
)

# Agent invocation
result = client.agents.invoke(
    agent_name="research_agent",
    input_data={"messages": [{"role": "user", "content": "Hello"}]}
)
```

### JavaScript SDK

```javascript
import { LangChainClient } from 'ai-assistant-sdk';

const client = new LangChainClient({ apiKey: 'your-api-key' });

// LLM request
const response = await client.llm.request({
  model: 'gpt-3.5-turbo',
  prompt: 'Hello, world!'
});

// Tool execution
const result = await client.tools.execute({
  toolName: 'search_tool',
  parameters: { query: 'machine learning' }
});

// Agent invocation
const result = await client.agents.invoke({
  agentName: 'research_agent',
  inputData: { messages: [{ role: 'user', content: 'Hello' }] }
});
```

## Webhooks

Configure webhooks to receive notifications about events:

### Webhook Events

- `llm.request.completed`
- `tool.execution.completed`
- `agent.invocation.completed`
- `conversation.created`
- `message.added`
- `system.alert`

### Webhook Configuration

**Endpoint:** `POST /api/langchain/webhooks`

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["llm.request.completed", "tool.execution.completed"],
  "secret": "webhook-secret"
}
```

**Webhook Payload:**
```json
{
  "event": "llm.request.completed",
  "data": {
    "model": "gpt-3.5-turbo",
    "duration": 1.5,
    "success": true
  },
  "timestamp": "2023-01-01T00:00:00Z"
}
```

## Support

For API support and questions:

- Documentation: [https://docs.ai-assistant.com](https://docs.ai-assistant.com)
- Status Page: [https://status.ai-assistant.com](https://status.ai-assistant.com)
- Support Email: support@ai-assistant.com
- Community Forum: [https://community.ai-assistant.com](https://community.ai-assistant.com)