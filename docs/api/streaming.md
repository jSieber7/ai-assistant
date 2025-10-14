# Streaming and Batching API

This section describes the streaming and batching capabilities of the AI Assistant System API.

## Streaming Responses

The API supports streaming responses for long-running operations, allowing clients to receive partial results as they become available.

### Enable Streaming

To enable streaming, include the `stream` parameter in your request:

```http
POST /api/agent/chat?stream=true
```

**Request Body:**
```json
{
  "message": "Explain quantum computing",
  "agent_id": "default"
}
```

### Streaming Response Format

When streaming is enabled, the response is sent as a series of Server-Sent Events (SSE):

```
data: {"type": "start", "message_id": "msg_123"}

data: {"type": "token", "content": "Quantum"}

data: {"type": "token", "content": " computing"}

data: {"type": "token", "content": " is"}

...

data: {"type": "end", "message_id": "msg_123"}
```

### Event Types

- `start`: Indicates the beginning of a streamed response
- `token`: Contains a partial piece of the response
- `error`: Indicates an error occurred during processing
- `end`: Marks the completion of the response

### Client Implementation

#### JavaScript Example

```javascript
const response = await fetch('/api/agent/chat?stream=true', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'Explain quantum computing',
    agent_id: 'default'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      handleStreamEvent(data);
    }
  }
}
```

#### Python Example

```python
import requests
import json

response = requests.post(
    'http://localhost:8000/api/agent/chat?stream=true',
    json={
        'message': 'Explain quantum computing',
        'agent_id': 'default'
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            handle_stream_event(data)
```

## Batching API

The batching API allows you to process multiple requests in a single API call, improving efficiency for bulk operations.

### Create Batch Request

```http
POST /api/batch
```

**Request Body:**
```json
{
  "requests": [
    {
      "id": "req_1",
      "method": "POST",
      "path": "/api/agent/chat",
      "body": {
        "message": "What is Python?",
        "agent_id": "default"
      }
    },
    {
      "id": "req_2",
      "method": "POST",
      "path": "/api/tools/search/execute",
      "body": {
        "parameters": {
          "query": "machine learning"
        }
      }
    }
  ]
}
```

### Batch Response Format

```json
{
  "responses": [
    {
      "id": "req_1",
      "status": 200,
      "body": {
        "response": "Python is a high-level programming language..."
      }
    },
    {
      "id": "req_2",
      "status": 200,
      "body": {
        "result": "Search results for machine learning..."
      }
    }
  ],
  "total_processing_time": 2.34
}
```

## Configuration

### Streaming Configuration

Configure streaming behavior in your environment:

```env
STREAM_ENABLED=true
STREAM_CHUNK_SIZE=100
STREAM_TIMEOUT=300
```

### Batching Configuration

Configure batching behavior:

```env
BATCH_ENABLED=true
BATCH_MAX_SIZE=10
BATCH_TIMEOUT=30
```

## Performance Considerations

1. **Streaming**: Use streaming for long-running operations to provide better user experience
2. **Batching**: Group similar operations to reduce overhead
3. **Timeouts**: Set appropriate timeouts to prevent resource exhaustion
4. **Error Handling**: Implement proper error handling for both streaming and batch operations

## Error Handling

### Streaming Errors

Errors during streaming are sent as error events:

```
data: {"type": "error", "message": "Processing failed", "code": 500}
```

### Batching Errors

Individual requests in a batch can fail without affecting others:

```json
{
  "responses": [
    {
      "id": "req_1",
      "status": 200,
      "body": {...}
    },
    {
      "id": "req_2",
      "status": 400,
      "error": "Invalid parameters"
    }
  ]
}