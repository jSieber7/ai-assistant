# Jina AI Reranker Integration

This document describes the Jina AI Reranker service integration in the AI Assistant system.

## Overview

The Jina AI Reranker is a powerful document reranking service that uses semantic similarity to reorder search results based on their relevance to a given query. This service is particularly useful for improving the quality of search results and content recommendations.

## Features

- **Semantic Reranking**: Uses advanced embedding models to understand semantic relevance
- **Multilingual Support**: Supports multiple languages with the `jina-reranker-v2-base-multilingual` model
- **Caching**: Built-in Redis caching to improve performance and reduce API calls
- **Monitoring**: Integrated with Prometheus for metrics collection
- **Health Checks**: Built-in health check endpoints
- **Rate Limiting**: Configurable rate limiting to manage API usage

## Architecture

The Jina Reranker service consists of:

1. **Docker Container**: A self-contained FastAPI service that proxies requests to Jina AI
2. **Integration Tool**: A Python tool that integrates with the AI Assistant's tool system
3. **Configuration**: Environment variables and YAML configuration files
4. **Monitoring**: Prometheus metrics and health checks

## Setup

### Prerequisites

1. **Jina AI API Key**: You need to obtain an API key from [Jina AI](https://jina.ai/)
2. **Redis**: The service uses Redis for caching
3. **Docker**: The service runs in a Docker container

### Configuration

1. Add your Jina AI API key to your `.env` file:
   ```env
   JINA_API_KEY=your-jina-api-key-here
   ```

2. Enable the Jina Reranker service:
   ```env
   JINA_RERANKER_ENABLED=true
   JINA_RERANKER_URL=http://jina-reranker:8080
   ```

3. Start the service with Docker Compose:
   ```bash
   docker compose --profile jina-reranker up
   ```

## Usage

### Direct API Usage

You can call the reranker service directly:

```bash
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "documents": [
      "Deep learning uses neural networks with multiple layers",
      "Gradient descent is an optimization algorithm",
      "Python is a popular programming language",
      "Adam optimizer combines momentum and RMSprop"
    ],
    "top_n": 3
  }'
```

### Integration with Tool System

The Jina Reranker is integrated as a tool in the AI Assistant system:

```python
from app.core.tools.jina_reranker_tool import JinaRerankerTool

# Create and use the tool
reranker = JinaRerankerTool()
result = await reranker.execute(
    query="machine learning algorithms",
    documents=[
        "Deep learning uses neural networks with multiple layers",
        "Gradient descent is an optimization algorithm",
        "Python is a popular programming language",
        "Adam optimizer combines momentum and RMSprop"
    ],
    top_n=3
)
```

### Reranking Search Results

The tool provides a convenience method for reranking search results:

```python
# Rerank search results
search_results = [
    {"content": "Document 1 content...", "url": "http://example.com/1"},
    {"content": "Document 2 content...", "url": "http://example.com/2"},
    # ... more results
]

reranked_results = await reranker.rerank_search_results(
    query="machine learning algorithms",
    search_results=search_results,
    top_n=5
)
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JINA_API_KEY` | - | Your Jina AI API key (required) |
| `JINA_RERANKER_ENABLED` | `false` | Enable/disable the reranker service |
| `JINA_RERANKER_URL` | `http://jina-reranker:8080` | Service URL |
| `JINA_RERANKER_MODEL` | `jina-reranker-v2-base-multilingual` | Model to use |
| `JINA_RERANKER_TIMEOUT` | `30` | Request timeout in seconds |
| `JINA_RERANKER_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `JINA_RERANKER_MAX_RETRIES` | `3` | Maximum retry attempts |

### Docker Configuration

The service can be configured using the `config/jina-reranker/config.yml` file:

```yaml
service:
  name: "jina-reranker"
  host: "0.0.0.0"
  port: 8080
  debug: false

jina_api:
  base_url: "https://api.jina.ai/v1/rerank"
  model: "jina-reranker-v2-base-multilingual"
  timeout: 30
  max_retries: 3
  retry_delay: 1

cache:
  enabled: true
  ttl: 3600
  max_size: 1000

rate_limiting:
  enabled: true
  requests_per_minute: 60
  burst_size: 10
```

## Monitoring

### Health Check

The service provides a health check endpoint:

```bash
curl http://localhost:8080/health
```

### Metrics

Prometheus metrics are available at `/metrics`:

- `jina_reranker_requests_total`: Total number of requests
- `jina_reranker_request_duration_seconds`: Request duration
- `jina_reranker_api_requests_total`: Jina API requests
- `jina_reranker_cache_hits_total`: Cache hits
- `jina_reranker_cache_misses_total`: Cache misses

### Logs

The service logs to `/app/logs/jina-reranker.log` and supports structured JSON logging.

## Docker Profiles

The Jina Reranker service can be started with different profiles:

- **Production**: `docker compose --profile production up`
- **Development**: `docker compose --profile dev up`
- **Jina Reranker only**: `docker compose --profile jina-reranker up`

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Jina AI API key is valid and has sufficient credits
2. **Connection Errors**: Check that the service is running and accessible
3. **Cache Issues**: Restart the Redis service if caching problems occur
4. **Timeout Errors**: Increase the timeout configuration if needed

### Debug Mode

Enable debug mode by setting `debug: true` in the configuration file or by using the development profile.

### Logs

Check the service logs for detailed error information:

```bash
docker logs ai-assistant-jina-reranker
```

## Performance Considerations

1. **Caching**: The service caches results to improve performance and reduce API costs
2. **Rate Limiting**: Configure appropriate rate limits to manage API usage
3. **Batch Processing**: Process multiple documents in a single request when possible
4. **Timeouts**: Set appropriate timeouts based on your use case

## Security

1. **API Key Security**: Store your Jina AI API key securely in environment variables
2. **Network Security**: The service is only accessible within the Docker network by default
3. **Input Validation**: The service validates all input parameters

## Integration Examples

### With Search Tools

```python
# Example: Enhancing search results with reranking
from app.core.tools.searxng_tool import SearxNGTool
from app.core.tools.jina_reranker_tool import JinaRerankerTool

# Perform search
search_tool = SearxNGTool()
search_results = await search_tool.execute(query="machine learning")

# Rerank results
reranker = JinaRerankerTool()
reranked_results = await reranker.rerank_search_results(
    query="machine learning",
    search_results=search_results.data["results"],
    top_n=10
)
```

### With Content Processing

```python
# Example: Reranking content for multi-writer system
from app.core.tools.jina_reranker_tool import JinaRerankerTool

# Content pieces from different writers
content_pieces = [
    {"content": "Content from writer 1...", "writer_id": "technical_1"},
    {"content": "Content from writer 2...", "writer_id": "creative_1"},
    # ... more content
]

reranker = JinaRerankerTool()
result = await reranker.execute(
    query="technical explanation of machine learning",
    documents=[piece["content"] for piece in content_pieces],
    top_n=3
)

# Use reranked results for further processing
```

## API Reference

### POST /rerank

Rerank documents based on semantic relevance to a query.

**Request Body:**
```json
{
  "query": "string",
  "documents": ["string"],
  "top_n": "integer (optional)",
  "model": "string (optional)"
}
```

**Response:**
```json
{
  "results": [
    {
      "index": "integer",
      "document": "string",
      "relevance_score": "float"
    }
  ],
  "model": "string",
  "query": "string",
  "total_documents": "integer",
  "cached": "boolean"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "jina-reranker"
}
```

### GET /metrics

Prometheus metrics endpoint (only available when monitoring is enabled).

## Support

For issues related to:
- **Jina AI API**: Contact Jina AI support
- **Docker Service**: Check the logs and configuration
- **Integration**: Refer to the AI Assistant documentation

## License

This integration follows the same license as the AI Assistant system.