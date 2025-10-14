# Caching Architecture

The AI Assistant System includes a sophisticated multi-layer caching architecture designed to improve performance and reduce API costs.

## Overview

The caching system is built around several key components:
- **Cache Layers**: Multiple storage backends (memory, Redis)
- **Compression**: Reduces memory usage for cached content
- **Batching**: Groups similar operations for efficiency
- **Integration**: Seamless integration with agents and tools

## Cache Layers

### Memory Cache
The fastest cache layer, storing data in application memory.

```python
from app.core.caching.layers.memory import MemoryCache

cache = MemoryCache(max_size=1000, ttl=3600)
```

### Redis Cache
Distributed cache layer for multi-instance deployments.

```python
from app.core.caching.layers.redis_cache import RedisCache

cache = RedisCache(redis_url="redis://localhost:6379/0")
```

## Compression

The caching system includes intelligent compression to reduce memory usage:

```python
from app.core.caching.compression.compressor import CacheCompressor

compressor = CacheCompressor()
compressed_data = compressor.compress(large_data)
```

## Batching

Batch processing groups similar operations to reduce overhead:

```python
from app.core.caching.batching.batch_processor import BatchProcessor

processor = BatchProcessor(batch_size=10, timeout=5.0)
```

## Integration with Agents

The agent system includes automatic caching of responses:

```python
from app.core.caching.integration.agent_cache import AgentCache

agent_cache = AgentCache(cache_layer=redis_cache)
```

## Integration with Tools

Tool responses are automatically cached when enabled:

```python
from app.core.caching.integration.tool_cache import ToolCache

tool_cache = ToolCache(cache_layer=memory_cache)
```

## Configuration

Configure caching in your environment:

```env
CACHE_ENABLED=true
CACHE_TTL=3600
REDIS_URL=redis://localhost:6379/0
CACHE_COMPRESSION=true
```

## Monitoring

Monitor cache performance using the built-in metrics:

```python
from app.core.caching.monitoring.metrics import CacheMetrics

metrics = CacheMetrics()
hit_rate = metrics.get_hit_rate()
```

## Best Practices

1. **Choose appropriate TTL**: Set expiration times based on data volatility
2. **Use compression**: Enable compression for large cached objects
3. **Monitor hit rates**: Track cache effectiveness
4. **Layer appropriately**: Use memory cache for hot data, Redis for shared data