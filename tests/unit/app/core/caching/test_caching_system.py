"""
Unit tests for the caching system as a whole.

Tests for the integration of all caching components together.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.caching.base import (
    CacheBackend,
    CacheLayer,
    MultiLayerCache,
    generate_cache_key,
    CacheOperation,
    CacheEvent,
    CacheMetrics,
    PerformanceMonitor,
)
from app.core.caching.layers.memory import MemoryCache
from app.core.caching.layers.redis_cache import RedisCache
from app.core.caching.compression.compressor import (
    ResultCompressor,
    CompressedCacheBackend,
    CompressionAlgorithm,
)
from app.core.caching.batching.batch_processor import (
    BatchProcessor,
    BatchProcessorManager,
)


class TestCachingSystemIntegration:
    """Test cases for the integration of caching components."""

    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache."""
        return MemoryCache(
            name="test_memory",
            max_size=100,
            default_ttl=60,
            cleanup_interval=0,
        )

    @pytest.fixture
    def redis_cache(self):
        """Create a mock Redis cache."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock Redis client
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.set.return_value = True
            mock_client.delete.return_value = 1
            mock_client.exists.return_value = 0
            mock_client.keys.return_value = []
            mock_client.flushdb.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            cache = RedisCache(
                name="test_redis",
                host="localhost",
                port=6379,
                db=0,
                password=None,
                default_ttl=60,
            )
            
            # Manually set up connection
            cache._client = mock_client
            cache._connected = True
            
            return cache

    @pytest.fixture
    def multi_layer_cache(self, memory_cache, redis_cache):
        """Create a multi-layer cache with memory and Redis layers."""
        cache = MultiLayerCache()
        
        # Add memory layer (higher priority)
        memory_layer = CacheLayer(
            backend=memory_cache,
            name="memory",
            priority=0,
            read_only=False,
            default_ttl=60,
        )
        cache.add_layer(memory_layer)
        
        # Add Redis layer (lower priority)
        redis_layer = CacheLayer(
            backend=redis_cache,
            name="redis",
            priority=1,
            read_only=False,
            default_ttl=120,
        )
        cache.add_layer(redis_layer)
        
        return cache

    @pytest.mark.asyncio
    async def test_multi_layer_cache_hit_in_first_layer(self, multi_layer_cache):
        """Test cache hit in the first (highest priority) layer."""
        # Set a value in the first layer (memory)
        await multi_layer_cache.layers[0].backend.set("test_key", "memory_value")
        
        # Get the value
        result = await multi_layer_cache.get("test_key")
        
        # Should get the value from memory
        assert result == "memory_value"

    @pytest.mark.asyncio
    async def test_multi_layer_cache_hit_in_second_layer(self, multi_layer_cache):
        """Test cache hit in the second (lower priority) layer."""
        # Set a value in the second layer (Redis)
        await multi_layer_cache.layers[1].backend.set("test_key", "redis_value")
        
        # Get the value
        result = await multi_layer_cache.get("test_key")
        
        # Should get the value from Redis
        assert result == "redis_value"

    @pytest.mark.asyncio
    async def test_multi_layer_cache_read_through(self, multi_layer_cache):
        """Test read-through functionality."""
        # Enable read-through
        multi_layer_cache.set_read_through(True)
        
        # Set a value in the second layer (Redis)
        await multi_layer_cache.layers[1].backend.set("test_key", "redis_value")
        
        # Get the value (should populate first layer)
        result = await multi_layer_cache.get("test_key")
        
        # Should get the value from Redis
        assert result == "redis_value"
        
        # Check that first layer now has the value
        first_layer_value = await multi_layer_cache.layers[0].backend.get("test_key")
        assert first_layer_value == "redis_value"

    @pytest.mark.asyncio
    async def test_multi_layer_cache_write_through(self, multi_layer_cache):
        """Test write-through functionality."""
        # Enable write-through
        multi_layer_cache.set_write_through(True)
        
        # Set a value
        result = await multi_layer_cache.set("test_key", "test_value")
        
        # Should succeed
        assert result is True
        
        # Check that both layers have the value
        first_layer_value = await multi_layer_cache.layers[0].backend.get("test_key")
        second_layer_value = await multi_layer_cache.layers[1].backend.get("test_key")
        
        assert first_layer_value == "test_value"
        assert second_layer_value == "test_value"

    @pytest.mark.asyncio
    async def test_multi_layer_cache_write_to_highest_priority_only(self, multi_layer_cache):
        """Test writing to only the highest priority layer."""
        # Disable write-through
        multi_layer_cache.set_write_through(False)
        
        # Set a value
        result = await multi_layer_cache.set("test_key", "test_value")
        
        # Should succeed
        assert result is True
        
        # Check that only the first layer has the value
        first_layer_value = await multi_layer_cache.layers[0].backend.get("test_key")
        second_layer_value = await multi_layer_cache.layers[1].backend.get("test_key")
        
        assert first_layer_value == "test_value"
        assert second_layer_value is None

    @pytest.mark.asyncio
    async def test_multi_layer_cache_delete(self, multi_layer_cache):
        """Test deleting a key from all layers."""
        # Set a value in both layers
        await multi_layer_cache.layers[0].backend.set("test_key", "memory_value")
        await multi_layer_cache.layers[1].backend.set("test_key", "redis_value")
        
        # Delete the key
        result = await multi_layer_cache.delete("test_key")
        
        # Should succeed
        assert result is True
        
        # Check that both layers no longer have the value
        first_layer_exists = await multi_layer_cache.layers[0].backend.exists("test_key")
        second_layer_exists = await multi_layer_cache.layers[1].backend.exists("test_key")
        
        assert first_layer_exists is False
        assert second_layer_exists is False

    @pytest.mark.asyncio
    async def test_multi_layer_cache_exists(self, multi_layer_cache):
        """Test checking if a key exists in any layer."""
        # Set a value in the second layer only
        await multi_layer_cache.layers[1].backend.set("test_key", "redis_value")
        
        # Check if key exists
        result = await multi_layer_cache.exists("test_key")
        
        # Should return True
        assert result is True

    @pytest.mark.asyncio
    async def test_multi_layer_cache_clear(self, multi_layer_cache):
        """Test clearing all layers."""
        # Set some values in both layers
        await multi_layer_cache.layers[0].backend.set("key1", "value1")
        await multi_layer_cache.layers[1].backend.set("key2", "value2")
        
        # Clear all layers
        result = await multi_layer_cache.clear()
        
        # Should succeed
        assert result is True
        
        # Check that both layers are cleared
        first_layer_keys = await multi_layer_cache.layers[0].backend.keys("*")
        second_layer_keys = await multi_layer_cache.layers[1].backend.keys("*")
        
        assert len(first_layer_keys) == 0
        assert len(second_layer_keys) == 0

    @pytest.mark.asyncio
    async def test_multi_layer_cache_keys(self, multi_layer_cache):
        """Test getting keys from all layers."""
        # Set some values in both layers
        await multi_layer_cache.layers[0].backend.set("key1", "value1")
        await multi_layer_cache.layers[1].backend.set("key2", "value2")
        await multi_layer_cache.layers[1].backend.set("key1", "value3")  # Duplicate
        
        # Get all keys
        keys = await multi_layer_cache.keys("*")
        
        # Should return unique keys
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.asyncio
    async def test_multi_layer_cache_disable_layer(self, multi_layer_cache):
        """Test disabling a layer."""
        # Disable the first layer
        multi_layer_cache.disable_layer("memory")
        
        # Set a value
        result = await multi_layer_cache.set("test_key", "test_value")
        
        # Should succeed
        assert result is True
        
        # Check that only the second layer has the value
        first_layer_value = await multi_layer_cache.layers[0].backend.get("test_key")
        second_layer_value = await multi_layer_cache.layers[1].backend.get("test_key")
        
        assert first_layer_value is None  # Layer is disabled
        assert second_layer_value == "test_value"

    @pytest.mark.asyncio
    async def test_multi_layer_cache_enable_layer(self, multi_layer_cache):
        """Test enabling a layer."""
        # Disable the first layer
        multi_layer_cache.disable_layer("memory")
        
        # Enable it again
        result = multi_layer_cache.enable_layer("memory")
        
        # Should succeed
        assert result is True
        
        # Check that the layer is enabled
        assert multi_layer_cache.layers[0].enabled is True

    @pytest.mark.asyncio
    async def test_compressed_cache_backend_integration(self, memory_cache):
        """Test integrating compression with cache backend."""
        # Create a compressor
        compressor = ResultCompressor(
            default_algorithm=CompressionAlgorithm.GZIP,
            min_size_for_compression=10,
        )
        
        # Create a compressed cache backend
        compressed_backend = CompressedCacheBackend(
            backend=memory_cache,
            compressor=compressor,
            compress_threshold=10,
        )
        
        # Set a small value (should not be compressed)
        result = await compressed_backend.set("small_key", "small")
        assert result is True
        
        # Get the small value
        value = await compressed_backend.get("small_key")
        assert value == "small"
        
        # Set a large value (should be compressed)
        large_value = "x" * 100
        result = await compressed_backend.set("large_key", large_value)
        assert result is True
        
        # Get the large value
        value = await compressed_backend.get("large_key")
        assert value == large_value

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self, memory_cache):
        """Test integrating batch processor with cache."""
        # Create a batch processor that uses the cache
        cache_results = {}
        
        def process_batch(requests):
            results = []
            for request in requests:
                key = f"batch_{request}"
                
                # Check cache first
                if key in cache_results:
                    results.append(cache_results[key])
                else:
                    # Process and cache the result
                    result = f"processed_{request}"
                    cache_results[key] = result
                    results.append(result)
            
            return results
        
        processor = BatchProcessor(
            name="cache_processor",
            process_batch_fn=process_batch,
            max_batch_size=3,
            max_wait_time=0.1,
        )
        
        await processor.start()
        
        try:
            # Submit first batch
            result1 = await processor.submit_request("item1")
            assert result1.result == "processed_item1"
            
            # Submit second batch with one cached item
            result2 = await processor.submit_request("item1")  # Should be cached
            result3 = await processor.submit_request("item2")
            
            assert result2.result == "processed_item1"  # From cache
            assert result3.result == "processed_item2"
            
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation with various inputs."""
        # Test with prefix only
        key = generate_cache_key("test_prefix")
        assert key == "test_prefix"
        
        # Test with prefix and args
        key = generate_cache_key("test_prefix", "arg1", "arg2")
        assert key == "test_prefix:arg1:arg2"
        
        # Test with prefix and kwargs
        key = generate_cache_key("test_prefix", key1="value1", key2="value2")
        assert key == "test_prefix:key1=value1:key2=value2"
        
        # Test with prefix, args, and kwargs
        key = generate_cache_key("test_prefix", "arg1", key1="value1", key2="value2")
        assert key == "test_prefix:arg1:key1=value1:key2=value2"
        
        # Test with special characters
        key = generate_cache_key("test prefix", "arg/with/slashes", "arg with spaces")
        assert key == "test_prefix:arg/with/slashes:arg_with_spaces"
        
        # Test with long key (should be hashed)
        long_args = ["x" * 100] * 3
        key = generate_cache_key("test_prefix", *long_args)
        assert len(key) < 200
        assert key.startswith("test_prefix:")
        assert ":" not in key[12:]  # Only prefix and one colon for hash

    @pytest.mark.asyncio
    async def test_cache_operation_tracking(self):
        """Test cache operation tracking."""
        # Create a cache operation
        operation = CacheOperation("get", "test_key")
        
        # Complete the operation successfully
        operation.complete(True)
        
        # Check the operation details
        assert operation.operation.value == "get"
        assert operation.key == "test_key"
        assert operation.success is True
        assert operation.error is None
        assert operation.duration is not None
        assert operation.duration > 0

    @pytest.mark.asyncio
    async def test_cache_event_creation(self):
        """Test cache event creation."""
        # Create a cache operation
        operation = CacheOperation("get", "test_key")
        operation.complete(True)
        
        # Create a cache event
        event = CacheEvent(
            event_type="cache_hit",
            operation=operation,
            layer_name="memory",
            metadata={"hit_rate": 0.8},
        )
        
        # Check the event details
        assert event.event_type == "cache_hit"
        assert event.operation == operation
        assert event.layer_name == "memory"
        assert event.metadata == {"hit_rate": 0.8}
        assert event.timestamp > 0
        
        # Convert to dictionary
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "cache_hit"
        assert event_dict["operation"] == "get"
        assert event_dict["key"] == "test_key"
        assert event_dict["layer_name"] == "memory"
        assert event_dict["success"] is True
        assert event_dict["duration"] is not None
        assert event_dict["metadata"] == {"hit_rate": 0.8}

    @pytest.mark.asyncio
    async def test_cache_metrics_tracking(self):
        """Test cache metrics tracking."""
        # Create cache metrics
        metrics = CacheMetrics()
        
        # Record some operations
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        metrics.record_delete()
        metrics.record_error()
        
        # Check the metrics
        assert metrics.hits == 2
        assert metrics.misses == 1
        assert metrics.sets == 1
        assert metrics.deletes == 1
        assert metrics.errors == 1
        
        # Check derived metrics
        assert metrics.hit_rate() == 2/3
        assert metrics.total_operations() == 6
        
        # Get stats as dictionary
        stats = metrics.get_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == 2/3
        assert stats["total_cache_operations"] == 6

    @pytest.mark.asyncio
    async def test_performance_monitor_integration(self, memory_cache):
        """Test performance monitor integration with cache."""
        # Create a performance monitor
        monitor = PerformanceMonitor()
        
        # Start the monitor
        await monitor.start()
        
        try:
            # Create and record some operations
            for i in range(10):
                operation = CacheOperation("get", f"key{i}")
                
                # Simulate some processing time
                time.sleep(0.001)
                
                # Complete the operation
                operation.complete(True)
                
                # Record the operation
                monitor.record_operation(operation)
            
            # Record some hits and misses
            for i in range(7):
                monitor.record_hit()
            for i in range(3):
                monitor.record_miss()
            
            # Get statistics
            stats = await monitor.get_stats()
            
            # Check the statistics
            assert stats["enabled"] is True
            assert stats["total_response_time_samples"] == 10
            assert stats["average_response_time"] > 0
            assert stats["p95_response_time"] > 0
            assert stats["p99_response_time"] > 0
            
            # Check metrics
            metrics = stats["cache_hits"]
            assert metrics == 7
            
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_batch_processor_manager_integration(self, memory_cache):
        """Test batch processor manager integration."""
        # Create a manager
        manager = BatchProcessorManager()
        
        # Create a batch processor that uses the cache
        cache_results = {}
        
        def process_batch(requests):
            results = []
            for request in requests:
                key = f"batch_{request}"
                
                # Check cache first
                if key in cache_results:
                    results.append(cache_results[key])
                else:
                    # Process and cache the result
                    result = f"processed_{request}"
                    cache_results[key] = result
                    results.append(result)
            
            return results
        
        processor = BatchProcessor(
            name="cache_processor",
            process_batch_fn=process_batch,
            max_batch_size=3,
            max_wait_time=0.1,
        )
        
        # Register the processor
        await manager.register_processor("cache_processor", processor)
        
        try:
            # Submit a request
            result = await manager.submit_request("cache_processor", "test_item")
            
            # Should get a result
            assert result.result == "processed_test_item"
            assert result.error is None
            
            # Get stats
            stats = await manager.get_stats()
            assert stats["total_processors"] == 1
            assert "cache_processor" in stats["processors"]
            
        finally:
            await manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_full_caching_system_integration(self, memory_cache, redis_cache):
        """Test the full integration of the caching system."""
        # Create a compressor
        compressor = ResultCompressor(
            default_algorithm=CompressionAlgorithm.GZIP,
            min_size_for_compression=10,
        )
        
        # Create a compressed Redis backend
        compressed_redis = CompressedCacheBackend(
            backend=redis_cache,
            compressor=compressor,
            compress_threshold=10,
        )
        
        # Create a multi-layer cache
        cache = MultiLayerCache()
        
        # Add memory layer (highest priority)
        memory_layer = CacheLayer(
            backend=memory_cache,
            name="memory",
            priority=0,
            read_only=False,
            default_ttl=60,
        )
        cache.add_layer(memory_layer)
        
        # Add compressed Redis layer (lower priority)
        redis_layer = CacheLayer(
            backend=compressed_redis,
            name="redis",
            priority=1,
            read_only=False,
            default_ttl=120,
        )
        cache.add_layer(redis_layer)
        
        # Create a performance monitor
        monitor = PerformanceMonitor()
        await monitor.start()
        
        try:
            # Test setting and getting values
            small_value = "small"
            large_value = "x" * 100
            
            # Set small value (should be in memory)
            result = await cache.set("small_key", small_value)
            assert result is True
            
            # Set large value (should be compressed in Redis)
            result = await cache.set("large_key", large_value)
            assert result is True
            
            # Get small value (should be from memory)
            value = await cache.get("small_key")
            assert value == small_value
            
            # Get large value (should be from Redis, decompressed)
            value = await cache.get("large_key")
            assert value == large_value
            
            # Check if keys exist
            assert await cache.exists("small_key") is True
            assert await cache.exists("large_key") is True
            assert await cache.exists("nonexistent") is False
            
            # Get all keys
            keys = await cache.keys("*")
            assert len(keys) == 2
            assert "small_key" in keys
            assert "large_key" in keys
            
            # Get cache statistics
            cache_stats = cache.get_stats()
            assert cache_stats["total_layers"] == 2
            assert cache_stats["enabled_layers"] == 2
            
            # Get performance statistics
            perf_stats = await monitor.get_stats()
            assert perf_stats["enabled"] is True
            
        finally:
            await monitor.stop()
            await cache.close()