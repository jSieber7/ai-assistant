"""
Integration tests for caching components.

This module tests integration between different caching layers and components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import redis
from dataclasses import dataclass

from app.core.caching.layers.memory import MemoryCache
from app.core.caching.layers.redis_cache import RedisCache
from app.core.caching.compression.compressor import CacheCompressor
from app.core.caching.base import CacheResult, CacheLayer
from app.core.caching.batching.batch_processor import BatchProcessor


@dataclass
class MockCacheLayer(CacheLayer):
    """Mock cache layer for testing"""
    
    def __init__(self, name="mock_layer", latency=0.01):
        super().__init__(name)
        self.data = {}
        self.latency = latency
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
    
    async def get(self, key: str) -> Optional[CacheResult]:
        await asyncio.sleep(self.latency)
        self.get_count += 1
        
        if key in self.data:
            value, timestamp, ttl = self.data[key]
            if ttl and time.time() > timestamp + ttl:
                del self.data[key]
                return None
            return CacheResult(
                key=key,
                value=value,
                hit=True,
                timestamp=timestamp,
                ttl=ttl
            )
        return CacheResult(key=key, value=None, hit=False)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        await asyncio.sleep(self.latency)
        self.set_count += 1
        self.data[key] = (value, time.time(), ttl)
        return True
    
    async def delete(self, key: str) -> bool:
        await asyncio.sleep(self.latency)
        self.delete_count += 1
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    async def clear(self) -> bool:
        self.data.clear()
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.data),
            "get_count": self.get_count,
            "set_count": self.set_count,
            "delete_count": self.delete_count
        }


class TestCachingIntegration:
    """Test caching integration between components"""

    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache instance"""
        cache = MemoryCache()
        cache.configure(ttl=300, max_size=1000)
        return cache

    @pytest.fixture
    def redis_cache(self):
        """Create a Redis cache instance"""
        cache = RedisCache()
        cache.configure(
            host="localhost",
            port=6379,
            db=0,
            ttl=300,
            max_connections=10
        )
        return cache

    @pytest.fixture
    def mock_redis_cache(self):
        """Create a mock Redis cache instance"""
        return MockCacheLayer("redis_mock", latency=0.02)

    @pytest.fixture
    def mock_memory_cache(self):
        """Create a mock memory cache instance"""
        return MockCacheLayer("memory_mock", latency=0.01)

    @pytest.fixture
    def compressor(self):
        """Create a cache compressor instance"""
        return CacheCompressor()

    @pytest.fixture
    def batch_processor(self):
        """Create a batch processor instance"""
        return BatchProcessor(batch_size=10, flush_interval=1.0)

    @pytest.mark.asyncio
    async def test_memory_cache_basic_operations(self, memory_cache):
        """Test basic memory cache operations"""
        # Test set and get
        await memory_cache.set("test_key", "test_value", ttl=60)
        result = await memory_cache.get("test_key")
        
        assert result.success is True
        assert result.data.hit is True
        assert result.data.value == "test_value"
        
        # Test non-existent key
        result = await memory_cache.get("nonexistent_key")
        assert result.success is True
        assert result.data.hit is False
        assert result.data.value is None

    @pytest.mark.asyncio
    async def test_memory_cache_ttl_expiration(self, memory_cache):
        """Test memory cache TTL expiration"""
        # Set with short TTL
        await memory_cache.set("ttl_key", "ttl_value", ttl=1)
        
        # Should be available immediately
        result = await memory_cache.get("ttl_key")
        assert result.data.hit is True
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await memory_cache.get("ttl_key")
        assert result.data.hit is False

    @pytest.mark.asyncio
    async def test_memory_cache_lru_eviction(self, memory_cache):
        """Test memory cache LRU eviction"""
        # Configure small cache
        memory_cache.configure(ttl=300, max_size=2)
        
        # Fill cache beyond capacity
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")  # Should evict key1
        
        # Check eviction
        result1 = await memory_cache.get("key1")
        assert result1.data.hit is False  # Should be evicted
        
        result2 = await memory_cache.get("key2")
        assert result2.data.hit is True  # Should still be present
        
        result3 = await memory_cache.get("key3")
        assert result3.data.hit is True  # Should be present

    @pytest.mark.asyncio
    async def test_cache_layer_fallback(self, mock_memory_cache, mock_redis_cache):
        """Test cache layer fallback mechanism"""
        from app.core.caching.base import MultiLayerCache
        
        # Create multi-layer cache with memory as L1 and Redis as L2
        multi_cache = MultiLayerCache()
        multi_cache.add_layer(mock_memory_cache)
        multi_cache.add_layer(mock_redis_cache)
        
        # Set value in L2 only
        await mock_redis_cache.set("fallback_key", "fallback_value")
        
        # Get should check L1 first, then fallback to L2
        result = await multi_cache.get("fallback_key")
        
        assert result.success is True
        assert result.data.value == "fallback_value"
        assert result.data.hit is True
        
        # Check that L1 was checked first
        assert mock_memory_cache.get_count == 1
        assert mock_redis_cache.get_count == 1

    @pytest.mark.asyncio
    async def test_cache_write_through(self, mock_memory_cache, mock_redis_cache):
        """Test cache write-through mechanism"""
        from app.core.caching.base import MultiLayerCache
        
        # Create multi-layer cache with write-through
        multi_cache = MultiLayerCache(write_through=True)
        multi_cache.add_layer(mock_memory_cache)
        multi_cache.add_layer(mock_redis_cache)
        
        # Set value should write to all layers
        await multi_cache.set("write_through_key", "write_through_value")
        
        # Check both layers have the value
        memory_result = await mock_memory_cache.get("write_through_key")
        redis_result = await mock_redis_cache.get("write_through_key")
        
        assert memory_result.hit is True
        assert memory_result.value == "write_through_value"
        assert redis_result.hit is True
        assert redis_result.value == "write_through_value"
        
        # Check set was called on both layers
        assert mock_memory_cache.set_count == 1
        assert mock_redis_cache.set_count == 1

    @pytest.mark.asyncio
    async def test_cache_compression_integration(self, memory_cache, compressor):
        """Test cache compression integration"""
        from app.core.caching.compression.compressor import CompressedCache
        
        # Create compressed cache
        compressed_cache = CompressedCache(memory_cache, compressor)
        
        # Store large data
        large_data = "x" * 1000  # 1KB of data
        await compressed_cache.set("compressed_key", large_data)
        
        # Retrieve and verify
        result = await compressed_cache.get("compressed_key")
        
        assert result.success is True
        assert result.data.hit is True
        assert result.data.value == large_data

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self, batch_processor):
        """Test batch processor integration"""
        from app.core.caching.batching.batch_processor import BatchOperation
        
        # Mock operation
        executed_operations = []
        
        async def mock_execute_batch(operations):
            executed_operations.extend(operations)
            return [True] * len(operations)
        
        batch_processor.execute_batch = mock_execute_batch
        
        # Add operations to batch
        for i in range(5):
            op = BatchOperation(
                type="set",
                key=f"batch_key_{i}",
                value=f"batch_value_{i}"
            )
            await batch_processor.add_operation(op)
        
        # Wait for batch to be processed
        await asyncio.sleep(1.1)
        
        # Check operations were executed
        assert len(executed_operations) == 5
        assert executed_operations[0].key == "batch_key_0"
        assert executed_operations[0].value == "batch_value_0"

    @pytest.mark.asyncio
    async def test_cache_performance_monitoring(self, memory_cache):
        """Test cache performance monitoring"""
        from app.core.caching.monitoring.metrics import CacheMetrics
        
        metrics = CacheMetrics()
        metrics.track_cache(memory_cache)
        
        # Perform some operations
        await memory_cache.set("perf_key", "perf_value")
        await memory_cache.get("perf_key")
        await memory_cache.get("nonexistent_key")
        
        # Get metrics
        cache_metrics = await metrics.get_metrics("memory_cache")
        
        assert "hit_rate" in cache_metrics
        assert "miss_rate" in cache_metrics
        assert "avg_get_time" in cache_metrics
        assert "avg_set_time" in cache_metrics

    @pytest.mark.asyncio
    async def test_cache_invalidation_propagation(self, mock_memory_cache, mock_redis_cache):
        """Test cache invalidation propagation"""
        from app.core.caching.base import MultiLayerCache
        
        # Create multi-layer cache
        multi_cache = MultiLayerCache(write_through=True)
        multi_cache.add_layer(mock_memory_cache)
        multi_cache.add_layer(mock_redis_cache)
        
        # Set value in both layers
        await multi_cache.set("invalidate_key", "original_value")
        
        # Verify it's in both layers
        memory_result = await mock_memory_cache.get("invalidate_key")
        redis_result = await mock_redis_cache.get("invalidate_key")
        
        assert memory_result.hit is True
        assert redis_result.hit is True
        
        # Delete from multi-layer cache
        await multi_cache.delete("invalidate_key")
        
        # Verify it's deleted from both layers
        memory_result = await mock_memory_cache.get("invalidate_key")
        redis_result = await mock_redis_cache.get("invalidate_key")
        
        assert memory_result.hit is False
        assert redis_result.hit is False

    @pytest.mark.asyncio
    async def test_cache_warming_strategy(self, memory_cache):
        """Test cache warming strategy"""
        from app.core.caching.base import CacheWarmer
        
        # Create cache warmer
        warmer = CacheWarmer(memory_cache)
        
        # Define warm-up data
        warm_data = {
            "warm_key_1": "warm_value_1",
            "warm_key_2": "warm_value_2",
            "warm_key_3": "warm_value_3"
        }
        
        # Warm cache
        await warmer.warm_cache(warm_data)
        
        # Verify data is in cache
        for key, expected_value in warm_data.items():
            result = await memory_cache.get(key)
            assert result.data.hit is True
            assert result.data.value == expected_value

    @pytest.mark.asyncio
    async def test_cache_consistency_check(self, mock_memory_cache, mock_redis_cache):
        """Test cache consistency check"""
        from app.core.caching.base import CacheConsistencyChecker
        
        # Create consistency checker
        checker = CacheConsistencyChecker()
        checker.add_layer(mock_memory_cache)
        checker.add_layer(mock_redis_cache)
        
        # Set different values in each layer
        await mock_memory_cache.set("consistency_key", "memory_value")
        await mock_redis_cache.set("consistency_key", "redis_value")
        
        # Check consistency
        inconsistencies = await checker.check_consistency()
        
        assert len(inconsistencies) == 1
        assert inconsistencies[0]["key"] == "consistency_key"
        assert inconsistencies[0]["memory_mock"] == "memory_value"
        assert inconsistencies[0]["redis_mock"] == "redis_value"

    @pytest.mark.asyncio
    async def test_cache_disaster_recovery(self, mock_memory_cache, mock_redis_cache):
        """Test cache disaster recovery"""
        from app.core.caching.base import CacheRecoveryManager
        
        # Create recovery manager
        recovery = CacheRecoveryManager()
        recovery.add_layer(mock_memory_cache)
        recovery.add_layer(mock_redis_cache)
        
        # Populate cache with data
        test_data = {
            "recovery_key_1": "recovery_value_1",
            "recovery_key_2": "recovery_value_2"
        }
        
        for key, value in test_data.items():
            await mock_memory_cache.set(key, value)
        
        # Simulate layer failure
        await mock_memory_cache.clear()
        
        # Recover from other layer
        await recovery.recover_layer("memory_mock", from_layers=["redis_mock"])
        
        # Verify recovery
        for key, expected_value in test_data.items():
            result = await mock_memory_cache.get(key)
            assert result.data.hit is True
            assert result.data.value == expected_value

    @pytest.mark.asyncio
    async def test_cache_sharding_strategy(self):
        """Test cache sharding strategy"""
        from app.core.caching.base import ShardedCache
        
        # Create multiple cache shards
        shards = [MockCacheLayer(f"shard_{i}") for i in range(3)]
        
        # Create sharded cache
        sharded_cache = ShardedCache(shards)
        
        # Set values
        await sharded_cache.set("shard_key_1", "value_1")
        await sharded_cache.set("shard_key_2", "value_2")
        await sharded_cache.set("shard_key_3", "value_3")
        
        # Get values
        result1 = await sharded_cache.get("shard_key_1")
        result2 = await sharded_cache.get("shard_key_2")
        result3 = await sharded_cache.get("shard_key_3")
        
        assert result1.data.hit is True
        assert result1.data.value == "value_1"
        assert result2.data.hit is True
        assert result2.data.value == "value_2"
        assert result3.data.hit is True
        assert result3.data.value == "value_3"
        
        # Verify distribution across shards
        total_operations = sum(shard.get_count for shard in shards)
        assert total_operations == 3  # One get per key

    @pytest.mark.asyncio
    async def test_cache_event_handling(self, memory_cache):
        """Test cache event handling"""
        from app.core.caching.base import CacheEventHandler
        
        events = []
        
        # Create event handler
        async def on_cache_hit(key, value):
            events.append({"type": "hit", "key": key, "value": value})
        
        async def on_cache_miss(key):
            events.append({"type": "miss", "key": key})
        
        async def on_cache_set(key, value):
            events.append({"type": "set", "key": key, "value": value})
        
        # Register event handlers
        handler = CacheEventHandler()
        handler.on_hit = on_cache_hit
        handler.on_miss = on_cache_miss
        handler.on_set = on_cache_set
        
        memory_cache.add_event_handler(handler)
        
        # Perform operations
        await memory_cache.set("event_key", "event_value")
        await memory_cache.get("event_key")
        await memory_cache.get("nonexistent_event_key")
        
        # Check events
        assert len(events) == 3
        assert events[0]["type"] == "set"
        assert events[1]["type"] == "hit"
        assert events[2]["type"] == "miss"

    @pytest.mark.asyncio
    async def test_cache_serialization_integration(self, memory_cache):
        """Test cache serialization integration"""
        from app.core.caching.base import CacheSerializer
        
        # Create serializer
        serializer = CacheSerializer()
        
        # Store complex object
        complex_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "datetime": datetime.now()
        }
        
        # Serialize and store
        serialized = serializer.serialize(complex_data)
        await memory_cache.set("serialization_key", serialized)
        
        # Retrieve and deserialize
        result = await memory_cache.get("serialization_key")
        deserialized = serializer.deserialize(result.data.value)
        
        assert deserialized["string"] == "test"
        assert deserialized["number"] == 42
        assert deserialized["list"] == [1, 2, 3]
        assert deserialized["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_cache_health_monitoring(self, mock_memory_cache, mock_redis_cache):
        """Test cache health monitoring"""
        from app.core.caching.monitoring.health import CacheHealthMonitor
        
        # Create health monitor
        monitor = CacheHealthMonitor()
        monitor.add_layer(mock_memory_cache)
        monitor.add_layer(mock_redis_cache)
        
        # Get health status
        health_status = await monitor.get_health_status()
        
        assert "memory_mock" in health_status
        assert "redis_mock" in health_status
        assert "status" in health_status["memory_mock"]
        assert "response_time" in health_status["memory_mock"]
        assert "error_rate" in health_status["memory_mock"]