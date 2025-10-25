"""
Unit tests for the base caching module.

Tests for the core abstractions, cache layers, and multi-layer cache implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from app.core.caching.base import (
    CacheableOperation,
    CacheBackend,
    CacheLayer,
    MultiLayerCache,
    generate_cache_key,
    CacheOperation,
    CacheEvent,
    CacheMetrics,
    PerformanceMonitor,
)


class MockCacheBackend(CacheBackend):
    """Mock cache backend for testing."""

    def __init__(self, name: str = "mock_backend", fail_operations: bool = False):
        self.name = name
        self.fail_operations = fail_operations
        self._data: Dict[str, Any] = {}
        self._stats = {
            "get_count": 0,
            "set_count": 0,
            "delete_count": 0,
            "exists_count": 0,
            "clear_count": 0,
            "keys_count": 0,
        }

    async def get(self, key: str) -> Optional[Any]:
        self._stats["get_count"] += 1
        if self.fail_operations:
            raise Exception(f"Failed to get key '{key}'")
        return self._data.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._stats["set_count"] += 1
        if self.fail_operations:
            raise Exception(f"Failed to set key '{key}'")
        self._data[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self._stats["delete_count"] += 1
        if self.fail_operations:
            raise Exception(f"Failed to delete key '{key}'")
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        self._stats["exists_count"] += 1
        if self.fail_operations:
            raise Exception(f"Failed to check existence of key '{key}'")
        return key in self._data

    async def clear(self) -> bool:
        self._stats["clear_count"] += 1
        if self.fail_operations:
            raise Exception("Failed to clear cache")
        self._data.clear()
        return True

    async def keys(self, pattern: str = "*") -> List[str]:
        self._stats["keys_count"] += 1
        if self.fail_operations:
            raise Exception(f"Failed to get keys with pattern '{pattern}'")
        if pattern == "*":
            return list(self._data.keys())
        # Simple pattern matching
        if "*" in pattern:
            prefix, suffix = pattern.split("*", 1)
            return [k for k in self._data.keys() if k.startswith(prefix) and k.endswith(suffix)]
        else:
            return [k] if pattern in self._data else []

    async def close(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_size": len(self._data),
            **self._stats,
        }


class TestCacheableOperation:
    """Test cases for CacheableOperation enum."""

    def test_cacheable_operation_values(self):
        """Test that CacheableOperation has the expected values."""
        assert CacheableOperation.GET.value == "get"
        assert CacheableOperation.SET.value == "set"
        assert CacheableOperation.DELETE.value == "delete"
        assert CacheableOperation.EXISTS.value == "exists"
        assert CacheableOperation.CLEAR.value == "clear"
        assert CacheableOperation.KEYS.value == "keys"


class TestCacheLayer:
    """Test cases for CacheLayer class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock cache backend."""
        return MockCacheBackend()

    @pytest.fixture
    def cache_layer(self, mock_backend):
        """Create a cache layer with a mock backend."""
        return CacheLayer(
            backend=mock_backend,
            name="test_layer",
            priority=1,
            read_only=False,
            default_ttl=60,
        )

    @pytest.mark.asyncio
    async def test_cache_layer_get(self, cache_layer):
        """Test getting a value from the cache layer."""
        # Set a value first
        await cache_layer.set("test_key", "test_value")
        
        # Get the value
        result = await cache_layer.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_cache_layer_get_disabled(self, cache_layer):
        """Test getting a value from a disabled cache layer."""
        cache_layer.enabled = False
        result = await cache_layer.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_layer_set(self, cache_layer):
        """Test setting a value in the cache layer."""
        result = await cache_layer.set("test_key", "test_value")
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_layer_set_readonly(self, cache_layer):
        """Test setting a value in a read-only cache layer."""
        cache_layer.read_only = True
        result = await cache_layer.set("test_key", "test_value")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_layer_set_disabled(self, cache_layer):
        """Test setting a value in a disabled cache layer."""
        cache_layer.enabled = False
        result = await cache_layer.set("test_key", "test_value")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_layer_delete(self, cache_layer):
        """Test deleting a key from the cache layer."""
        # Set a value first
        await cache_layer.set("test_key", "test_value")
        
        # Delete the key
        result = await cache_layer.delete("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_layer_delete_nonexistent(self, cache_layer):
        """Test deleting a non-existent key."""
        result = await cache_layer.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_layer_exists(self, cache_layer):
        """Test checking if a key exists in the cache layer."""
        # Set a value first
        await cache_layer.set("test_key", "test_value")
        
        # Check if key exists
        result = await cache_layer.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_layer_exists_nonexistent(self, cache_layer):
        """Test checking if a non-existent key exists."""
        result = await cache_layer.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_layer_clear(self, cache_layer):
        """Test clearing all keys from the cache layer."""
        # Set some values
        await cache_layer.set("key1", "value1")
        await cache_layer.set("key2", "value2")
        
        # Clear the cache
        result = await cache_layer.clear()
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_layer_keys(self, cache_layer):
        """Test getting keys matching a pattern."""
        # Set some values
        await cache_layer.set("test_key1", "value1")
        await cache_layer.set("test_key2", "value2")
        await cache_layer.set("other_key", "value3")
        
        # Get all keys
        all_keys = await cache_layer.keys("*")
        assert len(all_keys) == 3
        
        # Get keys with pattern
        test_keys = await cache_layer.keys("test_*")
        assert len(test_keys) == 2
        assert "test_key1" in test_keys
        assert "test_key2" in test_keys

    @pytest.mark.asyncio
    async def test_cache_layer_close(self, cache_layer):
        """Test closing the cache layer."""
        # This should not raise an exception
        await cache_layer.close()

    def test_cache_layer_get_stats(self, cache_layer):
        """Test getting statistics for the cache layer."""
        stats = cache_layer.get_stats()
        assert stats["name"] == "test_layer"
        assert stats["priority"] == 1
        assert stats["read_only"] is False
        assert stats["enabled"] is True
        assert stats["default_ttl"] == 60


class TestMultiLayerCache:
    """Test cases for MultiLayerCache class."""

    @pytest.fixture
    def mock_backend1(self):
        """Create a mock cache backend."""
        return MockCacheBackend("backend1")

    @pytest.fixture
    def mock_backend2(self):
        """Create another mock cache backend."""
        return MockCacheBackend("backend2")

    @pytest.fixture
    def cache_layer1(self, mock_backend1):
        """Create a cache layer with higher priority."""
        return CacheLayer(
            backend=mock_backend1,
            name="layer1",
            priority=0,  # Higher priority
            read_only=False,
            default_ttl=60,
        )

    @pytest.fixture
    def cache_layer2(self, mock_backend2):
        """Create a cache layer with lower priority."""
        return CacheLayer(
            backend=mock_backend2,
            name="layer2",
            priority=1,  # Lower priority
            read_only=False,
            default_ttl=120,
        )

    @pytest.fixture
    def readonly_layer(self):
        """Create a read-only cache layer."""
        return CacheLayer(
            backend=MockCacheBackend("readonly"),
            name="readonly",
            priority=2,
            read_only=True,
        )

    @pytest.fixture
    def multi_layer_cache(self, cache_layer1, cache_layer2):
        """Create a multi-layer cache with two layers."""
        cache = MultiLayerCache()
        cache.add_layer(cache_layer1)
        cache.add_layer(cache_layer2)
        return cache

    def test_add_layer(self, multi_layer_cache):
        """Test adding a layer to the multi-layer cache."""
        # Layers should be sorted by priority (lowest first)
        assert len(multi_layer_cache.layers) == 2
        assert multi_layer_cache.layers[0].name == "layer1"  # Priority 0
        assert multi_layer_cache.layers[1].name == "layer2"  # Priority 1

    def test_remove_layer(self, multi_layer_cache):
        """Test removing a layer from the multi-layer cache."""
        # Remove a layer
        result = multi_layer_cache.remove_layer("layer1")
        assert result is True
        assert len(multi_layer_cache.layers) == 1
        assert multi_layer_cache.layers[0].name == "layer2"

    def test_remove_nonexistent_layer(self, multi_layer_cache):
        """Test removing a non-existent layer."""
        result = multi_layer_cache.remove_layer("nonexistent")
        assert result is False

    def test_get_layer(self, multi_layer_cache):
        """Test getting a layer by name."""
        layer = multi_layer_cache.get_layer("layer1")
        assert layer is not None
        assert layer.name == "layer1"

    def test_get_nonexistent_layer(self, multi_layer_cache):
        """Test getting a non-existent layer."""
        layer = multi_layer_cache.get_layer("nonexistent")
        assert layer is None

    @pytest.mark.asyncio
    async def test_get_hit_in_first_layer(self, multi_layer_cache):
        """Test getting a value that exists in the first layer."""
        # Set a value in the first layer (higher priority)
        await multi_layer_cache.layers[0].set("test_key", "layer1_value")
        
        # Get the value
        result = await multi_layer_cache.get("test_key")
        assert result == "layer1_value"

    @pytest.mark.asyncio
    async def test_get_hit_in_second_layer(self, multi_layer_cache):
        """Test getting a value that exists only in the second layer."""
        # Set a value in the second layer (lower priority)
        await multi_layer_cache.layers[1].set("test_key", "layer2_value")
        
        # Get the value
        result = await multi_layer_cache.get("test_key")
        assert result == "layer2_value"

    @pytest.mark.asyncio
    async def test_get_miss(self, multi_layer_cache):
        """Test getting a value that doesn't exist in any layer."""
        result = await multi_layer_cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_read_through(self, multi_layer_cache):
        """Test read-through functionality."""
        # Enable read-through
        multi_layer_cache.set_read_through(True)
        
        # Set a value in the second layer (lower priority)
        await multi_layer_cache.layers[1].set("test_key", "layer2_value")
        
        # Get the value (should populate the first layer)
        result = await multi_layer_cache.get("test_key")
        assert result == "layer2_value"
        
        # Check that the first layer now has the value
        first_layer_value = await multi_layer_cache.layers[0].get("test_key")
        assert first_layer_value == "layer2_value"

    @pytest.mark.asyncio
    async def test_write_through(self, multi_layer_cache):
        """Test write-through functionality."""
        # Enable write-through
        multi_layer_cache.set_write_through(True)
        
        # Set a value
        result = await multi_layer_cache.set("test_key", "test_value")
        assert result is True
        
        # Check that both layers have the value
        layer1_value = await multi_layer_cache.layers[0].get("test_key")
        layer2_value = await multi_layer_cache.layers[1].get("test_key")
        assert layer1_value == "test_value"
        assert layer2_value == "test_value"

    @pytest.mark.asyncio
    async def test_write_to_highest_priority_only(self, multi_layer_cache):
        """Test writing to only the highest priority layer."""
        # Disable write-through
        multi_layer_cache.set_write_through(False)
        
        # Set a value
        result = await multi_layer_cache.set("test_key", "test_value")
        assert result is True
        
        # Check that only the first layer has the value
        layer1_value = await multi_layer_cache.layers[0].get("test_key")
        layer2_value = await multi_layer_cache.layers[1].get("test_key")
        assert layer1_value == "test_value"
        assert layer2_value is None

    @pytest.mark.asyncio
    async def test_delete(self, multi_layer_cache):
        """Test deleting a key from all layers."""
        # Set a value in both layers
        await multi_layer_cache.layers[0].set("test_key", "value1")
        await multi_layer_cache.layers[1].set("test_key", "value2")
        
        # Delete the key
        result = await multi_layer_cache.delete("test_key")
        assert result is True
        
        # Check that the key is deleted from both layers
        layer1_exists = await multi_layer_cache.layers[0].exists("test_key")
        layer2_exists = await multi_layer_cache.layers[1].exists("test_key")
        assert layer1_exists is False
        assert layer2_exists is False

    @pytest.mark.asyncio
    async def test_exists(self, multi_layer_cache):
        """Test checking if a key exists in any layer."""
        # Set a value in the second layer only
        await multi_layer_cache.layers[1].set("test_key", "value")
        
        # Check if key exists
        result = await multi_layer_cache.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear(self, multi_layer_cache):
        """Test clearing all keys from all layers."""
        # Set some values in both layers
        await multi_layer_cache.layers[0].set("key1", "value1")
        await multi_layer_cache.layers[1].set("key2", "value2")
        
        # Clear all layers
        result = await multi_layer_cache.clear()
        assert result is True
        
        # Check that all layers are cleared
        layer1_keys = await multi_layer_cache.layers[0].keys("*")
        layer2_keys = await multi_layer_cache.layers[1].keys("*")
        assert len(layer1_keys) == 0
        assert len(layer2_keys) == 0

    @pytest.mark.asyncio
    async def test_keys(self, multi_layer_cache):
        """Test getting keys from all layers."""
        # Set some values in both layers
        await multi_layer_cache.layers[0].set("key1", "value1")
        await multi_layer_cache.layers[1].set("key2", "value2")
        await multi_layer_cache.layers[1].set("key1", "value3")  # Duplicate key
        
        # Get all keys
        keys = await multi_layer_cache.keys("*")
        assert len(keys) == 2  # Should deduplicate
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.asyncio
    async def test_close(self, multi_layer_cache):
        """Test closing all layers."""
        # This should not raise an exception
        await multi_layer_cache.close()

    def test_enable_disable_layer(self, multi_layer_cache):
        """Test enabling and disabling layers."""
        # Disable a layer
        result = multi_layer_cache.disable_layer("layer1")
        assert result is True
        assert multi_layer_cache.layers[0].enabled is False
        
        # Re-enable the layer
        result = multi_layer_cache.enable_layer("layer1")
        assert result is True
        assert multi_layer_cache.layers[0].enabled is True
        
        # Try to disable a non-existent layer
        result = multi_layer_cache.disable_layer("nonexistent")
        assert result is False

    def test_get_stats(self, multi_layer_cache):
        """Test getting statistics for the multi-layer cache."""
        stats = multi_layer_cache.get_stats()
        assert stats["total_layers"] == 2
        assert stats["enabled_layers"] == 2
        assert stats["write_through"] is True
        assert stats["read_through"] is True
        assert len(stats["layers"]) == 2


class TestGenerateCacheKey:
    """Test cases for the generate_cache_key function."""

    def test_generate_cache_key_with_prefix_only(self):
        """Test generating a cache key with only a prefix."""
        key = generate_cache_key("test_prefix")
        assert key == "test_prefix"

    def test_generate_cache_key_with_args(self):
        """Test generating a cache key with positional arguments."""
        key = generate_cache_key("test_prefix", "arg1", "arg2")
        assert key == "test_prefix:arg1:arg2"

    def test_generate_cache_key_with_kwargs(self):
        """Test generating a cache key with keyword arguments."""
        key = generate_cache_key("test_prefix", key1="value1", key2="value2")
        assert key == "test_prefix:key1=value1:key2=value2"

    def test_generate_cache_key_with_args_and_kwargs(self):
        """Test generating a cache key with both positional and keyword arguments."""
        key = generate_cache_key("test_prefix", "arg1", key1="value1", key2="value2")
        assert key == "test_prefix:arg1:key1=value1:key2=value2"

    def test_generate_cache_key_with_sorted_kwargs(self):
        """Test that keyword arguments are sorted consistently."""
        key1 = generate_cache_key("test_prefix", b="value2", a="value1")
        key2 = generate_cache_key("test_prefix", a="value1", b="value2")
        assert key1 == key2

    def test_generate_cache_key_with_long_key(self):
        """Test generating a cache key that exceeds the maximum length."""
        # Create a very long key
        long_args = ["x" * 100] * 3  # This will create a key longer than 200 characters
        key = generate_cache_key("test_prefix", *long_args)
        
        # Should be hashed
        assert len(key) < 200
        assert key.startswith("test_prefix:")
        assert ":" not in key[12:]  # Only the prefix and one colon for the hash

    def test_generate_cache_key_with_special_characters(self):
        """Test generating a cache key with special characters."""
        key = generate_cache_key("test prefix", "arg/with/slashes", "arg with spaces")
        assert key == "test_prefix:arg/with/slashes:arg_with_spaces"


class TestCacheOperation:
    """Test cases for CacheOperation class."""

    def test_cache_operation_initialization(self):
        """Test initializing a cache operation."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        assert operation.operation == CacheableOperation.GET
        assert operation.key == "test_key"
        assert operation.start_time > 0
        assert operation.end_time is None
        assert operation.success is None
        assert operation.error is None
        assert operation.duration is None

    def test_cache_operation_complete_success(self):
        """Test completing a cache operation successfully."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        time.sleep(0.01)  # Small delay to ensure duration > 0
        operation.complete(True)
        
        assert operation.end_time is not None
        assert operation.success is True
        assert operation.error is None
        assert operation.duration is not None
        assert operation.duration > 0

    def test_cache_operation_complete_failure(self):
        """Test completing a cache operation with failure."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        time.sleep(0.01)  # Small delay to ensure duration > 0
        operation.complete(False, "Test error")
        
        assert operation.end_time is not None
        assert operation.success is False
        assert operation.error == "Test error"
        assert operation.duration is not None
        assert operation.duration > 0


class TestCacheEvent:
    """Test cases for CacheEvent class."""

    def test_cache_event_initialization(self):
        """Test initializing a cache event."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        event = CacheEvent(
            event_type="test_event",
            operation=operation,
            layer_name="test_layer",
            metadata={"key": "value"},
        )
        
        assert event.event_type == "test_event"
        assert event.operation == operation
        assert event.layer_name == "test_layer"
        assert event.metadata == {"key": "value"}
        assert event.timestamp > 0

    def test_cache_event_to_dict(self):
        """Test converting a cache event to a dictionary."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        operation.complete(True)
        
        event = CacheEvent(
            event_type="test_event",
            operation=operation,
            layer_name="test_layer",
            metadata={"key": "value"},
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "test_event"
        assert event_dict["operation"] == "get"
        assert event_dict["key"] == "test_key"
        assert event_dict["layer_name"] == "test_layer"
        assert event_dict["success"] is True
        assert event_dict["duration"] is not None
        assert event_dict["error"] is None
        assert event_dict["timestamp"] is not None
        assert event_dict["metadata"] == {"key": "value"}


class TestCacheMetrics:
    """Test cases for CacheMetrics class."""

    def test_cache_metrics_initialization(self):
        """Test initializing cache metrics."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0

    def test_record_hit(self):
        """Test recording a cache hit."""
        metrics = CacheMetrics()
        metrics.record_hit()
        assert metrics.hits == 1

    def test_record_miss(self):
        """Test recording a cache miss."""
        metrics = CacheMetrics()
        metrics.record_miss()
        assert metrics.misses == 1

    def test_record_set(self):
        """Test recording a cache set operation."""
        metrics = CacheMetrics()
        metrics.record_set()
        assert metrics.sets == 1

    def test_record_delete(self):
        """Test recording a cache delete operation."""
        metrics = CacheMetrics()
        metrics.record_delete()
        assert metrics.deletes == 1

    def test_record_error(self):
        """Test recording a cache error."""
        metrics = CacheMetrics()
        metrics.record_error()
        assert metrics.errors == 1

    def test_hit_rate(self):
        """Test calculating the cache hit rate."""
        metrics = CacheMetrics()
        
        # No operations yet
        assert metrics.hit_rate() == 0.0
        
        # Record some hits and misses
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        
        # Hit rate should be 2/3 = 0.666...
        assert abs(metrics.hit_rate() - 0.666666) < 0.000001

    def test_total_operations(self):
        """Test calculating total cache operations."""
        metrics = CacheMetrics()
        
        # No operations yet
        assert metrics.total_operations() == 0
        
        # Record some operations
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        metrics.record_delete()
        metrics.record_error()
        
        # Total should be 5
        assert metrics.total_operations() == 5

    def test_reset(self):
        """Test resetting cache metrics."""
        metrics = CacheMetrics()
        
        # Record some operations
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        metrics.record_delete()
        metrics.record_error()
        
        # Reset metrics
        metrics.reset()
        
        # All should be zero
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0

    def test_get_stats(self):
        """Test getting cache metrics as a dictionary."""
        metrics = CacheMetrics()
        
        # Record some operations
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        
        stats = metrics.get_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_sets"] == 1
        assert stats["cache_deletes"] == 0
        assert stats["cache_errors"] == 0
        assert abs(stats["cache_hit_rate"] - 0.666666) < 0.000001
        assert stats["total_cache_operations"] == 4


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""

    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor."""
        return PerformanceMonitor()

    @pytest.mark.asyncio
    async def test_performance_monitor_start_stop(self, performance_monitor):
        """Test starting and stopping the performance monitor."""
        # Initially disabled
        assert performance_monitor._enabled is False
        
        # Start the monitor
        await performance_monitor.start()
        assert performance_monitor._enabled is True
        
        # Stop the monitor
        await performance_monitor.stop()
        assert performance_monitor._enabled is False

    def test_record_operation_disabled(self, performance_monitor):
        """Test recording an operation when the monitor is disabled."""
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        operation.complete(True)
        
        # Record the operation (should be ignored)
        performance_monitor.record_operation(operation)
        
        # Check that no stats were recorded
        stats = performance_monitor._metrics.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_sets"] == 0
        assert stats["cache_deletes"] == 0
        assert stats["cache_errors"] == 0

    @pytest.mark.asyncio
    async def test_record_operation_enabled(self, performance_monitor):
        """Test recording an operation when the monitor is enabled."""
        await performance_monitor.start()
        
        # Record a GET operation
        operation = CacheOperation(CacheableOperation.GET, "test_key")
        operation.complete(True)
        performance_monitor.record_operation(operation)
        
        # Record a SET operation
        operation = CacheOperation(CacheableOperation.SET, "test_key")
        operation.complete(True)
        performance_monitor.record_operation(operation)
        
        # Check that stats were recorded
        stats = performance_monitor._metrics.get_stats()
        assert stats["cache_sets"] == 1

    def test_record_hit(self, performance_monitor):
        """Test recording a cache hit."""
        # This should not record anything when disabled
        performance_monitor.record_hit()
        stats = performance_monitor._metrics.get_stats()
        assert stats["cache_hits"] == 0

    def test_record_miss(self, performance_monitor):
        """Test recording a cache miss."""
        # This should not record anything when disabled
        performance_monitor.record_miss()
        stats = performance_monitor._metrics.get_stats()
        assert stats["cache_misses"] == 0

    def test_get_average_response_time(self, performance_monitor):
        """Test calculating average response time."""
        # Initially no response times
        assert performance_monitor.get_average_response_time() == 0.0
        
        # Add some response times
        performance_monitor._response_times = [0.1, 0.2, 0.3]
        
        # Average should be 0.2
        assert performance_monitor.get_average_response_time() == 0.2

    def test_get_percentile_response_time(self, performance_monitor):
        """Test calculating percentile response time."""
        # Initially no response times
        assert performance_monitor.get_percentile_response_time(95) == 0.0
        
        # Add some response times
        performance_monitor._response_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 95th percentile should be the last value (0.5)
        assert performance_monitor.get_percentile_response_time(95) == 0.5
        
        # 50th percentile should be the middle value (0.3)
        assert performance_monitor.get_percentile_response_time(50) == 0.3

    @pytest.mark.asyncio
    async def test_get_stats(self, performance_monitor):
        """Test getting performance statistics."""
        await performance_monitor.start()
        
        # Record some operations
        performance_monitor.record_hit()
        performance_monitor.record_miss()
        performance_monitor._response_times = [0.1, 0.2, 0.3]
        
        stats = await performance_monitor.get_stats()
        assert stats["enabled"] is True
        assert stats["total_response_time_samples"] == 3
        assert stats["average_response_time"] == 0.2
        assert stats["p95_response_time"] == 0.3
        assert stats["p99_response_time"] == 0.3

    def test_reset(self, performance_monitor):
        """Test resetting performance metrics."""
        # Add some data
        performance_monitor._metrics.record_hit()
        performance_monitor._metrics.record_miss()
        performance_monitor._response_times = [0.1, 0.2, 0.3]
        
        # Reset
        performance_monitor.reset()
        
        # Check that everything is reset
        stats = performance_monitor._metrics.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert len(performance_monitor._response_times) == 0