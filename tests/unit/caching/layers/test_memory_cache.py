"""
Tests for memory cache layer.
"""

import pytest
import asyncio
from app.core.caching.layers.memory import MemoryCache, MemoryCacheLayer


class TestMemoryCache:
    """Test MemoryCache class."""

    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache instance."""
        return MemoryCache(max_size=100, default_ttl=300)

    @pytest.mark.asyncio
    async def test_initialization(self, memory_cache):
        """Test memory cache initialization."""
        assert memory_cache.max_size == 100
        assert memory_cache.default_ttl == 300
        assert memory_cache._cache == {}
        assert memory_cache._stats["hits"] == 0
        assert memory_cache._stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_cache):
        """Test setting and getting a value."""
        # Set a value
        result = await memory_cache.set("test_key", "test_value", 300)
        assert result is True

        # Get the value
        result = await memory_cache.get("test_key")
        assert result == "test_value"

        # Check that hit count increased
        assert memory_cache._stats["hits"] == 1
        assert memory_cache._stats["sets"] == 1
        assert memory_cache._stats["current_size"] == 1
        assert 'test_key' in memory_cache._cache

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, memory_cache):
        """Test setting a value with custom TTL."""
        # Set a value with short TTL
        result = await memory_cache.set("test_key", "test_value", 1)
        assert result is True

        # Value should be available immediately
        result = await memory_cache.get("test_key")
        assert result == "test_value"

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Value should be expired
        result = await memory_cache.get("test_key")
        assert result is None
        assert memory_cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_cache):
        """Test getting a nonexistent key."""
        result = await memory_cache.get("nonexistent_key")
        assert result is None
        assert memory_cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_delete_existing_key(self, memory_cache):
        """Test deleting an existing key."""
        # Set a value
        await memory_cache.set("test_key", "test_value")

        # Delete the key
        result = await memory_cache.delete("test_key")
        assert result is True

        # Key should be gone
        result = await memory_cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, memory_cache):
        """Test deleting a nonexistent key."""
        result = await memory_cache.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_existing_key(self, memory_cache):
        """Test checking existence of an existing key."""
        await memory_cache.set("test_key", "test_value")
        result = await memory_cache.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_nonexistent_key(self, memory_cache):
        """Test checking existence of a nonexistent key."""
        result = await memory_cache.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_expired_key(self, memory_cache):
        """Test checking existence of an expired key."""
        # Set a value with short TTL
        await memory_cache.set("test_key", "test_value", 0.1)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Key should not exist
        result = await memory_cache.exists("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, memory_cache):
        """Test clearing the cache."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")

        # Clear the cache
        result = await memory_cache.clear()
        assert result is True

        # Cache should be empty
        assert memory_cache._cache == {}

        # Values should be gone
        result = await memory_cache.get("key1")
        assert result is None
        result = await memory_cache.get("key2")
        assert result is None

    @pytest.mark.asyncio
    async def test_max_size_eviction(self, memory_cache):
        """Test eviction when max size is reached."""
        # Create a small cache
        small_cache = MemoryCache(max_size=2, default_ttl=300)

        # Fill the cache
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")

        # Add a third key - should trigger eviction
        await small_cache.set("key3", "value3")

        # Cache should have exactly 2 items
        assert len(small_cache._cache) == 2

        # The least recently used key (key1) should be evicted
        result = await small_cache.get("key1")
        assert result is None

        # Other keys should still be there
        result = await small_cache.get("key2")
        assert result == "value2"
        result = await small_cache.get("key3")
        assert result == "value3"

    @pytest.mark.asyncio
    async def test_lru_eviction_policy(self, memory_cache):
        """Test LRU eviction policy."""
        # Create a small cache
        small_cache = MemoryCache(max_size=2, default_ttl=300)

        # Fill the cache
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")

        # Access key1 to make it more recently used
        await small_cache.get("key1")

        # Add a third key - key2 should be evicted (least recently used)
        await small_cache.set("key3", "value3")

        # key2 should be evicted
        result = await small_cache.get("key2")
        assert result is None

        # key1 and key3 should be present
        result = await small_cache.get("key1")
        assert result == "value1"
        result = await small_cache.get("key3")
        assert result == "value3"

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_cache):
        """Test getting cache statistics."""
        # Perform some operations
        await memory_cache.set("key1", "value1")
        await memory_cache.get("key1")  # Hit
        await memory_cache.get("key2")  # Miss
        await memory_cache.delete("key1")

        stats = await memory_cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["current_size"] == 0  # key1 was deleted
        assert stats["max_size"] == 100
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 gets

    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, memory_cache):
        """Test cleanup of expired entries."""
        # Set some values with short TTL
        await memory_cache.set("key1", "value1", 0.1)
        await memory_cache.set("key2", "value2", 0.1)

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Cleanup should remove expired entries
        await memory_cache._cleanup_expired()

        # Cache should be empty
        assert memory_cache._cache == {}


    @pytest.mark.asyncio
    async def test_close(self, memory_cache):
        """Test closing the cache."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")

        # Close the cache
        await memory_cache.close()

        # Cache should be cleared
        assert memory_cache._cache == {}

class TestMemoryCacheLayer:
    """Test MemoryCacheLayer class."""

    @pytest.fixture
    def memory_cache_layer(self):
        """Create a memory cache layer."""
        return MemoryCacheLayer(
            name="test_layer",
            priority=0,
            max_size=100,
            default_ttl=300,
            cleanup_interval=60,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, memory_cache_layer):
        """Test layer initialization."""
        assert memory_cache_layer.layer.name == "test_layer"
        assert memory_cache_layer.layer.priority == 0
        assert memory_cache_layer.layer.read_only is False
        assert memory_cache_layer.layer.default_ttl == 300
        assert memory_cache_layer.layer is not None
        assert isinstance(memory_cache_layer.layer.backend, MemoryCache)

    @pytest.mark.asyncio
    async def test_get(self, memory_cache_layer):
        """Test layer get method."""
        # Set a value
        await memory_cache_layer.layer.set("test_key", "test_value")

        # Get the value
        result = await memory_cache_layer.layer.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set(self, memory_cache_layer):
        """Test layer set method."""
        result = await memory_cache_layer.layer.set("test_key", "test_value")
        assert result is True

        # Verify the value was set
        result = await memory_cache_layer.layer.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_read_only(self, memory_cache_layer):
        """Test layer set method in read-only mode."""
        memory_cache_layer.layer.read_only = True
        result = await memory_cache_layer.layer.set("test_key", "test_value")
        assert result is False

        # Value should not be set
        result = await memory_cache_layer.layer.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, memory_cache_layer):
        """Test layer delete method."""
        # Set a value
        await memory_cache_layer.layer.set("test_key", "test_value")

        # Delete the value
        result = await memory_cache_layer.layer.delete("test_key")
        assert result is True

        # Value should be gone
        result = await memory_cache_layer.layer.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_read_only(self, memory_cache_layer):
        """Test layer delete method in read-only mode."""
        memory_cache_layer.layer.read_only = True

        # Set a value (should fail in read-only mode)
        await memory_cache_layer.layer.set("test_key", "test_value")

        # Delete should also fail
        result = await memory_cache_layer.layer.delete("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, memory_cache_layer):
        """Test layer exists method."""
        # Set a value
        await memory_cache_layer.layer.set("test_key", "test_value")

        # Check existence
        result = await memory_cache_layer.layer.exists("test_key")
        assert result is True

        # Check nonexistence
        result = await memory_cache_layer.layer.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, memory_cache_layer):
        """Test layer clear method."""
        # Set some values
        await memory_cache_layer.layer.set("key1", "value1")
        await memory_cache_layer.layer.set("key2", "value2")

        # Clear the cache
        result = await memory_cache_layer.layer.clear()
        assert result is True

        # Values should be gone
        result = await memory_cache_layer.layer.get("key1")
        assert result is None
        result = await memory_cache_layer.layer.get("key2")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self, memory_cache_layer):
        """Test layer close method."""
        # Set a value
        await memory_cache_layer.layer.set("test_key", "test_value")

        # Close the layer
        await memory_cache_layer.layer.close()

        # Cache should be cleared
        result = await memory_cache_layer.layer.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_cache_layer):
        """Test layer get_stats method."""
        # Perform some operations
        await memory_cache_layer.layer.set("key1", "value1")
        await memory_cache_layer.layer.get("key1")
        await memory_cache_layer.layer.get("key2")
    
        # Get layer stats (synchronous method)
        stats = memory_cache_layer.layer.get_stats()

        assert stats["name"] == "test_layer"
        assert stats["priority"] == 0
        assert stats["read_only"] is False
        assert stats["enabled"] is True
        assert stats["default_ttl"] == 300

        # Get backend stats (async method)
        backend_stats = await memory_cache_layer.cache.get_stats()
        assert "hits" in backend_stats
        assert "misses" in backend_stats
        assert "sets" in backend_stats


class TestMemoryCacheIntegration:
    """Integration tests for memory cache."""

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to memory cache."""
        cache = MemoryCache(max_size=100, default_ttl=300)

        # Set initial value
        await cache.set("counter", 0)

        # Define increment function
        async def increment_counter():
            current = await cache.get("counter")
            if current is None:
                current = 0
            await cache.set("counter", current + 1)

        # Run multiple increments concurrently
        tasks = [increment_counter() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Check final value (should be 10 if no race conditions in our test)
        # Note: This test is for basic functionality, not thread safety
        # since asyncio is single-threaded
        final_value = await cache.get("counter")
        assert final_value == 10

    @pytest.mark.asyncio
    async def test_large_data_handling(self):
        """Test handling of large data."""
        cache = MemoryCache(max_size=5, default_ttl=300)  # Small cache

        # Add items until cache is full
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}" * 100)  # Large values

        # Cache should not exceed max size
        stats = await cache.get_stats()
        assert stats["current_size"] <= 5

    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        cache = MemoryCache(max_size=100, default_ttl=300)

        # Perform many operations
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")
            await cache.get(f"key{i}")

        # Check performance metrics
        stats = await cache.get_stats()
        assert stats["hits"] == 100
        assert stats["sets"] == 100
        assert stats["hit_rate"] == 1.0
