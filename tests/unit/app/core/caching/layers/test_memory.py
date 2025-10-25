"""
Unit tests for the memory cache layer implementation.

Tests for the in-memory cache with TTL support, LRU eviction, and cleanup functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock

from app.core.caching.layers.memory import (
    CacheEntry,
    MemoryCache,
    MemoryCacheLayer,
)


class TestCacheEntry:
    """Test cases for CacheEntry class."""

    def test_cache_entry_initialization(self):
        """Test initializing a cache entry."""
        entry = CacheEntry(
            value="test_value",
            expires_at=1234567890.0,
            created_at=1234567880.0,
            access_count=5,
            last_accessed=1234567895.0,
        )
        
        assert entry.value == "test_value"
        assert entry.expires_at == 1234567890.0
        assert entry.created_at == 1234567880.0
        assert entry.access_count == 5
        assert entry.last_accessed == 1234567895.0

    def test_cache_entry_defaults(self):
        """Test cache entry with default values."""
        entry = CacheEntry(
            value="test_value",
            expires_at=None,
            created_at=1234567880.0,
        )
        
        assert entry.value == "test_value"
        assert entry.expires_at is None
        assert entry.created_at == 1234567880.0
        assert entry.access_count == 0
        assert entry.last_accessed == 0.0


class TestMemoryCache:
    """Test cases for MemoryCache class."""

    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache instance."""
        return MemoryCache(
            name="test_cache",
            max_size=3,
            default_ttl=60,
            cleanup_interval=0,  # Disable automatic cleanup for tests
        )

    @pytest.fixture
    def memory_cache_with_ttl(self):
        """Create a memory cache with a short TTL for testing expiration."""
        return MemoryCache(
            name="test_cache_ttl",
            max_size=10,
            default_ttl=0.1,  # 100ms TTL
            cleanup_interval=0,  # Disable automatic cleanup for tests
        )

    @pytest.mark.asyncio
    async def test_memory_cache_initialization(self):
        """Test initializing a memory cache."""
        cache = MemoryCache(
            name="test_cache",
            max_size=100,
            default_ttl=30,
            cleanup_interval=60,
        )
        
        assert cache.name == "test_cache"
        assert cache.max_size == 100
        assert cache.default_ttl == 30
        assert cache.cleanup_interval == 60
        assert len(cache._cache) == 0
        assert cache._stats["current_size"] == 0

    @pytest.mark.asyncio
    async def test_memory_cache_start_stop(self):
        """Test starting and stopping the memory cache."""
        cache = MemoryCache(cleanup_interval=0.1)  # Short interval for testing
        
        # Start the cache
        await cache.start()
        assert cache._cleanup_task is not None
        
        # Stop the cache
        await cache.stop()
        assert cache._cleanup_task is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_cache):
        """Test setting and getting a value."""
        # Set a value
        result = await memory_cache.set("test_key", "test_value")
        assert result is True
        
        # Get the value
        value = await memory_cache.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_cache):
        """Test getting a non-existent key."""
        value = await memory_cache.get("nonexistent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, memory_cache):
        """Test setting a value with a custom TTL."""
        # Set a value with a short TTL
        result = await memory_cache.set("test_key", "test_value", ttl=0.1)
        assert result is True
        
        # Get the value immediately
        value = await memory_cache.get("test_key")
        assert value == "test_value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Get the value after expiration
        value = await memory_cache.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(self, memory_cache_with_ttl):
        """Test setting a value with the default TTL."""
        # Set a value (should use default TTL)
        result = await memory_cache_with_ttl.set("test_key", "test_value")
        assert result is True
        
        # Get the value immediately
        value = await memory_cache_with_ttl.get("test_key")
        assert value == "test_value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Get the value after expiration
        value = await memory_cache_with_ttl.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_without_ttl(self):
        """Test setting a value without expiration."""
        cache = MemoryCache(default_ttl=None)  # No default TTL
        
        # Set a value
        result = await cache.set("test_key", "test_value")
        assert result is True
        
        # Get the value immediately
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Wait a bit and get the value again (should still be there)
        await asyncio.sleep(0.1)
        value = await cache.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_lru_eviction(self, memory_cache):
        """Test LRU eviction when cache is full."""
        # Fill the cache to max size
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # All keys should be present
        assert await memory_cache.get("key1") == "value1"
        assert await memory_cache.get("key2") == "value2"
        assert await memory_cache.get("key3") == "value3"
        
        # Add one more key (should evict key1, the least recently used)
        await memory_cache.set("key4", "value4")
        
        # key1 should be evicted, others should remain
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") == "value2"
        assert await memory_cache.get("key3") == "value3"
        assert await memory_cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_lru_eviction_with_access(self, memory_cache):
        """Test LRU eviction with access updates."""
        # Fill the cache to max size
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        await memory_cache.get("key1")
        
        # Add one more key (should evict key2, the least recently used)
        await memory_cache.set("key4", "value4")
        
        # key2 should be evicted, others should remain
        assert await memory_cache.get("key1") == "value1"  # Was accessed
        assert await memory_cache.get("key2") is None  # Was evicted
        assert await memory_cache.get("key3") == "value3"
        assert await memory_cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_delete(self, memory_cache):
        """Test deleting a key."""
        # Set a value
        await memory_cache.set("test_key", "test_value")
        
        # Delete the key
        result = await memory_cache.delete("test_key")
        assert result is True
        
        # Key should be gone
        value = await memory_cache.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, memory_cache):
        """Test deleting a non-existent key."""
        result = await memory_cache.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, memory_cache):
        """Test checking if a key exists."""
        # Set a value
        await memory_cache.set("test_key", "test_value")
        
        # Check if key exists
        result = await memory_cache.exists("test_key")
        assert result is True
        
        # Check if non-existent key exists
        result = await memory_cache.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_expired(self, memory_cache_with_ttl):
        """Test checking if an expired key exists."""
        # Set a value with short TTL
        await memory_cache_with_ttl.set("test_key", "test_value")
        
        # Check if key exists immediately
        result = await memory_cache_with_ttl.exists("test_key")
        assert result is True
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Check if key exists after expiration
        result = await memory_cache_with_ttl.exists("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, memory_cache):
        """Test clearing all keys."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Clear the cache
        result = await memory_cache.clear()
        assert result is True
        
        # All keys should be gone
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") is None
        assert await memory_cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_keys_all(self, memory_cache):
        """Test getting all keys."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Get all keys
        keys = await memory_cache.keys("*")
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    @pytest.mark.asyncio
    async def test_keys_with_pattern(self, memory_cache):
        """Test getting keys with a pattern."""
        # Set some values
        await memory_cache.set("test_key1", "value1")
        await memory_cache.set("test_key2", "value2")
        await memory_cache.set("other_key", "value3")
        
        # Get keys with pattern
        keys = await memory_cache.keys("test_*")
        assert len(keys) == 2
        assert "test_key1" in keys
        assert "test_key2" in keys
        assert "other_key" not in keys

    @pytest.mark.asyncio
    async def test_keys_exact_match(self, memory_cache):
        """Test getting keys with exact match."""
        # Set some values
        await memory_cache.set("test_key", "value1")
        await memory_cache.set("other_key", "value2")
        
        # Get exact match
        keys = await memory_cache.keys("test_key")
        assert len(keys) == 1
        assert "test_key" in keys

    @pytest.mark.asyncio
    async def test_keys_no_match(self, memory_cache):
        """Test getting keys with no matching pattern."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        # Get keys with non-matching pattern
        keys = await memory_cache.keys("nonexistent_*")
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_close(self, memory_cache):
        """Test closing the cache."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        # Close the cache
        await memory_cache.close()
        
        # Cache should be empty
        keys = await memory_cache.keys("*")
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, memory_cache_with_ttl):
        """Test cleaning up expired entries."""
        # Set some values with short TTL
        await memory_cache_with_ttl.set("key1", "value1")
        await memory_cache_with_ttl.set("key2", "value2")
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Clean up expired entries
        removed_count = await memory_cache_with_ttl._cleanup_expired()
        
        # Should have removed 2 entries
        assert removed_count == 2
        
        # Cache should be empty
        keys = await memory_cache_with_ttl.keys("*")
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_cleanup_loop(self):
        """Test the cleanup loop."""
        # Create a cache with short cleanup interval
        cache = MemoryCache(
            name="test_cache",
            max_size=10,
            default_ttl=0.1,  # 100ms TTL
            cleanup_interval=0.05,  # 50ms cleanup interval
        )
        
        # Start the cache
        await cache.start()
        
        # Set a value
        await cache.set("test_key", "test_value")
        
        # Value should exist initially
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Wait for cleanup to run
        await asyncio.sleep(0.2)
        
        # Value should be gone after cleanup
        value = await cache.get("test_key")
        assert value is None
        
        # Stop the cache
        await cache.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_cache):
        """Test getting cache statistics."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        # Get some values
        await memory_cache.get("key1")  # Hit
        await memory_cache.get("key2")  # Hit
        await memory_cache.get("nonexistent")  # Miss
        
        # Delete a value
        await memory_cache.delete("key1")
        
        # Get stats
        stats = await memory_cache.get_stats()
        assert stats["name"] == "test_cache"
        assert stats["max_size"] == 3
        assert stats["current_size"] == 1  # Only key2 remains
        assert stats["utilization"] == 1/3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        assert stats["sets"] == 2
        assert stats["deletes"] == 1
        assert stats["default_ttl"] == 60

    @pytest.mark.asyncio
    async def test_get_detailed_stats(self, memory_cache):
        """Test getting detailed cache statistics."""
        # Set some values
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        # Access key1 multiple times
        await memory_cache.get("key1")
        await memory_cache.get("key1")
        
        # Wait a bit for age differences
        await asyncio.sleep(0.01)
        
        # Set another value
        await memory_cache.set("key3", "value3")
        
        # Get detailed stats
        stats = await memory_cache.get_detailed_stats()
        
        # Check basic stats
        assert stats["name"] == "test_cache"
        assert stats["current_size"] == 3
        
        # Check entry stats
        entry_stats = stats["entry_stats"]
        assert entry_stats["total_entries"] == 3
        assert entry_stats["average_access_count"] >= 1  # At least 1 access on average
        assert entry_stats["oldest_entry_age"] >= entry_stats["newest_entry_age"]

    @pytest.mark.asyncio
    async def test_access_count_updates(self, memory_cache):
        """Test that access count is updated correctly."""
        # Set a value
        await memory_cache.set("test_key", "test_value")
        
        # Get the value multiple times
        await memory_cache.get("test_key")
        await memory_cache.get("test_key")
        await memory_cache.get("test_key")
        
        # Check the entry directly
        entry = memory_cache._cache["test_key"]
        assert entry.access_count == 3

    @pytest.mark.asyncio
    async def test_last_accessed_updates(self, memory_cache):
        """Test that last accessed time is updated correctly."""
        # Set a value
        await memory_cache.set("test_key", "test_value")
        initial_time = time.time()
        
        # Wait a bit
        await asyncio.sleep(0.01)
        
        # Get the value
        await memory_cache.get("test_key")
        access_time = time.time()
        
        # Check the entry directly
        entry = memory_cache._cache["test_key"]
        assert entry.last_accessed >= initial_time
        assert entry.last_accessed <= access_time

    @pytest.mark.asyncio
    async def test_eviction_oldest(self, memory_cache):
        """Test that the oldest entry is evicted."""
        # Fill the cache
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Access key3 to make it most recently used
        await memory_cache.get("key3")
        
        # Add one more to trigger eviction
        await memory_cache.set("key4", "value4")
        
        # key1 should be evicted (oldest)
        assert await memory_cache.get("key1") is None
        assert await memory_cache.get("key2") == "value2"
        assert await memory_cache.get("key3") == "value3"
        assert await memory_cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_eviction_stats(self, memory_cache):
        """Test that eviction statistics are updated."""
        # Fill the cache
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Add one more to trigger eviction
        await memory_cache.set("key4", "value4")
        
        # Check eviction stats
        assert memory_cache._stats["evictions"] == 1

    @pytest.mark.asyncio
    async def test_set_existing_key(self, memory_cache):
        """Test setting an existing key."""
        # Set a value
        await memory_cache.set("test_key", "value1")
        
        # Set the same key with a new value
        await memory_cache.set("test_key", "value2")
        
        # Get the value
        value = await memory_cache.get("test_key")
        assert value == "value2"
        
        # Should still be only one item in cache
        stats = await memory_cache.get_stats()
        assert stats["current_size"] == 1

    @pytest.mark.asyncio
    async def test_set_existing_key_no_eviction(self, memory_cache):
        """Test that setting an existing key doesn't trigger eviction."""
        # Fill the cache
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Update an existing key
        await memory_cache.set("key1", "new_value1")
        
        # All keys should still be present
        assert await memory_cache.get("key1") == "new_value1"
        assert await memory_cache.get("key2") == "value2"
        assert await memory_cache.get("key3") == "value3"
        
        # No eviction should have occurred
        assert memory_cache._stats["evictions"] == 0


class TestMemoryCacheLayer:
    """Test cases for MemoryCacheLayer class."""

    @pytest.fixture
    def memory_cache_layer(self):
        """Create a memory cache layer."""
        return MemoryCacheLayer(
            name="test_layer",
            priority=1,
            max_size=100,
            default_ttl=60,
            cleanup_interval=0,
        )

    @pytest.mark.asyncio
    async def test_memory_cache_layer_initialization(self, memory_cache_layer):
        """Test initializing a memory cache layer."""
        assert memory_cache_layer.cache.name == "test_layer"
        assert memory_cache_layer.cache.max_size == 100
        assert memory_cache_layer.cache.default_ttl == 60
        assert memory_cache_layer.layer.name == "test_layer"
        assert memory_cache_layer.layer.priority == 1

    @pytest.mark.asyncio
    async def test_memory_cache_layer_start_stop(self, memory_cache_layer):
        """Test starting and stopping a memory cache layer."""
        # Start the layer
        await memory_cache_layer.start()
        
        # Use the cache
        await memory_cache_layer.layer.backend.set("test_key", "test_value")
        value = await memory_cache_layer.layer.backend.get("test_key")
        assert value == "test_value"
        
        # Stop the layer
        await memory_cache_layer.stop()
        
        # Cache should be empty
        keys = await memory_cache_layer.layer.backend.keys("*")
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_memory_cache_layer_integration(self, memory_cache_layer):
        """Test that the memory cache layer works correctly."""
        # Start the layer
        await memory_cache_layer.start()
        
        # Test layer operations
        assert await memory_cache_layer.layer.set("test_key", "test_value") is True
        assert await memory_cache_layer.layer.get("test_key") == "test_value"
        assert await memory_cache_layer.layer.exists("test_key") is True
        assert await memory_cache_layer.layer.delete("test_key") is True
        assert await memory_cache_layer.layer.exists("test_key") is False
        
        # Stop the layer
        await memory_cache_layer.stop()