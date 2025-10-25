"""
Unit tests for Redis cache layer implementation.

Tests for Redis-based cache backend with connection management, serialization, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from app.core.caching.layers.redis_cache import (
    RedisCache,
    RedisCacheLayer,
)


class TestRedisCache:
    """Test cases for RedisCache class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = 0
        mock_client.keys.return_value = []
        mock_client.flushdb.return_value = True
        mock_client.info.return_value = {
            "redis_version": "6.0.0",
            "uptime_in_seconds": 1000,
            "connected_clients": 5,
        }
        mock_client.execute_command.return_value = {
            "used_memory": 1024,
            "used_memory_human": "1KB",
        }
        return mock_client

    @pytest.fixture
    def redis_cache(self):
        """Create a Redis cache instance."""
        return RedisCache(
            name="test_redis",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            default_ttl=60,
            max_connections=10,
            connection_timeout=5,
            retry_attempts=3,
            retry_delay=0.1,
        )

    @pytest.fixture
    def redis_cache_with_auth(self):
        """Create a Redis cache instance with authentication."""
        return RedisCache(
            name="test_redis_auth",
            host="localhost",
            port=6379,
            db=1,
            password="test_password",
            default_ttl=30,
        )

    @pytest.mark.asyncio
    async def test_redis_cache_initialization(self, redis_cache):
        """Test initializing a Redis cache."""
        assert redis_cache.name == "test_redis"
        assert redis_cache.host == "localhost"
        assert redis_cache.port == 6379
        assert redis_cache.db == 0
        assert redis_cache.password is None
        assert redis_cache.default_ttl == 60
        assert redis_cache.max_connections == 10
        assert redis_cache.connection_timeout == 5
        assert redis_cache.retry_attempts == 3
        assert redis_cache.retry_delay == 0.1
        assert redis_cache._client is None
        assert redis_cache._connected is False

    @pytest.mark.asyncio
    async def test_ensure_connected_success(self, redis_cache, mock_redis):
        """Test successful connection to Redis."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect to Redis
            result = await redis_cache._ensure_connected()
            
            # Should return True and set up client
            assert result is True
            assert redis_cache._client is not None
            assert redis_cache._connected is True

    @pytest.mark.asyncio
    async def test_ensure_connected_failure(self, redis_cache):
        """Test failed connection to Redis."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection failure
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_redis.Redis.return_value = mock_client
            
            # Try to connect
            result = await redis_cache._ensure_connected()
            
            # Should return False and not set up client
            assert result is False
            assert redis_cache._client is None
            assert redis_cache._connected is False
            assert redis_cache._stats["connection_errors"] == 1

    @pytest.mark.asyncio
    async def test_ensure_connected_already_connected(self, redis_cache, mock_redis):
        """Test that _ensure_connected returns early if already connected."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect once
            await redis_cache._ensure_connected()
            
            # Call again (should return early)
            result = await redis_cache._ensure_connected()
            
            # Should return True without creating new client
            assert result is True
            assert redis_cache._client is mock_redis.Redis.return_value

    @pytest.mark.asyncio
    async def test_retry_operation_success(self, redis_cache, mock_redis):
        """Test successful operation with retries."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect first
            await redis_cache._ensure_connected()
            
            # Mock operation
            operation = AsyncMock(return_value="success")
            
            # Retry operation
            result = await redis_cache._retry_operation(operation, "arg1", kwarg1="value1")
            
            # Should return success
            assert result == "success"
            operation.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_retry_operation_with_retries(self, redis_cache):
        """Test operation that fails initially but succeeds after retries."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock operation that fails twice then succeeds
            operation = AsyncMock(side_effect=[Exception("Fail 1"), Exception("Fail 2"), "success"])
            
            # Retry operation
            result = await redis_cache._retry_operation(operation)
            
            # Should return success after retries
            assert result == "success"
            assert operation.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_operation_exhausted(self, redis_cache):
        """Test operation that fails after all retries."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock operation that always fails
            operation = AsyncMock(side_effect=Exception("Always fails"))
            
            # Retry operation (should raise exception)
            with pytest.raises(Exception, match="Always fails"):
                await redis_cache._retry_operation(operation)
            
            # Should have tried 3 times
            assert operation.call_count == 3
            assert redis_cache._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_get_hit(self, redis_cache, mock_redis):
        """Test getting a value that exists in Redis."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock get returning a JSON string
            mock_redis.Redis.return_value.get.return_value = json.dumps({"key": "value"})
            
            # Get value
            result = await redis_cache.get("test_key")
            
            # Should return deserialized value
            assert result == {"key": "value"}
            assert redis_cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_get_miss(self, redis_cache, mock_redis):
        """Test getting a value that doesn't exist in Redis."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock get returning None
            mock_redis.Redis.return_value.get.return_value = None
            
            # Get value
            result = await redis_cache.get("test_key")
            
            # Should return None
            assert result is None
            assert redis_cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_raw_string(self, redis_cache, mock_redis):
        """Test getting a raw string value."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock get returning a raw string
            mock_redis.Redis.return_value.get.return_value = "raw_string"
            
            # Get value
            result = await redis_cache.get("test_key")
            
            # Should return the raw string
            assert result == "raw_string"
            assert redis_cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_get_bytes(self, redis_cache, mock_redis):
        """Test getting bytes value."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock get returning bytes
            mock_redis.Redis.return_value.get.return_value = b"raw_bytes"
            
            # Get value
            result = await redis_cache.get("test_key")
            
            # Should return the bytes
            assert result == b"raw_bytes"
            assert redis_cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_get_invalid_json(self, redis_cache, mock_redis):
        """Test getting invalid JSON."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock get returning invalid JSON
            mock_redis.Redis.return_value.get.return_value = "invalid_json"
            
            # Get value
            result = await redis_cache.get("test_key")
            
            # Should return the raw string
            assert result == "invalid_json"
            assert redis_cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, redis_cache, mock_redis):
        """Test setting a value with TTL."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock set returning True
            mock_redis.Redis.return_value.set.return_value = True
            
            # Set value with TTL
            result = await redis_cache.set("test_key", {"key": "value"}, ttl=30)
            
            # Should return True
            assert result is True
            assert redis_cache._stats["sets"] == 1
            
            # Check that set was called with TTL
            mock_redis.Redis.return_value.set.assert_called_once_with(
                "test_key", json.dumps({"key": "value"}), ex=30
            )

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(self, redis_cache, mock_redis):
        """Test setting a value with default TTL."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock set returning True
            mock_redis.Redis.return_value.set.return_value = True
            
            # Set value without TTL
            result = await redis_cache.set("test_key", {"key": "value"})
            
            # Should return True
            assert result is True
            assert redis_cache._stats["sets"] == 1
            
            # Check that set was called with default TTL
            mock_redis.Redis.return_value.set.assert_called_once_with(
                "test_key", json.dumps({"key": "value"}), ex=60
            )

    @pytest.mark.asyncio
    async def test_set_without_ttl(self):
        """Test setting a value without TTL."""
        cache = RedisCache(default_ttl=None)
        
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            # Connect
            await cache._ensure_connected()
            
            # Mock set returning True
            mock_client.set.return_value = True
            
            # Set value without TTL
            result = await cache.set("test_key", {"key": "value"})
            
            # Should return True
            assert result is True
            assert cache._stats["sets"] == 1
            
            # Check that set was called without TTL
            mock_client.set.assert_called_once_with(
                "test_key", json.dumps({"key": "value"})
            )

    @pytest.mark.asyncio
    async def test_set_raw_value(self, redis_cache, mock_redis):
        """Test setting a raw string value."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock set returning True
            mock_redis.Redis.return_value.set.return_value = True
            
            # Set raw string value
            result = await redis_cache.set("test_key", "raw_string")
            
            # Should return True
            assert result is True
            assert redis_cache._stats["sets"] == 1
            
            # Check that set was called with raw string
            mock_redis.Redis.return_value.set.assert_called_once_with(
                "test_key", "raw_string", ex=60
            )

    @pytest.mark.asyncio
    async def test_delete_success(self, redis_cache, mock_redis):
        """Test successfully deleting a key."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock delete returning 1 (key was deleted)
            mock_redis.Redis.return_value.delete.return_value = 1
            
            # Delete key
            result = await redis_cache.delete("test_key")
            
            # Should return True
            assert result is True
            assert redis_cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_delete_not_found(self, redis_cache, mock_redis):
        """Test deleting a key that doesn't exist."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock delete returning 0 (key was not found)
            mock_redis.Redis.return_value.delete.return_value = 0
            
            # Delete key
            result = await redis_cache.delete("test_key")
            
            # Should return False
            assert result is False
            assert redis_cache._stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_exists_true(self, redis_cache, mock_redis):
        """Test checking if a key exists (exists)."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock exists returning 1 (key exists)
            mock_redis.Redis.return_value.exists.return_value = 1
            
            # Check if key exists
            result = await redis_cache.exists("test_key")
            
            # Should return True
            assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, redis_cache, mock_redis):
        """Test checking if a key exists (doesn't exist)."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock exists returning 0 (key doesn't exist)
            mock_redis.Redis.return_value.exists.return_value = 0
            
            # Check if key exists
            result = await redis_cache.exists("test_key")
            
            # Should return False
            assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, redis_cache, mock_redis):
        """Test clearing all keys."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock flushdb returning True
            mock_redis.Redis.return_value.flushdb.return_value = True
            
            # Clear cache
            result = await redis_cache.clear()
            
            # Should return True
            assert result is True

    @pytest.mark.asyncio
    async def test_keys_all(self, redis_cache, mock_redis):
        """Test getting all keys."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock keys returning a list of bytes
            mock_redis.Redis.return_value.keys.return_value = [b"key1", b"key2", b"key3"]
            
            # Get all keys
            result = await redis_cache.keys("*")
            
            # Should return decoded strings
            assert result == ["key1", "key2", "key3"]

    @pytest.mark.asyncio
    async def test_keys_with_pattern(self, redis_cache, mock_redis):
        """Test getting keys with a pattern."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Mock keys returning a list of strings
            mock_redis.Redis.return_value.keys.return_value = ["test_key1", "test_key2"]
            
            # Get keys with pattern
            result = await redis_cache.keys("test_*")
            
            # Should return the matching keys
            assert result == ["test_key1", "test_key2"]

    @pytest.mark.asyncio
    async def test_close(self, redis_cache, mock_redis):
        """Test closing the Redis connection."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Close connection
            await redis_cache.close()
            
            # Should have called close on client
            mock_redis.Redis.return_value.close.assert_called_once()
            
            # Should reset connection state
            assert redis_cache._connected is False
            assert redis_cache._client is None

    @pytest.mark.asyncio
    async def test_get_info(self, redis_cache, mock_redis):
        """Test getting Redis server information."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Get info
            result = await redis_cache.get_info()
            
            # Should return the info dictionary
            assert result == {
                "redis_version": "6.0.0",
                "uptime_in_seconds": 1000,
                "connected_clients": 5,
            }

    @pytest.mark.asyncio
    async def test_get_info_not_connected(self, redis_cache):
        """Test getting info when not connected."""
        # Get info without connecting
        result = await redis_cache.get_info()
        
        # Should return empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_memory_info(self, redis_cache, mock_redis):
        """Test getting Redis memory information."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Get memory info
            result = await redis_cache.get_memory_info()
            
            # Should return the memory info dictionary
            assert result == {
                "used_memory": 1024,
                "used_memory_human": "1KB",
                "used_memory_peak": 0,
                "used_memory_peak_human": "0B",
                "used_memory_rss": 0,
                "used_memory_rss_human": "0B",
            }

    @pytest.mark.asyncio
    async def test_get_memory_info_fallback(self, redis_cache):
        """Test getting memory info with fallback to MEMORY STATS."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.info.side_effect = [
                Exception("Info failed"),
                {"used_memory": 1024, "used_memory_human": "1KB"},
            ]
            mock_client.execute_command.return_value = {"used_memory": 1024}
            mock_redis.Redis.return_value = mock_client
            
            # Connect
            await redis_cache._ensure_connected()
            
            # Get memory info
            result = await redis_cache.get_memory_info()
            
            # Should return the memory info from execute_command
            assert result == {"used_memory": 1024}

    @pytest.mark.asyncio
    async def test_get_stats(self, redis_cache, mock_redis):
        """Test getting cache statistics."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Set up some stats
            redis_cache._stats["hits"] = 10
            redis_cache._stats["misses"] = 5
            redis_cache._stats["sets"] = 8
            redis_cache._stats["deletes"] = 2
            redis_cache._stats["errors"] = 1
            
            # Get stats
            result = await redis_cache.get_stats()
            
            # Should return the stats dictionary
            assert result["name"] == "test_redis"
            assert result["connected"] is True
            assert result["host"] == "localhost"
            assert result["port"] == 6379
            assert result["db"] == 0
            assert result["hits"] == 10
            assert result["misses"] == 5
            assert result["hit_rate"] == 10/15
            assert result["sets"] == 8
            assert result["deletes"] == 2
            assert result["errors"] == 1
            assert result["default_ttl"] == 60

    @pytest.mark.asyncio
    async def test_get_stats_with_redis_info(self, redis_cache, mock_redis):
        """Test getting stats with Redis server info."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Get stats
            result = await redis_cache.get_stats()
            
            # Should include Redis server info
            assert "redis_info" in result
            assert result["redis_info"]["version"] == "6.0.0"
            assert result["redis_info"]["uptime_seconds"] == 1000
            assert result["redis_info"]["connected_clients"] == 5

    @pytest.mark.asyncio
    async def test_ping_success(self, redis_cache, mock_redis):
        """Test successful ping to Redis."""
        with patch('app.core.caching.layers.redis_cache.redis', mock_redis):
            # Connect
            await redis_cache._ensure_connected()
            
            # Ping Redis
            result = await redis_cache.ping()
            
            # Should return True
            assert result is True

    @pytest.mark.asyncio
    async def test_ping_failure(self, redis_cache):
        """Test failed ping to Redis."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection failure
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Ping failed")
            mock_redis.Redis.return_value = mock_client
            
            # Try to ping
            result = await redis_cache.ping()
            
            # Should return False
            assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, redis_cache):
        """Test handling of connection errors."""
        with patch('app.core.caching.layers.redis_cache.redis') as mock_redis:
            # Mock connection failure
            mock_redis.Redis.side_effect = Exception("Connection failed")
            
            # Try to get a value (should fail gracefully)
            result = await redis_cache.get("test_key")
            
            # Should return None
            assert result is None
            assert redis_cache._stats["connection_errors"] == 1

    @pytest.mark.asyncio
    async def test_operation_without_connection(self, redis_cache):
        """Test operations without connection."""
        # Don't connect, just try to get a value
        result = await redis_cache.get("test_key")
        
        # Should return None
        assert result is None


class TestRedisCacheLayer:
    """Test cases for RedisCacheLayer class."""

    @pytest.fixture
    def redis_cache_layer(self):
        """Create a Redis cache layer."""
        return RedisCacheLayer(
            name="test_layer",
            priority=1,
            host="localhost",
            port=6379,
            db=0,
            password=None,
            default_ttl=60,
            max_connections=10,
            connection_timeout=5,
            retry_attempts=3,
            retry_delay=0.1,
        )

    def test_redis_cache_layer_initialization(self, redis_cache_layer):
        """Test initializing a Redis cache layer."""
        assert redis_cache_layer.cache.name == "test_layer"
        assert redis_cache_layer.cache.host == "localhost"
        assert redis_cache_layer.cache.port == 6379
        assert redis_cache_layer.cache.db == 0
        assert redis_cache_layer.cache.default_ttl == 60
        assert redis_cache_layer.layer.name == "test_layer"
        assert redis_cache_layer.layer.priority == 1

    @pytest.mark.asyncio
    async def test_redis_cache_layer_ping(self, redis_cache_layer):
        """Test pinging Redis through the layer."""
        with patch.object(redis_cache_layer.cache, 'ping', return_value=True) as mock_ping:
            # Ping Redis
            result = await redis_cache_layer.ping()
            
            # Should return True
            assert result is True
            mock_ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_layer_get_info(self, redis_cache_layer):
        """Test getting Redis info through the layer."""
        mock_info = {"redis_version": "6.0.0"}
        with patch.object(redis_cache_layer.cache, 'get_info', return_value=mock_info) as mock_get_info:
            # Get info
            result = await redis_cache_layer.get_info()
            
            # Should return the info
            assert result == mock_info
            mock_get_info.assert_called_once()


class TestRedisCacheImportError:
    """Test cases for Redis import error handling."""

    def test_redis_not_installed(self):
        """Test behavior when Redis is not installed."""
        with patch('app.core.caching.layers.redis_cache._redis_available', False):
            # Should raise ImportError when creating RedisCache
            with pytest.raises(ImportError, match="Redis is not installed"):
                RedisCache()