"""
Redis cache implementation for distributed caching.

This module provides a Redis-based cache backend for distributed caching
across multiple instances of the AI Assistant.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
import logging

from ..base import CacheBackend, CacheLayer

# Try to import Redis, but make it optional
_redis_available = False
try:
    import redis.asyncio as redis

    _redis_available = True
except ImportError:
    # Try synchronous redis as fallback
    try:
        import sync_redis

        # Create async wrapper for synchronous redis
        class AsyncRedisWrapper:
            def __init__(self, *args, **kwargs):
                self._redis = sync_redis.Redis(*args, **kwargs)

            async def get(self, key: str) -> Optional[bytes]:
                return self._redis.get(key)

            async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
                result = self._redis.set(key, value, ex=ex)
                return result is True

            async def delete(self, key: str) -> bool:
                result = self._redis.delete(key)
                return result > 0

            async def exists(self, key: str) -> bool:
                return self._redis.exists(key) > 0

            async def keys(self, pattern: str = "*") -> List[str]:
                return [
                    key.decode("utf-8") if isinstance(key, bytes) else key
                    for key in self._redis.keys(pattern)
                ]

            async def close(self) -> None:
                self._redis.close()

        redis = AsyncRedisWrapper
        _redis_available = True
    except ImportError:
        # Redis is not available at all
        redis = None
        _redis_available = False

logger = logging.getLogger(__name__)


class RedisCache(CacheBackend):
    """
    Redis-based cache backend for distributed caching.

    Provides a Redis implementation of the cache backend with support
    for TTL, serialization, and connection management.
    """

    def __init__(
        self,
        name: str = "redis_cache",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
        max_connections: int = 10,
        connection_timeout: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
    ):

        if not _redis_available:
            raise ImportError(
                "Redis is not installed. Please install the 'redis' package to use RedisCache. "
                "Run: pip install redis"
            )
        """
        Initialize the Redis cache.
        
        Args:
            name: Cache name for identification
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (optional)
            default_ttl: Default time-to-live in seconds
            max_connections: Maximum number of connections
            connection_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts for operations
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "connection_errors": 0,
        }

    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established."""
        if self._connected and self._client:
            return True

        async with self._lock:
            if self._connected and self._client:
                return True

            try:
                # Create Redis client
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    max_connections=self.max_connections,
                    socket_connect_timeout=self.connection_timeout,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                await self._client.ping()
                self._connected = True
                logger.info(
                    f"Connected to Redis at {self.host}:{self.port} (db {self.db})"
                )
                return True

            except Exception as e:
                self._stats["connection_errors"] += 1
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
                self._client = None
                return False

    async def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """
        Retry a Redis operation with exponential backoff.

        Args:
            operation: The operation function to retry
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                if not await self._ensure_connected():
                    raise ConnectionError("Redis connection failed")

                result = await operation(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(
                        self.retry_delay * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    self._stats["errors"] += 1
                    logger.error(
                        f"Redis operation failed after {self.retry_attempts} attempts: {e}"
                    )
                    raise last_exception

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""

        async def _get():
            value = await self._client.get(key)
            if value is None:
                self._stats["misses"] += 1
                return None

            try:
                # Try to deserialize JSON
                decoded_value = (
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
                deserialized = json.loads(decoded_value)
                self._stats["hits"] += 1
                return deserialized
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Return raw value if not JSON
                self._stats["hits"] += 1
                return value

        return await self._retry_operation(_get)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis."""

        async def _set():
            # Serialize value to JSON if it's a complex type
            if not isinstance(value, (str, bytes, int, float)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = value

            ttl_seconds = ttl or self.default_ttl

            if ttl_seconds is not None:
                result = await self._client.set(key, serialized_value, ex=ttl_seconds)
            else:
                result = await self._client.set(key, serialized_value)

            self._stats["sets"] += 1
            return result is True

        return await self._retry_operation(_set)

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""

        async def _delete():
            result = await self._client.delete(key)
            self._stats["deletes"] += 1
            return result > 0

        return await self._retry_operation(_delete)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""

        async def _exists():
            result = await self._client.exists(key)
            return result > 0

        return await self._retry_operation(_exists)

    async def clear(self) -> bool:
        """Clear all keys from Redis (use with caution!)."""

        async def _clear():
            # Note: This clears the entire database!
            result = await self._client.flushdb()
            return result is True

        logger.warning(f"Clearing entire Redis database {self.db}")
        return await self._retry_operation(_clear)

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern from Redis."""

        async def _keys():
            redis_keys = await self._client.keys(pattern)
            return [
                key.decode("utf-8") if isinstance(key, bytes) else key
                for key in redis_keys
            ]

        return await self._retry_operation(_keys)

    async def close(self) -> None:
        """Close the Redis connection."""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._connected = False
                self._client = None
                logger.info("Closed Redis connection")

    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""

        async def _info():
            if not await self._ensure_connected():
                return {}

            info = await self._client.info()
            return info

        try:
            return await self._retry_operation(_info)
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}

    async def get_memory_info(self) -> Dict[str, Any]:
        """Get Redis memory information."""

        async def _memory_info():
            if not await self._ensure_connected():
                return {}

            try:
                # Try to get memory stats
                memory_info = await self._client.info("memory")
                return {
                    "used_memory": memory_info.get("used_memory", 0),
                    "used_memory_human": memory_info.get("used_memory_human", "0B"),
                    "used_memory_peak": memory_info.get("used_memory_peak", 0),
                    "used_memory_peak_human": memory_info.get(
                        "used_memory_peak_human", "0B"
                    ),
                    "used_memory_rss": memory_info.get("used_memory_rss", 0),
                    "used_memory_rss_human": memory_info.get(
                        "used_memory_rss_human", "0B"
                    ),
                }
            except Exception:
                # Fallback to basic memory command
                try:
                    memory_stats = await self._client.execute_command("MEMORY STATS")
                    return memory_stats
                except Exception:
                    return {}

        try:
            return await self._retry_operation(_memory_info)
        except Exception as e:
            logger.error(f"Failed to get Redis memory info: {e}")
            return {}

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0
            else 0.0
        )

        stats = {
            "name": self.name,
            "connected": self._connected,
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "errors": self._stats["errors"],
            "connection_errors": self._stats["connection_errors"],
            "default_ttl": self.default_ttl,
        }

        # Add Redis server info if connected
        if self._connected:
            try:
                info = await self.get_info()
                memory_info = await self.get_memory_info()

                stats["redis_info"] = {
                    "version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": memory_info.get("used_memory", 0),
                    "used_memory_human": memory_info.get("used_memory_human", "0B"),
                }
            except Exception as e:
                logger.warning(f"Failed to get Redis server info: {e}")

        return stats

    async def ping(self) -> bool:
        """Ping Redis server to check connectivity."""
        try:
            if not await self._ensure_connected():
                return False

            result = await self._client.ping()
            return result is True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False


# Convenience class for creating Redis cache layers
class RedisCacheLayer:
    """
    Convenience class for creating Redis cache layers.

    Wraps RedisCache with CacheLayer interface.
    """

    def __init__(
        self,
        name: str = "redis_cache",
        priority: int = 1,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
        max_connections: int = 10,
        connection_timeout: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
    ):
        """
        Initialize the Redis cache layer.

        Args:
            name: Layer name
            priority: Layer priority
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            default_ttl: Default TTL
            max_connections: Max connections
            connection_timeout: Connection timeout
            retry_attempts: Retry attempts
            retry_delay: Retry delay
        """
        self.cache = RedisCache(
            name=name,
            host=host,
            port=port,
            db=db,
            password=password,
            default_ttl=default_ttl,
            max_connections=max_connections,
            connection_timeout=connection_timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )
        self.layer = CacheLayer(
            backend=self.cache,
            name=name,
            priority=priority,
            read_only=False,
            default_ttl=default_ttl,
        )

    async def ping(self) -> bool:
        """Ping Redis server."""
        return await self.cache.ping()

    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        return await self.cache.get_info()
