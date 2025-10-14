"""
In-memory cache implementation with TTL support.

This module provides a fast, in-memory cache backend using Python's
built-in data structures with support for TTL and size limits.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging
from collections import OrderedDict

from ..base import CacheBackend, CacheLayer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    value: Any
    expires_at: Optional[float]  # None for no expiration
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class MemoryCache(CacheBackend):
    """
    In-memory cache with TTL and size limits.

    Uses an ordered dictionary to maintain insertion order for LRU eviction.
    """

    def __init__(
        self,
        name: str = "memory_cache",
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
        cleanup_interval: int = 60,
    ):
        """
        Initialize the memory cache.

        Args:
            name: Cache name for identification
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds (None for no expiration)
            cleanup_interval: Interval for automatic cleanup in seconds
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Use OrderedDict for LRU eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "current_size": 0,
        }

    async def start(self) -> None:
        """Start the cache and begin cleanup task."""
        if self.cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the cache and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.expires_at is not None and time.time() > entry.expires_at:
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["current_size"] = len(self._cache)
                return None

            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Move to end for LRU (most recently used)
            self._cache.move_to_end(key)

            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        async with self._lock:
            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.default_ttl is not None:
                expires_at = time.time() + self.default_ttl

            # Create cache entry
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
                created_at=time.time(),
                last_accessed=time.time(),
            )

            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_oldest()

            # Store the entry
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as recently used

            self._stats["sets"] += 1
            self._stats["current_size"] = len(self._cache)
            return True

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                self._stats["current_size"] = len(self._cache)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache (and is not expired)."""
        async with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.expires_at is not None and time.time() > entry.expires_at:
                del self._cache[key]
                self._stats["current_size"] = len(self._cache)
                return False

            return True

    async def clear(self) -> bool:
        """Clear all keys from the cache."""
        async with self._lock:
            self._cache.clear()
            self._stats["current_size"] = 0
            return True

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern."""
        async with self._lock:
            # Simple pattern matching (supports * wildcard)
            if pattern == "*":
                return list(self._cache.keys())

            # Basic wildcard matching
            if "*" in pattern:
                prefix, suffix = pattern.split("*", 1)
                matching_keys = []
                for key in self._cache.keys():
                    if key.startswith(prefix) and key.endswith(suffix):
                        matching_keys.append(key)
                return matching_keys
            else:
                # Exact match
                return [pattern] if pattern in self._cache else []

    async def close(self) -> None:
        """Close the cache and release resources."""
        await self.stop()
        await self.clear()

    async def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return

        # Remove the first item (oldest)
        key, _ = self._cache.popitem(last=False)
        self._stats["evictions"] += 1
        logger.debug(f"Evicted key '{key}' from memory cache '{self.name}'")

    async def _cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        async with self._lock:
            current_time = time.time()
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.expires_at is not None and current_time > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(
                    f"Cleaned up {len(expired_keys)} expired entries from memory cache '{self.name}'"
                )

            self._stats["current_size"] = len(self._cache)
            return len(expired_keys)

    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cache cleanup: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            hit_rate = (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            )

            return {
                "name": self.name,
                "max_size": self.max_size,
                "current_size": self._stats["current_size"],
                "utilization": (
                    self._stats["current_size"] / self.max_size
                    if self.max_size > 0
                    else 0.0
                ),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
                "default_ttl": self.default_ttl,
            }

    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics including entry information."""
        async with self._lock:
            stats = await self.get_stats()

            # Add entry-level statistics
            entry_stats = {
                "total_entries": len(self._cache),
                "expired_entries": 0,
                "average_access_count": 0,
                "oldest_entry_age": 0,
                "newest_entry_age": 0,
            }

            if self._cache:
                current_time = time.time()
                access_counts = []
                entry_ages = []

                for entry in self._cache.values():
                    if entry.expires_at is not None and current_time > entry.expires_at:
                        entry_stats["expired_entries"] += 1

                    access_counts.append(entry.access_count)
                    entry_ages.append(current_time - entry.created_at)

                if access_counts:
                    entry_stats["average_access_count"] = int(
                        sum(access_counts) / len(access_counts)
                    )
                if entry_ages:
                    entry_stats["oldest_entry_age"] = int(max(entry_ages))
                    entry_stats["newest_entry_age"] = int(min(entry_ages))

            stats["entry_stats"] = entry_stats
            return stats


# Convenience class for creating cache layers
class MemoryCacheLayer:
    """
    Convenience class for creating memory cache layers.

    Wraps MemoryCache with CacheLayer interface.
    """

    def __init__(
        self,
        name: str = "memory_cache",
        priority: int = 0,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
        cleanup_interval: int = 60,
    ):
        """
        Initialize the memory cache layer.

        Args:
            name: Layer name
            priority: Layer priority
            max_size: Maximum cache size
            default_ttl: Default TTL
            cleanup_interval: Cleanup interval
        """
        self.cache = MemoryCache(
            name=name,
            max_size=max_size,
            default_ttl=default_ttl,
            cleanup_interval=cleanup_interval,
        )
        self.layer = CacheLayer(
            backend=self.cache,
            name=name,
            priority=priority,
            read_only=False,
            default_ttl=default_ttl,
        )

    async def start(self) -> None:
        """Start the cache."""
        await self.cache.start()

    async def stop(self) -> None:
        """Stop the cache."""
        await self.cache.stop()
