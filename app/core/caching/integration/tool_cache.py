"""
Tool cache integration for caching tool execution results.

This module provides caching integration for the tool registry system,
allowing tool results to be cached to improve performance and reduce
external API calls.
"""

import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import logging

from ...tools.registry import ToolRegistry, BaseTool, ToolResult
from ..base import MultiLayerCache, generate_cache_key
from ..compression.compressor import ResultCompressor

logger = logging.getLogger(__name__)


@dataclass
class CachedToolResult:
    """Enhanced tool result with caching metadata."""

    result: ToolResult
    cache_key: str
    cached_at: float
    ttl: int
    hit_count: int = 0


class ToolCache:
    """
    Cache manager for tool execution results.

    Provides caching capabilities for tool results with configurable
    TTL, compression, and multi-layer caching support.
    """

    def __init__(self, cache: MultiLayerCache, default_ttl: int = 3600):
        """
        Initialize the tool cache.

        Args:
            cache: Multi-layer cache instance
            default_ttl: Default time-to-live for cached results (seconds)
        """
        self.cache = cache
        self.default_ttl = default_ttl
        self.compressor = ResultCompressor()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tool_executions": 0,
            "average_execution_time": 0.0,
            "total_cache_savings": 0,
        }

    def generate_tool_cache_key(self, tool_name: str, *args, **kwargs) -> str:
        """
        Generate a cache key for a tool execution.

        Args:
            tool_name: Name of the tool
            *args: Tool arguments
            **kwargs: Tool keyword arguments

        Returns:
            Cache key string
        """
        return generate_cache_key(f"tool:{tool_name}", *args, **kwargs)

    async def execute_with_cache(
        self, tool: BaseTool, *args, ttl: Optional[int] = None, **kwargs
    ) -> ToolResult:
        """
        Execute a tool with caching support.

        Args:
            tool: The tool to execute
            *args: Tool arguments
            ttl: Custom TTL for this execution (optional)
            **kwargs: Tool keyword arguments

        Returns:
            Tool result (from cache or fresh execution)
        """
        cache_key = self.generate_tool_cache_key(tool.name, *args, **kwargs)
        ttl = ttl or self.default_ttl

        # Try to get from cache first
        cached_result = await self._get_cached_result(cache_key)
        if cached_result is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for tool '{tool.name}' with key {cache_key}")
            return cached_result.result

        # Cache miss - execute the tool
        self._stats["cache_misses"] += 1
        self._stats["tool_executions"] += 1

        start_time = time.time()
        result = await tool.execute(*args, **kwargs)
        execution_time = time.time() - start_time

        # Update execution time statistics
        self._update_execution_stats(execution_time)

        # Cache the result
        await self._cache_result(cache_key, result, ttl)

        logger.debug(
            f"Tool '{tool.name}' executed in {execution_time:.3f}s (cache miss)"
        )
        return result

    async def _get_cached_result(self, cache_key: str) -> Optional[CachedToolResult]:
        """Get a cached tool result."""
        cached_data = await self.cache.get(cache_key)
        if cached_data is None:
            return None

        # Deserialize the cached result
        if isinstance(cached_data, dict) and "cached_result" in cached_data:
            data = cached_data
            # Update hit count
            data["hit_count"] = data.get("hit_count", 0) + 1
            # Update the cache with new hit count
            await self.cache.set(cache_key, data, data.get("ttl", self.default_ttl))

            return CachedToolResult(
                result=data["cached_result"],
                cache_key=cache_key,
                cached_at=data["cached_at"],
                ttl=data["ttl"],
                hit_count=data["hit_count"],
            )

        return None

    async def _cache_result(self, cache_key: str, result: ToolResult, ttl: int) -> None:
        """Cache a tool result."""
        cache_data = {
            "cached_result": result,
            "cached_at": time.time(),
            "ttl": ttl,
            "hit_count": 0,
        }

        await self.cache.set(cache_key, cache_data, ttl)

    def _update_execution_stats(self, execution_time: float) -> None:
        """Update execution time statistics."""
        total_executions = self._stats["tool_executions"]
        current_avg = self._stats["average_execution_time"]

        # Calculate new average
        if total_executions == 1:
            self._stats["average_execution_time"] = execution_time
        else:
            self._stats["average_execution_time"] = (
                current_avg * (total_executions - 1) + execution_time
            ) / total_executions

    async def invalidate_tool_cache(self, tool_name: str, pattern: str = "*") -> int:
        """
        Invalidate cached results for a tool.

        Args:
            tool_name: Name of the tool
            pattern: Pattern to match cache keys

        Returns:
            Number of cache entries invalidated
        """
        cache_pattern = f"tool:{tool_name}:{pattern}"
        keys = []

        # Get matching keys from all cache layers
        for layer in self.cache.layers:
            layer_keys = await layer.keys(cache_pattern)
            keys.extend(layer_keys)

        # Remove duplicates
        keys = list(set(keys))

        # Delete the keys
        for key in keys:
            await self.cache.delete(key)

        logger.info(f"Invalidated {len(keys)} cache entries for tool '{tool_name}'")
        return len(keys)

    async def clear_all_tool_cache(self) -> int:
        """Clear all tool cache entries."""
        keys = []

        # Get all tool cache keys from all layers
        for layer in self.cache.layers:
            layer_keys = await layer.keys("tool:*")
            keys.extend(layer_keys)

        # Remove duplicates
        keys = list(set(keys))

        # Delete the keys
        for key in keys:
            await self.cache.delete(key)

        logger.info(f"Cleared {len(keys)} tool cache entries")
        return len(keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get tool cache statistics."""
        cache_stats = self.cache.get_stats()
        stats = self._stats.copy()
        stats.update(
            {
                "cache_hit_rate": (
                    self._stats["cache_hits"]
                    / (self._stats["cache_hits"] + self._stats["cache_misses"])
                    if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                    else 0.0
                ),
                "cache_layers": cache_stats,
            }
        )
        return stats


class CachedToolRegistry(ToolRegistry):
    """
    Tool registry with built-in caching support.

    Extends the standard ToolRegistry to provide automatic caching
    of tool execution results.
    """

    def __init__(self, cache: Optional[MultiLayerCache] = None):
        """
        Initialize the cached tool registry.

        Args:
            cache: Multi-layer cache instance (creates default if None)
        """
        super().__init__()
        self.tool_cache = ToolCache(cache or self._create_default_cache())

    def _create_default_cache(self) -> MultiLayerCache:
        """Create a default multi-layer cache for tool results."""
        from ..layers.memory import MemoryCache
        from ..layers.redis_cache import RedisCache

        cache = MultiLayerCache()

        # Add memory cache layer (high priority)
        memory_cache = MemoryCache(
            name="tool_memory_cache",
            priority=0,
            max_size=1000,
            default_ttl=300,  # 5 minutes for memory cache
        )
        cache.add_layer(memory_cache.layer)

        # Add Redis cache layer (lower priority)
        redis_cache = RedisCache(
            name="tool_redis_cache",
            priority=1,
            default_ttl=3600,  # 1 hour for Redis cache
        )
        cache.add_layer(redis_cache.layer)

        return cache

    async def execute_tool_with_cache(
        self, tool_name: str, *args, ttl: Optional[int] = None, **kwargs
    ) -> ToolResult:
        """
        Execute a tool with caching support.

        Args:
            tool_name: Name of the tool to execute
            *args: Tool arguments
            ttl: Custom TTL for this execution (optional)
            **kwargs: Tool keyword arguments

        Returns:
            Tool result (from cache or fresh execution)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        if not tool.enabled:
            raise ValueError(f"Tool '{tool_name}' is disabled")

        return await self.tool_cache.execute_with_cache(tool, *args, ttl=ttl, **kwargs)

    async def invalidate_tool_cache(self, tool_name: str, pattern: str = "*") -> int:
        """
        Invalidate cached results for a specific tool.

        Args:
            tool_name: Name of the tool
            pattern: Pattern to match cache keys

        Returns:
            Number of cache entries invalidated
        """
        return await self.tool_cache.invalidate_tool_cache(tool_name, pattern)

    async def clear_all_cache(self) -> int:
        """
        Clear all cached tool results.

        Returns:
            Number of cache entries cleared
        """
        return await self.tool_cache.clear_all_tool_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        return self.tool_cache.get_stats()


def cache_tool_result(ttl: int = 3600) -> Callable:
    """
    Decorator for caching tool execution results.

    Args:
        ttl: Time-to-live for cached results (seconds)

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Only cache if this is a tool execution method
            if hasattr(self, "name") and hasattr(self, "execute"):
                # Use the tool's cache if available
                if hasattr(self, "_registry") and hasattr(self._registry, "tool_cache"):
                    cache = self._registry.tool_cache
                    return await cache.execute_with_cache(
                        self, *args, ttl=ttl, **kwargs
                    )

            # Fallback to direct execution
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


# Global cached tool registry instance
_cached_tool_registry: Optional[CachedToolRegistry] = None


def get_cached_tool_registry() -> CachedToolRegistry:
    """Get the global cached tool registry instance."""
    global _cached_tool_registry
    if _cached_tool_registry is None:
        _cached_tool_registry = CachedToolRegistry()
    return _cached_tool_registry
