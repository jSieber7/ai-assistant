"""
Agent cache integration for caching agent processing results.

This module provides caching integration for the agent system,
allowing agent responses and processing results to be cached to
improve performance and reduce LLM API calls.
"""

import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import logging

from ...agents.management.registry import AgentRegistry
from ...agents.base.base import BaseAgent, AgentResult
from ..base import MultiLayerCache, generate_cache_key
from ..compression.compressor import ResultCompressor

logger = logging.getLogger(__name__)


@dataclass
class CachedAgentResult:
    """Enhanced agent result with caching metadata."""

    result: AgentResult
    cache_key: str
    cached_at: float
    ttl: int
    hit_count: int = 0
    conversation_id: Optional[str] = None


class AgentCache:
    """
    Cache manager for agent processing results.

    Provides caching capabilities for agent responses with configurable
    TTL, compression, and multi-layer caching support.
    """

    def __init__(self, cache: MultiLayerCache, default_ttl: int = 1800):
        """
        Initialize the agent cache.

        Args:
            cache: Multi-layer cache instance
            default_ttl: Default time-to-live for cached results (seconds)
        """
        self.cache = cache
        self.default_ttl = default_ttl
        self.compressor = ResultCompressor()
        self._stats = {
            "agent_cache_hits": 0,
            "agent_cache_misses": 0,
            "agent_processings": 0,
            "agent_average_processing_time": 0.0,
            "agent_total_cache_savings": 0,
        }

    def generate_agent_cache_key(
        self,
        agent_name: str,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a cache key for agent processing.

        Args:
            agent_name: Name of the agent
            message: User message
            conversation_id: Conversation ID (optional)
            context: Additional context (optional)

        Returns:
            Cache key string
        """
        # Include conversation ID in cache key if provided
        key_parts = [f"agent:{agent_name}", message]
        if conversation_id:
            key_parts.append(conversation_id)

        if context:
            # Sort context keys for consistency
            sorted_context = {k: context[k] for k in sorted(context.keys())}
            key_parts.append(str(sorted_context))

        return generate_cache_key(":".join(key_parts))

    async def process_with_cache(
        self,
        agent: BaseAgent,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> AgentResult:
        """
        Process a message with an agent using caching support.

        Args:
            agent: The agent to use
            message: User message to process
            conversation_id: Conversation ID for context
            context: Additional context information
            ttl: Custom TTL for this processing (optional)

        Returns:
            Agent result (from cache or fresh processing)
        """
        cache_key = self.generate_agent_cache_key(
            agent.name, message, conversation_id, context
        )
        ttl = ttl or self.default_ttl

        # Try to get from cache first
        cached_result = await self._get_cached_result(cache_key)
        if cached_result is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for agent '{agent.name}' with key {cache_key}")
            return cached_result.result

        # Cache miss - process with agent
        self._stats["cache_misses"] += 1
        self._stats["agent_processings"] += 1

        start_time = time.time()
        result = await agent.process_message(message, conversation_id, context)
        processing_time = time.time() - start_time

        # Update processing time statistics
        self._update_processing_stats(processing_time)

        # Cache the result (only if successful)
        if result.success:
            await self._cache_result(cache_key, result, ttl, conversation_id)

        logger.debug(
            f"Agent '{agent.name}' processed message in {processing_time:.3f}s (cache miss)"
        )
        return result

    async def _get_cached_result(self, cache_key: str) -> Optional[CachedAgentResult]:
        """Get a cached agent result."""
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

            return CachedAgentResult(
                result=data["cached_result"],
                cache_key=cache_key,
                cached_at=data["cached_at"],
                ttl=data["ttl"],
                hit_count=data["hit_count"],
                conversation_id=data.get("conversation_id"),
            )

        return None

    async def _cache_result(
        self,
        cache_key: str,
        result: AgentResult,
        ttl: int,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Cache an agent result."""
        cache_data = {
            "cached_result": result,
            "cached_at": time.time(),
            "ttl": ttl,
            "hit_count": 0,
            "conversation_id": conversation_id,
        }

        await self.cache.set(cache_key, cache_data, ttl)

    def _update_processing_stats(self, processing_time: float) -> None:
        """Update processing time statistics."""
        total_processings = self._stats["agent_processings"]
        current_avg = self._stats["average_processing_time"]

        # Calculate new average
        if total_processings == 1:
            self._stats["average_processing_time"] = processing_time
        else:
            self._stats["average_processing_time"] = (
                current_avg * (total_processings - 1) + processing_time
            ) / total_processings

    async def invalidate_agent_cache(
        self, agent_name: str, pattern: str = "*", conversation_id: Optional[str] = None
    ) -> int:
        """
        Invalidate cached results for an agent.

        Args:
            agent_name: Name of the agent
            pattern: Pattern to match cache keys
            conversation_id: Specific conversation ID to invalidate (optional)

        Returns:
            Number of cache entries invalidated
        """
        if conversation_id:
            cache_pattern = f"agent:{agent_name}:*:{conversation_id}:*"
        else:
            cache_pattern = f"agent:{agent_name}:{pattern}"

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

        logger.info(f"Invalidated {len(keys)} cache entries for agent '{agent_name}'")
        return len(keys)

    async def invalidate_conversation_cache(self, conversation_id: str) -> int:
        """
        Invalidate all cache entries for a specific conversation.

        Args:
            conversation_id: The conversation ID to invalidate

        Returns:
            Number of cache entries invalidated
        """
        keys = []

        # Get all conversation cache keys from all layers
        for layer in self.cache.layers:
            layer_keys = await layer.keys(f"*:{conversation_id}:*")
            keys.extend(layer_keys)

        # Remove duplicates
        keys = list(set(keys))

        # Delete the keys
        for key in keys:
            await self.cache.delete(key)

        logger.info(
            f"Invalidated {len(keys)} cache entries for conversation '{conversation_id}'"
        )
        return len(keys)

    async def clear_all_agent_cache(self) -> int:
        """Clear all agent cache entries."""
        keys = []

        # Get all agent cache keys from all layers
        for layer in self.cache.layers:
            layer_keys = await layer.keys("agent:*")
            keys.extend(layer_keys)

        # Remove duplicates
        keys = list(set(keys))

        # Delete the keys
        for key in keys:
            await self.cache.delete(key)

        logger.info(f"Cleared {len(keys)} agent cache entries")
        return len(keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent cache statistics."""
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


class CachedAgentRegistry(AgentRegistry):
    """
    Agent registry with built-in caching support.

    Extends the standard AgentRegistry to provide automatic caching
    of agent processing results.
    """

    def __init__(self, cache: Optional[MultiLayerCache] = None):
        """
        Initialize the cached agent registry.

        Args:
            cache: Multi-layer cache instance (creates default if None)
        """
        super().__init__()
        self.agent_cache = AgentCache(cache or self._create_default_cache())

    def _create_default_cache(self) -> MultiLayerCache:
        """Create a default multi-layer cache for agent results."""
        from ..layers.memory import MemoryCache
        from ..layers.redis_cache import RedisCache

        cache = MultiLayerCache()

        # Add memory cache layer (high priority)
        memory_cache = MemoryCache(
            name="agent_memory_cache",
            priority=0,
            max_size=500,  # Smaller than tool cache since agent responses are larger
            default_ttl=600,  # 10 minutes for memory cache
        )
        cache.add_layer(memory_cache.layer)

        # Add Redis cache layer (lower priority)
        redis_cache = RedisCache(
            name="agent_redis_cache",
            priority=1,
            default_ttl=1800,  # 30 minutes for Redis cache
        )
        cache.add_layer(redis_cache.layer)

        return cache

    async def process_message_with_cache(
        self,
        message: str,
        agent_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> AgentResult:
        """
        Process a message with caching support.

        Args:
            message: User message to process
            agent_name: Specific agent to use (optional)
            conversation_id: Conversation ID for context
            context: Additional context information
            ttl: Custom TTL for this processing (optional)

        Returns:
            Agent result (from cache or fresh processing)
        """
        agent = None

        if agent_name:
            agent = self.get_agent(agent_name)
            if not agent:
                error_msg = f"Agent '{agent_name}' not found"
                logger.error(error_msg)
                return AgentResult(
                    success=False,
                    response="Requested agent is not available.",
                    error=error_msg,
                    agent_name="unknown",
                    execution_time=0.0,
                    conversation_id=conversation_id,
                )
        else:
            agent = self.find_relevant_agent(message, context)
            if not agent:
                error_msg = "No suitable agent found"
                logger.error(error_msg)
                return AgentResult(
                    success=False,
                    response="No agents are currently available.",
                    error=error_msg,
                    agent_name="unknown",
                    execution_time=0.0,
                    conversation_id=conversation_id,
                )

        return await self.agent_cache.process_with_cache(
            agent, message, conversation_id, context, ttl
        )

    async def invalidate_agent_cache(
        self, agent_name: str, pattern: str = "*", conversation_id: Optional[str] = None
    ) -> int:
        """
        Invalidate cached results for a specific agent.

        Args:
            agent_name: Name of the agent
            pattern: Pattern to match cache keys
            conversation_id: Specific conversation ID to invalidate (optional)

        Returns:
            Number of cache entries invalidated
        """
        return await self.agent_cache.invalidate_agent_cache(
            agent_name, pattern, conversation_id
        )

    async def invalidate_conversation_cache(self, conversation_id: str) -> int:
        """
        Invalidate all cache entries for a specific conversation.

        Args:
            conversation_id: The conversation ID to invalidate

        Returns:
            Number of cache entries invalidated
        """
        return await self.agent_cache.invalidate_conversation_cache(conversation_id)

    async def clear_all_cache(self) -> int:
        """
        Clear all cached agent results.

        Returns:
            Number of cache entries cleared
        """
        return await self.agent_cache.clear_all_agent_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        return self.agent_cache.get_stats()


def cache_agent_response(ttl: int = 1800) -> Callable:
    """
    Decorator for caching agent responses.

    Args:
        ttl: Time-to-live for cached responses (seconds)

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(
            self,
            message: str,
            conversation_id: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
        ):
            # Only cache if this is an agent processing method
            if hasattr(self, "name") and hasattr(self, "process_message"):
                # Use the agent's cache if available
                if hasattr(self, "_registry") and hasattr(
                    self._registry, "agent_cache"
                ):
                    cache = self._registry.agent_cache
                    return await cache.process_with_cache(
                        self, message, conversation_id, context, ttl
                    )

            # Fallback to direct processing
            return await func(self, message, conversation_id, context)

        return wrapper

    return decorator


# Global cached agent registry instance
_cached_agent_registry: Optional[CachedAgentRegistry] = None


def get_cached_agent_registry() -> CachedAgentRegistry:
    """Get the global cached agent registry instance."""
    global _cached_agent_registry
    if _cached_agent_registry is None:
        _cached_agent_registry = CachedAgentRegistry()
    return _cached_agent_registry
