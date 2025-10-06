"""
Base classes and interfaces for the caching system.

This module defines the core abstractions for cache backends, layers,
and the multi-layer cache architecture.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheableOperation(Enum):
    """Types of cacheable operations."""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    EXISTS = "exists"
    CLEAR = "clear"
    KEYS = "keys"


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol defining the interface for cache backends."""

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        ...

    async def clear(self) -> bool:
        """Clear all keys from the cache."""
        ...

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern."""
        ...

    async def close(self) -> None:
        """Close the cache backend and release resources."""
        ...


class CacheLayer:
    """
    A single cache layer with priority and configuration.

    Each layer can have different characteristics (memory, Redis, etc.)
    and is ordered by priority for read operations.
    """

    def __init__(
        self,
        backend: CacheBackend,
        name: str,
        priority: int = 0,
        read_only: bool = False,
        default_ttl: Optional[int] = None,
    ):
        """
        Initialize a cache layer.

        Args:
            backend: The cache backend implementation
            name: Unique name for this layer
            priority: Priority for read operations (lower = higher priority)
            read_only: Whether this layer is read-only
            default_ttl: Default TTL for this layer
        """
        self.backend = backend
        self.name = name
        self.priority = priority
        self.read_only = read_only
        self.default_ttl = default_ttl
        self.enabled = True

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from this layer."""
        if not self.enabled:
            return None
        return await self.backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in this layer."""
        if not self.enabled or self.read_only:
            return False
        return await self.backend.set(key, value, ttl or self.default_ttl)

    async def delete(self, key: str) -> bool:
        """Delete a key from this layer."""
        if not self.enabled or self.read_only:
            return False
        return await self.backend.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in this layer."""
        if not self.enabled:
            return False
        return await self.backend.exists(key)

    async def clear(self) -> bool:
        """Clear all keys from this layer."""
        if not self.enabled or self.read_only:
            return False
        return await self.backend.clear()

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern from this layer."""
        if not self.enabled:
            return []
        return await self.backend.keys(pattern)

    async def close(self) -> None:
        """Close this cache layer."""
        await self.backend.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this layer."""
        return {
            "name": self.name,
            "priority": self.priority,
            "read_only": self.read_only,
            "enabled": self.enabled,
            "default_ttl": self.default_ttl,
        }


class MultiLayerCache:
    """
    Multi-layer cache that coordinates multiple cache layers.

    Reads are attempted from layers in priority order (lowest priority first).
    Writes are performed on all writable layers.
    """

    def __init__(self):
        """Initialize the multi-layer cache."""
        self.layers: List[CacheLayer] = []
        self._write_through = True
        self._read_through = True

    def add_layer(self, layer: CacheLayer) -> None:
        """
        Add a cache layer.

        Args:
            layer: The cache layer to add
        """
        self.layers.append(layer)
        # Sort layers by priority (lowest first for read order)
        self.layers.sort(key=lambda x: x.priority)

    def remove_layer(self, name: str) -> bool:
        """
        Remove a cache layer by name.

        Args:
            name: Name of the layer to remove

        Returns:
            True if layer was removed, False if not found
        """
        for i, layer in enumerate(self.layers):
            if layer.name == name:
                self.layers.pop(i)
                return True
        return False

    def get_layer(self, name: str) -> Optional[CacheLayer]:
        """
        Get a cache layer by name.

        Args:
            name: Name of the layer to get

        Returns:
            The cache layer or None if not found
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Reads from layers in priority order (lowest priority first).
        If read-through is enabled, higher layers are populated from lower ones.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.layers:
            return None

        # Try to get from layers in priority order (lowest priority first)
        for layer in self.layers:
            if not layer.enabled:
                continue

            value = await layer.get(key)
            if value is not None:
                logger.debug(f"Cache hit in layer '{layer.name}' for key '{key}'")

                # Populate higher priority layers if read-through is enabled
                if self._read_through:
                    await self._populate_higher_layers(layer, key, value)

                return value

        logger.debug(f"Cache miss for key '{key}'")
        return None

    async def _populate_higher_layers(
        self, source_layer: CacheLayer, key: str, value: Any
    ) -> None:
        """Populate higher priority layers with a value from a lower priority layer."""
        for layer in self.layers:
            if (
                layer.priority < source_layer.priority
                and layer.enabled
                and not layer.read_only
            ):
                try:
                    await layer.set(key, value)
                    logger.debug(
                        f"Populated layer '{layer.name}' from '{source_layer.name}'"
                    )
                except Exception as e:
                    logger.warning(f"Failed to populate layer '{layer.name}': {e}")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.

        Writes to all writable layers if write-through is enabled.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if at least one layer was written to successfully
        """
        if not self.layers:
            return False

        success = False

        if self._write_through:
            # Write to all writable layers
            for layer in self.layers:
                if layer.enabled and not layer.read_only:
                    try:
                        layer_success = await layer.set(key, value, ttl)
                        if layer_success:
                            success = True
                            logger.debug(f"Set key '{key}' in layer '{layer.name}'")
                    except Exception as e:
                        logger.warning(
                            f"Failed to set key '{key}' in layer '{layer.name}': {e}"
                        )
        else:
            # Write only to the highest priority writable layer
            for layer in self.layers:
                if layer.enabled and not layer.read_only:
                    try:
                        success = await layer.set(key, value, ttl)
                        if success:
                            logger.debug(f"Set key '{key}' in layer '{layer.name}'")
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to set key '{key}' in layer '{layer.name}': {e}"
                        )

        return success

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted from at least one layer
        """
        if not self.layers:
            return False

        success = False

        for layer in self.layers:
            if layer.enabled and not layer.read_only:
                try:
                    layer_success = await layer.delete(key)
                    if layer_success:
                        success = True
                        logger.debug(f"Deleted key '{key}' from layer '{layer.name}'")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete key '{key}' from layer '{layer.name}': {e}"
                    )

        return success

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in any cache layer.

        Args:
            key: Cache key to check

        Returns:
            True if key exists in any layer
        """
        if not self.layers:
            return False

        for layer in self.layers:
            if layer.enabled:
                try:
                    if await layer.exists(key):
                        return True
                except Exception as e:
                    logger.warning(
                        f"Failed to check existence in layer '{layer.name}': {e}"
                    )

        return False

    async def clear(self) -> bool:
        """
        Clear all keys from all cache layers.

        Returns:
            True if all layers were cleared successfully
        """
        if not self.layers:
            return True

        success = True

        for layer in self.layers:
            if layer.enabled and not layer.read_only:
                try:
                    layer_success = await layer.clear()
                    if not layer_success:
                        success = False
                    logger.debug(f"Cleared layer '{layer.name}'")
                except Exception as e:
                    logger.warning(f"Failed to clear layer '{layer.name}': {e}")
                    success = False

        return success

    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern from all layers.

        Args:
            pattern: Pattern to match keys

        Returns:
            List of matching keys (deduplicated)
        """
        if not self.layers:
            return []

        all_keys = []

        for layer in self.layers:
            if layer.enabled:
                try:
                    layer_keys = await layer.keys(pattern)
                    all_keys.extend(layer_keys)
                except Exception as e:
                    logger.warning(f"Failed to get keys from layer '{layer.name}': {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in all_keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)

        return unique_keys

    async def close(self) -> None:
        """Close all cache layers."""
        for layer in self.layers:
            try:
                await layer.close()
            except Exception as e:
                logger.warning(f"Error closing layer '{layer.name}': {e}")

    def enable_layer(self, name: str) -> bool:
        """
        Enable a cache layer.

        Args:
            name: Name of the layer to enable

        Returns:
            True if layer was enabled, False if not found
        """
        layer = self.get_layer(name)
        if layer:
            layer.enabled = True
            return True
        return False

    def disable_layer(self, name: str) -> bool:
        """
        Disable a cache layer.

        Args:
            name: Name of the layer to disable

        Returns:
            True if layer was disabled, False if not found
        """
        layer = self.get_layer(name)
        if layer:
            layer.enabled = False
            return True
        return False

    def set_write_through(self, enabled: bool) -> None:
        """Enable or disable write-through mode."""
        self._write_through = enabled

    def set_read_through(self, enabled: bool) -> None:
        """Enable or disable read-through mode."""
        self._read_through = enabled

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the multi-layer cache."""
        layer_stats = []
        for layer in self.layers:
            stats = layer.get_stats()
            layer_stats.append(stats)

        return {
            "total_layers": len(self.layers),
            "enabled_layers": len([l for l in self.layers if l.enabled]),
            "write_through": self._write_through,
            "read_through": self._read_through,
            "layers": layer_stats,
        }


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.

    Args:
        prefix: Key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Generated cache key
    """
    # Convert args and kwargs to a consistent string representation
    parts = [prefix]

    if args:
        # Convert args to string representation
        args_str = ":".join(str(arg) for arg in args)
        parts.append(args_str)

    if kwargs:
        # Sort kwargs by key for consistency
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = ":".join(f"{k}={v}" for k, v in sorted_kwargs)
        parts.append(kwargs_str)

    # Join parts and create hash for long keys
    key = ":".join(parts)

    # If key is too long, use a hash
    if len(key) > 200:
        hash_obj = hashlib.md5(key.encode())
        key = f"{prefix}:{hash_obj.hexdigest()}"

    # Replace problematic characters
    key = key.replace(" ", "_").replace("/", "-")

    return key


class CacheOperation:
    """Represents a cache operation with timing and metadata."""

    def __init__(self, operation: CacheableOperation, key: str):
        """
        Initialize a cache operation.

        Args:
            operation: Type of operation
            key: Cache key involved
        """
        self.operation = operation
        self.key = key
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.success: Optional[bool] = None
        self.error: Optional[str] = None
        self.duration: Optional[float] = None

    def complete(self, success: bool, error: Optional[str] = None) -> None:
        """
        Mark the operation as completed.

        Args:
            success: Whether the operation was successful
            error: Error message if operation failed
        """
        self.end_time = time.time()
        self.success = success
        self.error = error
        self.duration = self.end_time - self.start_time


class CacheEvent:
    """Event emitted by the cache system."""

    def __init__(
        self,
        event_type: str,
        operation: CacheOperation,
        layer_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a cache event.

        Args:
            event_type: Type of event
            operation: The cache operation
            layer_name: Name of the cache layer involved
            metadata: Additional event metadata
        """
        self.event_type = event_type
        self.operation = operation
        self.layer_name = layer_name
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "operation": self.operation.operation.value,
            "key": self.operation.key,
            "layer_name": self.layer_name,
            "success": self.operation.success,
            "duration": self.operation.duration,
            "error": self.operation.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class CacheMetrics:
    """
    Tracks cache performance metrics.

    This class collects statistics about cache operations including
    hits, misses, sets, deletes, and errors.
    """

    def __init__(self):
        """Initialize cache metrics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def record_set(self) -> None:
        """Record a cache set operation."""
        self.sets += 1

    def record_delete(self) -> None:
        """Record a cache delete operation."""
        self.deletes += 1

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    def hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total_gets = self.hits + self.misses
        if total_gets == 0:
            return 0.0
        return self.hits / total_gets

    def total_operations(self) -> int:
        """Calculate total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes + self.errors

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics as a dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": self.hit_rate(),
            "total_operations": self.total_operations(),
        }


class PerformanceMonitor:
    """
    Monitors cache performance and collects metrics.

    Tracks response times, hit rates, and other performance indicators
    for the caching system.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self._metrics = CacheMetrics()
        self._response_times = []
        self._enabled = False

    async def start(self) -> None:
        """Start the performance monitor."""
        self._enabled = True
        logger.info("Performance monitor started")

    async def stop(self) -> None:
        """Stop the performance monitor."""
        self._enabled = False
        logger.info("Performance monitor stopped")

    def record_operation(self, operation: CacheOperation) -> None:
        """Record a cache operation."""
        if not self._enabled:
            return

        if operation.success:
            if operation.operation == CacheableOperation.GET:
                if operation.duration is not None:
                    self._response_times.append(operation.duration)

            # Record metrics based on operation type
            if operation.operation == CacheableOperation.GET:
                # This would be determined by whether the get was a hit or miss
                # For simplicity, we'll assume it's recorded elsewhere
                pass
            elif operation.operation == CacheableOperation.SET:
                self._metrics.record_set()
            elif operation.operation == CacheableOperation.DELETE:
                self._metrics.record_delete()
        else:
            self._metrics.record_error()

    def record_hit(self) -> None:
        """Record a cache hit."""
        if self._enabled:
            self._metrics.record_hit()

    def record_miss(self) -> None:
        """Record a cache miss."""
        if self._enabled:
            self._metrics.record_miss()

    def get_average_response_time(self) -> float:
        """Calculate average response time for cache operations."""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)

    def get_percentile_response_time(self, percentile: float) -> float:
        """Get the response time at a given percentile."""
        if not self._response_times:
            return 0.0

        sorted_times = sorted(self._response_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]

    async def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._metrics.get_stats()
        stats.update(
            {
                "enabled": self._enabled,
                "total_response_time_samples": len(self._response_times),
                "average_response_time": self.get_average_response_time(),
                "p95_response_time": self.get_percentile_response_time(95),
                "p99_response_time": self.get_percentile_response_time(99),
            }
        )
        return stats

    def reset(self) -> None:
        """Reset all performance metrics."""
        self._metrics.reset()
        self._response_times.clear()
