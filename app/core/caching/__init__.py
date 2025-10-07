"""
Caching System for AI Assistant

This module provides a comprehensive caching system with multiple layers,
compression, batching, and connection pooling for optimal performance.
"""

from typing import Any, Dict, Optional, List
import asyncio
import logging

from .base import (
    CacheBackend,
    CacheLayer,
    MultiLayerCache,
    CacheOperation,
    CacheEvent,
    CacheMetrics,
    PerformanceMonitor,
)
from .layers.memory import MemoryCache, MemoryCacheLayer

# Import Redis cache only if Redis is available
try:
    import redis
    from .layers.redis_cache import RedisCache, RedisCacheLayer

    REDIS_AVAILABLE = True
except ImportError:
    # Create dummy classes if Redis is not available
    class RedisCache:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Redis is not installed. Please install the 'redis' package to use RedisCache."
            )

    class RedisCacheLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Redis is not installed. Please install the 'redis' package to use RedisCacheLayer."
            )

    REDIS_AVAILABLE = False
from .compression.compressor import (
    ResultCompressor,
    CompressionAlgorithm,
    CompressedCacheBackend,
    get_default_compressor,
)
from .batching.batch_processor import (
    BatchProcessor,
    BatchProcessorManager,
    get_batch_processor_manager,
)
from .pooling.connection_pool import (
    ConnectionPool,
    ConnectionPoolManager,
    get_connection_pool_manager,
)

logger = logging.getLogger(__name__)


class CachingSystem:
    """
    Main caching system that integrates all caching components.

    Provides a unified interface for caching with multi-layer support,
    compression, batching, and connection pooling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the caching system.

        Args:
            config: Configuration dictionary for caching system
        """
        self.config = config or {}
        self._cache: Optional[MultiLayerCache] = None
        self._compressor: Optional[ResultCompressor] = None
        self._batch_processor_manager: Optional[BatchProcessorManager] = None
        self._connection_pool_manager: Optional[ConnectionPoolManager] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the caching system with all components."""
        if self._initialized:
            return

        logger.info("Initializing caching system...")

        # Initialize cache layers
        await self._initialize_cache()

        # Initialize compression
        await self._initialize_compression()

        # Initialize batching
        await self._initialize_batching()

        # Initialize connection pooling
        await self._initialize_connection_pooling()

        # Initialize performance monitoring
        await self._initialize_performance_monitoring()

        self._initialized = True
        logger.info("Caching system initialized successfully")

    async def _initialize_cache(self) -> None:
        """Initialize the multi-layer cache."""
        cache_layers = []

        # Memory cache layer (highest priority)
        if self.config.get("memory_cache_enabled", True):
            memory_layer = MemoryCacheLayer(
                name="memory_cache",
                priority=0,
                max_size=self.config.get("memory_cache_max_size", 500),
                default_ttl=self.config.get("default_ttl", 300),
                cleanup_interval=self.config.get("memory_cache_cleanup_interval", 60),
            )
            cache_layers.append(memory_layer.layer)

        # Redis cache layer (lower priority)
        if self.config.get("redis_cache_enabled", False):
            redis_layer = RedisCacheLayer(
                name="redis_cache",
                priority=1,
                host=self.config.get("redis_host", "localhost"),
                port=self.config.get("redis_port", 6379),
                db=self.config.get("redis_db", 0),
                password=self.config.get("redis_password"),
                default_ttl=self.config.get("default_ttl", 300),
                max_connections=self.config.get("redis_max_connections", 10),
                connection_timeout=self.config.get("redis_connection_timeout", 5),
                retry_attempts=self.config.get("redis_retry_attempts", 3),
            )
            cache_layers.append(redis_layer.layer)

        # Create multi-layer cache
        self._cache = MultiLayerCache()

        # Add layers to the cache
        for layer in cache_layers:
            self._cache.add_layer(layer)

        # Configure cache behavior
        self._cache.set_write_through(self.config.get("write_through", True))
        self._cache.set_read_through(self.config.get("read_through", True))

    async def _initialize_compression(self) -> None:
        """Initialize the compression system."""
        compression_enabled = self.config.get("compression_enabled", True)

        if compression_enabled:
            algorithm_name = self.config.get("compression_algorithm", "gzip")
            try:
                algorithm = CompressionAlgorithm(algorithm_name)
            except ValueError:
                logger.warning(
                    f"Unknown compression algorithm: {algorithm_name}, using gzip"
                )
                algorithm = CompressionAlgorithm.GZIP

            self._compressor = ResultCompressor(
                default_algorithm=algorithm,
                min_size_for_compression=self.config.get("compression_threshold", 100),
                compression_level=self.config.get("compression_level", 6),
            )
        else:
            self._compressor = ResultCompressor(
                default_algorithm=CompressionAlgorithm.NONE,
                min_size_for_compression=0,  # Disable compression
            )

    async def _initialize_batching(self) -> None:
        """Initialize the batching system."""
        batching_enabled = self.config.get("batching_enabled", True)

        if batching_enabled:
            self._batch_processor_manager = await get_batch_processor_manager()

            # Register default batch processors
            # These would be configured based on the application's needs
            from .batching.batch_processor import (
                create_tool_execution_batch_processor,
                create_agent_processing_batch_processor,
            )

            tool_processor = create_tool_execution_batch_processor()
            await self._batch_processor_manager.register_processor(
                "tool_execution", tool_processor
            )

            agent_processor = create_agent_processing_batch_processor()
            await self._batch_processor_manager.register_processor(
                "agent_processing", agent_processor
            )

    async def _initialize_connection_pooling(self) -> None:
        """Initialize the connection pooling system."""
        connection_pooling_enabled = self.config.get("connection_pooling_enabled", True)

        if connection_pooling_enabled:
            self._connection_pool_manager = await get_connection_pool_manager()

            # Register default connection pools
            # These would be configured based on the application's needs
            from .pooling.connection_pool import HTTPConnectionPool

            http_pool = HTTPConnectionPool(
                name="http_pool",
                max_size=self.config.get("max_http_connections", 10),
                min_size=2,
                connection_timeout=self.config.get("connection_timeout", 5.0),
                acquire_timeout=self.config.get("acquire_timeout", 10.0),
            )
            await self._connection_pool_manager.register_pool("http", http_pool)

    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring."""
        monitoring_enabled = self.config.get("monitoring_enabled", True)

        if monitoring_enabled:
            self._performance_monitor = PerformanceMonitor()
            await self._performance_monitor.start()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not self._initialized or not self._cache:
            return None

        return await self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        if not self._initialized or not self._cache:
            return False

        # Apply compression if enabled and compressor is available
        if self._compressor and self._should_compress(value):
            try:
                compression_result = await self._compressor.compress(value)
                # Store compressed value with metadata
                compressed_value = {
                    "compressed": True,
                    "algorithm": compression_result.algorithm.value,
                    "data": compression_result.compressed_data,
                    "original_size": compression_result.original_size,
                }
                return await self._cache.set(key, compressed_value, ttl)
            except Exception as e:
                logger.error(f"Failed to compress value for key '{key}': {e}")
                # Fall back to storing uncompressed
                return await self._cache.set(key, value, ttl)
        else:
            return await self._cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if not self._initialized or not self._cache:
            return False

        return await self._cache.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if not self._initialized or not self._cache:
            return False

        return await self._cache.exists(key)

    async def clear(self) -> bool:
        """Clear all keys from the cache."""
        if not self._initialized or not self._cache:
            return False

        return await self._cache.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching system statistics."""
        stats = {
            "initialized": self._initialized,
            "cache_enabled": self._cache is not None,
            "compression_enabled": self._compressor is not None,
            "batching_enabled": self._batch_processor_manager is not None,
            "connection_pooling_enabled": self._connection_pool_manager is not None,
            "monitoring_enabled": self._performance_monitor is not None,
        }

        if self._cache:
            cache_stats = self._cache.get_stats()
            stats["cache"] = cache_stats

        if self._compressor:
            compression_stats = self._compressor.get_stats()
            stats["compression"] = compression_stats

        if self._batch_processor_manager:
            batching_stats = self._batch_processor_manager.get_stats()
            stats["batching"] = batching_stats

        if self._connection_pool_manager:
            pooling_stats = self._connection_pool_manager.get_stats()
            stats["connection_pooling"] = pooling_stats

        if self._performance_monitor:
            monitoring_stats = await self._performance_monitor.get_stats()
            stats["monitoring"] = monitoring_stats

        return stats

    def _should_compress(self, value: Any) -> bool:
        """Determine if a value should be compressed."""
        if not self._compressor:
            return False

        # Simple heuristic: compress if value is large enough
        try:
            # Estimate size by serializing
            import pickle

            serialized_size = len(pickle.dumps(value))
            return serialized_size >= self.config.get("compression_threshold", 100)
        except Exception:
            # If we can't determine size, don't compress
            return False

    async def shutdown(self) -> None:
        """Shutdown the caching system and release resources."""
        if not self._initialized:
            return

        logger.info("Shutting down caching system...")

        # Shutdown components in reverse order
        if self._performance_monitor:
            await self._performance_monitor.stop()

        if self._connection_pool_manager:
            await self._connection_pool_manager.close_all()

        if self._batch_processor_manager:
            await self._batch_processor_manager.shutdown_all()

        if self._cache:
            await self._cache.close()

        self._initialized = False
        logger.info("Caching system shutdown complete")


# Global caching system instance
_caching_system: Optional[CachingSystem] = None


async def get_caching_system(config: Optional[Dict[str, Any]] = None) -> CachingSystem:
    """
    Get the global caching system instance.

    Args:
        config: Configuration for the caching system

    Returns:
        The caching system instance
    """
    global _caching_system
    if _caching_system is None:
        _caching_system = CachingSystem(config)
        await _caching_system.initialize()
    return _caching_system


async def shutdown_caching_system() -> None:
    """Shutdown the global caching system."""
    global _caching_system
    if _caching_system is not None:
        await _caching_system.shutdown()
        _caching_system = None


# Convenience functions for common caching operations


async def cache_get(key: str) -> Optional[Any]:
    """Convenience function to get a value from the cache."""
    caching_system = await get_caching_system()
    return await caching_system.get(key)


async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Convenience function to set a value in the cache."""
    caching_system = await get_caching_system()
    return await caching_system.set(key, value, ttl)


async def cache_delete(key: str) -> bool:
    """Convenience function to delete a key from the cache."""
    caching_system = await get_caching_system()
    return await caching_system.delete(key)


async def cache_exists(key: str) -> bool:
    """Convenience function to check if a key exists in the cache."""
    caching_system = await get_caching_system()
    return await caching_system.exists(key)


async def cache_clear() -> bool:
    """Convenience function to clear the cache."""
    caching_system = await get_caching_system()
    return await caching_system.clear()


async def get_cache_stats() -> Dict[str, Any]:
    """Convenience function to get cache statistics."""
    caching_system = await get_caching_system()
    return caching_system.get_stats()


# Context manager for temporary caching configuration


class CachingContext:
    """Context manager for temporary caching configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the caching context.

        Args:
            config: Configuration for the caching system
        """
        self.config = config
        self.original_system: Optional[CachingSystem] = None

    async def __aenter__(self):
        """Enter the context and configure the caching system."""
        global _caching_system
        self.original_system = _caching_system
        _caching_system = None  # Force recreation with new config
        return await get_caching_system(self.config)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore the original caching system."""
        global _caching_system
        await shutdown_caching_system()
        _caching_system = self.original_system


# Export main classes and functions
__all__ = [
    # Core classes
    "CachingSystem",
    "CacheBackend",
    "CacheLayer",
    "MultiLayerCache",
    # Cache implementations
    "MemoryCache",
    "MemoryCacheLayer",
    "RedisCache",
    "RedisCacheLayer",
    # Compression
    "ResultCompressor",
    "CompressionAlgorithm",
    "CompressedCacheBackend",
    # Batching
    "BatchProcessor",
    "BatchProcessorManager",
    # Connection pooling
    "ConnectionPool",
    "ConnectionPoolManager",
    # Performance monitoring
    "PerformanceMonitor",
    "CacheMetrics",
    # Utility functions
    "get_caching_system",
    "shutdown_caching_system",
    "cache_get",
    "cache_set",
    "cache_delete",
    "cache_exists",
    "cache_clear",
    "get_cache_stats",
    "CachingContext",
]
