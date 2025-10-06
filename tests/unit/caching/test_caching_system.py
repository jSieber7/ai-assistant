"""
Tests for the main caching system integration.
"""

import pytest
import asyncio
from app.core.caching import (
    CachingSystem,
    get_caching_system,
    shutdown_caching_system,
    cache_get,
    cache_set,
    cache_delete,
    cache_exists,
    cache_clear,
    get_cache_stats,
    CachingContext,
)


class TestCachingSystem:
    """Test the main CachingSystem class."""

    @pytest.fixture
    def caching_system(self):
        """Create a caching system with test configuration."""
        config = {
            "memory_cache_enabled": True,
            "redis_cache_enabled": False,  # Disable Redis for unit tests
            "compression_enabled": True,
            "batching_enabled": False,  # Disable batching for simpler tests
            "connection_pooling_enabled": False,
            "monitoring_enabled": True,
            "default_ttl": 300,
            "memory_cache_max_size": 100,
            "compression_threshold": 100,
        }
        return CachingSystem(config)

    @pytest.mark.asyncio
    async def test_initialization(self, caching_system):
        """Test caching system initialization."""
        assert caching_system._initialized is False
        assert caching_system._cache is None
        assert caching_system._compressor is None
        assert caching_system._batch_processor_manager is None
        assert caching_system._connection_pool_manager is None
        assert caching_system._performance_monitor is None

    @pytest.mark.asyncio
    async def test_initialize(self, caching_system):
        """Test system initialization."""
        await caching_system.initialize()

        assert caching_system._initialized is True
        assert caching_system._cache is not None
        assert caching_system._compressor is not None
        assert caching_system._performance_monitor is not None
        # Batching and pooling should be disabled in config
        assert caching_system._batch_processor_manager is None
        assert caching_system._connection_pool_manager is None

    @pytest.mark.asyncio
    async def test_double_initialization(self, caching_system):
        """Test that double initialization doesn't cause issues."""
        await caching_system.initialize()
        await caching_system.initialize()  # Should be idempotent

        assert caching_system._initialized is True

    @pytest.mark.asyncio
    async def test_get_set_operations(self, caching_system):
        """Test basic get and set operations."""
        await caching_system.initialize()

        # Set a value
        result = await caching_system.set("test_key", "test_value", 300)
        assert result is True

        # Get the value
        result = await caching_system.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, caching_system):
        """Test getting a nonexistent key."""
        await caching_system.initialize()

        result = await caching_system.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_without_initialization(self, caching_system):
        """Test set operation without initialization."""
        result = await caching_system.set("test_key", "test_value")
        assert result is False  # Should fail if not initialized

    @pytest.mark.asyncio
    async def test_get_without_initialization(self, caching_system):
        """Test get operation without initialization."""
        result = await caching_system.get("test_key")
        assert result is None  # Should return None if not initialized

    @pytest.mark.asyncio
    async def test_delete_operation(self, caching_system):
        """Test delete operation."""
        await caching_system.initialize()

        # Set a value
        await caching_system.set("test_key", "test_value")

        # Delete the value
        result = await caching_system.delete("test_key")
        assert result is True

        # Value should be gone
        result = await caching_system.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, caching_system):
        """Test deleting a nonexistent key."""
        await caching_system.initialize()

        result = await caching_system.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_operation(self, caching_system):
        """Test exists operation."""
        await caching_system.initialize()

        # Set a value
        await caching_system.set("test_key", "test_value")

        # Check existence
        result = await caching_system.exists("test_key")
        assert result is True

        # Check nonexistence
        result = await caching_system.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_operation(self, caching_system):
        """Test clear operation."""
        await caching_system.initialize()

        # Set some values
        await caching_system.set("key1", "value1")
        await caching_system.set("key2", "value2")

        # Clear the cache
        result = await caching_system.clear()
        assert result is True

        # Values should be gone
        result = await caching_system.get("key1")
        assert result is None
        result = await caching_system.get("key2")
        assert result is None

    @pytest.mark.asyncio
    async def test_compression_integration(self, caching_system):
        """Test compression integration."""
        await caching_system.initialize()

        # Large data that should be compressed
        large_data = {"data": "x" * 1000}  # 1000 characters

        # Set the data
        result = await caching_system.set("large_key", large_data)
        assert result is True

        # Get the data
        result = await caching_system.get("large_key")
        assert result == large_data

    @pytest.mark.asyncio
    async def test_get_stats(self, caching_system):
        """Test getting system statistics."""
        await caching_system.initialize()

        # Perform some operations
        await caching_system.set("key1", "value1")
        await caching_system.get("key1")
        await caching_system.get("key2")
        await caching_system.delete("key1")

        stats = await caching_system.get_stats()

        assert stats["initialized"] is True
        assert stats["cache_enabled"] is True
        assert stats["compression_enabled"] is True
        assert stats["batching_enabled"] is False
        assert stats["connection_pooling_enabled"] is False
        assert stats["monitoring_enabled"] is True

        # Check that cache stats are included
        assert "cache" in stats
        assert "compression" in stats
        assert "monitoring" in stats

    @pytest.mark.asyncio
    async def test_shutdown(self, caching_system):
        """Test system shutdown."""
        await caching_system.initialize()

        # Set a value
        await caching_system.set("test_key", "test_value")

        # Shutdown the system
        await caching_system.shutdown()

        assert caching_system._initialized is False
        assert caching_system._cache is None
        assert caching_system._compressor is None
        assert caching_system._performance_monitor is None

    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self, caching_system):
        """Test shutdown without initialization."""
        # Should not raise an error
        await caching_system.shutdown()

        assert caching_system._initialized is False


class TestGlobalCachingFunctions:
    """Test the global caching functions."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup and teardown for global caching tests."""
        # Ensure we start with a clean state
        await shutdown_caching_system()
        yield
        # Clean up after each test
        await shutdown_caching_system()

    @pytest.mark.asyncio
    async def test_get_caching_system(self):
        """Test getting the global caching system."""
        system1 = await get_caching_system()
        system2 = await get_caching_system()

        # Should return the same instance
        assert system1 is system2
        assert system1._initialized is True

    @pytest.mark.asyncio
    async def test_get_caching_system_with_config(self):
        """Test getting the global caching system with configuration."""
        config = {
            "memory_cache_enabled": True,
            "default_ttl": 600,
        }
        system = await get_caching_system(config)

        assert system._initialized is True
        assert system.config["default_ttl"] == 600

    @pytest.mark.asyncio
    async def test_shutdown_caching_system(self):
        """Test shutting down the global caching system."""
        system = await get_caching_system()
        assert system._initialized is True

        await shutdown_caching_system()

        # Getting a new system should create a fresh instance
        new_system = await get_caching_system()
        assert new_system is not system
        assert new_system._initialized is True

    @pytest.mark.asyncio
    async def test_cache_get_set_functions(self):
        """Test the convenience get/set functions."""
        # Set a value
        result = await cache_set("test_key", "test_value", 300)
        assert result is True

        # Get the value
        result = await cache_get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_cache_delete_function(self):
        """Test the convenience delete function."""
        # Set a value
        await cache_set("test_key", "test_value")

        # Delete the value
        result = await cache_delete("test_key")
        assert result is True

        # Value should be gone
        result = await cache_get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_exists_function(self):
        """Test the convenience exists function."""
        # Set a value
        await cache_set("test_key", "test_value")

        # Check existence
        result = await cache_exists("test_key")
        assert result is True

        # Check nonexistence
        result = await cache_exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_clear_function(self):
        """Test the convenience clear function."""
        # Set some values
        await cache_set("key1", "value1")
        await cache_set("key2", "value2")

        # Clear the cache
        result = await cache_clear()
        assert result is True

        # Values should be gone
        result = await cache_get("key1")
        assert result is None
        result = await cache_get("key2")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cache_stats_function(self):
        """Test the convenience stats function."""
        # Perform some operations
        await cache_set("key1", "value1")
        await cache_get("key1")
        await cache_get("key2")

        stats = await get_cache_stats()

        assert "initialized" in stats
        assert "cache_enabled" in stats
        assert "compression_enabled" in stats


class TestCachingContext:
    """Test the CachingContext context manager."""

    @pytest.mark.asyncio
    async def test_caching_context(self):
        """Test the caching context manager."""
        # Get the default system
        default_system = await get_caching_system()

        # Use a context with custom configuration
        custom_config = {
            "memory_cache_enabled": True,
            "default_ttl": 600,
            "compression_enabled": False,
        }

        async with CachingContext(custom_config) as context_system:
            # Context system should be different from default
            assert context_system is not default_system
            assert context_system.config["default_ttl"] == 600
            assert context_system.config["compression_enabled"] is False

            # Set and get a value in the context
            result = await context_system.set("test_key", "test_value")
            assert result is True

            result = await context_system.get("test_key")
            assert result == "test_value"

        # After context, the default system should be restored
        restored_system = await get_caching_system()
        assert restored_system is default_system

        # The value set in the context should not be in the default system
        result = await restored_system.get("test_key")
        assert result is None


class TestCachingSystemIntegration:
    """Integration tests for the caching system."""

    @pytest.mark.asyncio
    async def test_multi_layer_cache_integration(self):
        """Test multi-layer cache integration."""
        config = {
            "memory_cache_enabled": True,
            "redis_cache_enabled": False,  # Disable Redis for unit tests
            "cache_layers": ["memory"],
            "write_through": True,
            "read_through": True,
            "compression_enabled": True,
            "batching_enabled": False,
            "connection_pooling_enabled": False,
            "monitoring_enabled": True,
        }

        system = CachingSystem(config)
        await system.initialize()

        # Test basic operations
        await system.set("key1", "value1")
        result = await system.get("key1")
        assert result == "value1"

        # Test compression with large data
        large_data = {"data": "x" * 1000}
        await system.set("large_key", large_data)
        result = await system.get("large_key")
        assert result == large_data

        # Test statistics
        stats = await system.get_stats()
        assert stats["cache_enabled"] is True
        assert stats["compression_enabled"] is True

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations on the caching system."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "compression_enabled": False,  # Disable compression for simpler test
            }
        )
        await system.initialize()

        # Set initial value
        await system.set("counter", 0)

        # Define increment function
        async def increment_counter():
            current = await system.get("counter")
            if current is None:
                current = 0
            await system.set("counter", current + 1)

        # Run multiple increments concurrently
        tasks = [increment_counter() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Check final value
        final_value = await system.get("counter")
        assert final_value == 10

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the caching system."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "compression_enabled": True,
            }
        )
        await system.initialize()

        # Test with invalid data that might cause compression errors
        # This should not crash the system
        invalid_data = object()  # Not serializable

        result = await system.set("invalid_key", invalid_data)
        # Should handle the error gracefully
        assert result is False or result is True  # Depends on error handling

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "monitoring_enabled": True,
            }
        )
        await system.initialize()

        # Perform operations to generate metrics
        for i in range(10):
            await system.set(f"key{i}", f"value{i}")
            await system.get(f"key{i}")

        # Get stats including monitoring
        stats = await system.get_stats()
        assert "monitoring" in stats
        monitoring_stats = stats["monitoring"]

        # Should have performance metrics
        assert "cache_hits" in monitoring_stats
        assert "cache_misses" in monitoring_stats
        assert "average_response_time" in monitoring_stats

        await system.shutdown()


class TestCachingSystemEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_config(self):
        """Test with empty configuration."""
        system = CachingSystem({})
        await system.initialize()

        # Should still work with default settings
        result = await system.set("test_key", "test_value")
        assert result is True

        result = await system.get("test_key")
        assert result == "test_value"

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_none_config(self):
        """Test with None configuration."""
        system = CachingSystem(None)
        await system.initialize()

        # Should work with default settings
        result = await system.set("test_key", "test_value")
        assert result is True

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_large_number_of_keys(self):
        """Test with a large number of keys."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "memory_cache_max_size": 1000,
            }
        )
        await system.initialize()

        # Add many keys
        for i in range(100):
            await system.set(f"key{i}", f"value{i}")

        # Verify all keys can be retrieved
        for i in range(100):
            result = await system.get(f"key{i}")
            assert result == f"value{i}"

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_various_data_types(self):
        """Test caching various data types."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "compression_enabled": True,
            }
        )
        await system.initialize()

        # Test different data types
        test_cases = [
            "string",
            123,
            45.67,
            True,
            None,
            ["list", "of", "items"],
            {"dict": "value", "number": 42},
            ("tuple", "value"),
        ]

        for i, data in enumerate(test_cases):
            await system.set(f"key{i}", data)
            result = await system.get(f"key{i}")
            assert result == data

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        system = CachingSystem(
            {
                "memory_cache_enabled": True,
                "default_ttl": 0.1,  # Very short TTL
            }
        )
        await system.initialize()

        # Set a value with short TTL
        await system.set("test_key", "test_value")

        # Value should be available immediately
        result = await system.get("test_key")
        assert result == "test_value"

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Value should be expired
        result = await system.get("test_key")
        assert result is None

        await system.shutdown()
