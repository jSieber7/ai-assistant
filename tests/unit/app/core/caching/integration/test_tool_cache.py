"""
Unit tests for tool cache integration components.

This module tests the caching integration for the tool system,
including tool result caching and cached tool registry.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

# Import the components we're testing
from app.core.caching.integration.tool_cache import (
    CachedToolResult,
    ToolCache,
    CachedToolRegistry,
    cache_tool_result,
    get_cached_tool_registry,
)


class TestCachedToolResult:
    """Test cases for CachedToolResult dataclass."""

    def test_cached_tool_result_creation(self):
        """Test creating a CachedToolResult."""
        mock_result = Mock()
        cache_key = "test_key"
        cached_at = time.time()
        ttl = 3600
        hit_count = 5

        cached_result = CachedToolResult(
            result=mock_result,
            cache_key=cache_key,
            cached_at=cached_at,
            ttl=ttl,
            hit_count=hit_count,
        )

        assert cached_result.result == mock_result
        assert cached_result.cache_key == cache_key
        assert cached_result.cached_at == cached_at
        assert cached_result.ttl == ttl
        assert cached_result.hit_count == hit_count

    def test_cached_tool_result_defaults(self):
        """Test creating a CachedToolResult with default values."""
        mock_result = Mock()
        cache_key = "test_key"
        cached_at = time.time()
        ttl = 3600

        cached_result = CachedToolResult(
            result=mock_result,
            cache_key=cache_key,
            cached_at=cached_at,
            ttl=ttl,
        )

        assert cached_result.result == mock_result
        assert cached_result.cache_key == cache_key
        assert cached_result.cached_at == cached_at
        assert cached_result.ttl == ttl
        assert cached_result.hit_count == 0


class TestToolCache:
    """Test cases for ToolCache class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock MultiLayerCache for testing."""
        cache = Mock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.get_stats.return_value = {"memory": {"hits": 10, "misses": 5}}
        cache.layers = [Mock(), Mock()]
        cache.layers[0].keys = AsyncMock(return_value=[])
        cache.layers[1].keys = AsyncMock(return_value=[])
        return cache

    @pytest.fixture
    def tool_cache(self, mock_cache):
        """Create a ToolCache instance for testing."""
        return ToolCache(mock_cache, default_ttl=3600)

    def test_init(self, tool_cache, mock_cache):
        """Test ToolCache initialization."""
        assert tool_cache.cache == mock_cache
        assert tool_cache.default_ttl == 3600
        assert tool_cache.compressor is not None
        assert tool_cache._stats["cache_hits"] == 0
        assert tool_cache._stats["cache_misses"] == 0

    def test_generate_tool_cache_key(self, tool_cache):
        """Test generating a tool cache key."""
        key = tool_cache.generate_tool_cache_key(
            tool_name="test_tool",
            arg1="value1",
            arg2="value2",
            param1="param_value"
        )

        assert "tool:test_tool" in key
        assert "value1" in key
        assert "value2" in key
        assert "param_value" in key

    @pytest.mark.asyncio
    async def test_execute_with_cache_hit(self, tool_cache, mock_cache):
        """Test executing with cache hit."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"

        mock_result = Mock()
        cached_result_data = {
            "cached_result": mock_result,
            "cached_at": time.time(),
            "ttl": 3600,
            "hit_count": 1,
        }
        mock_cache.get.return_value = cached_result_data

        # Call the function
        result = await tool_cache.execute_with_cache(
            tool=mock_tool,
            arg1="value1",
            param1="param_value"
        )

        # Verify the result
        assert result == mock_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_not_called()

        # Verify stats
        assert tool_cache._stats["cache_hits"] == 1
        assert tool_cache._stats["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_cache_miss(self, tool_cache, mock_cache):
        """Test executing with cache miss."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock()

        mock_result = Mock()
        mock_tool.execute.return_value = mock_result

        mock_cache.get.return_value = None

        # Call the function
        result = await tool_cache.execute_with_cache(
            tool=mock_tool,
            arg1="value1",
            param1="param_value"
        )

        # Verify the result
        assert result == mock_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

        # Verify tool was called
        mock_tool.execute.assert_called_once_with("value1", param1="param_value")

        # Verify stats
        assert tool_cache._stats["cache_hits"] == 0
        assert tool_cache._stats["cache_misses"] == 1
        assert tool_cache._stats["tool_executions"] == 1

    @pytest.mark.asyncio
    async def test_execute_with_custom_ttl(self, tool_cache, mock_cache):
        """Test executing with custom TTL."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock()

        mock_result = Mock()
        mock_tool.execute.return_value = mock_result

        mock_cache.get.return_value = None

        # Call the function with custom TTL
        result = await tool_cache.execute_with_cache(
            tool=mock_tool,
            arg1="value1",
            ttl=7200  # Custom TTL
        )

        # Verify the result
        assert result == mock_result

        # Verify cache was called with custom TTL
        mock_cache.set.assert_called_once()
        args, kwargs = mock_cache.set.call_args
        assert args[2] == 7200  # TTL should be the custom value

    @pytest.mark.asyncio
    async def test_get_cached_result(self, tool_cache, mock_cache):
        """Test getting a cached result."""
        # Setup mocks
        mock_result = Mock()
        cached_result_data = {
            "cached_result": mock_result,
            "cached_at": time.time(),
            "ttl": 3600,
            "hit_count": 1,
        }
        mock_cache.get.return_value = cached_result_data

        # Call the function
        result = await tool_cache._get_cached_result("test_key")

        # Verify the result
        assert result is not None
        assert result.result == mock_result
        assert result.cache_key == "test_key"
        assert result.hit_count == 2  # Should be incremented

        # Verify cache was updated with new hit count
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_result_none(self, tool_cache, mock_cache):
        """Test getting a cached result when none exists."""
        mock_cache.get.return_value = None

        # Call the function
        result = await tool_cache._get_cached_result("test_key")

        # Verify the result
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_result(self, tool_cache, mock_cache):
        """Test caching a result."""
        mock_result = Mock()

        # Call the function
        await tool_cache._cache_result("test_key", mock_result, 3600)

        # Verify cache was called
        mock_cache.set.assert_called_once()
        args, kwargs = mock_cache.set.call_args
        assert args[0] == "test_key"
        assert args[1]["cached_result"] == mock_result
        assert args[1]["ttl"] == 3600

    def test_update_execution_stats_first(self, tool_cache):
        """Test updating execution stats for the first time."""
        tool_cache._update_execution_stats(1.5)

        assert tool_cache._stats["average_execution_time"] == 1.5

    def test_update_execution_stats_multiple(self, tool_cache):
        """Test updating execution stats with multiple values."""
        # First update
        tool_cache._update_execution_stats(1.0)
        assert tool_cache._stats["average_execution_time"] == 1.0

        # Second update
        tool_cache._update_execution_stats(2.0)
        # (1.0 * 1 + 2.0) / 2 = 1.5
        assert tool_cache._stats["average_execution_time"] == 1.5

        # Third update
        tool_cache._update_execution_stats(3.0)
        # (1.5 * 2 + 3.0) / 3 = 2.0
        assert tool_cache._stats["average_execution_time"] == 2.0

    @pytest.mark.asyncio
    async def test_invalidate_tool_cache(self, tool_cache, mock_cache):
        """Test invalidating tool cache."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["tool:test_tool:key1", "tool:test_tool:key2"]
        mock_cache.layers[1].keys.return_value = ["tool:test_tool:key2", "tool:test_tool:key3"]

        # Call the function
        count = await tool_cache.invalidate_tool_cache("test_tool")

        # Verify the result
        assert count == 3  # Should be unique keys

        # Verify delete was called for each key
        assert mock_cache.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_invalidate_tool_cache_with_pattern(self, tool_cache, mock_cache):
        """Test invalidating tool cache with pattern."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["tool:test_tool:pattern1", "tool:test_tool:pattern2"]
        mock_cache.layers[1].keys.return_value = []

        # Call the function
        count = await tool_cache.invalidate_tool_cache("test_tool", pattern="pattern*")

        # Verify the result
        assert count == 2

        # Verify delete was called
        assert mock_cache.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_all_tool_cache(self, tool_cache, mock_cache):
        """Test clearing all tool cache."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["tool:test_tool:key1", "tool:other_tool:key2"]
        mock_cache.layers[1].keys.return_value = ["tool:test_tool:key3"]

        # Call the function
        count = await tool_cache.clear_all_tool_cache()

        # Verify the result
        assert count == 3

        # Verify delete was called for each key
        assert mock_cache.delete.call_count == 3

    def test_get_stats(self, tool_cache, mock_cache):
        """Test getting cache statistics."""
        # Setup some stats
        tool_cache._stats["cache_hits"] = 8
        tool_cache._stats["cache_misses"] == 2

        # Call the function
        stats = tool_cache.get_stats()

        # Verify the result
        assert stats["cache_hits"] == 8
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == 0.8  # 8 / (8 + 2)
        assert "cache_layers" in stats


class TestCachedToolRegistry:
    """Test cases for CachedToolRegistry class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock MultiLayerCache for testing."""
        cache = Mock()
        return cache

    @pytest.fixture
    def cached_tool_registry(self, mock_cache):
        """Create a CachedToolRegistry instance for testing."""
        with patch("app.core.caching.integration.tool_cache.ToolRegistry.__init__", return_value=None):
            registry = CachedToolRegistry(mock_cache)
            return registry

    def test_init_with_cache(self, mock_cache):
        """Test CachedToolRegistry initialization with cache."""
        with patch("app.core.caching.integration.tool_cache.ToolRegistry.__init__", return_value=None):
            registry = CachedToolRegistry(mock_cache)

            assert registry.tool_cache.cache == mock_cache
            assert registry.tool_cache.default_ttl == 3600

    @patch("app.core.caching.integration.tool_cache.MemoryCache")
    @patch("app.core.caching.integration.tool_cache.RedisCache")
    @patch("app.core.caching.integration.tool_cache.MultiLayerCache")
    def test_create_default_cache(self, mock_multi_layer, mock_redis, mock_memory):
        """Test creating default cache."""
        # Setup mocks
        mock_cache_instance = Mock()
        mock_multi_layer.return_value = mock_cache_instance

        mock_memory_layer = Mock()
        mock_memory.return_value.layer = mock_memory_layer

        mock_redis_layer = Mock()
        mock_redis.return_value.layer = mock_redis_layer

        with patch("app.core.caching.integration.tool_cache.ToolRegistry.__init__", return_value=None):
            registry = CachedToolRegistry()

            # Verify cache was created
            mock_multi_layer.assert_called_once()
            mock_memory.assert_called_once()
            mock_redis.assert_called_once()

            # Verify layers were added
            assert mock_cache_instance.add_layer.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache_success(self, cached_tool_registry):
        """Test executing tool with cache success."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.enabled = True
        cached_tool_registry.get_tool = Mock(return_value=mock_tool)
        cached_tool_registry.tool_cache.execute_with_cache = AsyncMock()
        mock_result = Mock()
        cached_tool_registry.tool_cache.execute_with_cache.return_value = mock_result

        # Call the function
        result = await cached_tool_registry.execute_tool_with_cache(
            tool_name="test_tool",
            arg1="value1",
            param1="param_value"
        )

        # Verify the result
        assert result == mock_result
        cached_tool_registry.get_tool.assert_called_once_with("test_tool")
        cached_tool_registry.tool_cache.execute_with_cache.assert_called_once_with(
            mock_tool, "value1", ttl=None, param1="param_value"
        )

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache_not_found(self, cached_tool_registry):
        """Test executing tool with cache when tool not found."""
        # Setup mocks
        cached_tool_registry.get_tool = Mock(return_value=None)

        # Call the function and verify exception
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found"):
            await cached_tool_registry.execute_tool_with_cache(
                tool_name="nonexistent_tool",
                arg1="value1"
            )

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache_disabled(self, cached_tool_registry):
        """Test executing tool with cache when tool is disabled."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.enabled = False
        cached_tool_registry.get_tool = Mock(return_value=mock_tool)

        # Call the function and verify exception
        with pytest.raises(ValueError, match="Tool 'test_tool' is disabled"):
            await cached_tool_registry.execute_tool_with_cache(
                tool_name="test_tool",
                arg1="value1"
            )

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache_custom_ttl(self, cached_tool_registry):
        """Test executing tool with cache using custom TTL."""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.enabled = True
        cached_tool_registry.get_tool = Mock(return_value=mock_tool)
        cached_tool_registry.tool_cache.execute_with_cache = AsyncMock()
        mock_result = Mock()
        cached_tool_registry.tool_cache.execute_with_cache.return_value = mock_result

        # Call the function with custom TTL
        result = await cached_tool_registry.execute_tool_with_cache(
            tool_name="test_tool",
            arg1="value1",
            ttl=7200
        )

        # Verify the result
        assert result == mock_result
        cached_tool_registry.tool_cache.execute_with_cache.assert_called_once_with(
            mock_tool, "value1", ttl=7200
        )

    @pytest.mark.asyncio
    async def test_invalidate_tool_cache(self, cached_tool_registry):
        """Test invalidating tool cache."""
        # Setup mocks
        cached_tool_registry.tool_cache.invalidate_tool_cache = AsyncMock(return_value=5)

        # Call the function
        count = await cached_tool_registry.invalidate_tool_cache("test_tool")

        # Verify the result
        assert count == 5
        cached_tool_registry.tool_cache.invalidate_tool_cache.assert_called_once_with("test_tool", "*")

    @pytest.mark.asyncio
    async def test_invalidate_tool_cache_with_pattern(self, cached_tool_registry):
        """Test invalidating tool cache with pattern."""
        # Setup mocks
        cached_tool_registry.tool_cache.invalidate_tool_cache = AsyncMock(return_value=3)

        # Call the function
        count = await cached_tool_registry.invalidate_tool_cache("test_tool", pattern="test*")

        # Verify the result
        assert count == 3
        cached_tool_registry.tool_cache.invalidate_tool_cache.assert_called_once_with("test_tool", "test*")

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cached_tool_registry):
        """Test clearing all cache."""
        # Setup mocks
        cached_tool_registry.tool_cache.clear_all_tool_cache = AsyncMock(return_value=10)

        # Call the function
        count = await cached_tool_registry.clear_all_cache()

        # Verify the result
        assert count == 10
        cached_tool_registry.tool_cache.clear_all_tool_cache.assert_called_once()

    def test_get_cache_stats(self, cached_tool_registry):
        """Test getting cache statistics."""
        # Setup mocks
        mock_stats = {"hits": 10, "misses": 5}
        cached_tool_registry.tool_cache.get_stats = Mock(return_value=mock_stats)

        # Call the function
        stats = cached_tool_registry.get_cache_stats()

        # Verify the result
        assert stats == mock_stats
        cached_tool_registry.tool_cache.get_stats.assert_called_once()


class TestCacheToolResultDecorator:
    """Test cases for cache_tool_result decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_tool_cache(self):
        """Test decorator with tool that has cache."""
        # Setup mocks
        mock_cache = Mock()
        mock_cache.execute_with_cache = AsyncMock()
        mock_result = Mock()
        mock_cache.execute_with_cache.return_value = mock_result

        mock_registry = Mock()
        mock_registry.tool_cache = mock_cache

        # Create a mock tool with cache
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool._registry = mock_registry

        # Apply the decorator
        decorated_func = cache_tool_result(ttl=3600)(mock_tool.execute)

        # Call the decorated function
        result = await decorated_func(mock_tool, "arg1", "arg2", param1="param_value")

        # Verify the result
        assert result == mock_result
        mock_cache.execute_with_cache.assert_called_once_with(
            mock_tool, "arg1", "arg2", ttl=3600, param1="param_value"
        )

    @pytest.mark.asyncio
    async def test_decorator_without_tool_cache(self):
        """Test decorator with tool that doesn't have cache."""
        # Create a mock tool without cache
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock()
        mock_result = Mock()
        mock_tool.execute.return_value = mock_result

        # Apply the decorator
        decorated_func = cache_tool_result(ttl=3600)(mock_tool.execute)

        # Call the decorated function
        result = await decorated_func(mock_tool, "arg1", "arg2", param1="param_value")

        # Verify the result
        assert result == mock_result
        mock_tool.execute.assert_called_once_with(
            mock_tool, "arg1", "arg2", param1="param_value"
        )

    @pytest.mark.asyncio
    async def test_decorator_without_tool_attributes(self):
        """Test decorator with object that doesn't have tool attributes."""
        # Create a mock object without tool attributes
        mock_obj = Mock()
        mock_obj.execute = AsyncMock()
        mock_result = Mock()
        mock_obj.execute.return_value = mock_result

        # Apply the decorator
        decorated_func = cache_tool_result(ttl=3600)(mock_obj.execute)

        # Call the decorated function
        result = await decorated_func(mock_obj, "arg1", "arg2", param1="param_value")

        # Verify the result
        assert result == mock_result
        mock_obj.execute.assert_called_once_with(
            mock_obj, "arg1", "arg2", param1="param_value"
        )


class TestGetCachedToolRegistry:
    """Test cases for get_cached_tool_registry function."""

    @patch("app.core.caching.integration.tool_cache._cached_tool_registry", None)
    def test_get_cached_tool_registry_first_call(self):
        """Test getting cached tool registry for the first time."""
        with patch("app.core.caching.integration.tool_cache.CachedToolRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Call the function
            result = get_cached_tool_registry()

            # Verify the result
            assert result == mock_registry
            mock_registry_class.assert_called_once()

    @patch("app.core.caching.integration.tool_cache._cached_tool_registry")
    def test_get_cached_tool_registry_subsequent_call(self, mock_global_registry):
        """Test getting cached tool registry for subsequent calls."""
        # Call the function
        result = get_cached_tool_registry()

        # Verify the result
        assert result == mock_global_registry