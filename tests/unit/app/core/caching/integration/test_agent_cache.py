"""
Unit tests for agent cache integration components.

This module tests the caching integration for the agent system,
including agent result caching and cached agent registry.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

# Import the components we're testing
from app.core.caching.integration.agent_cache import (
    CachedAgentResult,
    AgentCache,
    CachedAgentRegistry,
    cache_agent_response,
    get_cached_agent_registry,
)


class TestCachedAgentResult:
    """Test cases for CachedAgentResult dataclass."""

    def test_cached_agent_result_creation(self):
        """Test creating a CachedAgentResult."""
        mock_result = Mock()
        cache_key = "test_key"
        cached_at = time.time()
        ttl = 1800
        hit_count = 5
        conversation_id = "conv_123"

        cached_result = CachedAgentResult(
            result=mock_result,
            cache_key=cache_key,
            cached_at=cached_at,
            ttl=ttl,
            hit_count=hit_count,
            conversation_id=conversation_id,
        )

        assert cached_result.result == mock_result
        assert cached_result.cache_key == cache_key
        assert cached_result.cached_at == cached_at
        assert cached_result.ttl == ttl
        assert cached_result.hit_count == hit_count
        assert cached_result.conversation_id == conversation_id

    def test_cached_agent_result_defaults(self):
        """Test creating a CachedAgentResult with default values."""
        mock_result = Mock()
        cache_key = "test_key"
        cached_at = time.time()
        ttl = 1800

        cached_result = CachedAgentResult(
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
        assert cached_result.conversation_id is None


class TestAgentCache:
    """Test cases for AgentCache class."""

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
    def agent_cache(self, mock_cache):
        """Create an AgentCache instance for testing."""
        return AgentCache(mock_cache, default_ttl=1800)

    def test_init(self, agent_cache, mock_cache):
        """Test AgentCache initialization."""
        assert agent_cache.cache == mock_cache
        assert agent_cache.default_ttl == 1800
        assert agent_cache.compressor is not None
        assert agent_cache._stats["agent_cache_hits"] == 0
        assert agent_cache._stats["agent_cache_misses"] == 0

    def test_generate_agent_cache_key_basic(self, agent_cache):
        """Test generating a basic agent cache key."""
        key = agent_cache.generate_agent_cache_key(
            agent_name="test_agent",
            message="Hello world"
        )

        assert "agent:test_agent" in key
        assert "Hello world" in key

    def test_generate_agent_cache_key_with_conversation(self, agent_cache):
        """Test generating an agent cache key with conversation ID."""
        key = agent_cache.generate_agent_cache_key(
            agent_name="test_agent",
            message="Hello world",
            conversation_id="conv_123"
        )

        assert "agent:test_agent" in key
        assert "Hello world" in key
        assert "conv_123" in key

    def test_generate_agent_cache_key_with_context(self, agent_cache):
        """Test generating an agent cache key with context."""
        context = {"user_id": "123", "session": "abc"}
        key = agent_cache.generate_agent_cache_key(
            agent_name="test_agent",
            message="Hello world",
            context=context
        )

        assert "agent:test_agent" in key
        assert "Hello world" in key
        assert "user_id" in key
        assert "session" in key

    @pytest.mark.asyncio
    async def test_process_with_cache_hit(self, agent_cache, mock_cache):
        """Test processing with cache hit."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        mock_result = Mock()
        cached_result_data = {
            "cached_result": mock_result,
            "cached_at": time.time(),
            "ttl": 1800,
            "hit_count": 1,
            "conversation_id": "conv_123",
        }
        mock_cache.get.return_value = cached_result_data

        # Call the function
        result = await agent_cache.process_with_cache(
            agent=mock_agent,
            message="Hello world",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result == mock_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_not_called()

        # Verify stats
        assert agent_cache._stats["agent_cache_hits"] == 1
        assert agent_cache._stats["agent_cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_process_with_cache_miss(self, agent_cache, mock_cache):
        """Test processing with cache miss."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.process_message = AsyncMock()

        mock_result = Mock()
        mock_result.success = True
        mock_agent.process_message.return_value = mock_result

        mock_cache.get.return_value = None

        # Call the function
        result = await agent_cache.process_with_cache(
            agent=mock_agent,
            message="Hello world",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result == mock_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

        # Verify agent was called
        mock_agent.process_message.assert_called_once_with("Hello world", "conv_123", None)

        # Verify stats
        assert agent_cache._stats["agent_cache_hits"] == 0
        assert agent_cache._stats["agent_cache_misses"] == 1
        assert agent_cache._stats["agent_processings"] == 1

    @pytest.mark.asyncio
    async def test_process_with_cache_miss_failure(self, agent_cache, mock_cache):
        """Test processing with cache miss but agent failure."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.process_message = AsyncMock()

        mock_result = Mock()
        mock_result.success = False
        mock_agent.process_message.return_value = mock_result

        mock_cache.get.return_value = None

        # Call the function
        result = await agent_cache.process_with_cache(
            agent=mock_agent,
            message="Hello world",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result == mock_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_not_called()  # Should not cache failed results

        # Verify agent was called
        mock_agent.process_message.assert_called_once_with("Hello world", "conv_123", None)

        # Verify stats
        assert agent_cache._stats["agent_cache_hits"] == 0
        assert agent_cache._stats["agent_cache_misses"] == 1
        assert agent_cache._stats["agent_processings"] == 1

    @pytest.mark.asyncio
    async def test_get_cached_result(self, agent_cache, mock_cache):
        """Test getting a cached result."""
        # Setup mocks
        mock_result = Mock()
        cached_result_data = {
            "cached_result": mock_result,
            "cached_at": time.time(),
            "ttl": 1800,
            "hit_count": 1,
            "conversation_id": "conv_123",
        }
        mock_cache.get.return_value = cached_result_data

        # Call the function
        result = await agent_cache._get_cached_result("test_key")

        # Verify the result
        assert result is not None
        assert result.result == mock_result
        assert result.cache_key == "test_key"
        assert result.hit_count == 2  # Should be incremented

        # Verify cache was updated with new hit count
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_result_none(self, agent_cache, mock_cache):
        """Test getting a cached result when none exists."""
        mock_cache.get.return_value = None

        # Call the function
        result = await agent_cache._get_cached_result("test_key")

        # Verify the result
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_result(self, agent_cache, mock_cache):
        """Test caching a result."""
        mock_result = Mock()

        # Call the function
        await agent_cache._cache_result("test_key", mock_result, 1800, "conv_123")

        # Verify cache was called
        mock_cache.set.assert_called_once()
        args, kwargs = mock_cache.set.call_args
        assert args[0] == "test_key"
        assert args[1]["cached_result"] == mock_result
        assert args[1]["ttl"] == 1800
        assert args[1]["conversation_id"] == "conv_123"

    def test_update_processing_stats_first(self, agent_cache):
        """Test updating processing stats for the first time."""
        agent_cache._update_processing_stats(1.5)

        assert agent_cache._stats["agent_average_processing_time"] == 1.5

    def test_update_processing_stats_multiple(self, agent_cache):
        """Test updating processing stats with multiple values."""
        # First update
        agent_cache._update_processing_stats(1.0)
        assert agent_cache._stats["agent_average_processing_time"] == 1.0

        # Second update
        agent_cache._update_processing_stats(2.0)
        # (1.0 * 1 + 2.0) / 2 = 1.5
        assert agent_cache._stats["agent_average_processing_time"] == 1.5

        # Third update
        agent_cache._update_processing_stats(3.0)
        # (1.5 * 2 + 3.0) / 3 = 2.0
        assert agent_cache._stats["agent_average_processing_time"] == 2.0

    @pytest.mark.asyncio
    async def test_invalidate_agent_cache(self, agent_cache, mock_cache):
        """Test invalidating agent cache."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["agent:test_agent:key1", "agent:test_agent:key2"]
        mock_cache.layers[1].keys.return_value = ["agent:test_agent:key2", "agent:test_agent:key3"]

        # Call the function
        count = await agent_cache.invalidate_agent_cache("test_agent")

        # Verify the result
        assert count == 3  # Should be unique keys

        # Verify delete was called for each key
        assert mock_cache.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_invalidate_agent_cache_with_conversation(self, agent_cache, mock_cache):
        """Test invalidating agent cache for a specific conversation."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["agent:test_agent:*:conv_123:*"]
        mock_cache.layers[1].keys.return_value = []

        # Call the function
        count = await agent_cache.invalidate_agent_cache("test_agent", conversation_id="conv_123")

        # Verify the result
        assert count == 1

        # Verify delete was called
        assert mock_cache.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_conversation_cache(self, agent_cache, mock_cache):
        """Test invalidating conversation cache."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["agent:*:conv_123:*", "tool:*:conv_123:*"]
        mock_cache.layers[1].keys.return_value = ["agent:*:conv_123:*"]

        # Call the function
        count = await agent_cache.invalidate_conversation_cache("conv_123")

        # Verify the result
        assert count == 3  # Should be unique keys

        # Verify delete was called for each key
        assert mock_cache.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_clear_all_agent_cache(self, agent_cache, mock_cache):
        """Test clearing all agent cache."""
        # Setup mocks
        mock_cache.layers[0].keys.return_value = ["agent:test_agent:key1", "agent:other_agent:key2"]
        mock_cache.layers[1].keys.return_value = ["agent:test_agent:key3"]

        # Call the function
        count = await agent_cache.clear_all_agent_cache()

        # Verify the result
        assert count == 3

        # Verify delete was called for each key
        assert mock_cache.delete.call_count == 3

    def test_get_stats(self, agent_cache, mock_cache):
        """Test getting cache statistics."""
        # Setup some stats
        agent_cache._stats["agent_cache_hits"] = 8
        agent_cache._stats["agent_cache_misses"] = 2

        # Call the function
        stats = agent_cache.get_stats()

        # Verify the result
        assert stats["agent_cache_hits"] == 8
        assert stats["agent_cache_misses"] == 2
        assert stats["cache_hit_rate"] == 0.8  # 8 / (8 + 2)
        assert "cache_layers" in stats


class TestCachedAgentRegistry:
    """Test cases for CachedAgentRegistry class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock MultiLayerCache for testing."""
        cache = Mock()
        return cache

    @pytest.fixture
    def cached_agent_registry(self, mock_cache):
        """Create a CachedAgentRegistry instance for testing."""
        with patch("app.core.caching.integration.agent_cache.AgentRegistry.__init__", return_value=None):
            registry = CachedAgentRegistry(mock_cache)
            return registry

    def test_init_with_cache(self, mock_cache):
        """Test CachedAgentRegistry initialization with cache."""
        with patch("app.core.caching.integration.agent_cache.AgentRegistry.__init__", return_value=None):
            registry = CachedAgentRegistry(mock_cache)

            assert registry.agent_cache.cache == mock_cache
            assert registry.agent_cache.default_ttl == 1800

    @patch("app.core.caching.integration.agent_cache.MemoryCache")
    @patch("app.core.caching.integration.agent_cache.RedisCache")
    @patch("app.core.caching.integration.agent_cache.MultiLayerCache")
    def test_create_default_cache(self, mock_multi_layer, mock_redis, mock_memory):
        """Test creating default cache."""
        # Setup mocks
        mock_cache_instance = Mock()
        mock_multi_layer.return_value = mock_cache_instance

        mock_memory_layer = Mock()
        mock_memory.return_value.layer = mock_memory_layer

        mock_redis_layer = Mock()
        mock_redis.return_value.layer = mock_redis_layer

        with patch("app.core.caching.integration.agent_cache.AgentRegistry.__init__", return_value=None):
            registry = CachedAgentRegistry()

            # Verify cache was created
            mock_multi_layer.assert_called_once()
            mock_memory.assert_called_once()
            mock_redis.assert_called_once()

            # Verify layers were added
            assert mock_cache_instance.add_layer.call_count == 2

    @pytest.mark.asyncio
    async def test_process_message_with_cache_success(self, cached_agent_registry):
        """Test processing message with cache success."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        cached_agent_registry.get_agent = Mock(return_value=mock_agent)
        cached_agent_registry.agent_cache.process_with_cache = AsyncMock()
        mock_result = Mock()
        cached_agent_registry.agent_cache.process_with_cache.return_value = mock_result

        # Call the function
        result = await cached_agent_registry.process_message_with_cache(
            message="Hello world",
            agent_name="test_agent",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result == mock_result
        cached_agent_registry.get_agent.assert_called_once_with("test_agent")
        cached_agent_registry.agent_cache.process_with_cache.assert_called_once_with(
            mock_agent, "Hello world", "conv_123", None, None
        )

    @pytest.mark.asyncio
    async def test_process_message_with_cache_agent_not_found(self, cached_agent_registry):
        """Test processing message with cache when agent not found."""
        # Setup mocks
        cached_agent_registry.get_agent = Mock(return_value=None)

        # Call the function
        result = await cached_agent_registry.process_message_with_cache(
            message="Hello world",
            agent_name="nonexistent_agent",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result.success is False
        assert "not found" in result.error
        assert result.agent_name == "unknown"
        assert result.conversation_id == "conv_123"

    @pytest.mark.asyncio
    async def test_process_message_with_cache_no_agent_name(self, cached_agent_registry):
        """Test processing message with cache without specifying agent name."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        cached_agent_registry.find_relevant_agent = Mock(return_value=mock_agent)
        cached_agent_registry.agent_cache.process_with_cache = AsyncMock()
        mock_result = Mock()
        cached_agent_registry.agent_cache.process_with_cache.return_value = mock_result

        # Call the function
        result = await cached_agent_registry.process_message_with_cache(
            message="Hello world",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result == mock_result
        cached_agent_registry.find_relevant_agent.assert_called_once()
        cached_agent_registry.agent_cache.process_with_cache.assert_called_once_with(
            mock_agent, "Hello world", "conv_123", None, None
        )

    @pytest.mark.asyncio
    async def test_process_message_with_cache_no_suitable_agent(self, cached_agent_registry):
        """Test processing message with cache when no suitable agent found."""
        # Setup mocks
        cached_agent_registry.find_relevant_agent = Mock(return_value=None)

        # Call the function
        result = await cached_agent_registry.process_message_with_cache(
            message="Hello world",
            conversation_id="conv_123"
        )

        # Verify the result
        assert result.success is False
        assert "No suitable agent found" in result.error
        assert result.agent_name == "unknown"
        assert result.conversation_id == "conv_123"

    @pytest.mark.asyncio
    async def test_invalidate_agent_cache(self, cached_agent_registry):
        """Test invalidating agent cache."""
        # Setup mocks
        cached_agent_registry.agent_cache.invalidate_agent_cache = AsyncMock(return_value=5)

        # Call the function
        count = await cached_agent_registry.invalidate_agent_cache("test_agent")

        # Verify the result
        assert count == 5
        cached_agent_registry.agent_cache.invalidate_agent_cache.assert_called_once_with("test_agent", "*", None)

    @pytest.mark.asyncio
    async def test_invalidate_conversation_cache(self, cached_agent_registry):
        """Test invalidating conversation cache."""
        # Setup mocks
        cached_agent_registry.agent_cache.invalidate_conversation_cache = AsyncMock(return_value=3)

        # Call the function
        count = await cached_agent_registry.invalidate_conversation_cache("conv_123")

        # Verify the result
        assert count == 3
        cached_agent_registry.agent_cache.invalidate_conversation_cache.assert_called_once_with("conv_123")

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cached_agent_registry):
        """Test clearing all cache."""
        # Setup mocks
        cached_agent_registry.agent_cache.clear_all_agent_cache = AsyncMock(return_value=10)

        # Call the function
        count = await cached_agent_registry.clear_all_cache()

        # Verify the result
        assert count == 10
        cached_agent_registry.agent_cache.clear_all_agent_cache.assert_called_once()

    def test_get_cache_stats(self, cached_agent_registry):
        """Test getting cache statistics."""
        # Setup mocks
        mock_stats = {"hits": 10, "misses": 5}
        cached_agent_registry.agent_cache.get_stats = Mock(return_value=mock_stats)

        # Call the function
        stats = cached_agent_registry.get_cache_stats()

        # Verify the result
        assert stats == mock_stats
        cached_agent_registry.agent_cache.get_stats.assert_called_once()


class TestCacheAgentResponseDecorator:
    """Test cases for cache_agent_response decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_agent_cache(self):
        """Test decorator with agent that has cache."""
        # Setup mocks
        mock_cache = Mock()
        mock_cache.process_with_cache = AsyncMock()
        mock_result = Mock()
        mock_cache.process_with_cache.return_value = mock_result

        mock_registry = Mock()
        mock_registry.agent_cache = mock_cache

        # Create a mock agent with cache
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent._registry = mock_registry

        # Apply the decorator
        decorated_func = cache_agent_response(ttl=1800)(mock_agent.process_message)

        # Call the decorated function
        result = await decorated_func(mock_agent, "Hello world", "conv_123", {"key": "value"})

        # Verify the result
        assert result == mock_result
        mock_cache.process_with_cache.assert_called_once_with(
            mock_agent, "Hello world", "conv_123", {"key": "value"}, 1800
        )

    @pytest.mark.asyncio
    async def test_decorator_without_agent_cache(self):
        """Test decorator with agent that doesn't have cache."""
        # Create a mock agent without cache
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.process_message = AsyncMock()
        mock_result = Mock()
        mock_agent.process_message.return_value = mock_result

        # Apply the decorator
        decorated_func = cache_agent_response(ttl=1800)(mock_agent.process_message)

        # Call the decorated function
        result = await decorated_func(mock_agent, "Hello world", "conv_123", {"key": "value"})

        # Verify the result
        assert result == mock_result
        mock_agent.process_message.assert_called_once_with(
            mock_agent, "Hello world", "conv_123", {"key": "value"}
        )

    @pytest.mark.asyncio
    async def test_decorator_without_agent_attributes(self):
        """Test decorator with object that doesn't have agent attributes."""
        # Create a mock object without agent attributes
        mock_obj = Mock()
        mock_obj.process_message = AsyncMock()
        mock_result = Mock()
        mock_obj.process_message.return_value = mock_result

        # Apply the decorator
        decorated_func = cache_agent_response(ttl=1800)(mock_obj.process_message)

        # Call the decorated function
        result = await decorated_func(mock_obj, "Hello world", "conv_123", {"key": "value"})

        # Verify the result
        assert result == mock_result
        mock_obj.process_message.assert_called_once_with(
            mock_obj, "Hello world", "conv_123", {"key": "value"}
        )


class TestGetCachedAgentRegistry:
    """Test cases for get_cached_agent_registry function."""

    @patch("app.core.caching.integration.agent_cache._cached_agent_registry", None)
    def test_get_cached_agent_registry_first_call(self):
        """Test getting cached agent registry for the first time."""
        with patch("app.core.caching.integration.agent_cache.CachedAgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Call the function
            result = get_cached_agent_registry()

            # Verify the result
            assert result == mock_registry
            mock_registry_class.assert_called_once()

    @patch("app.core.caching.integration.agent_cache._cached_agent_registry")
    def test_get_cached_agent_registry_subsequent_call(self, mock_global_registry):
        """Test getting cached agent registry for subsequent calls."""
        # Call the function
        result = get_cached_agent_registry()

        # Verify the result
        assert result == mock_global_registry