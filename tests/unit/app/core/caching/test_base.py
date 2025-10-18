"""
Tests for caching base classes and interfaces.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.core.caching.base import (
    CacheBackend,
    CacheLayer,
    MultiLayerCache,
    CacheMetrics,
)


@pytest.mark.unit
class TestCacheBackend:
    """Test CacheBackend interface."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock cache backend."""
        backend = Mock(spec=CacheBackend)
        backend.get = AsyncMock()
        backend.set = AsyncMock()
        backend.delete = AsyncMock()
        backend.exists = AsyncMock()
        backend.clear = AsyncMock()
        backend.close = AsyncMock()
        backend.get_stats = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_get(self, mock_backend):
        """Test get method."""
        mock_backend.get.return_value = "test_value"
        result = await mock_backend.get("test_key")
        assert result == "test_value"
        mock_backend.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set(self, mock_backend):
        """Test set method."""
        mock_backend.set.return_value = True
        result = await mock_backend.set("test_key", "test_value", 300)
        assert result is True
        mock_backend.set.assert_called_once_with("test_key", "test_value", 300)

    @pytest.mark.asyncio
    async def test_delete(self, mock_backend):
        """Test delete method."""
        mock_backend.delete.return_value = True
        result = await mock_backend.delete("test_key")
        assert result is True
        mock_backend.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists(self, mock_backend):
        """Test exists method."""
        mock_backend.exists.return_value = True
        result = await mock_backend.exists("test_key")
        assert result is True
        mock_backend.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear(self, mock_backend):
        """Test clear method."""
        mock_backend.clear.return_value = True
        result = await mock_backend.clear()
        assert result is True
        mock_backend.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, mock_backend):
        """Test close method."""
        await mock_backend.close()
        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_backend):
        """Test get_stats method."""
        mock_backend.get_stats.return_value = {"hits": 10, "misses": 5}
        result = await mock_backend.get_stats()
        assert result == {"hits": 10, "misses": 5}
        mock_backend.get_stats.assert_called_once()


@pytest.mark.unit
class TestCacheLayer:
    """Test CacheLayer class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = Mock(spec=CacheBackend)
        backend.get = AsyncMock()
        backend.set = AsyncMock()
        backend.delete = AsyncMock()
        backend.exists = AsyncMock()
        backend.clear = AsyncMock()
        backend.close = AsyncMock()
        backend.get_stats = AsyncMock()
        return backend

    @pytest.fixture
    def cache_layer(self, mock_backend):
        """Create a cache layer with mock backend."""
        from app.core.caching.base import CacheLayer

        return CacheLayer(
            backend=mock_backend,
            name="test_layer",
            priority=0,
            read_only=False,
            default_ttl=300,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, cache_layer, mock_backend):
        """Test layer initialization."""
        assert cache_layer.name == "test_layer"
        assert cache_layer.priority == 0
        assert cache_layer.read_only is False
        assert cache_layer.default_ttl == 300
        assert cache_layer.backend == mock_backend

    @pytest.mark.asyncio
    async def test_get(self, cache_layer, mock_backend):
        """Test layer get method."""
        mock_backend.get.return_value = "test_value"
        result = await cache_layer.get("test_key")
        assert result == "test_value"
        mock_backend.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set(self, cache_layer, mock_backend):
        """Test layer set method."""
        mock_backend.set.return_value = True
        result = await cache_layer.set("test_key", "test_value")
        assert result is True
        mock_backend.set.assert_called_once_with("test_key", "test_value", 300)

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache_layer, mock_backend):
        """Test layer set method with custom TTL."""
        mock_backend.set.return_value = True
        result = await cache_layer.set("test_key", "test_value", 600)
        assert result is True
        mock_backend.set.assert_called_once_with("test_key", "test_value", 600)

    @pytest.mark.asyncio
    async def test_set_read_only(self, cache_layer, mock_backend):
        """Test layer set method in read-only mode."""
        cache_layer.read_only = True
        result = await cache_layer.set("test_key", "test_value")
        assert result is False
        mock_backend.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete(self, cache_layer, mock_backend):
        """Test layer delete method."""
        mock_backend.delete.return_value = True
        result = await cache_layer.delete("test_key")
        assert result is True
        mock_backend.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_read_only(self, cache_layer, mock_backend):
        """Test layer delete method in read-only mode."""
        cache_layer.read_only = True
        result = await cache_layer.delete("test_key")
        assert result is False
        mock_backend.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_exists(self, cache_layer, mock_backend):
        """Test layer exists method."""
        mock_backend.exists.return_value = True
        result = await cache_layer.exists("test_key")
        assert result is True
        mock_backend.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear(self, cache_layer, mock_backend):
        """Test layer clear method."""
        mock_backend.clear.return_value = True
        result = await cache_layer.clear()
        assert result is True
        mock_backend.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_read_only(self, cache_layer, mock_backend):
        """Test layer clear method in read-only mode."""
        cache_layer.read_only = True
        result = await cache_layer.clear()
        assert result is False
        mock_backend.clear.assert_not_called()

    @pytest.mark.asyncio
    async def test_close(self, cache_layer, mock_backend):
        """Test layer close method."""
        await cache_layer.close()
        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_layer, mock_backend):
        """Test layer get_stats method."""
        mock_backend.get_stats.return_value = {"hits": 10, "misses": 5}
        result = cache_layer.get_stats()
        assert result == {
            "name": "test_layer",
            "priority": 0,
            "read_only": False,
            "enabled": True,
            "default_ttl": 300,
        }


@pytest.mark.unit
class TestMultiLayerCache:
    """Test MultiLayerCache class."""

    @pytest.fixture
    def mock_layers(self):
        """Create mock cache layers."""
        layers = []
        for i in range(3):
            layer = Mock(spec=CacheLayer)
            layer.name = f"layer_{i}"
            layer.priority = i
            layer.read_only = False
            layer.default_ttl = 300
            layer.enabled = True
            layer.get = AsyncMock()
            layer.set = AsyncMock()
            layer.delete = AsyncMock()
            layer.exists = AsyncMock()
            layer.clear = AsyncMock()
            layer.close = AsyncMock()
            layer.get_stats = AsyncMock()
            layers.append(layer)
        return layers

    @pytest.fixture
    def multi_layer_cache(self, mock_layers):
        """Create a multi-layer cache with mock layers."""
        cache = MultiLayerCache()
        for layer in mock_layers:
            cache.add_layer(layer)
        cache.set_write_through(True)
        cache.set_read_through(True)
        return cache

    @pytest.mark.asyncio
    async def test_initialization(self, multi_layer_cache, mock_layers):
        """Test multi-layer cache initialization."""
        assert multi_layer_cache.layers == mock_layers
        assert multi_layer_cache._write_through is True  # Use internal attribute
        assert multi_layer_cache._read_through is True  # Use internal attribute
        # Layers should be sorted by priority
        assert multi_layer_cache.layers[0].priority == 0
        assert multi_layer_cache.layers[1].priority == 1
        assert multi_layer_cache.layers[2].priority == 2

    @pytest.mark.asyncio
    async def test_get_found_in_first_layer(self, multi_layer_cache, mock_layers):
        """Test get when value is found in first layer."""
        mock_layers[0].get.return_value = "test_value"
        result = await multi_layer_cache.get("test_key")
        assert result == "test_value"
        mock_layers[0].get.assert_called_once_with("test_key")
        # Other layers should not be checked
        mock_layers[1].get.assert_not_called()
        mock_layers[2].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_found_in_second_layer(self, multi_layer_cache, mock_layers):
        """Test get when value is found in second layer."""
        mock_layers[0].get.return_value = None
        mock_layers[1].get.return_value = "test_value"
        result = await multi_layer_cache.get("test_key")
        assert result == "test_value"
        mock_layers[0].get.assert_called_once_with("test_key")
        mock_layers[1].get.assert_called_once_with("test_key")
        mock_layers[2].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_not_found(self, multi_layer_cache, mock_layers):
        """Test get when value is not found in any layer."""
        for layer in mock_layers:
            layer.get.return_value = None
        result = await multi_layer_cache.get("test_key")
        assert result is None
        for layer in mock_layers:
            layer.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_write_through(self, multi_layer_cache, mock_layers):
        """Test set with write-through enabled."""
        for layer in mock_layers:
            layer.set.return_value = True
        result = await multi_layer_cache.set("test_key", "test_value")
        assert result is True
        for layer in mock_layers:
            layer.set.assert_called_once_with("test_key", "test_value", None)

    @pytest.mark.asyncio
    async def test_set_write_through_partial_failure(
        self, multi_layer_cache, mock_layers
    ):
        """Test set with partial write-through failure."""
        mock_layers[0].set.return_value = True
        mock_layers[1].set.return_value = False  # Simulate failure
        mock_layers[2].set.return_value = True
        result = await multi_layer_cache.set("test_key", "test_value")
        assert result is False  # Should return False if any layer fails

    @pytest.mark.asyncio
    async def test_set_no_write_through(self, multi_layer_cache, mock_layers):
        """Test set with write-through disabled."""
        multi_layer_cache.set_write_through(False)
        mock_layers[0].set.return_value = True  # Only write to first layer
        result = await multi_layer_cache.set("test_key", "test_value")
        assert result is True
        mock_layers[0].set.assert_called_once_with("test_key", "test_value", None)
        # Other layers should not be written to
        mock_layers[1].set.assert_not_called()
        mock_layers[2].set.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete(self, multi_layer_cache, mock_layers):
        """Test delete operation."""
        for layer in mock_layers:
            layer.delete.return_value = True
        result = await multi_layer_cache.delete("test_key")
        assert result is True
        for layer in mock_layers:
            layer.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists(self, multi_layer_cache, mock_layers):
        """Test exists operation."""
        mock_layers[0].exists.return_value = False
        mock_layers[1].exists.return_value = True  # Found in second layer
        result = await multi_layer_cache.exists("test_key")
        assert result is True
        mock_layers[0].exists.assert_called_once_with("test_key")
        mock_layers[1].exists.assert_called_once_with("test_key")
        # Should stop after finding the key
        mock_layers[2].exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear(self, multi_layer_cache, mock_layers):
        """Test clear operation."""
        for layer in mock_layers:
            layer.clear.return_value = True
        result = await multi_layer_cache.clear()
        assert result is True
        for layer in mock_layers:
            layer.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, multi_layer_cache, mock_layers):
        """Test get_stats operation."""
        for i, layer in enumerate(mock_layers):
            layer.get_stats.return_value = {
                "name": f"layer_{i}",
                "priority": i,
                "read_only": False,
                "hits": i * 10,
                "misses": i * 5,
            }
        result = multi_layer_cache.get_stats()  # Remove 'await'
        assert "layers" in result
        assert len(result["layers"]) == 3
        assert result["total_layers"] == 3
        assert result["write_through"] is True
        assert result["read_through"] is True


@pytest.mark.unit
class TestCacheMetrics:
    """Test CacheMetrics class."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0

    def test_record_hit(self):
        """Test recording a hit."""
        metrics = CacheMetrics()
        metrics.record_hit()
        assert metrics.hits == 1
        assert metrics.misses == 0

    def test_record_miss(self):
        """Test recording a miss."""
        metrics = CacheMetrics()
        metrics.record_miss()
        assert metrics.hits == 0
        assert metrics.misses == 1

    def test_record_set(self):
        """Test recording a set."""
        metrics = CacheMetrics()
        metrics.record_set()
        assert metrics.sets == 1

    def test_record_delete(self):
        """Test recording a delete."""
        metrics = CacheMetrics()
        metrics.record_delete()
        assert metrics.deletes == 1

    def test_record_error(self):
        """Test recording an error."""
        metrics = CacheMetrics()
        metrics.record_error()
        assert metrics.errors == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        assert metrics.hit_rate() == 2 / 3  # 2 hits out of 3 operations

    def test_hit_rate_no_operations(self):
        """Test hit rate with no operations."""
        metrics = CacheMetrics()
        assert metrics.hit_rate() == 0.0

    def test_total_operations(self):
        """Test total operations calculation."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        metrics.record_delete()
        metrics.record_error()
        assert metrics.total_operations() == 5

    def test_reset(self):
        """Test metrics reset."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_miss()
        metrics.reset()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0
