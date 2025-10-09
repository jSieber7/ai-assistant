"""
Tests for result compression utilities.
"""

import pytest
import pickle
import gzip
import lzma
import zlib
import base64
from unittest.mock import Mock, AsyncMock
from app.core.caching.compression.compressor import (
    ResultCompressor,
    CompressionAlgorithm,
    CompressionResult,
    CompressedCacheBackend,
)


class TestCompressionAlgorithm:
    """Test CompressionAlgorithm enum."""

    def test_enum_values(self):
        """Test that all compression algorithms are defined."""
        algorithms = list(CompressionAlgorithm)
        expected_algorithms = ["none", "gzip", "lzma", "zlib", "base64"]

        for alg in expected_algorithms:
            assert CompressionAlgorithm(alg) is not None

        assert len(algorithms) == len(expected_algorithms)


class TestCompressionResult:
    """Test CompressionResult class."""

    def test_initialization(self):
        """Test result initialization."""
        result = CompressionResult(
            compressed_data=b"compressed",
            original_size=100,
            compressed_size=50,
            algorithm=CompressionAlgorithm.GZIP,
            compression_ratio=0.5,
        )

        assert result.compressed_data == b"compressed"
        assert result.original_size == 100
        assert result.compressed_size == 50
        assert result.algorithm == CompressionAlgorithm.GZIP
        assert result.compression_ratio == 0.5

    def test_space_saved(self):
        """Test space saved calculation."""
        result = CompressionResult(
            compressed_data=b"compressed",
            original_size=100,
            compressed_size=50,
            algorithm=CompressionAlgorithm.GZIP,
            compression_ratio=0.5,
        )

        assert result.space_saved == 50

    def test_compression_percentage(self):
        """Test compression percentage calculation."""
        result = CompressionResult(
            compressed_data=b"compressed",
            original_size=100,
            compressed_size=50,
            algorithm=CompressionAlgorithm.GZIP,
            compression_ratio=0.5,
        )

        assert result.compression_percentage == 50.0

    def test_compression_percentage_zero_original(self):
        """Test compression percentage with zero original size."""
        result = CompressionResult(
            compressed_data=b"",
            original_size=0,
            compressed_size=0,
            algorithm=CompressionAlgorithm.NONE,
            compression_ratio=1.0,
        )

        assert result.compression_percentage == 0.0


class TestResultCompressor:
    """Test ResultCompressor class."""

    @pytest.fixture
    def compressor(self):
        """Create a result compressor."""
        return ResultCompressor(
            default_algorithm=CompressionAlgorithm.GZIP,
            min_size_for_compression=10,
            compression_level=6,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, compressor):
        """Test compressor initialization."""
        assert compressor.default_algorithm == CompressionAlgorithm.GZIP
        assert compressor.min_size_for_compression == 10
        assert compressor.compression_level == 6
        assert compressor._stats["total_compressions"] == 0
        assert compressor._stats["total_decompressions"] == 0

    @pytest.mark.asyncio
    async def test_compress_none_algorithm(self, compressor):
        """Test compression with no algorithm (none)."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        result = await compressor.compress(test_data, CompressionAlgorithm.NONE)

        assert result.algorithm == CompressionAlgorithm.NONE
        assert result.original_size > 0
        assert result.compressed_size == result.original_size
        assert result.compression_ratio == 1.0
        assert result.compressed_data == pickle.dumps(test_data)

    @pytest.mark.asyncio
    async def test_compress_gzip(self, compressor):
        """Test compression with gzip algorithm."""
        # Use larger data that will actually compress
        test_data = {"key": "value" * 100, "numbers": list(range(1000))}
        result = await compressor.compress(test_data, CompressionAlgorithm.GZIP)

        assert result.algorithm == CompressionAlgorithm.GZIP
        assert result.original_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0

        # Verify decompression works
        decompressed = gzip.decompress(result.compressed_data)
        original = pickle.loads(decompressed)
        assert original == test_data

    @pytest.mark.asyncio
    async def test_compress_lzma(self, compressor):
        """Test compression with lzma algorithm."""
        # Use larger data that will actually compress
        test_data = {"key": "value" * 100, "numbers": list(range(1000))}
        result = await compressor.compress(test_data, CompressionAlgorithm.LZMA)

        assert result.algorithm == CompressionAlgorithm.LZMA
        assert result.original_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0

        # Verify decompression works
        decompressed = lzma.decompress(result.compressed_data)
        original = pickle.loads(decompressed)
        assert original == test_data

    @pytest.mark.asyncio
    async def test_compress_zlib(self, compressor):
        """Test compression with zlib algorithm."""
        # Use larger data that will actually compress
        test_data = {"key": "value" * 100, "numbers": list(range(1000))}
        result = await compressor.compress(test_data, CompressionAlgorithm.ZLIB)

        assert result.algorithm == CompressionAlgorithm.ZLIB
        assert result.original_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0

        # Verify decompression works
        decompressed = zlib.decompress(result.compressed_data)
        original = pickle.loads(decompressed)
        assert original == test_data

    @pytest.mark.asyncio
    async def test_compress_base64(self, compressor):
        """Test compression with base64 algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        result = await compressor.compress(test_data, CompressionAlgorithm.BASE64)

        assert result.algorithm == CompressionAlgorithm.BASE64
        assert result.original_size > 0
        assert result.compressed_size > result.original_size  # Base64 increases size
        assert result.compression_ratio > 1.0

        # Verify decompression works
        decompressed = base64.b64decode(result.compressed_data)
        original = pickle.loads(decompressed)
        assert original == test_data

    @pytest.mark.asyncio
    async def test_compress_small_data(self, compressor):
        """Test compression with data smaller than threshold."""
        test_data = 1  # Very small data (4 bytes when pickled)
        result = await compressor.compress(test_data, CompressionAlgorithm.GZIP)

        # Should use NONE algorithm for small data (below min_size_for_compression=10)
        assert result.algorithm == CompressionAlgorithm.NONE
        assert result.compressed_data == pickle.dumps(test_data)

    @pytest.mark.asyncio
    async def test_decompress_none(self, compressor):
        """Test decompression with no algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = pickle.dumps(test_data)

        result = await compressor.decompress(compressed_data, CompressionAlgorithm.NONE)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_decompress_gzip(self, compressor):
        """Test decompression with gzip algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = gzip.compress(pickle.dumps(test_data))

        result = await compressor.decompress(compressed_data, CompressionAlgorithm.GZIP)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_decompress_lzma(self, compressor):
        """Test decompression with lzma algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = lzma.compress(pickle.dumps(test_data))

        result = await compressor.decompress(compressed_data, CompressionAlgorithm.LZMA)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_decompress_zlib(self, compressor):
        """Test decompression with zlib algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = zlib.compress(pickle.dumps(test_data))

        result = await compressor.decompress(compressed_data, CompressionAlgorithm.ZLIB)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_decompress_base64(self, compressor):
        """Test decompression with base64 algorithm."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = base64.b64encode(pickle.dumps(test_data))

        result = await compressor.decompress(
            compressed_data, CompressionAlgorithm.BASE64
        )
        assert result == test_data

    @pytest.mark.asyncio
    async def test_decompress_invalid_algorithm(self, compressor):
        """Test decompression with invalid algorithm."""
        test_data = {"key": "value"}
        compressed_data = pickle.dumps(test_data)

        with pytest.raises(ValueError, match="Unsupported compression algorithm"):
            await compressor.decompress(compressed_data, "invalid_algorithm")

    @pytest.mark.asyncio
    async def test_auto_compress(self, compressor):
        """Test automatic algorithm selection."""
        test_data = {"key": "value", "numbers": list(range(100))}  # Medium size data
        result = await compressor.auto_compress(test_data)

        # Should use default algorithm (GZIP)
        assert result.algorithm == CompressionAlgorithm.GZIP
        assert result.compressed_size < result.original_size

    @pytest.mark.asyncio
    async def test_get_best_algorithm_small_data(self, compressor):
        """Test best algorithm selection for small data."""
        test_data = {"key": "value"}  # Small data
        algorithm = await compressor.get_best_algorithm(test_data)
        assert algorithm == CompressionAlgorithm.NONE

    @pytest.mark.asyncio
    async def test_get_best_algorithm_medium_data(self, compressor):
        """Test best algorithm selection for medium data."""
        test_data = {"key": "value" * 100}  # Medium size data
        algorithm = await compressor.get_best_algorithm(test_data)
        assert algorithm == CompressionAlgorithm.ZLIB  # Fast for small/medium data

    @pytest.mark.asyncio
    async def test_get_best_algorithm_large_data(self, compressor):
        """Test best algorithm selection for large data."""
        test_data = {"key": "value" * 1000}  # Large data
        algorithm = await compressor.get_best_algorithm(test_data)
        # For large data, it should choose an algorithm that compresses (not NONE)
        assert algorithm != CompressionAlgorithm.NONE
        assert algorithm in [
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.LZMA,
            CompressionAlgorithm.ZLIB,
        ]

    @pytest.mark.asyncio
    async def test_get_stats(self, compressor):
        """Test getting compressor statistics."""
        # Perform some operations
        test_data = {"key": "value"}
        await compressor.compress(test_data, CompressionAlgorithm.GZIP)
        # Use valid compressed data for decompression test
        compressed_data = pickle.dumps(test_data)
        await compressor.decompress(compressed_data, CompressionAlgorithm.NONE)

        stats = compressor.get_stats()

        assert stats["total_compressions"] == 1
        assert stats["total_decompressions"] == 1
        # For small data, bytes saved might be negative (overhead), so we don't check > 0
        assert "total_bytes_saved" in stats
        assert "compressions_by_algorithm" in stats
        assert "decompressions_by_algorithm" in stats
        assert stats["default_algorithm"] == "gzip"
        assert stats["min_size_for_compression"] == 10
        assert stats["compression_level"] == 6

    @pytest.mark.asyncio
    async def test_reset_stats(self, compressor):
        """Test resetting compressor statistics."""
        # Perform some operations
        test_data = {"key": "value"}
        await compressor.compress(test_data, CompressionAlgorithm.GZIP)

        # Reset stats
        compressor.reset_stats()

        stats = compressor.get_stats()
        assert stats["total_compressions"] == 0
        assert stats["total_decompressions"] == 0
        assert stats["total_bytes_saved"] == 0


class TestCompressedCacheBackend:
    """Test CompressedCacheBackend class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock cache backend."""
        backend = Mock()
        backend.get = AsyncMock()
        backend.set = AsyncMock()
        backend.delete = AsyncMock()
        backend.exists = AsyncMock()
        backend.clear = AsyncMock()
        backend.close = AsyncMock()
        backend.keys = AsyncMock()
        return backend

    @pytest.fixture
    def compressed_backend(self, mock_backend):
        """Create a compressed cache backend."""
        return CompressedCacheBackend(backend=mock_backend, compress_threshold=50)

    @pytest.mark.asyncio
    async def test_initialization(self, compressed_backend, mock_backend):
        """Test backend initialization."""
        assert compressed_backend.backend == mock_backend
        assert compressed_backend.compress_threshold == 50
        assert compressed_backend.compressor is not None
        assert compressed_backend.metadata_suffix == ":compression"

    @pytest.mark.asyncio
    async def test_get_compressed_value(self, compressed_backend, mock_backend):
        """Test getting a compressed value."""
        # Mock compressed value and metadata
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        compressed_data = gzip.compress(pickle.dumps(test_data))

        mock_backend.get.side_effect = [
            # Main value
            compressed_data,
            {  # Metadata
                "algorithm": "gzip",
                "original_size": 100,
                "compressed_size": 50,
                "compression_ratio": 0.5,
            },
        ]

        result = await compressed_backend.get("test_key")
        assert result == test_data

        # Should call get for both value and metadata
        assert mock_backend.get.call_count == 2
        mock_backend.get.assert_any_call("test_key")
        mock_backend.get.assert_any_call("test_key:compression")

    @pytest.mark.asyncio
    async def test_get_uncompressed_value(self, compressed_backend, mock_backend):
        """Test getting an uncompressed value."""
        test_data = {"key": "value"}
        mock_backend.get.side_effect = [test_data, None]  # No metadata

        result = await compressed_backend.get("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_get_decompression_error(self, compressed_backend, mock_backend):
        """Test getting a value with decompression error."""
        # Mock invalid compressed data
        compressed_data = b"invalid_compressed_data"
        mock_backend.get.side_effect = [
            compressed_data,  # Main value as bytes
            {  # Metadata
                "algorithm": "gzip",
                "original_size": 100,
                "compressed_size": 50,
                "compression_ratio": 0.5,
            },
        ]

        # Should return the compressed value as fallback
        result = await compressed_backend.get("test_key")
        assert result == compressed_data

    @pytest.mark.asyncio
    async def test_set_compress_large_data(self, compressed_backend, mock_backend):
        """Test setting a large value that should be compressed."""
        test_data = {"key": "value" * 100}  # Large data
        mock_backend.set.return_value = True

        result = await compressed_backend.set("test_key", test_data, 300)
        assert result is True

        # Should call set for both value and metadata
        assert mock_backend.set.call_count == 2

        # Get the actual calls and verify they have the right types
        calls = mock_backend.set.call_args_list

        # Find the value call and metadata call
        value_call = None
        metadata_call = None

        for call in calls:
            args, _ = call
            if args[0] == "test_key":
                value_call = args
            elif args[0] == "test_key:compression":
                metadata_call = args

        assert value_call is not None, "Value call not found"
        assert metadata_call is not None, "Metadata call not found"

        # Verify the value is bytes (compressed data)
        assert isinstance(
            value_call[1], bytes
        ), f"Expected bytes, got {type(value_call[1])}"

        # Verify the metadata is a dict with expected keys
        assert isinstance(
            metadata_call[1], dict
        ), f"Expected dict, got {type(metadata_call[1])}"
        assert "algorithm" in metadata_call[1]
        assert "original_size" in metadata_call[1]
        assert "compressed_size" in metadata_call[1]
        assert "compression_ratio" in metadata_call[1]

    @pytest.mark.asyncio
    async def test_set_compress_small_data(self, compressed_backend, mock_backend):
        """Test setting a small value that should not be compressed."""
        test_data = {"key": "value"}  # Small data
        mock_backend.set.return_value = True

        result = await compressed_backend.set("test_key", test_data, 300)
        assert result is True

        # Should call set only for the value (no compression)
        mock_backend.set.assert_called_once_with("test_key", test_data, 300)

    @pytest.mark.asyncio
    async def test_set_compression_metadata_failure(
        self, compressed_backend, mock_backend
    ):
        """Test setting with metadata storage failure."""
        test_data = {"key": "value" * 100}  # Large data
        mock_backend.set.side_effect = [True, False]  # Metadata storage fails

        result = await compressed_backend.set("test_key", test_data, 300)
        assert result is False

        # Should clean up the main value
        mock_backend.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete(self, compressed_backend, mock_backend):
        """Test deleting a key."""
        mock_backend.delete.return_value = True

        result = await compressed_backend.delete("test_key")
        assert result is True

        # Should delete both value and metadata
        assert mock_backend.delete.call_count == 2
        mock_backend.delete.assert_any_call("test_key")
        mock_backend.delete.assert_any_call("test_key:compression")

    @pytest.mark.asyncio
    async def test_exists(self, compressed_backend, mock_backend):
        """Test checking existence."""
        mock_backend.exists.return_value = True

        result = await compressed_backend.exists("test_key")
        assert result is True

        # Should only check the main key (not metadata)
        mock_backend.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear(self, compressed_backend, mock_backend):
        """Test clearing the cache."""
        mock_backend.clear.return_value = True

        result = await compressed_backend.clear()
        assert result is True

        mock_backend.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_keys(self, compressed_backend, mock_backend):
        """Test getting keys."""
        mock_backend.keys.return_value = [
            "key1",
            "key2",
            "key1:compression",
            "key2:compression",
        ]

        result = await compressed_backend.keys()
        assert result == ["key1", "key2"]  # Should filter out metadata keys

    @pytest.mark.asyncio
    async def test_close(self, compressed_backend, mock_backend):
        """Test closing the backend."""
        await compressed_backend.close()
        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_compression_stats(self, compressed_backend):
        """Test getting compression statistics."""
        stats = compressed_backend.get_compression_stats()
        assert "total_compressions" in stats
        assert "total_decompressions" in stats


class TestCompressionIntegration:
    """Integration tests for compression system."""

    @pytest.mark.asyncio
    async def test_end_to_end_compression(self):
        """Test end-to-end compression and decompression."""
        compressor = ResultCompressor()

        # Test data
        test_data = {
            "message": "Hello, world!",
            "numbers": list(range(1000)),
            "nested": {"key1": "value1", "key2": ["item1", "item2", "item3"]},
        }

        # Compress the data
        compression_result = await compressor.compress(
            test_data, CompressionAlgorithm.GZIP
        )

        # Verify compression worked
        assert compression_result.algorithm == CompressionAlgorithm.GZIP
        assert compression_result.compressed_size < compression_result.original_size
        assert compression_result.compression_ratio < 1.0

        # Decompress the data
        decompressed_data = await compressor.decompress(
            compression_result.compressed_data, compression_result.algorithm
        )

        # Verify data integrity
        assert decompressed_data == test_data

    @pytest.mark.asyncio
    async def test_compression_performance(self):
        """Test compression performance with different algorithms."""
        compressor = ResultCompressor()

        # Large test data
        test_data = {"data": "x" * 10000}

        algorithms = [
            CompressionAlgorithm.NONE,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.LZMA,
            CompressionAlgorithm.ZLIB,
        ]

        results = {}
        for algorithm in algorithms:
            result = await compressor.compress(test_data, algorithm)
            results[algorithm.value] = {
                "original_size": result.original_size,
                "compressed_size": result.compressed_size,
                "compression_ratio": result.compression_ratio,
                "space_saved": result.space_saved,
            }

        # Verify compression ratios make sense
        assert results["none"]["compression_ratio"] == 1.0
        assert results["gzip"]["compression_ratio"] < 1.0
        assert results["lzma"]["compression_ratio"] < 1.0
        assert results["zlib"]["compression_ratio"] < 1.0

        # For repetitive data like "x" * 10000, LZMA should have good compression
        # but we can't guarantee it's always the best, so we'll check that at least
        # one of the compression algorithms is better than NONE
        compressed_sizes = [
            results[alg]["compressed_size"] for alg in ["gzip", "lzma", "zlib"]
        ]
        assert min(compressed_sizes) < results["none"]["compressed_size"]
