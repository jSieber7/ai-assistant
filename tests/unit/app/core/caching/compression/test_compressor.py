"""
Unit tests for the compression module.

Tests for result compression utilities, compression algorithms, and compressed cache backend wrapper.
"""

import pytest
import asyncio
import gzip
import lzma
import zlib
import base64
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.caching.compression.compressor import (
    CompressionAlgorithm,
    CompressionResult,
    ResultCompressor,
    CompressedCacheBackend,
    get_default_compressor,
    compress_data,
    decompress_data,
    compress_string,
    decompress_string,
)


class TestCompressionAlgorithm:
    """Test cases for CompressionAlgorithm enum."""

    def test_compression_algorithm_values(self):
        """Test that CompressionAlgorithm has the expected values."""
        assert CompressionAlgorithm.NONE.value == "none"
        assert CompressionAlgorithm.GZIP.value == "gzip"
        assert CompressionAlgorithm.LZMA.value == "lzma"
        assert CompressionAlgorithm.ZLIB.value == "zlib"
        assert CompressionAlgorithm.BASE64.value == "base64"


class TestCompressionResult:
    """Test cases for CompressionResult class."""

    def test_compression_result_properties(self):
        """Test CompressionResult properties."""
        result = CompressionResult(
            compressed_data=b"compressed",
            original_size=100,
            compressed_size=25,
            algorithm=CompressionAlgorithm.GZIP,
            compression_ratio=0.25,
        )
        
        assert result.space_saved == 75
        assert result.compression_percentage == 75.0

    def test_compression_result_zero_size(self):
        """Test CompressionResult with zero original size."""
        result = CompressionResult(
            compressed_data=b"",
            original_size=0,
            compressed_size=0,
            algorithm=CompressionAlgorithm.NONE,
            compression_ratio=1.0,
        )
        
        assert result.space_saved == 0
        assert result.compression_percentage == 0.0


class TestResultCompressor:
    """Test cases for ResultCompressor class."""

    @pytest.fixture
    def compressor(self):
        """Create a result compressor."""
        return ResultCompressor(
            default_algorithm=CompressionAlgorithm.GZIP,
            min_size_for_compression=100,
            compression_level=6,
        )

    @pytest.mark.asyncio
    async def test_result_compressor_initialization(self):
        """Test initializing a result compressor."""
        compressor = ResultCompressor(
            default_algorithm=CompressionAlgorithm.LZMA,
            min_size_for_compression=50,
            compression_level=9,
        )
        
        assert compressor.default_algorithm == CompressionAlgorithm.LZMA
        assert compressor.min_size_for_compression == 50
        assert compressor.compression_level == 9
        assert compressor._stats["total_compressions"] == 0
        assert compressor._stats["total_decompressions"] == 0

    @pytest.mark.asyncio
    async def test_compress_none_algorithm(self, compressor):
        """Test compressing with NONE algorithm."""
        data = "test data"
        result = await compressor.compress(data, CompressionAlgorithm.NONE)
        
        assert result.algorithm == CompressionAlgorithm.NONE
        assert result.original_size > 0
        assert result.compressed_data == compressor._serialize_data(data)
        assert result.compressed_size == result.original_size
        assert result.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_compress_gzip(self, compressor):
        """Test compressing with GZIP algorithm."""
        data = "test data" * 100  # Make it large enough
        result = await compressor.compress(data, CompressionAlgorithm.GZIP)
        
        assert result.algorithm == CompressionAlgorithm.GZIP
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0
        
        # Verify it's valid GZIP data
        decompressed = gzip.decompress(result.compressed_data)
        assert decompressed == compressor._serialize_data(data)

    @pytest.mark.asyncio
    async def test_compress_lzma(self, compressor):
        """Test compressing with LZMA algorithm."""
        data = "test data" * 100  # Make it large enough
        result = await compressor.compress(data, CompressionAlgorithm.LZMA)
        
        assert result.algorithm == CompressionAlgorithm.LZMA
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0
        
        # Verify it's valid LZMA data
        decompressed = lzma.decompress(result.compressed_data)
        assert decompressed == compressor._serialize_data(data)

    @pytest.mark.asyncio
    async def test_compress_zlib(self, compressor):
        """Test compressing with ZLIB algorithm."""
        data = "test data" * 100  # Make it large enough
        result = await compressor.compress(data, CompressionAlgorithm.ZLIB)
        
        assert result.algorithm == CompressionAlgorithm.ZLIB
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compressed_size < result.original_size
        assert result.compression_ratio < 1.0
        
        # Verify it's valid ZLIB data
        decompressed = zlib.decompress(result.compressed_data)
        assert decompressed == compressor._serialize_data(data)

    @pytest.mark.asyncio
    async def test_compress_base64(self, compressor):
        """Test compressing with BASE64 algorithm."""
        data = "test data"
        result = await compressor.compress(data, CompressionAlgorithm.BASE64)
        
        assert result.algorithm == CompressionAlgorithm.BASE64
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compressed_size > result.original_size  # Base64 increases size
        assert result.compression_ratio > 1.0
        
        # Verify it's valid Base64 data
        decoded = base64.b64decode(result.compressed_data)
        assert decoded == compressor._serialize_data(data)

    @pytest.mark.asyncio
    async def test_compress_small_data(self, compressor):
        """Test compressing small data (below threshold)."""
        data = "small"  # Smaller than min_size_for_compression
        result = await compressor.compress(data)
        
        # Should use NONE algorithm for small data
        assert result.algorithm == CompressionAlgorithm.NONE
        assert result.compressed_size == result.original_size
        assert result.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_compress_default_algorithm(self, compressor):
        """Test compressing with default algorithm."""
        data = "test data" * 100  # Make it large enough
        result = await compressor.compress(data)  # No algorithm specified
        
        # Should use default algorithm
        assert result.algorithm == CompressionAlgorithm.GZIP

    @pytest.mark.asyncio
    async def test_compress_unsupported_algorithm(self, compressor):
        """Test compressing with unsupported algorithm."""
        data = "test data"
        
        # Should raise ValueError for unsupported algorithm
        with pytest.raises(ValueError, match="Unsupported compression algorithm"):
            await compressor.compress(data, "unsupported")

    @pytest.mark.asyncio
    async def test_decompress_none_algorithm(self, compressor):
        """Test decompressing NONE algorithm."""
        data = "test data"
        compressed = await compressor.compress(data, CompressionAlgorithm.NONE)
        
        result = await compressor.decompress(compressed.compressed_data, CompressionAlgorithm.NONE)
        assert result == data

    @pytest.mark.asyncio
    async def test_decompress_gzip(self, compressor):
        """Test decompressing GZIP algorithm."""
        data = "test data" * 100
        compressed = await compressor.compress(data, CompressionAlgorithm.GZIP)
        
        result = await compressor.decompress(compressed.compressed_data, CompressionAlgorithm.GZIP)
        assert result == data

    @pytest.mark.asyncio
    async def test_decompress_lzma(self, compressor):
        """Test decompressing LZMA algorithm."""
        data = "test data" * 100
        compressed = await compressor.compress(data, CompressionAlgorithm.LZMA)
        
        result = await compressor.decompress(compressed.compressed_data, CompressionAlgorithm.LZMA)
        assert result == data

    @pytest.mark.asyncio
    async def test_decompress_zlib(self, compressor):
        """Test decompressing ZLIB algorithm."""
        data = "test data" * 100
        compressed = await compressor.compress(data, CompressionAlgorithm.ZLIB)
        
        result = await compressor.decompress(compressed.compressed_data, CompressionAlgorithm.ZLIB)
        assert result == data

    @pytest.mark.asyncio
    async def test_decompress_base64(self, compressor):
        """Test decompressing BASE64 algorithm."""
        data = "test data"
        compressed = await compressor.compress(data, CompressionAlgorithm.BASE64)
        
        result = await compressor.decompress(compressed.compressed_data, CompressionAlgorithm.BASE64)
        assert result == data

    @pytest.mark.asyncio
    async def test_decompress_unsupported_algorithm(self, compressor):
        """Test decompressing unsupported algorithm."""
        # Should raise ValueError for unsupported algorithm
        with pytest.raises(ValueError, match="Unsupported compression algorithm"):
            await compressor.decompress(b"data", "unsupported")

    @pytest.mark.asyncio
    async def test_serialize_deserialize_data(self, compressor):
        """Test serializing and deserializing data."""
        # Test with various data types
        test_cases = [
            "string",
            123,
            123.456,
            {"key": "value", "nested": {"a": 1, "b": 2}},
            [1, 2, 3, "four"],
            (1, 2, 3),
            True,
            None,
        ]
        
        for data in test_cases:
            # Serialize
            serialized = compressor._serialize_data(data)
            assert isinstance(serialized, bytes)
            
            # Deserialize
            deserialized = compressor._deserialize_data(serialized)
            assert deserialized == data

    @pytest.mark.asyncio
    async def test_deserialize_invalid_data(self, compressor):
        """Test deserializing invalid data."""
        # Should raise ValueError for invalid pickle data
        with pytest.raises(ValueError, match="Failed to deserialize cached data"):
            compressor._deserialize_data(b"invalid pickle data")

    @pytest.mark.asyncio
    async def test_auto_compress(self, compressor):
        """Test auto_compress method."""
        data = "test data" * 100
        result = await compressor.auto_compress(data)
        
        # Should use default algorithm
        assert result.algorithm == CompressionAlgorithm.GZIP

    @pytest.mark.asyncio
    async def test_get_best_algorithm(self, compressor):
        """Test get_best_algorithm method."""
        # Small data should use NONE
        algorithm = await compressor.get_best_algorithm("small")
        assert algorithm == CompressionAlgorithm.NONE
        
        # Medium data should use ZLIB
        medium_data = "x" * 500
        algorithm = await compressor.get_best_algorithm(medium_data)
        assert algorithm == CompressionAlgorithm.ZLIB
        
        # Large data should use GZIP
        large_data = "x" * 5000
        algorithm = await compressor.get_best_algorithm(large_data)
        assert algorithm == CompressionAlgorithm.GZIP
        
        # Very large data should use LZMA
        very_large_data = "x" * 50000
        algorithm = await compressor.get_best_algorithm(very_large_data)
        assert algorithm == CompressionAlgorithm.LZMA

    @pytest.mark.asyncio
    async def test_get_stats(self, compressor):
        """Test getting compression statistics."""
        # Perform some operations
        await compressor.compress("test1")
        await compressor.compress("test2")
        await compressor.decompress(b"data", CompressionAlgorithm.NONE)
        
        stats = compressor.get_stats()
        
        assert stats["total_compressions"] == 2
        assert stats["total_decompressions"] == 1
        assert stats["default_algorithm"] == "gzip"
        assert stats["min_size_for_compression"] == 100
        assert stats["compression_level"] == 6
        assert "compressions_by_algorithm" in stats
        assert "decompressions_by_algorithm" in stats

    @pytest.mark.asyncio
    async def test_reset_stats(self, compressor):
        """Test resetting compression statistics."""
        # Perform some operations
        await compressor.compress("test1")
        await compressor.decompress(b"data", CompressionAlgorithm.NONE)
        
        # Reset stats
        compressor.reset_stats()
        
        # Check that stats are reset
        stats = compressor.get_stats()
        assert stats["total_compressions"] == 0
        assert stats["total_decompressions"] == 0
        assert all(count == 0 for count in stats["compressions_by_algorithm"].values())
        assert all(count == 0 for count in stats["decompressions_by_algorithm"].values())


class TestCompressedCacheBackend:
    """Test cases for CompressedCacheBackend class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock cache backend."""
        backend = AsyncMock()
        backend.get.return_value = None
        backend.set.return_value = True
        backend.delete.return_value = True
        backend.exists.return_value = False
        backend.clear.return_value = True
        backend.keys.return_value = []
        backend.close.return_value = None
        return backend

    @pytest.fixture
    def compressed_backend(self, mock_backend):
        """Create a compressed cache backend."""
        return CompressedCacheBackend(
            backend=mock_backend,
            compress_threshold=100,
        )

    @pytest.mark.asyncio
    async def test_compressed_backend_initialization(self, mock_backend):
        """Test initializing a compressed cache backend."""
        compressor = ResultCompressor()
        backend = CompressedCacheBackend(
            backend=mock_backend,
            compressor=compressor,
            compress_threshold=50,
        )
        
        assert backend.backend == mock_backend
        assert backend.compressor == compressor
        assert backend.compress_threshold == 50

    @pytest.mark.asyncio
    async def test_get_uncompressed_value(self, compressed_backend, mock_backend):
        """Test getting an uncompressed value."""
        # Mock backend returning uncompressed value
        mock_backend.get.return_value = "uncompressed_value"
        
        result = await compressed_backend.get("test_key")
        
        assert result == "uncompressed_value"
        mock_backend.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_compressed_value(self, compressed_backend, mock_backend):
        """Test getting a compressed value."""
        # Create compressed data
        compressor = ResultCompressor()
        original_data = {"key": "value", "nested": {"data": [1, 2, 3]}}
        compression_result = await compressor.compress(original_data)
        
        # Mock backend returning compressed value and metadata
        mock_backend.get.side_effect = [
            compression_result.compressed_data,  # Main value
            {  # Metadata
                "algorithm": compression_result.algorithm.value,
                "original_size": compression_result.original_size,
                "compressed_size": compression_result.compressed_size,
                "compression_ratio": compression_result.compression_ratio,
            }
        ]
        
        result = await compressed_backend.get("test_key")
        
        assert result == original_data
        assert mock_backend.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_compressed_value_no_metadata(self, compressed_backend, mock_backend):
        """Test getting a compressed value without metadata."""
        # Create compressed data
        compressor = ResultCompressor()
        original_data = {"key": "value"}
        compression_result = await compressor.compress(original_data)
        
        # Mock backend returning compressed value but no metadata
        mock_backend.get.return_value = compression_result.compressed_data
        
        result = await compressed_backend.get("test_key")
        
        # Should return compressed data as fallback
        assert result == compression_result.compressed_data

    @pytest.mark.asyncio
    async def test_get_compressed_value_decompression_error(self, compressed_backend, mock_backend):
        """Test getting a compressed value with decompression error."""
        # Mock backend returning compressed data and metadata
        mock_backend.get.side_effect = [
            b"invalid_compressed_data",  # Main value
            {"algorithm": "gzip"}  # Metadata
        ]
        
        result = await compressed_backend.get("test_key")
        
        # Should return compressed data as fallback
        assert result == b"invalid_compressed_data"

    @pytest.mark.asyncio
    async def test_get_nonexistent_value(self, compressed_backend, mock_backend):
        """Test getting a non-existent value."""
        # Mock backend returning None
        mock_backend.get.return_value = None
        
        result = await compressed_backend.get("test_key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_small_value(self, compressed_backend, mock_backend):
        """Test setting a small value (below compression threshold)."""
        # Set a small value
        result = await compressed_backend.set("test_key", "small_value")
        
        assert result is True
        mock_backend.set.assert_called_once_with("test_key", "small_value", None)

    @pytest.mark.asyncio
    async def test_set_large_value(self, compressed_backend, mock_backend):
        """Test setting a large value (above compression threshold)."""
        # Set a large value
        large_value = "x" * 200  # Larger than threshold
        result = await compressed_backend.set("test_key", large_value)
        
        assert result is True
        
        # Should have called set twice (once for data, once for metadata)
        assert mock_backend.set.call_count == 2

    @pytest.mark.asyncio
    async def test_set_large_value_with_ttl(self, compressed_backend, mock_backend):
        """Test setting a large value with TTL."""
        # Set a large value with TTL
        large_value = "x" * 200  # Larger than threshold
        result = await compressed_backend.set("test_key", large_value, ttl=60)
        
        assert result is True
        
        # Check that both calls used the same TTL
        calls = mock_backend.set.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["ttl"] == 60  # First call (data)
        assert calls[1][1]["ttl"] == 60  # Second call (metadata)

    @pytest.mark.asyncio
    async def test_set_metadata_failure(self, compressed_backend, mock_backend):
        """Test setting a large value when metadata storage fails."""
        # Mock backend to fail on metadata storage
        mock_backend.set.side_effect = [True, False]  # First succeeds, second fails
        
        large_value = "x" * 200  # Larger than threshold
        result = await compressed_backend.set("test_key", large_value)
        
        assert result is False
        
        # Should have tried to clean up the main value
        assert mock_backend.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_delete(self, compressed_backend, mock_backend):
        """Test deleting a key."""
        result = await compressed_backend.delete("test_key")
        
        assert result is True
        
        # Should delete both the main value and metadata
        assert mock_backend.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_exists(self, compressed_backend, mock_backend):
        """Test checking if a key exists."""
        mock_backend.exists.return_value = True
        
        result = await compressed_backend.exists("test_key")
        
        assert result is True
        mock_backend.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear(self, compressed_backend, mock_backend):
        """Test clearing all keys."""
        result = await compressed_backend.clear()
        
        assert result is True
        mock_backend.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_keys(self, compressed_backend, mock_backend):
        """Test getting keys."""
        # Mock backend returning keys including metadata keys
        mock_backend.keys.return_value = ["key1", "key2", "key1:compression", "key3:compression"]
        
        result = await compressed_backend.keys("*")
        
        # Should filter out metadata keys
        assert result == ["key1", "key2"]
        mock_backend.keys.assert_called_once_with("*")

    @pytest.mark.asyncio
    async def test_close(self, compressed_backend, mock_backend):
        """Test closing the backend."""
        await compressed_backend.close()
        
        mock_backend.close.assert_called_once()

    def test_get_compression_stats(self, compressed_backend):
        """Test getting compression statistics."""
        stats = compressed_backend.get_compression_stats()
        
        # Should return the compressor's stats
        assert "total_compressions" in stats
        assert "total_decompressions" in stats


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @pytest.mark.asyncio
    async def test_get_default_compressor(self):
        """Test getting the default compressor."""
        compressor1 = get_default_compressor()
        compressor2 = get_default_compressor()
        
        # Should return the same instance
        assert compressor1 is compressor2
        assert isinstance(compressor1, ResultCompressor)

    @pytest.mark.asyncio
    async def test_compress_data(self):
        """Test compress_data utility function."""
        data = {"key": "value"}
        result = await compress_data(data, CompressionAlgorithm.GZIP)
        
        assert isinstance(result, CompressionResult)
        assert result.algorithm == CompressionAlgorithm.GZIP

    @pytest.mark.asyncio
    async def test_decompress_data(self):
        """Test decompress_data utility function."""
        compressor = ResultCompressor()
        data = {"key": "value"}
        compressed = await compressor.compress(data, CompressionAlgorithm.GZIP)
        
        result = await decompress_data(compressed.compressed_data, CompressionAlgorithm.GZIP)
        
        assert result == data

    @pytest.mark.asyncio
    async def test_compress_string(self):
        """Test compress_string utility function."""
        text = "This is a test string"
        result = await compress_string(text, CompressionAlgorithm.GZIP)
        
        assert isinstance(result, bytes)
        
        # Verify it's valid GZIP data
        decompressed = gzip.decompress(result).decode('utf-8')
        assert decompressed == text

    @pytest.mark.asyncio
    async def test_decompress_string(self):
        """Test decompress_string utility function."""
        text = "This is a test string"
        compressed = await compress_string(text, CompressionAlgorithm.GZIP)
        
        result = await decompress_string(compressed, CompressionAlgorithm.GZIP)
        
        assert result == text

    @pytest.mark.asyncio
    async def test_example_usage(self):
        """Test the example usage function."""
        # This is just a smoke test to ensure it doesn't crash
        from app.core.caching.compression.compressor import example_usage
        
        # Run the example
        await example_usage()