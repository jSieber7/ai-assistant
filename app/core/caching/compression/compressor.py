"""
Result compression utilities for efficient storage and transmission.

This module provides compression algorithms and utilities for compressing
cache results to reduce storage space and network bandwidth usage.
"""

import gzip
import lzma
import zlib
import pickle
import base64
from typing import Any, Dict, List, Optional
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""

    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"
    BASE64 = "base64"  # Simple encoding, not compression


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    compressed_data: bytes
    original_size: int
    compressed_size: int
    algorithm: CompressionAlgorithm
    compression_ratio: float

    @property
    def space_saved(self) -> int:
        """Calculate the space saved by compression."""
        return self.original_size - self.compressed_size

    @property
    def compression_percentage(self) -> float:
        """Calculate the compression percentage."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100


class ResultCompressor:
    """
    Compressor for cache results with multiple algorithm support.

    Provides compression and decompression with algorithm selection
    based on data characteristics and performance requirements.
    """

    def __init__(
        self,
        default_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        min_size_for_compression: int = 100,
        compression_level: int = 6,
    ):
        """
        Initialize the result compressor.

        Args:
            default_algorithm: Default compression algorithm to use
            min_size_for_compression: Minimum data size to apply compression
            compression_level: Compression level (1-9 for gzip/zlib, 0-9 for lzma)
        """
        self.default_algorithm = default_algorithm
        self.min_size_for_compression = min_size_for_compression
        self.compression_level = compression_level
        self._stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "total_bytes_saved": 0,
            "compressions_by_algorithm": {alg.value: 0 for alg in CompressionAlgorithm},
            "decompressions_by_algorithm": {
                alg.value: 0 for alg in CompressionAlgorithm
            },
        }

    async def compress(
        self, data: Any, algorithm: Optional[CompressionAlgorithm] = None
    ) -> CompressionResult:
        """
        Compress data using the specified algorithm.

        Args:
            data: The data to compress
            algorithm: Compression algorithm to use (defaults to default_algorithm)

        Returns:
            Compression result with compressed data and metadata
        """
        algorithm = algorithm or self.default_algorithm

        # Serialize data to bytes
        original_data = self._serialize_data(data)
        original_size = len(original_data)

        # Skip compression for small data
        if original_size < self.min_size_for_compression:
            algorithm = CompressionAlgorithm.NONE

        # Compress based on algorithm
        if algorithm == CompressionAlgorithm.NONE:
            compressed_data = original_data
        elif algorithm == CompressionAlgorithm.GZIP:
            compressed_data = gzip.compress(
                original_data, compresslevel=self.compression_level
            )
        elif algorithm == CompressionAlgorithm.LZMA:
            compressed_data = lzma.compress(
                original_data, preset=self.compression_level
            )
        elif algorithm == CompressionAlgorithm.ZLIB:
            compressed_data = zlib.compress(original_data, level=self.compression_level)
        elif algorithm == CompressionAlgorithm.BASE64:
            compressed_data = base64.b64encode(original_data)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

        compressed_size = len(compressed_data)
        compression_ratio = (
            compressed_size / original_size if original_size > 0 else 1.0
        )

        result = CompressionResult(
            compressed_data=compressed_data,
            original_size=original_size,
            compressed_size=compressed_size,
            algorithm=algorithm,
            compression_ratio=compression_ratio,
        )

        # Update statistics
        self._stats["total_compressions"] += 1
        self._stats["compressions_by_algorithm"][algorithm.value] += 1
        self._stats["total_bytes_saved"] += result.space_saved

        logger.debug(
            f"Compressed {original_size} bytes to {compressed_size} bytes "
            f"using {algorithm.value} (ratio: {compression_ratio:.2f})"
        )

        return result

    async def decompress(
        self, compressed_data: bytes, algorithm: CompressionAlgorithm
    ) -> Any:
        """
        Decompress data using the specified algorithm.

        Args:
            compressed_data: The compressed data
            algorithm: Compression algorithm used

        Returns:
            The original decompressed data
        """
        # Decompress based on algorithm
        if algorithm == CompressionAlgorithm.NONE:
            decompressed_data = compressed_data
        elif algorithm == CompressionAlgorithm.GZIP:
            decompressed_data = gzip.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.LZMA:
            decompressed_data = lzma.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            decompressed_data = zlib.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.BASE64:
            decompressed_data = base64.b64decode(compressed_data)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

        # Deserialize the data
        result = self._deserialize_data(decompressed_data)

        # Update statistics
        self._stats["total_decompressions"] += 1
        self._stats["decompressions_by_algorithm"][algorithm.value] += 1

        return result

    def _serialize_data(self, data: Any) -> bytes:
        """
        Serialize data to bytes for compression.

        Args:
            data: The data to serialize

        Returns:
            Serialized bytes
        """
        # Use pickle for general serialization
        # In production, you might want to use a more efficient serialization format
        return pickle.dumps(data)

    def _deserialize_data(self, data: bytes) -> Any:
        """
        Deserialize bytes back to original data.

        Args:
            data: Serialized bytes

        Returns:
            The original data
        """
        return pickle.loads(data)

    async def auto_compress(self, data: Any) -> CompressionResult:
        """
        Automatically select the best compression algorithm for the data.

        Args:
            data: The data to compress

        Returns:
            Compression result with the best algorithm
        """
        # For now, use the default algorithm
        # In a more sophisticated implementation, you might test multiple algorithms
        # and select the one with the best compression ratio
        return await self.compress(data, self.default_algorithm)

    async def get_best_algorithm(self, data: Any) -> CompressionAlgorithm:
        """
        Determine the best compression algorithm for the given data.

        Args:
            data: The data to analyze

        Returns:
            Recommended compression algorithm
        """
        # Simple heuristic-based algorithm selection
        serialized_data = self._serialize_data(data)
        data_size = len(serialized_data)

        if data_size < 100:
            return CompressionAlgorithm.NONE
        elif data_size < 1000:
            return CompressionAlgorithm.ZLIB  # Fast for small data
        elif data_size < 10000:
            return CompressionAlgorithm.GZIP  # Good balance for medium data
        else:
            return CompressionAlgorithm.LZMA  # Best compression for large data

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        total_compressions = self._stats["total_compressions"]
        total_decompressions = self._stats["total_decompressions"]

        return {
            "total_compressions": total_compressions,
            "total_decompressions": total_decompressions,
            "total_bytes_saved": self._stats["total_bytes_saved"],
            "average_bytes_saved_per_compression": (
                self._stats["total_bytes_saved"] / total_compressions
                if total_compressions > 0
                else 0
            ),
            "compressions_by_algorithm": self._stats["compressions_by_algorithm"],
            "decompressions_by_algorithm": self._stats["decompressions_by_algorithm"],
            "default_algorithm": self.default_algorithm.value,
            "min_size_for_compression": self.min_size_for_compression,
            "compression_level": self.compression_level,
        }

    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self._stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "total_bytes_saved": 0,
            "compressions_by_algorithm": {alg.value: 0 for alg in CompressionAlgorithm},
            "decompressions_by_algorithm": {
                alg.value: 0 for alg in CompressionAlgorithm
            },
        }


class CompressedCacheBackend:
    """
    Cache backend wrapper that automatically compresses stored values.

    Wraps an existing cache backend and adds compression/decompression
    transparently.
    """

    def __init__(
        self,
        backend: Any,  # Should be a CacheBackend
        compressor: Optional[ResultCompressor] = None,
        compress_threshold: int = 100,
    ):
        """
        Initialize the compressed cache backend.

        Args:
            backend: The underlying cache backend
            compressor: Result compressor instance
            compress_threshold: Minimum size to compress (bytes)
        """
        self.backend = backend
        self.compressor = compressor or ResultCompressor()
        self.compress_threshold = compress_threshold

        # Metadata key suffix for storing compression info
        self.metadata_suffix = ":compression"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache and decompress it."""
        # Get the compressed value
        compressed_value = await self.backend.get(key)
        if compressed_value is None:
            return None

        # Get compression metadata
        metadata_key = key + self.metadata_suffix
        metadata = await self.backend.get(metadata_key)

        if metadata is None:
            # No compression metadata, return as-is
            return compressed_value

        try:
            # Decompress the value
            algorithm = CompressionAlgorithm(metadata.get("algorithm", "none"))
            decompressed_value = await self.compressor.decompress(
                compressed_value, algorithm
            )
            return decompressed_value
        except Exception as e:
            logger.error(f"Failed to decompress value for key '{key}': {e}")
            # Return the compressed value as fallback
            return compressed_value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with compression."""
        # Check if compression should be applied
        serialized_size = len(self.compressor._serialize_data(value))

        if serialized_size >= self.compress_threshold:
            # Compress the value
            compression_result = await self.compressor.auto_compress(value)

            # Store compressed value
            success = await self.backend.set(
                key, compression_result.compressed_data, ttl
            )
            if not success:
                return False

            # Store compression metadata
            metadata = {
                "algorithm": compression_result.algorithm.value,
                "original_size": compression_result.original_size,
                "compressed_size": compression_result.compressed_size,
                "compression_ratio": compression_result.compression_ratio,
            }
            metadata_key = key + self.metadata_suffix

            # Set metadata with same TTL
            metadata_success = await self.backend.set(metadata_key, metadata, ttl)
            if not metadata_success:
                # Clean up the main value if metadata storage fails
                await self.backend.delete(key)
                return False

            return True
        else:
            # Store without compression
            return await self.backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete a key and its compression metadata."""
        # Delete the main value
        success = await self.backend.delete(key)

        # Delete metadata
        metadata_key = key + self.metadata_suffix
        await self.backend.delete(metadata_key)

        return success

    async def exists(self, key: str) -> bool:
        """Check if a key exists (ignores compression metadata)."""
        return await self.backend.exists(key)

    async def clear(self) -> bool:
        """Clear all keys and metadata."""
        return await self.backend.clear()

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern (excluding metadata keys)."""
        all_keys = await self.backend.keys(pattern)
        # Filter out metadata keys
        return [key for key in all_keys if not key.endswith(self.metadata_suffix)]

    async def close(self) -> None:
        """Close the underlying backend."""
        await self.backend.close()

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics from the compressor."""
        return self.compressor.get_stats()


# Global compressor instance
_default_compressor: Optional[ResultCompressor] = None


def get_default_compressor() -> ResultCompressor:
    """Get the default global compressor instance."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = ResultCompressor()
    return _default_compressor


# Utility functions for common compression tasks


async def compress_data(
    data: Any, algorithm: Optional[CompressionAlgorithm] = None
) -> CompressionResult:
    """Compress data using the default compressor."""
    compressor = get_default_compressor()
    return await compressor.compress(data, algorithm)


async def decompress_data(
    compressed_data: bytes, algorithm: CompressionAlgorithm
) -> Any:
    """Decompress data using the default compressor."""
    compressor = get_default_compressor()
    return await compressor.decompress(compressed_data, algorithm)


async def compress_string(
    text: str, algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
) -> bytes:
    """Compress a string efficiently."""
    compressor = get_default_compressor()
    result = await compressor.compress(text, algorithm)
    return result.compressed_data


async def decompress_string(
    compressed_data: bytes, algorithm: CompressionAlgorithm
) -> str:
    """Decompress data back to a string."""
    compressor = get_default_compressor()
    result = await compressor.decompress(compressed_data, algorithm)
    return result


# Example usage and testing


async def example_usage():
    """Example demonstrating compression usage."""
    compressor = ResultCompressor()

    # Example data
    sample_data = {
        "message": "Hello, world!",
        "numbers": list(range(1000)),
        "nested": {"key1": "value1", "key2": ["item1", "item2", "item3"]},
    }

    # Compress the data
    result = await compressor.compress(sample_data)
    print(f"Original size: {result.original_size} bytes")
    print(f"Compressed size: {result.compressed_size} bytes")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Space saved: {result.space_saved} bytes")
    print(f"Algorithm: {result.algorithm.value}")

    # Decompress the data
    decompressed = await compressor.decompress(result.compressed_data, result.algorithm)
    print(f"Decompression successful: {decompressed == sample_data}")

    # Get statistics
    stats = compressor.get_stats()
    print(f"Total compressions: {stats['total_compressions']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
