"""
Result compression utilities for reducing cache size.

This module provides compression utilities to reduce the size of cached data,
optimizing memory usage and network transfer for distributed caching.
"""

from .compressor import ResultCompressor, CompressionAlgorithm

__all__ = [
    "ResultCompressor",
    "CompressionAlgorithm",
]
