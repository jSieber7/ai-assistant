"""
Request batching system for optimizing external API calls.

This module provides request batching functionality to group multiple
similar requests into single batched requests, reducing the number of
external API calls and improving performance.
"""

from .batch_processor import BatchProcessor, BatchRequest, BatchResult

__all__ = [
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
]
