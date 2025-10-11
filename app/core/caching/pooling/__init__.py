"""
Connection pooling system for external API calls.

This module provides connection pooling for HTTP clients and other external
services to optimize performance and reduce connection overhead.
"""

from .connection_pool import ConnectionPool, HTTPConnectionPool

__all__ = [
    "ConnectionPool",
    "HTTPConnectionPool",
]
