"""
Connection pooling system for efficient resource management.

This module provides connection pooling for various types of connections
(HTTP, database, etc.) to optimize resource usage and performance.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum
import logging
import aiohttp
import httpx

logger = logging.getLogger(__name__)

# Type variable for connection types
T = TypeVar("T")


class ConnectionState(Enum):
    """State of a connection in the pool."""

    IDLE = "idle"
    IN_USE = "in_use"
    BROKEN = "broken"
    CLOSED = "closed"


@dataclass
class ConnectionStats:
    """Statistics for a connection."""

    created_at: float
    last_used: float
    use_count: int
    total_use_time: float
    state: ConnectionState


class Connection(Generic[T]):
    """
    Represents a single connection with metadata and lifecycle management.
    """

    def __init__(
        self, connection_id: str, connection_obj: T, created_at: Optional[float] = None
    ):
        """
        Initialize a connection.

        Args:
            connection_id: Unique identifier for the connection
            connection_obj: The actual connection object
            created_at: Creation timestamp (defaults to current time)
        """
        self.connection_id = connection_id
        self.connection_obj = connection_obj
        self.created_at = created_at or time.time()
        self.last_used = self.created_at
        self.use_count = 0
        self.total_use_time = 0.0
        self.state = ConnectionState.IDLE
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire the connection for use."""
        async with self._lock:
            if self.state != ConnectionState.IDLE:
                return False

            self.state = ConnectionState.IN_USE
            self.last_used = time.time()
            return True

    async def release(self, use_time: float) -> None:
        """Release the connection after use."""
        async with self._lock:
            if self.state == ConnectionState.IN_USE:
                self.state = ConnectionState.IDLE
                self.use_count += 1
                self.total_use_time += use_time

    async def mark_broken(self) -> None:
        """Mark the connection as broken."""
        async with self._lock:
            self.state = ConnectionState.BROKEN

    async def close(self) -> None:
        """Close the connection."""
        async with self._lock:
            self.state = ConnectionState.CLOSED

    def get_stats(self) -> ConnectionStats:
        """Get connection statistics."""
        return ConnectionStats(
            created_at=self.created_at,
            last_used=self.last_used,
            use_count=self.use_count,
            total_use_time=self.total_use_time,
            state=self.state,
        )


class ConnectionPool(Generic[T]):
    """
    Generic connection pool for managing connections of type T.

    Provides connection reuse, lifecycle management, and statistics.
    """

    def __init__(
        self,
        name: str,
        max_size: int = 10,
        min_size: int = 2,
        max_lifetime: int = 300,  # 5 minutes
        max_idle_time: int = 60,  # 1 minute
        connection_timeout: float = 5.0,
        acquire_timeout: float = 10.0,
    ):
        """
        Initialize the connection pool.

        Args:
            name: Pool name for identification
            max_size: Maximum number of connections
            min_size: Minimum number of connections to maintain
            max_lifetime: Maximum lifetime of a connection in seconds
            max_idle_time: Maximum idle time before closing a connection
            connection_timeout: Timeout for creating new connections
            acquire_timeout: Timeout for acquiring a connection
        """
        self.name = name
        self.max_size = max_size
        self.min_size = min_size
        self.max_lifetime = max_lifetime
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.acquire_timeout = acquire_timeout

        self.connections: Dict[str, Connection[T]] = {}
        self._available: List[Connection[T]] = []
        self._in_use: List[Connection[T]] = []
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "total_created": 0,
            "total_closed": 0,
            "total_acquired": 0,
            "total_released": 0,
            "total_errors": 0,
            "wait_time": 0.0,
        }

    async def start(self) -> None:
        """Start the connection pool and begin maintenance tasks."""
        # Create initial connections
        for _ in range(self.min_size):
            await self._create_connection()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._maintenance_loop())

    async def stop(self) -> None:
        """Stop the connection pool and close all connections."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            # Close all connections
            for connection in list(self.connections.values()):
                await connection.close()

            self.connections.clear()
            self._available.clear()
            self._in_use.clear()

    async def acquire(self) -> Optional[Connection[T]]:
        """
        Acquire a connection from the pool.

        Returns:
            A connection or None if timeout or error
        """
        start_time = time.time()

        try:
            async with asyncio.timeout(self.acquire_timeout):
                return await self._acquire_connection()
        except asyncio.TimeoutError:
            self._stats["total_errors"] += 1
            logger.warning(f"Timeout acquiring connection from pool '{self.name}'")
            return None
        except Exception as e:
            self._stats["total_errors"] += 1
            logger.error(f"Error acquiring connection from pool '{self.name}': {e}")
            return None
        finally:
            self._stats["wait_time"] += time.time() - start_time

    async def _acquire_connection(self) -> Optional[Connection[T]]:
        """Internal method to acquire a connection."""
        async with self._lock:
            # Try to get an available connection
            while self._available:
                connection = self._available.pop(0)

                # Check if connection is still valid
                if await self._is_connection_valid(connection):
                    if await connection.acquire():
                        self._in_use.append(connection)
                        self._stats["total_acquired"] += 1
                        return connection
                else:
                    # Remove invalid connection
                    await self._remove_connection(connection)

            # Create new connection if under max size
            if len(self.connections) < self.max_size:
                connection = await self._create_connection()
                if connection and await connection.acquire():
                    self._in_use.append(connection)
                    self._stats["total_acquired"] += 1
                    return connection

            # Wait for a connection to become available
            # This is a simple implementation - in production you might use a Condition
            await asyncio.sleep(0.1)
            return await self._acquire_connection()

    async def release(self, connection: Connection[T]) -> None:
        """
        Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        async with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)

                # Calculate use time
                use_time = time.time() - connection.last_used
                await connection.release(use_time)

                # Add back to available if still valid
                if await self._is_connection_valid(connection):
                    self._available.append(connection)
                else:
                    await self._remove_connection(connection)

                self._stats["total_released"] += 1

    async def _create_connection(self) -> Optional[Connection[T]]:
        """
        Create a new connection.

        This method should be overridden by subclasses to create
        specific types of connections.

        Returns:
            New connection or None if creation failed
        """
        # Base implementation - subclasses should override this
        connection_id = f"conn_{len(self.connections) + 1}"
        connection_obj = None  # Subclasses should provide actual connection

        connection = Connection(connection_id, connection_obj)
        self.connections[connection_id] = connection
        self._available.append(connection)
        self._stats["total_created"] += 1

        return connection

    async def _is_connection_valid(self, connection: Connection[T]) -> bool:
        """
        Check if a connection is still valid.

        Args:
            connection: The connection to check

        Returns:
            True if connection is valid
        """
        if connection.state != ConnectionState.IDLE:
            return False

        # Check lifetime
        if time.time() - connection.created_at > self.max_lifetime:
            return False

        # Check idle time
        if time.time() - connection.last_used > self.max_idle_time:
            return False

        return True

    async def _remove_connection(self, connection: Connection[T]) -> None:
        """Remove a connection from the pool."""
        if connection.connection_id in self.connections:
            del self.connections[connection.connection_id]

        if connection in self._available:
            self._available.remove(connection)

        if connection in self._in_use:
            self._in_use.remove(connection)

        await connection.close()
        self._stats["total_closed"] += 1

    async def _maintenance_loop(self) -> None:
        """Background task for pool maintenance."""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                await self._cleanup_connections()
                await self._maintain_pool_size()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection pool maintenance: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _cleanup_connections(self) -> None:
        """Clean up invalid connections."""
        async with self._lock:
            connections_to_remove = []

            for connection in list(self.connections.values()):
                if not await self._is_connection_valid(connection):
                    connections_to_remove.append(connection)

            for connection in connections_to_remove:
                await self._remove_connection(connection)

            if connections_to_remove:
                logger.debug(
                    f"Cleaned up {len(connections_to_remove)} connections from pool '{self.name}'"
                )

    async def _maintain_pool_size(self) -> None:
        """Maintain minimum pool size by creating connections if needed."""
        async with self._lock:
            current_size = len(self.connections)
            if current_size < self.min_size:
                needed = self.min_size - current_size
                for _ in range(needed):
                    await self._create_connection()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        current_size = len(self.connections)
        available_size = len(self._available)
        in_use_size = len(self._in_use)

        avg_wait_time = (
            self._stats["wait_time"] / self._stats["total_acquired"]
            if self._stats["total_acquired"] > 0
            else 0.0
        )

        return {
            "name": self.name,
            "current_size": current_size,
            "available_size": available_size,
            "in_use_size": in_use_size,
            "utilization": in_use_size / current_size if current_size > 0 else 0.0,
            "total_created": self._stats["total_created"],
            "total_closed": self._stats["total_closed"],
            "total_acquired": self._stats["total_acquired"],
            "total_released": self._stats["total_released"],
            "total_errors": self._stats["total_errors"],
            "average_wait_time": avg_wait_time,
            "max_size": self.max_size,
            "min_size": self.min_size,
        }


class HTTPConnectionPool(ConnectionPool[aiohttp.ClientSession]):
    """Connection pool for HTTP connections using aiohttp."""

    def __init__(
        self,
        name: str = "http_pool",
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize HTTP connection pool.

        Args:
            name: Pool name
            base_url: Base URL for connections
            headers: Default headers for connections
            **kwargs: Additional ConnectionPool arguments
        """
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.headers = headers or {}

    async def _create_connection(self) -> Optional[Connection[aiohttp.ClientSession]]:
        """Create a new aiohttp ClientSession."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            connector = aiohttp.TCPConnector(limit=1)  # One connection per session

            session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=self.headers,
                timeout=timeout,
                connector=connector,
            )

            connection_id = f"http_conn_{len(self.connections) + 1}"
            connection = Connection(connection_id, session)
            self.connections[connection_id] = connection
            self._available.append(connection)
            self._stats["total_created"] += 1

            return connection
        except Exception as e:
            logger.error(f"Failed to create HTTP connection: {e}")
            return None


class AIOHTTPConnectionPool(ConnectionPool[httpx.AsyncClient]):
    """Connection pool for HTTP connections using httpx."""

    def __init__(
        self,
        name: str = "httpx_pool",
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize httpx connection pool.

        Args:
            name: Pool name
            base_url: Base URL for connections
            headers: Default headers for connections
            **kwargs: Additional ConnectionPool arguments
        """
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.headers = headers or {}

    async def _create_connection(self) -> Optional[Connection[httpx.AsyncClient]]:
        """Create a new httpx AsyncClient."""
        try:
            timeout = httpx.Timeout(self.connection_timeout)
            limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)

            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=timeout,
                limits=limits,
            )

            connection_id = f"httpx_conn_{len(self.connections) + 1}"
            connection = Connection(connection_id, client)
            self.connections[connection_id] = connection
            self._available.append(connection)
            self._stats["total_created"] += 1

            return connection
        except Exception as e:
            logger.error(f"Failed to create httpx connection: {e}")
            return None


class ConnectionPoolManager:
    """
    Manager for multiple connection pools.

    Provides a centralized way to manage different types of connection pools.
    """

    def __init__(self):
        """Initialize the connection pool manager."""
        self.pools: Dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()

    async def register_pool(self, name: str, pool: ConnectionPool) -> None:
        """Register a connection pool."""
        async with self._lock:
            self.pools[name] = pool
            await pool.start()

    async def unregister_pool(self, name: str) -> None:
        """Unregister a connection pool."""
        async with self._lock:
            if name in self.pools:
                await self.pools[name].stop()
                del self.pools[name]

    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        async with self._lock:
            return self.pools.get(name)

    async def acquire_connection(self, pool_name: str) -> Optional[Connection]:
        """Acquire a connection from a pool."""
        pool = await self.get_pool(pool_name)
        if pool:
            return await pool.acquire()
        return None

    async def release_connection(self, pool_name: str, connection: Connection) -> None:
        """Release a connection back to a pool."""
        pool = await self.get_pool(pool_name)
        if pool:
            await pool.release(connection)

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        async with self._lock:
            stats = {
                "total_pools": len(self.pools),
                "pools": {},
            }

            for name, pool in self.pools.items():
                stats["pools"][name] = pool.get_stats()

            return stats

    async def close_all(self) -> None:
        """Close all connection pools."""
        async with self._lock:
            for name, pool in self.pools.items():
                await pool.stop()
            self.pools.clear()


# Global connection pool manager instance
_connection_pool_manager: Optional[ConnectionPoolManager] = None


async def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager instance."""
    global _connection_pool_manager
    if _connection_pool_manager is None:
        _connection_pool_manager = ConnectionPoolManager()
    return _connection_pool_manager


async def close_connection_pools() -> None:
    """Close all connection pools."""
    global _connection_pool_manager
    if _connection_pool_manager is not None:
        await _connection_pool_manager.close_all()
        _connection_pool_manager = None
