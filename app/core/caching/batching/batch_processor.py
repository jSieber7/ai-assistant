"""
Request batching system for efficient API calls.

This module provides a batching mechanism to aggregate multiple requests
and process them in batches to reduce external API calls and improve performance.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)

# Type variables for request and result types
T = TypeVar("T")  # Request type
R = TypeVar("R")  # Result type


class BatchState(Enum):
    """State of a batch."""

    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRequest(Generic[T]):
    """A single request in a batch."""

    request_id: str
    data: T
    created_at: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult(Generic[R]):
    """Result for a batch request."""

    request_id: str
    result: Optional[R] = None
    error: Optional[str] = None
    processed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch(Generic[T, R]):
    """A batch of requests to be processed together."""

    batch_id: str
    requests: List[BatchRequest[T]]
    created_at: float
    state: BatchState = BatchState.COLLECTING
    results: List[BatchResult[R]] = field(default_factory=list)
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None


class BatchProcessor(Generic[T, R]):
    """
    Processor for batching requests and processing them together.

    Collects requests over a time window or until a size limit is reached,
    then processes them as a batch.
    """

    def __init__(
        self,
        name: str,
        process_batch_fn: Callable[[List[T]], List[R]],
        max_batch_size: int = 10,
        max_wait_time: float = 0.1,  # 100ms
        max_queue_size: int = 1000,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
    ):
        """
        Initialize the batch processor.

        Args:
            name: Processor name for identification
            process_batch_fn: Function to process a batch of requests
            max_batch_size: Maximum number of requests per batch
            max_wait_time: Maximum time to wait before processing a batch (seconds)
            max_queue_size: Maximum number of requests in queue
            retry_attempts: Number of retry attempts for failed batches
            retry_delay: Delay between retries (seconds)
        """
        self.name = name
        self.process_batch_fn = process_batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Request queue and state
        self._queue: List[BatchRequest[T]] = []
        self._current_batch: Optional[Batch[T, R]] = None
        self._pending_results: Dict[str, asyncio.Future[BatchResult[R]]] = {}

        # Processing state
        self._lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Statistics
        self._stats = {
            "total_requests": 0,
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "queue_overflows": 0,
        }

    async def start(self) -> None:
        """Start the batch processor."""
        self._shutdown = False
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Started batch processor '{self.name}'")

    async def stop(self) -> None:
        """Stop the batch processor and process remaining requests."""
        self._shutdown = True

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Process any remaining requests
        if self._queue:
            await self._process_current_batch()

        logger.info(f"Stopped batch processor '{self.name}'")

    async def submit_request(
        self, data: T, priority: int = 0, metadata: Optional[Dict[str, Any]] = None
    ) -> BatchResult[R]:
        """
        Submit a request for batch processing.

        Args:
            data: The request data
            priority: Request priority (higher = processed first)
            metadata: Additional request metadata

        Returns:
            Batch result with the processed result
        """
        request_id = str(uuid.uuid4())
        request = BatchRequest(
            request_id=request_id,
            data=data,
            created_at=time.time(),
            priority=priority,
            metadata=metadata or {},
        )

        # Create future for the result
        future: asyncio.Future[BatchResult[R]] = asyncio.Future()

        async with self._lock:
            # Check queue size limit
            if len(self._queue) >= self.max_queue_size:
                self._stats["queue_overflows"] += 1
                error_result: BatchResult[R] = BatchResult(
                    request_id=request_id,
                    error="Queue overflow - request rejected",
                    processed_at=time.time(),
                )
                future.set_result(error_result)
                return error_result

            # Add to queue
            self._queue.append(request)
            self._pending_results[request_id] = future
            self._stats["total_requests"] += 1

            # Sort queue by priority (higher priority first)
            self._queue.sort(key=lambda r: r.priority, reverse=True)

        return await future

    async def _processing_loop(self) -> None:
        """Main processing loop for batches."""
        while not self._shutdown:
            try:
                # Check if we should process the current batch
                should_process = await self._should_process_batch()

                if should_process:
                    await self._process_current_batch()
                else:
                    await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    async def _should_process_batch(self) -> bool:
        """Determine if the current batch should be processed."""
        async with self._lock:
            if not self._queue:
                return False

            # Check if we've reached max batch size
            if len(self._queue) >= self.max_batch_size:
                return True

            # Check if the oldest request has been waiting too long
            if self._queue:
                oldest_request = min(self._queue, key=lambda r: r.created_at)
                wait_time = time.time() - oldest_request.created_at
                if wait_time >= self.max_wait_time:
                    return True

            return False

    async def _process_current_batch(self) -> None:
        """Process the current batch of requests."""
        async with self._lock:
            if not self._queue:
                return

            # Create batch from current queue
            batch_requests = self._queue[: self.max_batch_size]
            self._queue = self._queue[self.max_batch_size :]

            batch: Batch[T, R] = Batch(
                batch_id=str(uuid.uuid4()),
                requests=batch_requests,
                created_at=time.time(),
            )

            self._current_batch = batch
            self._stats["total_batches"] += 1

        # Process the batch
        await self._process_batch(batch)

    async def _process_batch(self, batch: Batch[T, R]) -> None:
        """Process a single batch."""
        batch.state = BatchState.PROCESSING
        batch.processing_started = time.time()

        logger.debug(
            f"Processing batch {batch.batch_id} with {len(batch.requests)} requests"
        )

        # Extract request data
        request_data = [req.data for req in batch.requests]

        # Process the batch with retries
        results = await self._process_with_retries(request_data, batch.batch_id)

        # Match results to requests
        batch.results = []
        for i, request in enumerate(batch.requests):
            result: BatchResult[R] = BatchResult(
                request_id=request.request_id,
                processed_at=time.time(),
                metadata=request.metadata,
            )

            if i < len(results):
                if isinstance(results[i], Exception):
                    result.error = str(results[i])
                else:
                    result.result = results[i]
            else:
                result.error = "No result returned for request"

            batch.results.append(result)

        batch.state = BatchState.COMPLETED
        batch.processing_completed = time.time()

        # Update statistics
        processing_time = batch.processing_completed - batch.processing_started
        self._stats["total_processing_time"] += processing_time
        self._stats["successful_batches"] += 1

        # Calculate average batch size
        total_batches = self._stats["total_batches"]
        current_avg = self._stats["average_batch_size"]
        self._stats["average_batch_size"] = (
            current_avg * (total_batches - 1) + len(batch.requests)
        ) / total_batches

        # Set results for pending futures
        await self._set_batch_results(batch)

    async def _process_with_retries(
        self, request_data: List[T], batch_id: str
    ) -> List[R]:
        """Process a batch with retry logic."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                # Process the batch
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.process_batch_fn, request_data
                )

                if len(results) != len(request_data):
                    logger.warning(
                        f"Batch {batch_id}: Expected {len(request_data)} results, "
                        f"got {len(results)}"
                    )

                return results

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Batch {batch_id} failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )

                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(
                        self.retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    # Return exceptions for all requests
                    return [e] * len(request_data)  # type: ignore[list-item]

        return [last_exception] * len(request_data)  # type: ignore[list-item]  # Should not reach here

    async def _set_batch_results(self, batch: Batch[T, R]) -> None:
        """Set results for all requests in the batch."""
        async with self._lock:
            for result in batch.results:
                if result.request_id in self._pending_results:
                    future = self._pending_results[result.request_id]
                    if not future.done():
                        future.set_result(result)
                    del self._pending_results[result.request_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        async with self._lock:
            stats = self._stats.copy()

            # Calculate derived metrics
            total_processing_time = stats["total_processing_time"]
            total_batches = stats["total_batches"]

            stats.update(
                {
                    "current_queue_size": len(self._queue),
                    "pending_results": len(self._pending_results),
                    "average_processing_time_per_batch": (
                        total_processing_time / total_batches
                        if total_batches > 0
                        else 0.0
                    ),
                    "average_processing_time_per_request": (
                        total_processing_time / stats["total_requests"]
                        if stats["total_requests"] > 0
                        else 0.0
                    ),
                    "success_rate": (
                        stats["successful_batches"] / total_batches
                        if total_batches > 0
                        else 0.0
                    ),
                }
            )

            return stats

    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including current state."""
        stats = self.get_stats()

        async with self._lock:
            current_batch_info = None
            if self._current_batch:
                current_batch = self._current_batch
                current_batch_info = {
                    "batch_id": current_batch.batch_id,
                    "state": current_batch.state.value,
                    "request_count": len(current_batch.requests),
                    "created_at": current_batch.created_at,
                    "processing_started": current_batch.processing_started,
                    "processing_completed": current_batch.processing_completed,
                }

            stats.update(
                {
                    "current_batch": current_batch_info,
                    "shutdown": self._shutdown,
                }
            )

        return stats


class BatchProcessorManager:
    """
    Manager for multiple batch processors.

    Provides a centralized way to manage different types of batch processors.
    """

    def __init__(self):
        """Initialize the batch processor manager."""
        self.processors: Dict[str, BatchProcessor] = {}
        self._lock = asyncio.Lock()

    async def register_processor(self, name: str, processor: BatchProcessor) -> None:
        """Register a batch processor."""
        async with self._lock:
            self.processors[name] = processor
            await processor.start()

    async def unregister_processor(self, name: str) -> None:
        """Unregister a batch processor."""
        async with self._lock:
            if name in self.processors:
                await self.processors[name].stop()
                del self.processors[name]

    async def get_processor(self, name: str) -> Optional[BatchProcessor]:
        """Get a batch processor by name."""
        async with self._lock:
            return self.processors.get(name)

    async def submit_request(
        self,
        processor_name: str,
        data: Any,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchResult:
        """
        Submit a request to a batch processor.

        Args:
            processor_name: Name of the processor
            data: Request data
            priority: Request priority
            metadata: Request metadata

        Returns:
            Batch result
        """
        processor = await self.get_processor(processor_name)
        if processor:
            return await processor.submit_request(data, priority, metadata)
        else:
            error_result: BatchResult[Any] = BatchResult(
                request_id=str(uuid.uuid4()),
                error=f"Processor '{processor_name}' not found",
                processed_at=time.time(),
            )
            return error_result

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all processors."""
        async with self._lock:
            stats = {
                "total_processors": len(self.processors),
                "processors": {},
            }

            for name, processor in self.processors.items():
                stats["processors"][name] = await processor.get_stats()

            return stats

    async def shutdown_all(self) -> None:
        """Shutdown all batch processors."""
        async with self._lock:
            for name, processor in self.processors.items():
                await processor.stop()
            self.processors.clear()


# Global batch processor manager instance
_batch_processor_manager: Optional[BatchProcessorManager] = None


async def get_batch_processor_manager() -> BatchProcessorManager:
    """Get the global batch processor manager instance."""
    global _batch_processor_manager
    if _batch_processor_manager is None:
        _batch_processor_manager = BatchProcessorManager()
    return _batch_processor_manager


async def shutdown_batch_processors() -> None:
    """Shutdown all batch processors."""
    global _batch_processor_manager
    if _batch_processor_manager is not None:
        await _batch_processor_manager.shutdown_all()
        _batch_processor_manager = None


# Example batch processing functions


def create_tool_execution_batch_processor() -> (
    BatchProcessor[Dict[str, Any], Dict[str, Any]]
):
    """
    Create a batch processor for tool execution requests.

    This processor batches multiple tool execution requests and processes
    them together to reduce external API calls.
    """

    def process_tool_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This would be implemented to batch process tool requests
        # For now, return mock results
        results = []
        for request in requests:
            # Simulate processing
            results.append(
                {
                    "success": True,
                    "result": f"Processed {request.get('tool_name', 'unknown')}",
                    "execution_time": 0.1,
                }
            )
        return results

    return BatchProcessor(
        name="tool_execution",
        process_batch_fn=process_tool_batch,
        max_batch_size=5,
        max_wait_time=0.05,  # 50ms
    )


def create_agent_processing_batch_processor() -> BatchProcessor[str, Dict[str, Any]]:
    """
    Create a batch processor for agent processing requests.

    This processor batches multiple agent processing requests for
    efficient LLM API usage.
    """

    def process_agent_batch(messages: List[str]) -> List[Dict[str, Any]]:
        # This would be implemented to batch process agent requests
        # For now, return mock results
        results = []
        for message in messages:
            # Simulate processing
            results.append(
                {
                    "success": True,
                    "response": f"Processed: {message}",
                    "agent_name": "batch_agent",
                    "execution_time": 0.2,
                }
            )
        return results

    return BatchProcessor(
        name="agent_processing",
        process_batch_fn=process_agent_batch,
        max_batch_size=3,
        max_wait_time=0.1,  # 100ms
    )
