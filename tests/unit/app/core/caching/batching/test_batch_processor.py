"""
Unit tests for the batching module.

Tests for the request batching system for efficient API calls, including batch processors
and the batch processor manager.
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.caching.batching.batch_processor import (
    BatchState,
    BatchRequest,
    BatchResult,
    Batch,
    BatchProcessor,
    BatchProcessorManager,
    get_batch_processor_manager,
    shutdown_batch_processors,
    create_tool_execution_batch_processor,
    create_agent_processing_batch_processor,
)


class TestBatchState:
    """Test cases for BatchState enum."""

    def test_batch_state_values(self):
        """Test that BatchState has the expected values."""
        assert BatchState.COLLECTING.value == "collecting"
        assert BatchState.PROCESSING.value == "processing"
        assert BatchState.COMPLETED.value == "completed"
        assert BatchState.FAILED.value == "failed"


class TestBatchRequest:
    """Test cases for BatchRequest class."""

    def test_batch_request_initialization(self):
        """Test initializing a batch request."""
        request_id = str(uuid.uuid4())
        created_at = time.time()
        metadata = {"key": "value"}
        
        request = BatchRequest(
            request_id=request_id,
            data="test_data",
            created_at=created_at,
            priority=5,
            metadata=metadata,
        )
        
        assert request.request_id == request_id
        assert request.data == "test_data"
        assert request.created_at == created_at
        assert request.priority == 5
        assert request.metadata == metadata

    def test_batch_request_defaults(self):
        """Test batch request with default values."""
        request = BatchRequest(
            request_id="test_id",
            data="test_data",
            created_at=1234567890.0,
        )
        
        assert request.request_id == "test_id"
        assert request.data == "test_data"
        assert request.created_at == 1234567890.0
        assert request.priority == 0
        assert request.metadata == {}


class TestBatchResult:
    """Test cases for BatchResult class."""

    def test_batch_result_initialization(self):
        """Test initializing a batch result."""
        request_id = str(uuid.uuid4())
        processed_at = time.time()
        metadata = {"key": "value"}
        
        result = BatchResult(
            request_id=request_id,
            result="test_result",
            error=None,
            processed_at=processed_at,
            metadata=metadata,
        )
        
        assert result.request_id == request_id
        assert result.result == "test_result"
        assert result.error is None
        assert result.processed_at == processed_at
        assert result.metadata == metadata

    def test_batch_result_defaults(self):
        """Test batch result with default values."""
        result = BatchResult(
            request_id="test_id",
        )
        
        assert result.request_id == "test_id"
        assert result.result is None
        assert result.error is None
        assert result.processed_at is None
        assert result.metadata == {}


class TestBatch:
    """Test cases for Batch class."""

    def test_batch_initialization(self):
        """Test initializing a batch."""
        batch_id = str(uuid.uuid4())
        created_at = time.time()
        requests = [
            BatchRequest("id1", "data1", created_at),
            BatchRequest("id2", "data2", created_at),
        ]
        
        batch = Batch(
            batch_id=batch_id,
            requests=requests,
            created_at=created_at,
            state=BatchState.PROCESSING,
        )
        
        assert batch.batch_id == batch_id
        assert batch.requests == requests
        assert batch.created_at == created_at
        assert batch.state == BatchState.PROCESSING
        assert batch.results == []
        assert batch.processing_started is None
        assert batch.processing_completed is None

    def test_batch_defaults(self):
        """Test batch with default values."""
        batch = Batch(
            batch_id="test_id",
            requests=[],
            created_at=1234567890.0,
        )
        
        assert batch.batch_id == "test_id"
        assert batch.requests == []
        assert batch.created_at == 1234567890.0
        assert batch.state == BatchState.COLLECTING
        assert batch.results == []
        assert batch.processing_started is None
        assert batch.processing_completed is None


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""

    @pytest.fixture
    def mock_process_batch_fn(self):
        """Create a mock process_batch function."""
        return AsyncMock(return_value=["result1", "result2", "result3"])

    @pytest.fixture
    def batch_processor(self, mock_process_batch_fn):
        """Create a batch processor."""
        return BatchProcessor(
            name="test_processor",
            process_batch_fn=mock_process_batch_fn,
            max_batch_size=3,
            max_wait_time=0.1,
            max_queue_size=10,
            retry_attempts=2,
            retry_delay=0.05,
        )

    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self, mock_process_batch_fn):
        """Test initializing a batch processor."""
        processor = BatchProcessor(
            name="test_processor",
            process_batch_fn=mock_process_batch_fn,
            max_batch_size=5,
            max_wait_time=0.2,
            max_queue_size=20,
            retry_attempts=3,
            retry_delay=0.1,
        )
        
        assert processor.name == "test_processor"
        assert processor.process_batch_fn == mock_process_batch_fn
        assert processor.max_batch_size == 5
        assert processor.max_wait_time == 0.2
        assert processor.max_queue_size == 20
        assert processor.retry_attempts == 3
        assert processor.retry_delay == 0.1
        assert len(processor._queue) == 0
        assert processor._current_batch is None
        assert processor._shutdown is False

    @pytest.mark.asyncio
    async def test_start_stop(self, batch_processor):
        """Test starting and stopping the batch processor."""
        # Start the processor
        await batch_processor.start()
        assert batch_processor._shutdown is False
        assert batch_processor._processing_task is not None
        
        # Stop the processor
        await batch_processor.stop()
        assert batch_processor._shutdown is True
        assert batch_processor._processing_task is None

    @pytest.mark.asyncio
    async def test_submit_request(self, batch_processor, mock_process_batch_fn):
        """Test submitting a request."""
        # Start the processor
        await batch_processor.start()
        
        # Submit a request
        result = await batch_processor.submit_request("test_data", priority=5, metadata={"key": "value"})
        
        # Should get a result
        assert isinstance(result, BatchResult)
        assert result.result == "result1"  # First result from mock function
        assert result.error is None
        
        # Check that the process function was called
        mock_process_batch_fn.assert_called_once()
        
        # Stop the processor
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_submit_request_queue_overflow(self, batch_processor):
        """Test submitting a request when queue is full."""
        # Create a processor with a small queue
        small_processor = BatchProcessor(
            name="small_processor",
            process_batch_fn=AsyncMock(return_value=[]),
            max_batch_size=1,
            max_wait_time=0.1,
            max_queue_size=2,
        )
        
        await small_processor.start()
        
        # Fill the queue
        await small_processor.submit_request("data1")
        await small_processor.submit_request("data2")
        
        # Try to submit one more (should be rejected)
        result = await small_processor.submit_request("data3")
        
        assert result.error == "Queue overflow - request rejected"
        assert small_processor._stats["queue_overflows"] == 1
        
        await small_processor.stop()

    @pytest.mark.asyncio
    async def test_submit_request_priority_ordering(self, batch_processor, mock_process_batch_fn):
        """Test that requests are ordered by priority."""
        # Start the processor
        await batch_processor.start()
        
        # Submit requests with different priorities
        result1 = batch_processor.submit_request("low_priority", priority=1)
        result2 = batch_processor.submit_request("high_priority", priority=10)
        result3 = batch_processor.submit_request("medium_priority", priority=5)
        
        # Wait for all to complete
        result1 = await result1
        result2 = await result2
        result3 = await result3
        
        # Check that the process function was called with high priority first
        call_args = mock_process_batch_fn.call_args[0][0]
        
        # Should be ordered by priority (high to low)
        assert call_args[0] == "high_priority"
        assert call_args[1] == "medium_priority"
        assert call_args[2] == "low_priority"
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_processing_loop_batch_size(self, batch_processor, mock_process_batch_fn):
        """Test that batch is processed when max_batch_size is reached."""
        # Start the processor
        await batch_processor.start()
        
        # Submit exactly max_batch_size requests
        results = []
        for i in range(batch_processor.max_batch_size):
            result = await batch_processor.submit_request(f"data{i}")
            results.append(result)
        
        # Wait for all to complete
        results = await asyncio.gather(*results)
        
        # Should have processed one batch
        assert mock_process_batch_fn.call_count == 1
        
        # All results should be successful
        for result in results:
            assert result.error is None
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_processing_loop_wait_time(self, batch_processor, mock_process_batch_fn):
        """Test that batch is processed when max_wait_time is reached."""
        # Start the processor
        await batch_processor.start()
        
        # Submit one request (less than max_batch_size)
        result = await batch_processor.submit_request("test_data")
        
        # Wait for processing to complete
        result = await result
        
        # Should have processed one batch (due to wait time)
        assert mock_process_batch_fn.call_count == 1
        assert result.error is None
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_process_with_retries_success(self, batch_processor, mock_process_batch_fn):
        """Test processing with retries on success."""
        # Start the processor
        await batch_processor.start()
        
        # Mock function that fails once then succeeds
        mock_process_batch_fn.side_effect = [
            Exception("First failure"),
            ["success"]
        ]
        
        # Submit a request
        result = await batch_processor.submit_request("test_data")
        
        # Should eventually succeed
        assert result.result == "success"
        assert result.error is None
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_process_with_retries_failure(self, batch_processor, mock_process_batch_fn):
        """Test processing with retries on failure."""
        # Start the processor
        await batch_processor.start()
        
        # Mock function that always fails
        mock_process_batch_fn.side_effect = Exception("Always fails")
        
        # Submit a request
        result = await batch_processor.submit_request("test_data")
        
        # Should fail after all retries
        assert result.result is None
        assert result.error == "Always fails"
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_mismatched_results_count(self, batch_processor, mock_process_batch_fn):
        """Test handling when process function returns wrong number of results."""
        # Start the processor
        await batch_processor.start()
        
        # Mock function returning fewer results than requests
        mock_process_batch_fn.return_value = ["result1"]  # Only one result for 3 requests
        
        # Submit multiple requests
        results = []
        for i in range(3):
            result = await batch_processor.submit_request(f"data{i}")
            results.append(result)
        
        # Wait for all to complete
        results = await asyncio.gather(*results)
        
        # First result should be successful, others should have error
        assert results[0].result == "result1"
        assert results[0].error is None
        
        for i in range(1, 3):
            assert results[i].result is None
            assert results[i].error == "No result returned for request"
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, batch_processor, mock_process_batch_fn):
        """Test getting batch processor statistics."""
        # Start the processor
        await batch_processor.start()
        
        # Submit some requests
        await batch_processor.submit_request("data1")
        await batch_processor.submit_request("data2")
        await batch_processor.submit_request("data3")
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Get stats
        stats = await batch_processor.get_stats()
        
        assert stats["total_requests"] == 3
        assert stats["total_batches"] >= 1
        assert stats["successful_batches"] >= 1
        assert stats["current_queue_size"] >= 0
        assert stats["pending_results"] >= 0
        assert stats["success_rate"] >= 0.0
        assert stats["average_processing_time_per_batch"] >= 0.0
        assert stats["average_processing_time_per_request"] >= 0.0
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_get_detailed_stats(self, batch_processor):
        """Test getting detailed statistics."""
        # Start the processor
        await batch_processor.start()
        
        # Get detailed stats
        stats = await batch_processor.get_detailed_stats()
        
        # Should include basic stats
        assert "total_requests" in stats
        assert "total_batches" in stats
        
        # Should include detailed info
        assert "current_batch" in stats
        assert "shutdown" in stats
        
        await batch_processor.stop()

    @pytest.mark.asyncio
    async def test_stop_with_remaining_requests(self, batch_processor, mock_process_batch_fn):
        """Test stopping processor with remaining requests."""
        # Start the processor
        await batch_processor.start()
        
        # Submit a request
        result_future = batch_processor.submit_request("test_data")
        
        # Stop the processor (should process remaining requests)
        await batch_processor.stop()
        
        # Request should still be processed
        result = await result_future
        assert result.error is None
        
        # Should have processed the remaining batch
        assert mock_process_batch_fn.call_count == 1


class TestBatchProcessorManager:
    """Test cases for BatchProcessorManager class."""

    @pytest.fixture
    def batch_processor_manager(self):
        """Create a batch processor manager."""
        return BatchProcessorManager()

    @pytest.fixture
    def mock_processor(self):
        """Create a mock batch processor."""
        processor = AsyncMock()
        processor.start = AsyncMock()
        processor.stop = AsyncMock()
        return processor

    @pytest.mark.asyncio
    async def test_register_processor(self, batch_processor_manager, mock_processor):
        """Test registering a batch processor."""
        # Register a processor
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Should have called start on the processor
        mock_processor.start.assert_called_once()
        
        # Should be in the processors dict
        assert "test_processor" in batch_processor_manager.processors
        assert batch_processor_manager.processors["test_processor"] == mock_processor

    @pytest.mark.asyncio
    async def test_unregister_processor(self, batch_processor_manager, mock_processor):
        """Test unregistering a batch processor."""
        # Register first
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Unregister
        await batch_processor_manager.unregister_processor("test_processor")
        
        # Should have called stop on the processor
        mock_processor.stop.assert_called_once()
        
        # Should be removed from the processors dict
        assert "test_processor" not in batch_processor_manager.processors

    @pytest.mark.asyncio
    async def test_get_processor(self, batch_processor_manager, mock_processor):
        """Test getting a batch processor."""
        # Register first
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Get the processor
        processor = await batch_processor_manager.get_processor("test_processor")
        
        assert processor == mock_processor
        
        # Get non-existent processor
        processor = await batch_processor_manager.get_processor("nonexistent")
        assert processor is None

    @pytest.mark.asyncio
    async def test_submit_request(self, batch_processor_manager, mock_processor):
        """Test submitting a request to a processor."""
        # Register first
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Mock the submit_request method
        mock_processor.submit_request = AsyncMock(return_value="test_result")
        
        # Submit a request
        result = await batch_processor_manager.submit_request(
            "test_processor", "test_data", priority=5, metadata={"key": "value"}
        )
        
        assert result == "test_result"
        mock_processor.submit_request.assert_called_once_with("test_data", priority=5, metadata={"key": "value"})

    @pytest.mark.asyncio
    async def test_submit_request_nonexistent_processor(self, batch_processor_manager):
        """Test submitting a request to a non-existent processor."""
        # Submit to non-existent processor
        result = await batch_processor_manager.submit_request(
            "nonexistent", "test_data"
        )
        
        # Should return an error result
        assert isinstance(result, BatchResult)
        assert result.result is None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_get_stats(self, batch_processor_manager, mock_processor):
        """Test getting statistics for all processors."""
        # Register a processor
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Mock the get_stats method
        mock_processor.get_stats = AsyncMock(return_value={"total_requests": 10})
        
        # Get stats
        stats = await batch_processor_manager.get_stats()
        
        assert stats["total_processors"] == 1
        assert "processors" in stats
        assert "test_processor" in stats["processors"]
        assert stats["processors"]["test_processor"] == {"total_requests": 10}

    @pytest.mark.asyncio
    async def test_shutdown_all(self, batch_processor_manager, mock_processor):
        """Test shutting down all processors."""
        # Register a processor
        await batch_processor_manager.register_processor("test_processor", mock_processor)
        
        # Shutdown all
        await batch_processor_manager.shutdown_all()
        
        # Should have called stop on the processor
        mock_processor.stop.assert_called_once()
        
        # Processors dict should be empty
        assert len(batch_processor_manager.processors) == 0


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @pytest.mark.asyncio
    async def test_get_batch_processor_manager(self):
        """Test getting the global batch processor manager."""
        manager1 = await get_batch_processor_manager()
        manager2 = await get_batch_processor_manager()
        
        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, BatchProcessorManager)

    @pytest.mark.asyncio
    async def test_shutdown_batch_processors(self):
        """Test shutting down all batch processors."""
        # Get the manager
        manager = await get_batch_processor_manager()
        
        # Mock the shutdown_all method
        manager.shutdown_all = AsyncMock()
        
        # Shutdown all processors
        await shutdown_batch_processors()
        
        # Should have called shutdown_all
        manager.shutdown_all.assert_called_once()

    def test_create_tool_execution_batch_processor(self):
        """Test creating a tool execution batch processor."""
        processor = create_tool_execution_batch_processor()
        
        assert processor.name == "tool_execution"
        assert processor.max_batch_size == 5
        assert processor.max_wait_time == 0.05
        assert callable(processor.process_batch_fn)

    def test_create_agent_processing_batch_processor(self):
        """Test creating an agent processing batch processor."""
        processor = create_agent_processing_batch_processor()
        
        assert processor.name == "agent_processing"
        assert processor.max_batch_size == 3
        assert processor.max_wait_time == 0.1
        assert callable(processor.process_batch_fn)

    @pytest.mark.asyncio
    async def test_tool_execution_batch_function(self):
        """Test the tool execution batch function."""
        processor = create_tool_execution_batch_processor()
        
        # Create test requests
        requests = [
            {"tool_name": "tool1", "args": {}},
            {"tool_name": "tool2", "args": {"param": "value"}},
        ]
        
        # Process the batch
        results = processor.process_batch_fn(requests)
        
        # Should return results for each request
        assert len(results) == 2
        assert all(result["success"] for result in results)
        assert all("Processed" in result["result"] for result in results)

    @pytest.mark.asyncio
    async def test_agent_processing_batch_function(self):
        """Test the agent processing batch function."""
        processor = create_agent_processing_batch_processor()
        
        # Create test messages
        messages = ["message1", "message2", "message3"]
        
        # Process the batch
        results = processor.process_batch_fn(messages)
        
        # Should return results for each message
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert all("Processed:" in result["response"] for result in results)