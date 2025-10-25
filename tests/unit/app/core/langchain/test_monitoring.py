"""
Unit tests for LangChain Monitoring System.

This module tests the comprehensive monitoring system for LangChain components,
including performance metrics, error tracking, and logging.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional

from app.core.langchain.monitoring import (
    LangChainMonitor,
    MetricType,
    ComponentType,
    MetricEvent,
    PerformanceMetrics,
    RequestTracker,
    langchain_monitor,
    track_llm_request,
    track_tool_execution,
    track_agent_execution,
    track_workflow_execution,
    track_request
)


class TestMetricType:
    """Test MetricType enum"""

    def test_metric_type_values(self):
        """Test that MetricType has expected values"""
        expected_types = [
            "counter",
            "gauge",
            "histogram",
            "timer"
        ]
        
        actual_types = [metric_type.value for metric_type in MetricType]
        assert actual_types == expected_types


class TestComponentType:
    """Test ComponentType enum"""

    def test_component_type_values(self):
        """Test that ComponentType has expected values"""
        expected_types = [
            "llm",
            "tool",
            "agent",
            "workflow",
            "memory",
            "chain"
        ]
        
        actual_types = [component_type.value for component_type in ComponentType]
        assert actual_types == expected_types


class TestMetricEvent:
    """Test MetricEvent dataclass"""

    def test_metric_event_defaults(self):
        """Test MetricEvent default values"""
        event = MetricEvent(
            name="test_metric",
            component_type=ComponentType.LLM,
            component_id="gpt-4",
            metric_type=MetricType.COUNTER,
            value=1.0,
            timestamp=datetime.now()
        )
        
        assert event.name == "test_metric"
        assert event.component_type == ComponentType.LLM
        assert event.component_id == "gpt-4"
        assert event.metric_type == MetricType.COUNTER
        assert event.value == 1.0
        assert event.metadata == {}

    def test_metric_event_with_values(self):
        """Test MetricEvent with provided values"""
        timestamp = datetime.now()
        metadata = {"request_id": "req-123", "user_id": "user-456"}
        
        event = MetricEvent(
            name="response_time",
            component_type=ComponentType.AGENT,
            component_id="conversational_agent",
            metric_type=MetricType.TIMER,
            value=2.5,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert event.name == "response_time"
        assert event.component_type == ComponentType.AGENT
        assert event.component_id == "conversational_agent"
        assert event.metric_type == MetricType.TIMER
        assert event.value == 2.5
        assert event.timestamp == timestamp
        assert event.metadata == metadata

    def test_metric_event_to_dict(self):
        """Test MetricEvent to_dict method"""
        timestamp = datetime.now()
        metadata = {"test": True}
        
        event = MetricEvent(
            name="test_metric",
            component_type=ComponentType.LLM,
            component_id="gpt-4",
            metric_type=MetricType.COUNTER,
            value=1.0,
            timestamp=timestamp,
            metadata=metadata
        )
        
        result = event.to_dict()
        
        assert result["name"] == "test_metric"
        assert result["component_type"] == "llm"
        assert result["component_id"] == "gpt-4"
        assert result["metric_type"] == "counter"
        assert result["value"] == 1.0
        assert result["timestamp"] == timestamp.isoformat()
        assert result["metadata"] == metadata


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values"""
        metrics = PerformanceMetrics(
            component_id="test_component",
            component_type=ComponentType.LLM
        )
        
        assert metrics.component_id == "test_component"
        assert metrics.component_type == ComponentType.LLM
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_duration == 0.0
        assert metrics.min_duration == float('inf')
        assert metrics.max_duration == 0.0
        assert metrics.average_duration == 0.0
        assert metrics.last_request_time is None
        assert metrics.error_rate == 0.0

    def test_performance_metrics_with_values(self):
        """Test PerformanceMetrics with provided values"""
        last_request_time = datetime.now()
        
        metrics = PerformanceMetrics(
            component_id="test_component",
            component_type=ComponentType.LLM,
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
            total_duration=45.5,
            min_duration=1.0,
            max_duration=10.0,
            average_duration=4.55,
            last_request_time=last_request_time,
            error_rate=0.1
        )
        
        assert metrics.component_id == "test_component"
        assert metrics.component_type == ComponentType.LLM
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 9
        assert metrics.failed_requests == 1
        assert metrics.total_duration == 45.5
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 10.0
        assert metrics.average_duration == 4.55
        assert metrics.last_request_time == last_request_time
        assert metrics.error_rate == 0.1

    def test_performance_metrics_update(self):
        """Test PerformanceMetrics update method"""
        metrics = PerformanceMetrics(
            component_id="test_component",
            component_type=ComponentType.LLM
        )
        
        # Update with successful request
        metrics.update(duration=2.5, success=True)
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_duration == 2.5
        assert metrics.min_duration == 2.5
        assert metrics.max_duration == 2.5
        assert metrics.average_duration == 2.5
        assert metrics.last_request_time is not None
        assert metrics.error_rate == 0.0
        
        # Update with failed request
        metrics.update(duration=1.0, success=False)
        
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.total_duration == 3.5
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 2.5
        assert metrics.average_duration == 1.75
        assert metrics.error_rate == 0.5

    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics to_dict method"""
        last_request_time = datetime.now()
        
        metrics = PerformanceMetrics(
            component_id="test_component",
            component_type=ComponentType.LLM,
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
            total_duration=45.5,
            min_duration=1.0,
            max_duration=10.0,
            average_duration=4.55,
            last_request_time=last_request_time,
            error_rate=0.1
        )
        
        result = metrics.to_dict()
        
        assert result["component_id"] == "test_component"
        assert result["component_type"] == "llm"
        assert result["total_requests"] == 10
        assert result["successful_requests"] == 9
        assert result["failed_requests"] == 1
        assert result["total_duration"] == 45.5
        assert result["min_duration"] == 1.0
        assert result["max_duration"] == 10.0
        assert result["average_duration"] == 4.55
        assert result["last_request_time"] == last_request_time.isoformat()
        assert result["error_rate"] == 0.1


class TestLangChainMonitor:
    """Test LangChainMonitor class"""

    @pytest.fixture
    def monitor_instance(self):
        """Create a fresh monitor instance for testing"""
        return LangChainMonitor()

    @pytest.fixture
    def mock_db_client(self):
        """Mock database client"""
        with patch('app.core.langchain.monitoring.get_langchain_client') as mock:
            mock_client = Mock()
            mock_client.log_performance = AsyncMock()
            mock_client.cleanup_old_metrics = AsyncMock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_initialize(self, monitor_instance, mock_db_client):
        """Test monitor initialization"""
        await monitor_instance.initialize()
        
        assert monitor_instance._initialized is True
        mock_db_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, monitor_instance, mock_db_client):
        """Test that initialize is idempotent"""
        await monitor_instance.initialize()
        await monitor_instance.initialize()
        
        assert monitor_instance._initialized is True
        # Should only initialize once
        mock_db_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, monitor_instance):
        """Test monitor initialization failure"""
        with patch('app.core.langchain.monitoring.get_langchain_client') as mock:
            mock.side_effect = Exception("Database connection failed")
            
            await monitor_instance.initialize()
            
            assert monitor_instance._initialized is True  # Still marked as initialized
            assert monitor_instance._db_client is None

    @pytest.mark.asyncio
    async def test_track_request(self, monitor_instance, mock_db_client):
        """Test tracking a request"""
        await monitor_instance.initialize()
        
        await monitor_instance.track_request(
            component_type=ComponentType.LLM,
            component_id="gpt-4",
            duration=2.5,
            success=True,
            metadata={"request_id": "req-123"}
        )
        
        # Check metrics were created
        key = "llm:gpt-4"
        assert key in monitor_instance._metrics
        metrics = monitor_instance._metrics[key]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_duration == 2.5
        assert metrics.average_duration == 2.5
        
        # Check event was created
        assert len(monitor_instance._events) == 1
        event = monitor_instance._events[0]
        assert event.name == "request_duration"
        assert event.component_type == ComponentType.LLM
        assert event.component_id == "gpt-4"
        assert event.metric_type == MetricType.TIMER
        assert event.value == 2.5
        assert event.metadata["success"] is True
        assert event.metadata["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_track_request_failure(self, monitor_instance):
        """Test tracking a failed request"""
        await monitor_instance.initialize()
        
        await monitor_instance.track_request(
            component_type=ComponentType.TOOL,
            component_id="search_tool",
            duration=1.0,
            success=False,
            metadata={"error": "API timeout"}
        )
        
        # Check metrics
        key = "tool:search_tool"
        metrics = monitor_instance._metrics[key]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_track_metric(self, monitor_instance):
        """Test tracking a custom metric"""
        await monitor_instance.initialize()
        
        await monitor_instance.track_metric(
            name="custom_metric",
            component_type=ComponentType.AGENT,
            component_id="conversational_agent",
            metric_type=MetricType.GAUGE,
            value=42.5,
            metadata={"unit": "items"}
        )
        
        # Check event was created
        assert len(monitor_instance._events) == 1
        event = monitor_instance._events[0]
        assert event.name == "custom_metric"
        assert event.component_type == ComponentType.AGENT
        assert event.component_id == "conversational_agent"
        assert event.metric_type == MetricType.GAUGE
        assert event.value == 42.5
        assert event.metadata["unit"] == "items"

    @pytest.mark.asyncio
    async def test_track_error(self, monitor_instance, mock_db_client):
        """Test tracking an error"""
        await monitor_instance.initialize()
        
        error = Exception("Test error")
        await monitor_instance.track_error(
            component_type=ComponentType.WORKFLOW,
            component_id="memory_workflow",
            error=error,
            metadata={"step": "validation"}
        )
        
        # Check failed request was tracked
        key = "workflow:memory_workflow"
        metrics = monitor_instance._metrics[key]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        
        # Check database logging was called
        mock_db_client.return_value.log_performance.assert_called_once()
        call_args = mock_db_client.return_value.log_performance.call_args[1]
        assert call_args["component_type"] == "workflow"
        assert call_args["component_id"] == "memory_workflow"
        assert call_args["metric_name"] == "error"
        assert call_args["metric_value"] == 1.0
        assert call_args["metadata"]["error_type"] == "Exception"
        assert call_args["metadata"]["error_message"] == "Test error"
        assert call_args["metadata"]["step"] == "validation"

    @pytest.mark.asyncio
    async def test_get_metrics(self, monitor_instance):
        """Test getting performance metrics"""
        await monitor_instance.initialize()
        
        # Create some metrics
        await monitor_instance.track_request(
            ComponentType.LLM, "gpt-4", 1.0, True
        )
        await monitor_instance.track_request(
            ComponentType.TOOL, "search_tool", 2.0, True
        )
        await monitor_instance.track_request(
            ComponentType.LLM, "gpt-3.5-turbo", 1.5, True
        )
        
        # Get all metrics
        all_metrics = await monitor_instance.get_metrics()
        assert len(all_metrics) == 3
        
        # Get filtered by component type
        llm_metrics = await monitor_instance.get_metrics(
            component_type=ComponentType.LLM
        )
        assert len(llm_metrics) == 2
        
        # Get filtered by component ID
        gpt4_metrics = await monitor_instance.get_metrics(
            component_type=ComponentType.LLM,
            component_id="gpt-4"
        )
        assert len(gpt4_metrics) == 1
        assert gpt4_metrics[0].component_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_events(self, monitor_instance):
        """Test getting metric events"""
        await monitor_instance.initialize()
        
        # Create some events
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        await monitor_instance.track_metric(
            "metric2", ComponentType.TOOL, "search_tool", MetricType.GAUGE, 2.0
        )
        await monitor_instance.track_metric(
            "metric3", ComponentType.AGENT, "conv_agent", MetricType.TIMER, 3.0
        )
        
        # Get all events
        all_events = await monitor_instance.get_events()
        assert len(all_events) == 3
        
        # Get limited events
        limited_events = await monitor_instance.get_events(limit=2)
        assert len(limited_events) == 2
        
        # Get filtered by component type
        llm_events = await monitor_instance.get_events(
            component_type=ComponentType.LLM
        )
        assert len(llm_events) == 1
        assert llm_events[0].component_type == ComponentType.LLM

    @pytest.mark.asyncio
    async def test_maybe_flush_events_buffer_full(self, monitor_instance, mock_db_client):
        """Test flushing events when buffer is full"""
        await monitor_instance.initialize()
        
        # Set small buffer size for testing
        monitor_instance._event_buffer_size = 2
        
        # Add events to fill buffer
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        await monitor_instance.track_metric(
            "metric2", ComponentType.TOOL, "search_tool", MetricType.GAUGE, 2.0
        )
        
        # Should have flushed
        assert len(monitor_instance._events) == 0
        assert mock_db_client.return_value.log_performance.call_count == 2

    @pytest.mark.asyncio
    async def test_maybe_flush_events_interval_passed(self, monitor_instance, mock_db_client):
        """Test flushing events when interval has passed"""
        await monitor_instance.initialize()
        
        # Set short interval for testing
        monitor_instance._flush_interval = 0.1  # 100ms
        monitor_instance._last_flush = datetime.now() - timedelta(seconds=1)
        
        # Add event
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        
        # Should have flushed
        assert len(monitor_instance._events) == 0
        mock_db_client.return_value.log_performance.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, monitor_instance, mock_db_client):
        """Test cleaning up old metrics"""
        await monitor_instance.initialize()
        
        await monitor_instance.cleanup_old_metrics(days=30)
        
        mock_db_client.return_value.cleanup_old_metrics.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_get_summary_stats(self, monitor_instance):
        """Test getting summary statistics"""
        await monitor_instance.initialize()
        
        # Create some metrics
        await monitor_instance.track_request(ComponentType.LLM, "gpt-4", 1.0, True)
        await monitor_instance.track_request(ComponentType.LLM, "gpt-3.5-turbo", 1.5, True)
        await monitor_instance.track_request(ComponentType.TOOL, "search_tool", 2.0, True)
        await monitor_instance.track_request(ComponentType.AGENT, "conv_agent", 2.5, False)
        
        result = await monitor_instance.get_summary_stats()
        
        assert result["total_components"] == 4
        assert result["total_requests"] == 4
        assert result["total_errors"] == 1
        assert len(result["components_by_type"]) == 3
        assert result["components_by_type"]["llm"] == 2
        assert result["components_by_type"]["tool"] == 1
        assert result["components_by_type"]["agent"] == 1
        assert "average_response_times" in result
        assert result["average_response_times"]["llm"] == 1.25  # (1.0 + 1.5) / 2
        assert result["average_response_times"]["tool"] == 2.0
        assert result["average_response_times"]["agent"] == 2.5

    @pytest.mark.asyncio
    async def test_flush_events(self, monitor_instance, mock_db_client):
        """Test flushing events to database"""
        await monitor_instance.initialize()
        
        # Add events
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        await monitor_instance.track_metric(
            "metric2", ComponentType.TOOL, "search_tool", MetricType.GAUGE, 2.0
        )
        
        # Manually flush
        await monitor_instance._flush_events()
        
        # Should have flushed
        assert len(monitor_instance._events) == 0
        assert mock_db_client.return_value.log_performance.call_count == 2

    @pytest.mark.asyncio
    async def test_flush_events_no_db_client(self, monitor_instance):
        """Test flushing events without database client"""
        await monitor_instance.initialize()
        monitor_instance._db_client = None
        
        # Add event
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        
        # Should not raise error
        await monitor_instance._flush_events()

    @pytest.mark.asyncio
    async def test_flush_events_error(self, monitor_instance, mock_db_client):
        """Test flushing events with database error"""
        await monitor_instance.initialize()
        
        # Make database fail
        mock_db_client.return_value.log_performance.side_effect = Exception("Database error")
        
        # Add event
        await monitor_instance.track_metric(
            "metric1", ComponentType.LLM, "gpt-4", MetricType.COUNTER, 1.0
        )
        
        # Should not raise error
        await monitor_instance._flush_events()
        
        # Events should still be cleared
        assert len(monitor_instance._events) == 0


class TestRequestTracker:
    """Test RequestTracker context manager"""

    @pytest.fixture
    def mock_monitor(self):
        """Mock monitor"""
        monitor = Mock()
        monitor.track_request = AsyncMock()
        return monitor

    @pytest.mark.asyncio
    async def test_request_tracker_success(self, mock_monitor):
        """Test request tracker with successful execution"""
        tracker = RequestTracker(
            monitor=mock_monitor,
            component_type=ComponentType.LLM,
            component_id="gpt-4",
            metadata={"request_id": "req-123"}
        )
        
        async with tracker:
            await asyncio.sleep(0.01)  # Simulate work
        
        # Verify tracking was called
        mock_monitor.track_request.assert_called_once()
        call_args = mock_monitor.track_request.call_args[1]
        assert call_args["component_type"] == ComponentType.LLM
        assert call_args["component_id"] == "gpt-4"
        assert call_args["success"] is True
        assert call_args["duration"] > 0
        assert call_args["metadata"]["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_request_tracker_failure(self, mock_monitor):
        """Test request tracker with failed execution"""
        tracker = RequestTracker(
            monitor=mock_monitor,
            component_type=ComponentType.TOOL,
            component_id="search_tool"
        )
        
        with pytest.raises(Exception, match="Test error"):
            async with tracker:
                await asyncio.sleep(0.01)  # Simulate work
                raise Exception("Test error")
        
        # Verify tracking was called with failure
        mock_monitor.track_request.assert_called_once()
        call_args = mock_monitor.track_request.call_args[1]
        assert call_args["component_type"] == ComponentType.TOOL
        assert call_args["component_id"] == "search_tool"
        assert call_args["success"] is False
        assert call_args["duration"] > 0

    @pytest.mark.asyncio
    async def test_request_tracker_no_start_time(self, mock_monitor):
        """Test request tracker when start_time is None"""
        tracker = RequestTracker(
            monitor=mock_monitor,
            component_type=ComponentType.AGENT,
            component_id="conv_agent"
        )
        
        # Manually set start_time to None to simulate edge case
        tracker.start_time = None
        
        async with tracker:
            pass
        
        # Verify tracking was called with duration 0.0
        mock_monitor.track_request.assert_called_once()
        call_args = mock_monitor.track_request.call_args[1]
        assert call_args["duration"] == 0.0


class TestHelperFunctions:
    """Test helper functions"""

    @pytest.mark.asyncio
    async def test_track_llm_request(self):
        """Test track_llm_request helper function"""
        with patch('app.core.langchain.monitoring.langchain_monitor') as mock_monitor:
            mock_monitor.track_request = AsyncMock()
            
            await track_llm_request(
                llm_id="gpt-4",
                duration=2.5,
                success=True,
                metadata={"request_id": "req-123"}
            )
            
            mock_monitor.track_request.assert_called_once()
            call_args = mock_monitor.track_request.call_args[1]
            assert call_args["component_type"] == ComponentType.LLM
            assert call_args["component_id"] == "gpt-4"
            assert call_args["duration"] == 2.5
            assert call_args["success"] is True
            assert call_args["metadata"]["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_track_tool_execution(self):
        """Test track_tool_execution helper function"""
        with patch('app.core.langchain.monitoring.langchain_monitor') as mock_monitor:
            mock_monitor.track_request = AsyncMock()
            
            await track_tool_execution(
                tool_id="search_tool",
                duration=1.5,
                success=False,
                metadata={"error": "timeout"}
            )
            
            mock_monitor.track_request.assert_called_once()
            call_args = mock_monitor.track_request.call_args[1]
            assert call_args["component_type"] == ComponentType.TOOL
            assert call_args["component_id"] == "search_tool"
            assert call_args["duration"] == 1.5
            assert call_args["success"] is False
            assert call_args["metadata"]["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_track_agent_execution(self):
        """Test track_agent_execution helper function"""
        with patch('app.core.langchain.monitoring.langchain_monitor') as mock_monitor:
            mock_monitor.track_request = AsyncMock()
            
            await track_agent_execution(
                agent_id="conversational_agent",
                duration=3.0,
                success=True
            )
            
            mock_monitor.track_request.assert_called_once()
            call_args = mock_monitor.track_request.call_args[1]
            assert call_args["component_type"] == ComponentType.AGENT
            assert call_args["component_id"] == "conversational_agent"
            assert call_args["duration"] == 3.0
            assert call_args["success"] is True

    @pytest.mark.asyncio
    async def test_track_workflow_execution(self):
        """Test track_workflow_execution helper function"""
        with patch('app.core.langchain.monitoring.langchain_monitor') as mock_monitor:
            mock_monitor.track_request = AsyncMock()
            
            await track_workflow_execution(
                workflow_id="memory_workflow",
                duration=5.0,
                success=True,
                metadata={"steps": 5}
            )
            
            mock_monitor.track_request.assert_called_once()
            call_args = mock_monitor.track_request.call_args[1]
            assert call_args["component_type"] == ComponentType.WORKFLOW
            assert call_args["component_id"] == "memory_workflow"
            assert call_args["duration"] == 5.0
            assert call_args["success"] is True
            assert call_args["metadata"]["steps"] == 5

    def test_track_request(self):
        """Test track_request helper function"""
        with patch('app.core.langchain.monitoring.langchain_monitor') as mock_monitor:
            tracker = RequestTracker(
                monitor=mock_monitor,
                component_type=ComponentType.LLM,
                component_id="gpt-4",
                metadata={"request_id": "req-123"}
            )
            
            result = track_request(
                ComponentType.LLM,
                "gpt-4",
                metadata={"request_id": "req-123"}
            )
            
            assert isinstance(result, RequestTracker)
            assert result.monitor == mock_monitor
            assert result.component_type == ComponentType.LLM
            assert result.component_id == "gpt-4"
            assert result.metadata["request_id"] == "req-123"


class TestGlobalMonitor:
    """Test global monitor instance"""

    def test_global_instance(self):
        """Test that global monitor instance exists"""
        from app.core.langchain.monitoring import langchain_monitor
        assert langchain_monitor is not None
        assert isinstance(langchain_monitor, LangChainMonitor)