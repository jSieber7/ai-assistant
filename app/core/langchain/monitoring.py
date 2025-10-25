"""
LangChain Monitoring and Observability System

This module provides comprehensive monitoring for LangChain components including:
- LLM performance metrics
- LangGraph workflow execution metrics
- Tool execution metrics
- Agent performance metrics
- Memory usage metrics
- Error tracking and logging
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

from app.core.config import settings
from app.core.secure_settings import secure_settings
from app.core.storage.langchain_client import get_langchain_client

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be tracked"""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class ComponentType(Enum):
    """LangChain component types"""
    
    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    WORKFLOW = "workflow"
    MEMORY = "memory"
    CHAIN = "chain"


@dataclass
class MetricEvent:
    """A single metric event"""
    
    name: str
    component_type: ComponentType
    component_id: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component"""
    
    component_id: str
    component_type: ComponentType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    average_duration: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    
    def update(self, duration: float, success: bool):
        """Update metrics with a new request"""
        self.total_requests += 1
        self.total_duration += duration
        self.last_request_time = datetime.now()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        # Update duration metrics
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.average_duration = self.total_duration / self.total_requests
        
        # Update error rate
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration": self.total_duration,
            "min_duration": self.min_duration if self.min_duration != float('inf') else 0.0,
            "max_duration": self.max_duration,
            "average_duration": self.average_duration,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "error_rate": self.error_rate
        }


class LangChainMonitor:
    """
    Comprehensive monitoring system for LangChain components.
    
    This monitor tracks performance metrics, errors, and usage patterns
    for all LangChain components in the system.
    """
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._events: List[MetricEvent] = []
        self._db_client = None
        self._initialized = False
        self._event_buffer_size = 1000
        self._flush_interval = 60  # seconds
        self._last_flush = datetime.now()
        
    async def initialize(self):
        """Initialize the monitoring system"""
        if self._initialized:
            return
            
        logger.info("Initializing LangChain Monitor...")
        
        # Initialize database client
        try:
            self._db_client = get_langchain_client()
            if self._db_client:
                logger.info("Initialized database client for monitoring")
            else:
                logger.warning("Failed to initialize database client for monitoring")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring database client: {str(e)}")
            
        self._initialized = True
        logger.info("LangChain Monitor initialized successfully")
        
    async def track_request(
        self,
        component_type: ComponentType,
        component_id: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a request to a LangChain component.
        
        Args:
            component_type: Type of component
            component_id: Unique identifier for the component
            duration: Request duration in seconds
            success: Whether the request was successful
            metadata: Additional metadata about the request
        """
        try:
            # Update performance metrics
            key = f"{component_type.value}:{component_id}"
            if key not in self._metrics:
                self._metrics[key] = PerformanceMetrics(
                    component_id=component_id,
                    component_type=component_type
                )
                
            self._metrics[key].update(duration, success)
            
            # Create metric event
            event = MetricEvent(
                name="request_duration",
                component_type=component_type,
                component_id=component_id,
                metric_type=MetricType.TIMER,
                value=duration,
                timestamp=datetime.now(),
                metadata={
                    "success": success,
                    **(metadata or {})
                }
            )
            
            self._events.append(event)
            
            # Flush events if buffer is full or interval has passed
            await self._maybe_flush_events()
            
        except Exception as e:
            logger.error(f"Failed to track request: {str(e)}")
            
    async def track_metric(
        self,
        name: str,
        component_type: ComponentType,
        component_id: str,
        metric_type: MetricType,
        value: Union[float, int],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a custom metric.
        
        Args:
            name: Name of the metric
            component_type: Type of component
            component_id: Unique identifier for the component
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata about the metric
        """
        try:
            event = MetricEvent(
                name=name,
                component_type=component_type,
                component_id=component_id,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self._events.append(event)
            
            # Flush events if buffer is full or interval has passed
            await self._maybe_flush_events()
            
        except Exception as e:
            logger.error(f"Failed to track metric: {str(e)}")
            
    async def track_error(
        self,
        component_type: ComponentType,
        component_id: str,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track an error from a LangChain component.
        
        Args:
            component_type: Type of component
            component_id: Unique identifier for the component
            error: The error that occurred
            metadata: Additional metadata about the error
        """
        try:
            # Track as failed request
            await self.track_request(
                component_type=component_type,
                component_id=component_id,
                duration=0.0,
                success=False,
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    **(metadata or {})
                }
            )
            
            # Log error to database if available
            if self._db_client:
                await self._db_client.log_performance(
                    component_type=component_type.value,
                    component_id=component_id,
                    metric_name="error",
                    metric_value=1.0,
                    metadata={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        **(metadata or {})
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to track error: {str(e)}")
            
    async def get_metrics(
        self,
        component_type: Optional[ComponentType] = None,
        component_id: Optional[str] = None
    ) -> List[PerformanceMetrics]:
        """
        Get performance metrics for components.
        
        Args:
            component_type: Filter by component type
            component_id: Filter by component ID
            
        Returns:
            List of performance metrics
        """
        metrics = list(self._metrics.values())
        
        # Apply filters
        if component_type:
            metrics = [m for m in metrics if m.component_type == component_type]
            
        if component_id:
            metrics = [m for m in metrics if m.component_id == component_id]
            
        return metrics
        
    async def get_events(
        self,
        component_type: Optional[ComponentType] = None,
        component_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MetricEvent]:
        """
        Get recent metric events.
        
        Args:
            component_type: Filter by component type
            component_id: Filter by component ID
            limit: Maximum number of events to return
            
        Returns:
            List of metric events
        """
        events = self._events[-limit:] if limit > 0 else self._events
        
        # Apply filters
        if component_type:
            events = [e for e in events if e.component_type == component_type]
            
        if component_id:
            events = [e for e in events if e.component_id == component_id]
            
        return events
        
    async def _maybe_flush_events(self):
        """Flush events to database if buffer is full or interval has passed"""
        if not self._db_client:
            return
            
        now = datetime.now()
        should_flush = (
            len(self._events) >= self._event_buffer_size or
            (now - self._last_flush).total_seconds() >= self._flush_interval
        )
        
        if should_flush:
            await self._flush_events()
            
    async def _flush_events(self):
        """Flush events to database"""
        if not self._db_client or not self._events:
            return
            
        try:
            # Batch insert events
            for event in self._events:
                await self._db_client.log_performance(
                    component_type=event.component_type.value,
                    component_id=event.component_id,
                    metric_name=event.name,
                    metric_value=event.value,
                    metadata=event.to_dict()
                )
                
            # Clear events
            self._events.clear()
            self._last_flush = datetime.now()
            
            logger.debug(f"Flushed {len(self._events)} events to database")
            
        except Exception as e:
            logger.error(f"Failed to flush events to database: {str(e)}")
            
    async def cleanup_old_metrics(self, days: int = 30):
        """
        Clean up old metrics data.
        
        Args:
            days: Age threshold in days
        """
        try:
            if self._db_client:
                await self._db_client.cleanup_old_metrics(days=days)
                logger.info(f"Cleaned up metrics older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {str(e)}")
            
    async def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all components"""
        stats = {
            "total_components": len(self._metrics),
            "total_requests": sum(m.total_requests for m in self._metrics.values()),
            "total_errors": sum(m.failed_requests for m in self._metrics.values()),
            "components_by_type": {},
            "average_response_times": {}
        }
        
        # Group by component type
        for metric in self._metrics.values():
            comp_type = metric.component_type.value
            if comp_type not in stats["components_by_type"]:
                stats["components_by_type"][comp_type] = 0
                stats["average_response_times"][comp_type] = []
                
            stats["components_by_type"][comp_type] += 1
            stats["average_response_times"][comp_type].append(metric.average_duration)
            
        # Calculate average response times by type
        for comp_type, times in stats["average_response_times"].items():
            stats["average_response_times"][comp_type] = sum(times) / len(times) if times else 0.0
            
        return stats


# Context manager for tracking requests
class RequestTracker:
    """Context manager for tracking LangChain component requests"""
    
    def __init__(
        self,
        monitor: LangChainMonitor,
        component_type: ComponentType,
        component_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.monitor = monitor
        self.component_type = component_type
        self.component_id = component_id
        self.metadata = metadata
        self.start_time = None
        self.success = True
        self.error = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0.0
        
        if exc_type is not None:
            self.success = False
            self.error = exc_val
            
        await self.monitor.track_request(
            component_type=self.component_type,
            component_id=self.component_id,
            duration=duration,
            success=self.success,
            metadata=self.metadata
        )
        
        # Don't suppress exceptions
        return False


# Global monitor instance
langchain_monitor = LangChainMonitor()


# Helper functions
async def track_llm_request(
    llm_id: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track an LLM request"""
    await langchain_monitor.track_request(
        component_type=ComponentType.LLM,
        component_id=llm_id,
        duration=duration,
        success=success,
        metadata=metadata
    )


async def track_tool_execution(
    tool_id: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track a tool execution"""
    await langchain_monitor.track_request(
        component_type=ComponentType.TOOL,
        component_id=tool_id,
        duration=duration,
        success=success,
        metadata=metadata
    )


async def track_agent_execution(
    agent_id: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track an agent execution"""
    await langchain_monitor.track_request(
        component_type=ComponentType.AGENT,
        component_id=agent_id,
        duration=duration,
        success=success,
        metadata=metadata
    )


async def track_workflow_execution(
    workflow_id: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track a workflow execution"""
    await langchain_monitor.track_request(
        component_type=ComponentType.WORKFLOW,
        component_id=workflow_id,
        duration=duration,
        success=success,
        metadata=metadata
    )


def track_request(component_type: ComponentType, component_id: str, metadata: Optional[Dict[str, Any]] = None):
    """Create a request tracker context manager"""
    return RequestTracker(
        monitor=langchain_monitor,
        component_type=component_type,
        component_id=component_id,
        metadata=metadata
    )