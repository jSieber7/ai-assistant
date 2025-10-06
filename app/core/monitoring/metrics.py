"""
Metrics collection system for the AI Assistant Tool System.

This module provides comprehensive metrics collection for:
- Tool execution metrics (success, failure, performance)
- Agent performance metrics
- API request metrics
- System health metrics
- Prometheus integration for observability
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry

from .config import monitoring_config

logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Metrics for individual tool execution"""

    tool_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    error_types: Dict[str, int] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time"""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100


@dataclass
class AgentMetrics:
    """Metrics for agent performance"""

    agent_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    tools_used: Dict[str, int] = None
    conversation_count: int = 0

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = {}

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time"""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100


class MetricsCollector:
    """
    Central metrics collector for the tool system.

    This class collects and aggregates metrics from tools, agents, and API endpoints.
    It provides Prometheus integration for external monitoring systems.
    """

    def __init__(self):
        self._tool_metrics: Dict[str, ToolMetrics] = {}
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._api_metrics: Dict[str, Any] = {}
        self._system_metrics: Dict[str, Any] = {}

        # Prometheus metrics
        self._registry = CollectorRegistry()
        self._setup_prometheus_metrics()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=5)

        # Metrics collection settings
        self._metrics_retention_hours = monitoring_config.metrics_retention_days * 24

        logger.info("MetricsCollector initialized")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Tool metrics
        self.tool_execution_counter = Counter(
            "tool_execution_total",
            "Total number of tool executions",
            ["tool_name", "status"],
            registry=self._registry,
        )

        self.tool_execution_duration = Histogram(
            "tool_execution_duration_seconds",
            "Tool execution duration in seconds",
            ["tool_name"],
            registry=self._registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Agent metrics
        self.agent_execution_counter = Counter(
            "agent_execution_total",
            "Total number of agent executions",
            ["agent_name", "status"],
            registry=self._registry,
        )

        self.agent_execution_duration = Histogram(
            "agent_execution_duration_seconds",
            "Agent execution duration in seconds",
            ["agent_name"],
            registry=self._registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # API metrics
        self.api_request_counter = Counter(
            "api_request_total",
            "Total number of API requests",
            ["endpoint", "method", "status_code"],
            registry=self._registry,
        )

        self.api_request_duration = Histogram(
            "api_request_duration_seconds",
            "API request duration in seconds",
            ["endpoint", "method"],
            registry=self._registry,
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # System metrics
        self.system_uptime = Gauge(
            "system_uptime_seconds", "System uptime in seconds", registry=self._registry
        )

        self.active_tools = Gauge(
            "active_tools_total", "Number of active tools", registry=self._registry
        )

        self.active_agents = Gauge(
            "active_agents_total", "Number of active agents", registry=self._registry
        )

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        error_type: str = None,
    ):
        """Record tool execution metrics"""
        if not monitoring_config.monitoring_enabled:
            return

        # Update internal metrics
        if tool_name not in self._tool_metrics:
            self._tool_metrics[tool_name] = ToolMetrics(tool_name=tool_name)

        metrics = self._tool_metrics[tool_name]
        metrics.execution_count += 1
        metrics.total_execution_time += execution_time
        metrics.last_execution_time = datetime.now()

        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            if error_type:
                metrics.error_types[error_type] = (
                    metrics.error_types.get(error_type, 0) + 1
                )

        # Update Prometheus metrics
        status = "success" if success else "failure"
        self.tool_execution_counter.labels(tool_name=tool_name, status=status).inc()
        self.tool_execution_duration.labels(tool_name=tool_name).observe(execution_time)

        if monitoring_config.log_metrics:
            logger.debug(
                f"Recorded tool execution: {tool_name}, success: {success}, "
                f"time: {execution_time:.3f}s"
            )

    def record_agent_execution(
        self,
        agent_name: str,
        success: bool,
        execution_time: float,
        tools_used: List[str] = None,
    ):
        """Record agent execution metrics"""
        if not monitoring_config.monitoring_enabled:
            return

        # Update internal metrics
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        metrics = self._agent_metrics[agent_name]
        metrics.execution_count += 1
        metrics.total_execution_time += execution_time

        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1

        # Track tools used
        if tools_used:
            for tool_name in tools_used:
                metrics.tools_used[tool_name] = metrics.tools_used.get(tool_name, 0) + 1

        # Update Prometheus metrics
        status = "success" if success else "failure"
        self.agent_execution_counter.labels(agent_name=agent_name, status=status).inc()
        self.agent_execution_duration.labels(agent_name=agent_name).observe(
            execution_time
        )

        if monitoring_config.log_metrics:
            logger.debug(
                f"Recorded agent execution: {agent_name}, success: {success}, "
                f"time: {execution_time:.3f}s"
            )

    def record_api_request(
        self, endpoint: str, method: str, status_code: int, duration: float
    ):
        """Record API request metrics"""
        if not monitoring_config.monitoring_enabled:
            return

        # Update Prometheus metrics
        self.api_request_counter.labels(
            endpoint=endpoint, method=method.upper(), status_code=str(status_code)
        ).inc()

        self.api_request_duration.labels(
            endpoint=endpoint, method=method.upper()
        ).observe(duration)

        if monitoring_config.log_metrics:
            logger.debug(
                f"Recorded API request: {method} {endpoint}, "
                f"status: {status_code}, duration: {duration:.3f}s"
            )

    def update_system_metrics(
        self, active_tools: int, active_agents: int, uptime_seconds: float
    ):
        """Update system-level metrics"""
        if not monitoring_config.monitoring_enabled:
            return

        self.active_tools.set(active_tools)
        self.active_agents.set(active_agents)
        self.system_uptime.set(uptime_seconds)

    def get_tool_metrics(self, tool_name: str = None) -> Dict[str, Any]:
        """Get metrics for specific tool or all tools"""
        if tool_name:
            metrics = self._tool_metrics.get(tool_name)
            return metrics.__dict__ if metrics else {}

        return {name: metrics.__dict__ for name, metrics in self._tool_metrics.items()}

    def get_agent_metrics(self, agent_name: str = None) -> Dict[str, Any]:
        """Get metrics for specific agent or all agents"""
        if agent_name:
            metrics = self._agent_metrics.get(agent_name)
            return metrics.__dict__ if metrics else {}

        return {name: metrics.__dict__ for name, metrics in self._agent_metrics.items()}

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide metrics summary"""
        total_tool_executions = sum(
            m.execution_count for m in self._tool_metrics.values()
        )
        total_agent_executions = sum(
            m.execution_count for m in self._agent_metrics.values()
        )

        tool_success_rate = 0.0
        if total_tool_executions > 0:
            total_success = sum(m.success_count for m in self._tool_metrics.values())
            tool_success_rate = (total_success / total_tool_executions) * 100

        agent_success_rate = 0.0
        if total_agent_executions > 0:
            total_success = sum(m.success_count for m in self._agent_metrics.values())
            agent_success_rate = (total_success / total_agent_executions) * 100

        return {
            "total_tool_executions": total_tool_executions,
            "total_agent_executions": total_agent_executions,
            "tool_success_rate": round(tool_success_rate, 2),
            "agent_success_rate": round(agent_success_rate, 2),
            "unique_tools_used": len(self._tool_metrics),
            "unique_agents_used": len(self._agent_metrics),
            "metrics_collection_enabled": monitoring_config.monitoring_enabled,
            "collection_level": monitoring_config.monitoring_level.value,
        }

    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus metrics in text format"""
        return generate_latest(self._registry)

    def cleanup_old_metrics(self):
        """Clean up old metrics data (placeholder for future implementation)"""
        # This would remove metrics older than retention period
        # For now, we'll keep all metrics in memory
        pass

    @contextmanager
    def measure_tool_execution(self, tool_name: str):
        """Context manager for measuring tool execution time"""
        start_time = time.time()
        success = False
        error_type = None

        try:
            yield
            success = True
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            execution_time = time.time() - start_time
            self.record_tool_execution(tool_name, success, execution_time, error_type)

    @contextmanager
    def measure_agent_execution(self, agent_name: str):
        """Context manager for measuring agent execution time"""
        start_time = time.time()
        success = False
        tools_used = []

        try:
            yield tools_used
            success = True
        except Exception:
            raise
        finally:
            execution_time = time.time() - start_time
            self.record_agent_execution(agent_name, success, execution_time, tools_used)


# Global metrics collector instance
metrics_collector = MetricsCollector()
