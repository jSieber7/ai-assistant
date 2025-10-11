"""
Performance monitoring and metrics for the caching system.

This module provides monitoring capabilities to track cache performance,
collect metrics, and generate reports on caching effectiveness.
"""

import time
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Metrics for cache performance tracking."""

    # Basic metrics
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

    # Timing metrics (in seconds)
    total_get_time: float = 0.0
    total_set_time: float = 0.0
    total_delete_time: float = 0.0

    # Size metrics
    current_size: int = 0
    max_size: int = 0
    memory_usage: int = 0

    # Derived metrics (computed properties)
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0

    @property
    def average_get_time(self) -> float:
        """Calculate average get operation time."""
        total_operations = self.hits + self.misses
        return self.total_get_time / total_operations if total_operations > 0 else 0.0

    @property
    def average_set_time(self) -> float:
        """Calculate average set operation time."""
        return self.total_set_time / self.sets if self.sets > 0 else 0.0

    @property
    def utilization_rate(self) -> float:
        """Calculate cache utilization rate."""
        return self.current_size / self.max_size if self.max_size > 0 else 0.0

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_get_time = 0.0
        self.total_set_time = 0.0
        self.total_delete_time = 0.0


@dataclass
class PerformanceSample:
    """A single performance sample."""

    timestamp: float
    operation: str
    duration: float
    success: bool
    key: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None


class PerformanceMonitor:
    """
    Performance monitor for caching operations.

    Collects detailed performance metrics and provides real-time
    monitoring and alerting capabilities.
    """

    def __init__(
        self,
        max_samples: int = 1000,
        sample_window: int = 60,  # seconds
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the performance monitor.

        Args:
            max_samples: Maximum number of samples to keep
            sample_window: Time window for sample retention (seconds)
            alert_thresholds: Thresholds for performance alerts
        """
        self.max_samples = max_samples
        self.sample_window = sample_window

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "high_latency": 1.0,  # 1 second
            "low_hit_rate": 0.3,  # 30% hit rate
            "high_error_rate": 0.1,  # 10% error rate
        }

        # Metrics storage
        self.metrics = CacheMetrics()
        self.samples: deque[PerformanceSample] = deque(maxlen=max_samples)
        self._lock = asyncio.Lock()

        # Alert handlers
        self.alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

        # Historical data for trend analysis
        self.historical_metrics: deque[CacheMetrics] = deque(maxlen=100)

    async def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        key: Optional[str] = None,
        size: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a cache operation.

        Args:
            operation: Type of operation (get, set, delete, etc.)
            duration: Operation duration in seconds
            success: Whether the operation was successful
            key: Cache key involved
            size: Size of data involved (for set operations)
            error: Error message if operation failed
        """
        sample = PerformanceSample(
            timestamp=time.time(),
            operation=operation,
            duration=duration,
            success=success,
            key=key,
            size=size,
            error=error,
        )

        async with self._lock:
            self.samples.append(sample)

            # Update metrics
            if operation == "get":
                if success:
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1
                self.metrics.total_get_time += duration
            elif operation == "set":
                self.metrics.sets += 1
                self.metrics.total_set_time += duration
            elif operation == "delete":
                self.metrics.deletes += 1
                self.metrics.total_delete_time += duration

            if not success:
                self.metrics.errors += 1

            # Check for alerts
            await self._check_alerts()

    async def update_size_metrics(
        self, current_size: int, max_size: int, memory_usage: int = 0
    ) -> None:
        """
        Update cache size metrics.

        Args:
            current_size: Current number of items in cache
            max_size: Maximum cache size
            memory_usage: Memory usage in bytes (optional)
        """
        async with self._lock:
            self.metrics.current_size = current_size
            self.metrics.max_size = max_size
            self.metrics.memory_usage = memory_usage

    async def get_recent_metrics(
        self, window_seconds: Optional[int] = None
    ) -> CacheMetrics:
        """
        Get metrics for a recent time window.

        Args:
            window_seconds: Time window in seconds (uses sample_window if None)

        Returns:
            Metrics for the specified time window
        """
        window = window_seconds or self.sample_window
        cutoff_time = time.time() - window

        recent_samples = [
            sample for sample in self.samples if sample.timestamp >= cutoff_time
        ]

        metrics = CacheMetrics()

        for sample in recent_samples:
            if sample.operation == "get":
                if sample.success:
                    metrics.hits += 1
                else:
                    metrics.misses += 1
                metrics.total_get_time += sample.duration
            elif sample.operation == "set":
                metrics.sets += 1
                metrics.total_set_time += sample.duration
            elif sample.operation == "delete":
                metrics.deletes += 1
                metrics.total_delete_time += sample.duration

            if not sample.success:
                metrics.errors += 1

        return metrics

    def get_detailed_stats(
        self, window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Dictionary with detailed statistics
        """
        window = window_seconds or self.sample_window
        cutoff_time = time.time() - window

        recent_samples = [
            sample for sample in self.samples if sample.timestamp >= cutoff_time
        ]

        if not recent_samples:
            return {}

        # Basic counts
        operations_by_type = defaultdict(list)
        for sample in recent_samples:
            operations_by_type[sample.operation].append(sample)

        stats = {
            "total_operations": len(recent_samples),
            "operations_by_type": {
                op: len(samples) for op, samples in operations_by_type.items()
            },
            "success_rate": len([s for s in recent_samples if s.success])
            / len(recent_samples),
        }

        # Timing statistics
        for op, samples in operations_by_type.items():
            if samples:
                durations = [s.duration for s in samples]
                stats[f"{op}_timing"] = {
                    "count": len(samples),
                    "min": min(durations),
                    "max": max(durations),
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "p95": (
                        statistics.quantiles(durations, n=20)[18]
                        if len(durations) >= 20
                        else max(durations)
                    ),
                    "p99": (
                        statistics.quantiles(durations, n=100)[98]
                        if len(durations) >= 100
                        else max(durations)
                    ),
                }

        # Error analysis
        error_samples = [s for s in recent_samples if not s.success]
        if error_samples:
            errors_by_type = defaultdict(int)
            for sample in error_samples:
                errors_by_type[sample.error or "unknown"] += 1

            stats["error_analysis"] = {
                "total_errors": len(error_samples),
                "error_rate": len(error_samples) / len(recent_samples),
                "errors_by_type": dict(errors_by_type),
            }

        return stats

    async def _check_alerts(self) -> None:
        """Check for performance alerts and trigger handlers."""
        recent_metrics = await self.get_recent_metrics(60)  # 1-minute window

        alerts = []

        # Check latency alert
        if recent_metrics.average_get_time > self.alert_thresholds["high_latency"]:
            alerts.append(
                {
                    "type": "high_latency",
                    "message": f"High cache latency detected: {recent_metrics.average_get_time:.3f}s",
                    "value": recent_metrics.average_get_time,
                    "threshold": self.alert_thresholds["high_latency"],
                }
            )

        # Check hit rate alert
        if recent_metrics.hit_rate < self.alert_thresholds["low_hit_rate"]:
            alerts.append(
                {
                    "type": "low_hit_rate",
                    "message": f"Low cache hit rate detected: {recent_metrics.hit_rate:.1%}",
                    "value": recent_metrics.hit_rate,
                    "threshold": self.alert_thresholds["low_hit_rate"],
                }
            )

        # Check error rate alert
        error_rate = recent_metrics.errors / (
            recent_metrics.hits
            + recent_metrics.misses
            + recent_metrics.sets
            + recent_metrics.deletes
        )
        if error_rate > self.alert_thresholds["high_error_rate"]:
            alerts.append(
                {
                    "type": "high_error_rate",
                    "message": f"High cache error rate detected: {error_rate:.1%}",
                    "value": error_rate,
                    "threshold": self.alert_thresholds["high_error_rate"],
                }
            )

        # Trigger alert handlers
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert["type"], alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def remove_alert_handler(
        self, handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Remove an alert handler function."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

    async def reset_metrics(self) -> None:
        """Reset all metrics and samples."""
        async with self._lock:
            self.metrics.reset()
            self.samples.clear()
            self.historical_metrics.clear()

    async def save_historical_snapshot(self) -> None:
        """Save current metrics as a historical snapshot."""
        async with self._lock:
            # Create a copy of current metrics
            snapshot = CacheMetrics()
            snapshot.hits = self.metrics.hits
            snapshot.misses = self.metrics.misses
            snapshot.sets = self.metrics.sets
            snapshot.deletes = self.metrics.deletes
            snapshot.errors = self.metrics.errors
            snapshot.total_get_time = self.metrics.total_get_time
            snapshot.total_set_time = self.metrics.total_set_time
            snapshot.total_delete_time = self.metrics.total_delete_time

            self.historical_metrics.append(snapshot)

    def get_historical_trends(self) -> Dict[str, List[float]]:
        """Get historical trends for key metrics."""
        trends = {
            "hit_rate": [],
            "average_get_time": [],
            "utilization_rate": [],
        }

        for metrics in self.historical_metrics:
            trends["hit_rate"].append(metrics.hit_rate)
            trends["average_get_time"].append(metrics.average_get_time)
            trends["utilization_rate"].append(metrics.utilization_rate)

        return trends


class CacheMetricsCollector:
    """
    Collector for aggregating metrics from multiple cache instances.

    Provides a centralized way to collect and analyze metrics
    from different cache layers and components.
    """

    def __init__(self):
        self.monitors: Dict[str, PerformanceMonitor] = {}
        self._lock = asyncio.Lock()

    async def register_monitor(self, name: str, monitor: PerformanceMonitor) -> None:
        """Register a performance monitor."""
        async with self._lock:
            self.monitors[name] = monitor

    async def unregister_monitor(self, name: str) -> None:
        """Unregister a performance monitor."""
        async with self._lock:
            if name in self.monitors:
                del self.monitors[name]

    async def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all registered monitors."""
        async with self._lock:
            aggregated = {
                "total_operations": 0,
                "total_hits": 0,
                "total_misses": 0,
                "total_sets": 0,
                "total_deletes": 0,
                "total_errors": 0,
                "monitors": {},
            }

            for name, monitor in self.monitors.items():
                metrics = monitor.metrics
                aggregated["total_operations"] += (
                    metrics.hits + metrics.misses + metrics.sets + metrics.deletes
                )
                aggregated["total_hits"] += metrics.hits
                aggregated["total_misses"] += metrics.misses
                aggregated["total_sets"] += metrics.sets
                aggregated["total_deletes"] += metrics.deletes
                aggregated["total_errors"] += metrics.errors

                aggregated["monitors"][name] = {
                    "hits": metrics.hits,
                    "misses": metrics.misses,
                    "sets": metrics.sets,
                    "deletes": metrics.deletes,
                    "errors": metrics.errors,
                    "hit_rate": metrics.hit_rate,
                    "average_get_time": metrics.average_get_time,
                    "utilization_rate": metrics.utilization_rate,
                }

            # Calculate overall hit rate
            total_requests = aggregated["total_hits"] + aggregated["total_misses"]
            aggregated["overall_hit_rate"] = (
                aggregated["total_hits"] / total_requests if total_requests > 0 else 0.0
            )

            return aggregated

    async def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        aggregated = await self.get_aggregated_metrics()

        report = {
            "summary": {
                "total_monitors": len(self.monitors),
                "total_operations": aggregated["total_operations"],
                "overall_hit_rate": aggregated["overall_hit_rate"],
                "total_errors": aggregated["total_errors"],
            },
            "monitor_details": aggregated["monitors"],
            "recommendations": await self._generate_recommendations(aggregated),
        }

        return report

    async def _generate_recommendations(self, aggregated: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        overall_hit_rate = aggregated["overall_hit_rate"]
        total_errors = aggregated["total_errors"]
        total_operations = aggregated["total_operations"]

        if overall_hit_rate < 0.3:
            recommendations.append(
                "Consider increasing cache TTL or implementing more aggressive caching strategies"
            )

        if overall_hit_rate > 0.8:
            recommendations.append(
                "Cache performance is excellent - consider reducing cache size to save memory"
            )

        error_rate = total_errors / total_operations if total_operations > 0 else 0.0
        if error_rate > 0.05:
            recommendations.append(
                "High error rate detected - investigate cache backend connectivity and configuration"
            )

        return recommendations


# Global metrics collector instance
_metrics_collector: Optional[CacheMetricsCollector] = None


async def get_metrics_collector() -> CacheMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = CacheMetricsCollector()
    return _metrics_collector


async def reset_global_metrics() -> None:
    """Reset all global metrics."""
    global _metrics_collector
    if _metrics_collector is not None:
        for monitor in _metrics_collector.monitors.values():
            await monitor.reset_metrics()
