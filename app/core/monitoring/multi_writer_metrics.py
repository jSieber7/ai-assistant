"""
Monitoring and metrics for multi-writer/checker system
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Create a custom registry for multi-writer metrics
multi_writer_registry = CollectorRegistry()

# Workflow metrics
WORKFLOW_STARTED = Counter(
    'multi_writer_workflows_started_total',
    'Total workflows started',
    registry=multi_writer_registry
)

WORKFLOW_COMPLETED = Counter(
    'multi_writer_workflows_completed_total',
    'Total workflows completed',
    registry=multi_writer_registry
)

WORKFLOW_FAILED = Counter(
    'multi_writer_workflows_failed_total',
    'Total workflows failed',
    registry=multi_writer_registry
)

# Stage timing metrics
SOURCE_PROCESSING_TIME = Histogram(
    'multi_writer_source_processing_seconds',
    'Time spent processing sources',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=multi_writer_registry
)

CONTENT_GENERATION_TIME = Histogram(
    'multi_writer_content_generation_seconds',
    'Time spent generating content',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=multi_writer_registry
)

QUALITY_CHECKING_TIME = Histogram(
    'multi_writer_quality_checking_seconds',
    'Time spent checking content',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=multi_writer_registry
)

TEMPLATE_RENDERING_TIME = Histogram(
    'multi_writer_template_rendering_seconds',
    'Time spent rendering templates',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=multi_writer_registry
)

# Quality metrics
QUALITY_SCORES = Histogram(
    'multi_writer_quality_scores',
    'Content quality scores',
    buckets=[50, 60, 70, 80, 90, 95, 100],
    registry=multi_writer_registry
)

WRITER_CONFIDENCE_SCORES = Histogram(
    'multi_writer_writer_confidence_scores',
    'Writer confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=multi_writer_registry
)

# System metrics
ACTIVE_WORKFLOWS = Gauge(
    'multi_writer_active_workflows',
    'Number of active workflows',
    registry=multi_writer_registry
)

FIRECRAWL_REQUESTS = Counter(
    'multi_writer_firecrawl_requests_total',
    'Total Firecrawl API requests',
    ['status'],  # success/failure
    registry=multi_writer_registry
)

FIRECRAWL_REQUEST_TIME = Histogram(
    'multi_writer_firecrawl_request_seconds',
    'Time spent on Firecrawl API requests',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=multi_writer_registry
)

# Content metrics
CONTENT_GENERATED = Counter(
    'multi_writer_content_generated_total',
    'Total content pieces generated',
    ['writer_specialty'],
    registry=multi_writer_registry
)

CHECKS_PERFORMED = Counter(
    'multi_writer_checks_performed_total',
    'Total checks performed',
    ['checker_focus'],
    registry=multi_writer_registry
)

# Storage metrics
DATABASE_OPERATIONS = Counter(
    'multi_writer_database_operations_total',
    'Total database operations',
    ['operation', 'status'],  # save/update/delete, success/failure
    registry=multi_writer_registry
)

DATABASE_OPERATION_TIME = Histogram(
    'multi_writer_database_operation_seconds',
    'Time spent on database operations',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=multi_writer_registry
)

class MultiWriterMetricsCollector:
    """Collects and manages metrics for the multi-writer system"""
    
    def __init__(self):
        self._active_workflows = set()
        self._workflow_start_times = {}
    
    def start_workflow(self, workflow_id: str):
        """Record workflow start"""
        WORKFLOW_STARTED.inc()
        self._active_workflows.add(workflow_id)
        self._workflow_start_times[workflow_id] = time.time()
        ACTIVE_WORKFLOWS.set(len(self._active_workflows))
    
    def complete_workflow(self, workflow_id: str, quality_score: Optional[float] = None):
        """Record workflow completion"""
        if workflow_id in self._active_workflows:
            self._active_workflows.remove(workflow_id)
            ACTIVE_WORKFLOWS.set(len(self._active_workflows))
        
        WORKFLOW_COMPLETED.inc()
        
        if quality_score is not None:
            QUALITY_SCORES.observe(quality_score)
        
        # Clean up start time
        if workflow_id in self._workflow_start_times:
            del self._workflow_start_times[workflow_id]
    
    def fail_workflow(self, workflow_id: str):
        """Record workflow failure"""
        if workflow_id in self._active_workflows:
            self._active_workflows.remove(workflow_id)
            ACTIVE_WORKFLOWS.set(len(self._active_workflows))
        
        WORKFLOW_FAILED.inc()
        
        # Clean up start time
        if workflow_id in self._workflow_start_times:
            del self._workflow_start_times[workflow_id]
    
    def record_source_processing_time(self, duration: float):
        """Record source processing time"""
        SOURCE_PROCESSING_TIME.observe(duration)
    
    def record_content_generation_time(self, duration: float):
        """Record content generation time"""
        CONTENT_GENERATION_TIME.observe(duration)
    
    def record_quality_checking_time(self, duration: float):
        """Record quality checking time"""
        QUALITY_CHECKING_TIME.observe(duration)
    
    def record_template_rendering_time(self, duration: float):
        """Record template rendering time"""
        TEMPLATE_RENDERING_TIME.observe(duration)
    
    def record_writer_confidence(self, specialty: str, confidence: float):
        """Record writer confidence score"""
        WRITER_CONFIDENCE_SCORES.observe(confidence)
        CONTENT_GENERATED.labels(writer_specialty=specialty).inc()
    
    def record_check_performed(self, focus_area: str):
        """Record check performed"""
        CHECKS_PERFORMED.labels(checker_focus=focus_area).inc()
    
    def record_firecrawl_request(self, success: bool, duration: float):
        """Record Firecrawl API request"""
        status = "success" if success else "failure"
        FIRECRAWL_REQUESTS.labels(status=status).inc()
        FIRECRAWL_REQUEST_TIME.observe(duration)
    
    def record_database_operation(self, operation: str, success: bool, duration: float):
        """Record database operation"""
        status = "success" if success else "failure"
        DATABASE_OPERATIONS.labels(operation=operation, status=status).inc()
        DATABASE_OPERATION_TIME.observe(duration)
    
    def get_workflow_duration(self, workflow_id: str) -> Optional[float]:
        """Get workflow duration if it's still active"""
        if workflow_id in self._workflow_start_times:
            return time.time() - self._workflow_start_times[workflow_id]
        return None
    
    def get_active_workflow_count(self) -> int:
        """Get number of active workflows"""
        return len(self._active_workflows)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        # This would typically be used for debugging or health checks
        return {
            "active_workflows": self.get_active_workflow_count(),
            "workflows_started": WORKFLOW_STARTED._value.get(),
            "workflows_completed": WORKFLOW_COMPLETED._value.get(),
            "workflows_failed": WORKFLOW_FAILED._value.get(),
            "firecrawl_requests": dict(FIRECRAWL_REQUESTS._value),
            "content_generated": dict(CONTENT_GENERATED._value),
            "checks_performed": dict(CHECKS_PERFORMED._value),
        }

# Global metrics collector instance
metrics_collector = MultiWriterMetricsCollector()

# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metric_func):
        self.metric_func = metric_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metric_func(duration)

# Decorator for timing functions
def timed(metric_func):
    """Decorator to time function execution"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with Timer(metric_func):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with Timer(metric_func):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Decorator for counting operations
def counted(counter, labels=None):
    """Decorator to count function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                return result
            except Exception:
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                return result
            except Exception:
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator