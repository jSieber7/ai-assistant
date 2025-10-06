"""
Monitoring middleware for FastAPI to track API requests and responses.

This module provides middleware that automatically tracks:
- Request/response timing
- Status codes
- Error rates
- API performance metrics
"""

import time
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import metrics_collector
from .config import monitoring_config

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests and responses.

    This middleware tracks request timing, status codes, and other metrics
    for all API endpoints.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._excluded_paths = {
            "/metrics",  # Prometheus metrics endpoint
            "/health",  # Health check endpoint
            "/docs",  # API documentation
            "/redoc",  # Alternative API documentation
            "/favicon.ico",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process incoming requests and track metrics"""

        # Skip monitoring for excluded paths
        if request.url.path in self._excluded_paths:
            return await call_next(request)

        # Skip if monitoring is disabled
        if not monitoring_config.monitoring_enabled:
            return await call_next(request)

        start_time = time.time()
        response = None

        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            success = status_code < 400

        except Exception as e:
            # Handle exceptions and track as errors
            status_code = 500
            success = False
            logger.error(f"Request failed: {request.method} {request.url.path} - {e}")
            raise

        finally:
            # Always record metrics, even if an exception occurred
            if response or not success:
                duration = time.time() - start_time

                # Record API metrics
                metrics_collector.record_api_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    duration=duration,
                )

                if monitoring_config.log_metrics:
                    logger.info(
                        f"API Request: {request.method} {request.url.path} - "
                        f"Status: {status_code} - Duration: {duration:.3f}s"
                    )

        return response


class InstrumentedAPIRoute(APIRoute):
    """
    Custom APIRoute that instruments route execution for detailed monitoring.

    This provides more granular monitoring at the route level, including
    tracking of individual endpoint performance.
    """

    def get_route_handler(self) -> Callable:
        """Get the route handler with instrumentation"""
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            start_time = time.time()
            response = None

            try:
                # Process the request
                response = await original_route_handler(request)
                status_code = response.status_code
                success = status_code < 400

            except Exception as e:
                status_code = 500
                success = False
                logger.error(f"Route handler failed: {self.path} - {e}")
                raise

            finally:
                # Record route-specific metrics
                if monitoring_config.monitoring_enabled:
                    duration = time.time() - start_time

                    # Use the route name as the endpoint identifier
                    endpoint_name = self.name or self.path

                    metrics_collector.record_api_request(
                        endpoint=endpoint_name,
                        method=self.methods[0] if self.methods else "UNKNOWN",
                        status_code=status_code,
                        duration=duration,
                    )

                    if (
                        monitoring_config.log_metrics
                        and monitoring_config.monitoring_level.value == "detailed"
                    ):
                        logger.debug(
                            f"Route Instrumentation: {endpoint_name} - "
                            f"Status: {status_code} - Duration: {duration:.3f}s"
                        )

            return response

        return custom_route_handler


class ToolExecutionMonitor:
    """
    Monitor for tracking tool execution within the system.

    This class provides decorators and context managers for monitoring
    tool and agent executions.
    """

    def __init__(self):
        self._active_executions: Dict[str, Dict[str, Any]] = {}

    def monitor_tool_execution(self, tool_name: str):
        """
        Decorator for monitoring tool execution.

        Usage:
            @monitor.monitor_tool_execution("my_tool")
            async def my_tool_function(*args, **kwargs):
                # tool logic here
        """

        def decorator(func):
            async def wrapper(*args, **kwargs):
                with metrics_collector.measure_tool_execution(tool_name):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def monitor_agent_execution(self, agent_name: str):
        """
        Decorator for monitoring agent execution.

        Usage:
            @monitor.monitor_agent_execution("my_agent")
            async def my_agent_function(*args, **kwargs):
                # agent logic here
        """

        def decorator(func):
            async def wrapper(*args, **kwargs):
                tools_used = []
                with metrics_collector.measure_agent_execution(
                    agent_name
                ) as tracked_tools:
                    # The tracked_tools list will be populated by tool executions
                    result = await func(*args, **kwargs)
                    # You can manually add tools to tracked_tools if needed
                    return result

            return wrapper

        return decorator

    def start_tool_execution(self, tool_name: str, context: Dict[str, Any] = None):
        """
        Start tracking a tool execution (manual instrumentation).

        Returns a context manager that automatically records metrics.
        """
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        self._active_executions[execution_id] = {
            "tool_name": tool_name,
            "start_time": time.time(),
            "context": context or {},
        }

        return execution_id

    def end_tool_execution(self, execution_id: str, success: bool, error: str = None):
        """
        End tracking of a tool execution (manual instrumentation).
        """
        if execution_id not in self._active_executions:
            logger.warning(f"Unknown execution ID: {execution_id}")
            return

        execution = self._active_executions.pop(execution_id)
        duration = time.time() - execution["start_time"]

        metrics_collector.record_tool_execution(
            tool_name=execution["tool_name"],
            success=success,
            execution_time=duration,
            error_type=error,
        )

    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active tool executions"""
        return self._active_executions.copy()


# Global monitoring instances
monitoring_middleware = MonitoringMiddleware
instrumented_route = InstrumentedAPIRoute
tool_monitor = ToolExecutionMonitor()
