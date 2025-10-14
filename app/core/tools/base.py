"""
Base tool interface and abstract classes for the AI Assistant tool system.

This module defines the core abstractions for tools, including the BaseTool
abstract class, ToolResult data structure, and tool-specific exceptions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import asyncio
import time
import logging

try:
    from app.core.monitoring.metrics import metrics_collector
    from app.core.monitoring.config import monitoring_config
except ImportError:
    # Fallback for when monitoring is not available
    metrics_collector = None  # type: ignore
    monitoring_config = None  # type: ignore

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Standardized result from tool execution"""

    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str
    execution_time: float
    metadata: Dict[str, Any] = {}


class ToolError(Exception):
    """Base exception for tool-related errors"""

    pass


class ToolTimeoutError(ToolError):
    """Tool execution timed out"""

    pass


class ToolConfigurationError(ToolError):
    """Tool is misconfigured"""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed"""

    pass


class BaseTool(ABC):
    """Abstract base class for all tools"""

    def __init__(self):
        self._enabled = True
        self._timeout = 30  # seconds
        self._last_used = None
        self._usage_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the tool's purpose"""
        pass

    @property
    def version(self) -> str:
        """Tool version (default: 1.0.0)"""
        return "1.0.0"

    @property
    def author(self) -> str:
        """Tool author/organization"""
        return "AI Assistant Team"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Expected parameters schema for the tool"""
        return {
            "query": {
                "type": str,
                "description": "The input query or question",
                "required": True,
            }
        }

    @property
    def keywords(self) -> List[str]:
        """Keywords that indicate when this tool should be used"""
        return []

    @property
    def categories(self) -> List[str]:
        """Tool categories for organization"""
        return ["general"]

    @property
    def timeout(self) -> int:
        """Maximum execution time in seconds"""
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        self._timeout = max(1, value)  # Minimum 1 second

    @property
    def enabled(self) -> bool:
        """Whether the tool is enabled"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result

        Raises:
            ToolError: If execution fails
        """
        pass

    async def execute_with_timeout(self, **kwargs) -> ToolResult:
        """Execute tool with timeout protection and monitoring integration"""
        start_time = time.time()
        success = False
        error_type = None

        try:
            # Set up timeout
            result = await asyncio.wait_for(
                self.execute(**kwargs), timeout=self.timeout
            )

            execution_time = time.time() - start_time
            self._last_used = time.time()
            self._usage_count += 1
            success = True

            return ToolResult(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time=execution_time,
                metadata={"execution_method": "direct"},
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Tool {self.name} timed out after {execution_time:.2f}s"
            error_type = "TimeoutError"
            logger.warning(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=self.name,
                execution_time=execution_time,
                metadata={"timeout": True},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool {self.name} failed: {str(e)}"
            error_type = type(e).__name__
            logger.error(error_msg)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=self.name,
                execution_time=execution_time,
                metadata={"exception_type": error_type},
            )

        finally:
            # Record metrics if monitoring is enabled
            if (
                metrics_collector
                and monitoring_config
                and monitoring_config.monitoring_enabled
                and monitoring_config.track_tool_performance
            ):
                execution_time = time.time() - start_time
                metrics_collector.record_tool_execution(
                    tool_name=self.name,
                    success=success,
                    execution_time=execution_time,
                    error_type=error_type,
                )

    def should_use(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this tool should be used for the given query

        Args:
            query: User query
            context: Additional context information

        Returns:
            Boolean indicating if tool should be used
        """
        if not self.enabled:
            return False

        # Basic keyword matching
        query_lower = query.lower()
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                return True

        return False

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "usage_count": self._usage_count,
            "last_used": self._last_used,
            "timeout": self.timeout,
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate tool parameters against expected schema"""
        expected_params = self.parameters

        for param_name, param_config in expected_params.items():
            if param_config.get("required", False) and param_name not in kwargs:
                return False

            if param_name in kwargs:
                param_value = kwargs[param_name]
                expected_type = param_config.get("type")

                if expected_type and not isinstance(param_value, expected_type):
                    return False

        return True
