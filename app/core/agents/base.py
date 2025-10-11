"""
Base agent interface and abstract classes for the AI Assistant agent system.

This module defines the core abstractions for agents, including the BaseAgent
abstract class, AgentResult data structure, and agent-specific exceptions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel
import time
import logging
import uuid

from ..tools.base import ToolResult
from ..tools.registry import ToolRegistry

try:
    from app.core.monitoring.metrics import metrics_collector
    from app.core.monitoring.config import monitoring_config
except ImportError:
    # Fallback for when monitoring is not available
    metrics_collector = None
    monitoring_config = None

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Standardized result from agent execution"""

    success: bool
    response: str
    tool_results: List[ToolResult] = []
    error: Optional[str] = None
    agent_name: str
    execution_time: float
    metadata: Dict[str, Any] = {}
    conversation_id: Optional[str] = None


class AgentError(Exception):
    """Base exception for agent-related errors"""

    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out"""

    pass


class AgentConfigurationError(AgentError):
    """Agent is misconfigured"""

    pass


class AgentExecutionError(AgentError):
    """Agent execution failed"""

    pass


class AgentState(Enum):
    """Agent execution states"""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    RESPONDING = "responding"
    ERROR = "error"


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, tool_registry: ToolRegistry, max_iterations: int = 5):
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._state = AgentState.IDLE
        self._conversation_history: List[Dict[str, Any]] = []
        self._current_conversation_id: Optional[str] = None
        self._usage_count = 0
        self._last_used = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the agent"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the agent's purpose"""
        pass

    @property
    def version(self) -> str:
        """Agent version (default: 1.0.0)"""
        return "1.0.0"

    @property
    def state(self) -> AgentState:
        """Current agent state"""
        return self._state

    @state.setter
    def state(self, value: AgentState):
        self._state = value
        logger.debug(f"Agent {self.name} state changed to {value.value}")

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Process a message and return agent response with monitoring integration

        Args:
            message: User message to process
            conversation_id: Optional conversation ID for context
            context: Additional context information

        Returns:
            AgentResult with response and tool execution results
        """
        start_time = time.time()
        success = False
        tools_used = []

        try:
            # Call the actual implementation
            result = await self._process_message_impl(message, conversation_id, context)
            success = True
            return result

        except Exception:
            # Re-raise the exception after recording metrics
            raise
        finally:
            # Record metrics if monitoring is enabled
            if (
                metrics_collector
                and monitoring_config
                and monitoring_config.monitoring_enabled
                and monitoring_config.track_agent_performance
            ):
                execution_time = time.time() - start_time
                metrics_collector.record_agent_execution(
                    agent_name=self.name,
                    success=success,
                    execution_time=execution_time,
                    tools_used=tools_used,
                )

    @abstractmethod
    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Internal implementation of message processing (to be implemented by subclasses)

        Args:
            message: User message to process
            conversation_id: Optional conversation ID for context
            context: Additional context information

        Returns:
            AgentResult with response and tool execution results
        """
        pass

    def start_conversation(self) -> str:
        """Start a new conversation and return conversation ID"""
        self._current_conversation_id = str(uuid.uuid4())
        self._conversation_history = []
        return self._current_conversation_id

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        if conversation_id != self._current_conversation_id:
            return []
        return self._conversation_history.copy()

    def add_to_conversation(
        self, role: str, content: str, metadata: Dict[str, Any] = None
    ):
        """Add a message to the current conversation"""
        if not self._current_conversation_id:
            self.start_conversation()

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._conversation_history.append(message)

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with proper state management"""
        self.state = AgentState.EXECUTING_TOOL

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(error_msg)
            self.state = AgentState.ERROR
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=tool_name,
                execution_time=0.0,
            )

        try:
            result = await tool.execute_with_timeout(**kwargs)
            self.state = AgentState.THINKING
            return result
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            self.state = AgentState.ERROR
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=tool_name,
                execution_time=0.0,
            )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get agent usage statistics"""
        return {
            "name": self.name,
            "usage_count": self._usage_count,
            "last_used": self._last_used,
            "current_conversation_id": self._current_conversation_id,
            "conversation_history_length": len(self._conversation_history),
        }

    def reset(self):
        """Reset agent state and clear conversation history"""
        self.state = AgentState.IDLE
        self._conversation_history = []
        self._current_conversation_id = None


class ToolCallingAgent(BaseAgent):
    """Base class for agents that can call tools"""

    def __init__(self, tool_registry: ToolRegistry, max_iterations: int = 5):
        super().__init__(tool_registry, max_iterations)
        self._max_tool_calls_per_message = 3

    @abstractmethod
    async def decide_tool_usage(
        self, message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Decide whether and which tools to use for the given message

        Args:
            message: User message
            context: Additional context

        Returns:
            Dictionary with tool usage decision
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        message: str,
        tool_results: List[ToolResult] = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate final response after tool execution

        Args:
            message: Original user message
            tool_results: Results from tool executions
            context: Additional context

        Returns:
            Generated response text
        """
        pass
