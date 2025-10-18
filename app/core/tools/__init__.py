"""
Core tool system for the AI Assistant.

This module provides the foundation for an extensible tool system that allows
the AI assistant to interact with various external systems and perform tasks.
"""

from .base.base import (
    BaseTool,
    ToolResult,
    ToolError,
    ToolTimeoutError,
    ToolConfigurationError,
    ToolExecutionError,
)
from .execution.registry import ToolRegistry
from .utilities.examples import CalculatorTool, TimeTool, EchoTool
from .web.searxng_tool import SearXNGTool
from .web.firecrawl_tool import FirecrawlTool
from .base.config import ToolSystemSettings, tool_settings
from .integration.langchain_integration import (
    LangChainToolWrapper,
    LangChainToolkit,
    ToolOutputParser,
    create_agent_with_tools,
    get_tool_descriptions,
    tool_selection_prompt,
    get_langchain_tools,
    is_tool_available,
)

# Global tool registry instance
tool_registry = ToolRegistry()

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolTimeoutError",
    "ToolConfigurationError",
    "ToolExecutionError",
    "ToolRegistry",
    "tool_registry",
    "CalculatorTool",
    "TimeTool",
    "EchoTool",
    "SearXNGTool",
    "FirecrawlTool",
    "ToolSystemSettings",
    "tool_settings",
    "LangChainToolWrapper",
    "LangChainToolkit",
    "ToolOutputParser",
    "create_agent_with_tools",
    "get_tool_descriptions",
    "tool_selection_prompt",
    "get_langchain_tools",
    "is_tool_available",
]
