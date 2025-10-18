"""
Agent system for intelligent tool selection and execution.

This module provides the core agent system that integrates LLM calls with
tool execution, enabling intelligent tool selection based on context and
query analysis.
"""

from .base.base import BaseAgent, AgentResult, AgentError
from .specialized.tool_agent import ToolAgent
from .specialized.firecrawl_agent import FirecrawlAgent
from .utilities.strategies import ToolSelectionStrategy, KeywordStrategy, LLMStrategy
from .management.registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentError",
    "ToolAgent",
    "FirecrawlAgent",
    "ToolSelectionStrategy",
    "KeywordStrategy",
    "LLMStrategy",
    "AgentRegistry",
]
