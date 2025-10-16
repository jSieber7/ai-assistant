"""
Agent system for intelligent tool selection and execution.

This module provides the core agent system that integrates LLM calls with
tool execution, enabling intelligent tool selection based on context and
query analysis.
"""

from .base import BaseAgent, AgentResult, AgentError
from .tool_agent import ToolAgent
from .firebase_scraper_agent import FirebaseScraperAgent
from .strategies import ToolSelectionStrategy, KeywordStrategy, LLMStrategy
from .registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentError",
    "ToolAgent",
    "FirebaseScraperAgent",
    "ToolSelectionStrategy",
    "KeywordStrategy",
    "LLMStrategy",
    "AgentRegistry",
]
