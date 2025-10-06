"""
Integration modules for connecting caching with tool registry and agent system.

This module provides integration points between the caching system and
the existing tool registry and agent system.
"""

from .tool_cache import ToolCache, CachedToolRegistry
from .agent_cache import AgentCache, CachedAgentRegistry

__all__ = [
    "ToolCache",
    "CachedToolRegistry",
    "AgentCache",
    "CachedAgentRegistry",
]
