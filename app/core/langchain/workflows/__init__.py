"""
LangGraph workflow definitions for AI Assistant.

This module contains various workflow implementations using LangGraph
for different types of agent behaviors and capabilities.
"""

from .conversational import create_conversational_workflow
from .tool_heavy import create_tool_heavy_workflow
from .multi_agent import create_multi_agent_workflow
from .researcher import create_researcher_workflow
from .analyst import create_analyst_workflow
from .synthesizer import create_synthesizer_workflow

__all__ = [
    "create_conversational_workflow",
    "create_tool_heavy_workflow", 
    "create_multi_agent_workflow",
    "create_researcher_workflow",
    "create_analyst_workflow",
    "create_synthesizer_workflow",
]