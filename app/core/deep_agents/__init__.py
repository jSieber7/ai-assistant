"""
Deep Agents integration for the AI Assistant.

This module provides a native integration with the deepagents library,
leveraging LangGraph for complex, multi-step task planning and execution.
"""

from .manager import DeepAgentManager, deep_agent_manager

__all__ = ["DeepAgentManager", "deep_agent_manager"]