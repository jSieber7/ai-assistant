"""
LangGraph-based collaboration agents.

This module provides LangGraph implementations of collaboration agents
for multi-agent workflows including debate systems and dynamic selection.
"""

from .debate_orchestrator import LangGraphDebateOrchestrator, DebateState, DebatePosition, DebateCritique, DebatePhase
from .dynamic_selector import LangGraphDynamicSelector, SelectorState, TaskAnalysis, SelectionResult

__all__ = [
    # Debate orchestrator
    "LangGraphDebateOrchestrator",
    "DebateState",
    "DebatePosition",
    "DebateCritique",
    "DebatePhase",
    
    # Dynamic selector
    "LangGraphDynamicSelector",
    "SelectorState",
    "TaskAnalysis",
    "SelectionResult",
]