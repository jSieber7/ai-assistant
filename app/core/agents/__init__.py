"""
Agent system for intelligent tool selection and execution.

This module provides core agent system that integrates LLM calls with
tool execution, enabling intelligent tool selection based on context and
query analysis.
"""

# Legacy agents
from .base.base import BaseAgent, AgentResult, AgentError
from .specialized.tool_agent import ToolAgent
from .specialized.firecrawl_agent import FirecrawlAgent
from .utilities.strategies import ToolSelectionStrategy, KeywordStrategy, LLMStrategy
from .management.registry import AgentRegistry

# LangGraph-based specialized agents
from .specialized_langchain import (
    SummarizeAgent,
    WebdriverAgent,
    ScraperAgent,
    SearchQueryAgent,
    ChainOfThoughtAgent,
    CreativeStoryAgent,
    ToolSelectionAgent,
    SemanticUnderstandingAgent,
    FactCheckerAgent,
)

# LangGraph-based content agents
from .content_langchain import (
    WriterAgent,
    MultiContentOrchestrator,
)

# LangGraph-based collaboration agents
from .collaboration_langchain import (
    LangGraphDebateOrchestrator,
    LangGraphDynamicSelector,
)

# LangGraph-based validation agents
from .validation_langchain import (
    LangGraphCheckerAgent,
    LangGraphCollaborativeChecker,
    LangGraphMasterCheckerAgent,
)

# LangGraph-based enhancement agents
from .enhancement_langchain import (
    LangGraphContextSharingSystem,
    LangGraphLearningSystem,
    LangGraphPersonalitySystem,
    ContextSharingState,
    LearningState,
    PersonalityState,
    SharedContext,
    PerformanceMetrics,
    PersonalityProfile,
    ContextType,
    LearningMode,
    PersuasionStyle,
)

# LangGraph-based memory agents
from .memory_langchain import (
    LangGraphMemoryAgent,
    LangGraphContextRetrieverAgent,
    LangGraphConversationSummarizerAgent,
    LangGraphMemoryManagerAgent,
    MemoryTaskType,
    MemoryAgentState,
    ContextRetrievalStrategy,
    ContextRetrieverState,
    SummaryType,
    SummaryLevel,
    ConversationSummarizerState,
    MemoryManagementTask,
    MemoryCleanupPolicy,
    MemoryManagerState,
)

__all__ = [
    # Legacy agents
    "BaseAgent",
    "AgentResult",
    "AgentError",
    "ToolAgent",
    "FirecrawlAgent",
    "ToolSelectionStrategy",
    "KeywordStrategy",
    "LLMStrategy",
    "AgentRegistry",
    
    # LangGraph-based specialized agents
    "SummarizeAgent",
    "WebdriverAgent",
    "ScraperAgent",
    "SearchQueryAgent",
    "ChainOfThoughtAgent",
    "CreativeStoryAgent",
    "ToolSelectionAgent",
    "SemanticUnderstandingAgent",
    "FactCheckerAgent",
    
    # LangGraph-based content agents
    "WriterAgent",
    "MultiContentOrchestrator",
    
    # LangGraph-based collaboration agents
    "LangGraphDebateOrchestrator",
    "LangGraphDynamicSelector",
    
    # LangGraph-based validation agents
    "LangGraphCheckerAgent",
    "LangGraphCollaborativeChecker",
    "LangGraphMasterCheckerAgent",
    
    # LangGraph-based enhancement agents
    "LangGraphContextSharingSystem",
    "LangGraphLearningSystem",
    "LangGraphPersonalitySystem",
    "ContextSharingState",
    "LearningState",
    "PersonalityState",
    "SharedContext",
    "PerformanceMetrics",
    "PersonalityProfile",
    "ContextType",
    "LearningMode",
    "PersuasionStyle",
    
    # LangGraph-based memory agents
    "LangGraphMemoryAgent",
    "LangGraphContextRetrieverAgent",
    "LangGraphConversationSummarizerAgent",
    "LangGraphMemoryManagerAgent",
    "MemoryTaskType",
    "MemoryAgentState",
    "ContextRetrievalStrategy",
    "ContextRetrieverState",
    "SummaryType",
    "SummaryLevel",
    "ConversationSummarizerState",
    "MemoryManagementTask",
    "MemoryCleanupPolicy",
    "MemoryManagerState",
]