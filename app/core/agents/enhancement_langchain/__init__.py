"""
LangGraph-based enhancement agents for the AI assistant system.

This module provides LangGraph implementations of enhancement agents that
manage context sharing, learning, and personality systems.
"""

from .context_sharing import (
    LangGraphContextSharingSystem,
    ContextSharingState,
    SharedContext,
    ContextRelationship,
    ContextType,
    ContextPriority,
    RelationshipType,
)

from .learning_system import (
    LangGraphLearningSystem,
    LearningState,
    PerformanceMetrics,
    InteractionPattern,
    LearningInsight,
    LearningMode,
    AdaptationType,
)

from .personality_system import (
    LangGraphPersonalitySystem,
    PersonalityState,
    PersonalityProfile,
    PersonalityTraits,
    VoiceCharacteristics,
    BehavioralPatterns,
    PersuasionStyle,
    WritingTone,
    DebateStyle,
)

__all__ = [
    # Context sharing
    "LangGraphContextSharingSystem",
    "ContextSharingState",
    "SharedContext",
    "ContextRelationship",
    "ContextType",
    "ContextPriority",
    "RelationshipType",
    
    # Learning system
    "LangGraphLearningSystem",
    "LearningState",
    "PerformanceMetrics",
    "InteractionPattern",
    "LearningInsight",
    "LearningMode",
    "AdaptationType",
    
    # Personality system
    "LangGraphPersonalitySystem",
    "PersonalityState",
    "PersonalityProfile",
    "PersonalityTraits",
    "VoiceCharacteristics",
    "BehavioralPatterns",
    "PersuasionStyle",
    "WritingTone",
    "DebateStyle",
]