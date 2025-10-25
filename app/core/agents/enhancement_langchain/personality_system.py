"""
LangGraph-based personality and voice modeling system for writer agents.

This module implements a personality system using LangGraph workflows that affects
writing style, debate behavior, and collaboration patterns.
"""

import logging
import time
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END

# Make CompiledGraph import optional to handle different langgraph versions
try:
    from langgraph.graph.graph import CompiledGraph
except ImportError:
    try:
        from langgraph.graph import CompiledGraph
    except ImportError:
        CompiledGraph = None

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PersuasionStyle(str, Enum):
    """Persuasion styles for debate and writing"""
    LOGICAL = "logical"
    EMOTIONAL = "emotional"
    AUTHORITY = "authority"
    COLLABORATIVE = "collaborative"
    CHALLENGING = "challenging"


class WritingTone(str, Enum):
    """Writing tones"""
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"


class DebateStyle(str, Enum):
    """Debate styles"""
    CONSTRUCTIVE = "constructive"
    AGGRESSIVE = "aggressive"
    DIPLOMATIC = "diplomatic"
    ANALYTICAL = "analytical"
    INSPIRATIONAL = "inspirational"


@dataclass
class PersonalityTraits:
    """Core personality traits for a writer"""
    confidence: float = 0.5  # 0.0-1.0
    creativity: float = 0.5  # 0.0-1.0
    analytical_depth: float = 0.5  # 0.0-1.0
    collaboration_tendency: float = 0.5  # 0.0-1.0
    debate_aggressiveness: float = 0.5  # 0.0-1.0
    openness_to_feedback: float = 0.5  # 0.0-1.0
    perfectionism: float = 0.5  # 0.0-1.0
    adaptability: float = 0.5  # 0.0-1.0
    leadership_tendency: float = 0.5  # 0.0-1.0
    risk_tolerance: float = 0.5  # 0.0-1.0


@dataclass
class VoiceCharacteristics:
    """Voice and style characteristics"""
    persuasion_style: PersuasionStyle = PersuasionStyle.LOGICAL
    writing_tone: WritingTone = WritingTone.PROFESSIONAL
    debate_style: DebateStyle = DebateStyle.CONSTRUCTIVE
    vocabulary_complexity: float = 0.5  # 0.0-1.0
    sentence_length_preference: float = 0.5  # 0.0-1.0
    use_of_metaphors: float = 0.3  # 0.0-1.0
    humor_level: float = 0.2  # 0.0-1.0
    formality_level: float = 0.7  # 0.0-1.0


@dataclass
class BehavioralPatterns:
    """Behavioral patterns affecting interaction"""
    response_speed: float = 0.5  # 0.0-1.0 (faster to slower)
    critique_style: str = "constructive"  # constructive, direct, gentle
    collaboration_preference: str = "balanced"  # leader, follower, balanced
    conflict_resolution: str = "diplomatic"  # diplomatic, direct, avoidant
    learning_style: str = "adaptive"  # adaptive, fixed, curious
    stress_response: str = "focused"  # focused, defensive, creative


@dataclass
class PersonalityProfile:
    """Complete personality profile for a writer agent"""
    writer_id: str
    specialty: str
    traits: PersonalityTraits
    voice: VoiceCharacteristics
    behavior: BehavioralPatterns
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class PersonalityState(BaseModel):
    """State for personality workflow"""
    # Input parameters
    action: str  # "create_profile", "get_profile", "apply_to_prompt", "apply_to_debate", "apply_to_critique", "update_history", "get_compatibility", "suggest_team"
    writer_id: Optional[str] = None
    specialty: Optional[str] = None
    template_name: Optional[str] = None
    customizations: Optional[Dict[str, Any]] = None
    base_prompt: Optional[str] = None
    base_response: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    target_writer_id: Optional[str] = None
    available_writers: Optional[List[str]] = None
    task_complexity: Optional[str] = None
    
    # Personality data
    profiles: Dict[str, PersonalityProfile] = field(default_factory=dict)
    personality_templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Results
    profile: Optional[PersonalityProfile] = None
    modified_prompt: Optional[str] = None
    modified_response: Optional[str] = None
    modified_critique: Optional[str] = None
    debate_strategy: Optional[Dict[str, Any]] = None
    compatibility_score: Optional[float] = None
    team_suggestion: Optional[Dict[str, Any]] = None
    
    # Output
    result: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class LangGraphPersonalitySystem:
    """LangGraph-based personality and voice modeling system"""
    
    def __init__(self):
        # Initialize workflow
        self.workflow = self._create_workflow()
        self.compiled_workflow = None
        
        # Storage for personality profiles
        self.profiles: Dict[str, PersonalityProfile] = {}
        self.personality_templates = self._create_personality_templates()
        
        logger.info("Initialized LangGraph Personality System")
    
    def _create_workflow(self) -> StateGraph:
        """Create personality workflow"""
        workflow = StateGraph(PersonalityState)
        
        # Add nodes
        workflow.add_node("route_action", self._route_action)
        workflow.add_node("create_profile", self._create_profile)
        workflow.add_node("get_profile", self._get_profile)
        workflow.add_node("apply_to_prompt", self._apply_to_prompt)
        workflow.add_node("apply_to_debate", self._apply_to_debate)
        workflow.add_node("apply_to_critique", self._apply_to_critique)
        workflow.add_node("update_history", self._update_history)
        workflow.add_node("get_compatibility", self._get_compatibility)
        workflow.add_node("suggest_team", self._suggest_team)
        
        # Set entry point
        workflow.set_entry_point("route_action")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "route_action",
            self._determine_action,
            {
                "create": "create_profile",
                "get": "get_profile",
                "prompt": "apply_to_prompt",
                "debate": "apply_to_debate",
                "critique": "apply_to_critique",
                "history": "update_history",
                "compatibility": "get_compatibility",
                "team": "suggest_team",
                "error": END
            }
        )
        
        # Add edges to end
        workflow.add_edge("create_profile", END)
        workflow.add_edge("get_profile", END)
        workflow.add_edge("apply_to_prompt", END)
        workflow.add_edge("apply_to_debate", END)
        workflow.add_edge("apply_to_critique", END)
        workflow.add_edge("update_history", END)
        workflow.add_edge("get_compatibility", END)
        workflow.add_edge("suggest_team", END)
        
        return workflow
    
    async def _route_action(self, state: PersonalityState) -> PersonalityState:
        """Route action based on input"""
        try:
            action = state.action.lower()
            
            if action in ["create", "create_profile", "new"]:
                state.action = "create"
            elif action in ["get", "get_profile", "retrieve", "find"]:
                state.action = "get"
            elif action in ["prompt", "apply_to_prompt", "enhance_prompt"]:
                state.action = "prompt"
            elif action in ["debate", "apply_to_debate", "enhance_debate"]:
                state.action = "debate"
            elif action in ["critique", "apply_to_critique", "enhance_critique"]:
                state.action = "critique"
            elif action in ["history", "update_history", "record"]:
                state.action = "history"
            elif action in ["compatibility", "get_compatibility", "compare"]:
                state.action = "compatibility"
            elif action in ["team", "suggest_team", "compose"]:
                state.action = "team"
            else:
                state.action = "error"
                state.error = f"Unknown action: {action}"
            
            # Copy current state to workflow state
            state.profiles = self.profiles.copy()
            state.personality_templates = self.personality_templates.copy()
            
        except Exception as e:
            logger.error(f"Error routing action: {str(e)}")
            state.action = "error"
            state.error = str(e)
        
        return state
    
    def _determine_action(self, state: PersonalityState) -> str:
        """Determine which action to take"""
        return state.action
    
    async def _create_profile(self, state: PersonalityState) -> PersonalityState:
        """Create a personality profile for a writer"""
        try:
            if not state.writer_id or not state.specialty:
                state.error = "Writer ID and specialty are required for profile creation"
                state.success = False
                return state
            
            # Select template or create default
            if state.template_name and state.template_name in state.personality_templates:
                template = state.personality_templates[state.template_name]
                traits = PersonalityTraits(**template["traits"].__dict__)
                voice = VoiceCharacteristics(**template["voice"].__dict__)
                behavior = BehavioralPatterns(**template["behavior"].__dict__)
            else:
                # Create balanced default profile
                traits = PersonalityTraits()
                voice = VoiceCharacteristics()
                behavior = BehavioralPatterns()
            
            # Apply customizations
            if state.customizations:
                traits = self._apply_trait_customizations(
                    traits, state.customizations.get("traits", {})
                )
                voice = self._apply_voice_customizations(
                    voice, state.customizations.get("voice", {})
                )
                behavior = self._apply_behavior_customizations(
                    behavior, state.customizations.get("behavior", {})
                )
            
            # Add some randomness for uniqueness
            traits = self._add_personality_variance(traits)
            
            profile = PersonalityProfile(
                writer_id=state.writer_id,
                specialty=state.specialty,
                traits=traits,
                voice=voice,
                behavior=behavior,
            )
            
            state.profiles[state.writer_id] = profile
            state.profile = profile
            
            state.result = f"Successfully created personality profile for {state.writer_id}"
            state.success = True
            
            logger.info(
                f"Created personality profile for {state.writer_id} using template: {state.template_name or 'default'}"
            )
            
        except Exception as e:
            logger.error(f"Error creating profile: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _get_profile(self, state: PersonalityState) -> PersonalityState:
        """Get personality profile for a writer"""
        try:
            if not state.writer_id:
                state.error = "Writer ID is required to get profile"
                state.success = False
                return state
            
            profile = state.profiles.get(state.writer_id)
            
            if profile:
                state.profile = profile
                state.result = f"Successfully retrieved profile for {state.writer_id}"
                state.success = True
            else:
                state.error = f"No profile found for writer {state.writer_id}"
                state.success = False
            
        except Exception as e:
            logger.error(f"Error getting profile: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _apply_to_prompt(self, state: PersonalityState) -> PersonalityState:
        """Apply personality to a prompt"""
        try:
            if not state.writer_id or not state.base_prompt:
                state.error = "Writer ID and base prompt are required"
                state.success = False
                return state
            
            profile = state.profiles.get(state.writer_id)
            if not profile:
                state.error = f"No profile found for writer {state.writer_id}"
                state.success = False
                return state
            
            personality_instructions = self._generate_personality_instructions(
                profile, state.context
            )
            
            modified_prompt = f"{state.base_prompt}\n\n{personality_instructions}"
            state.modified_prompt = modified_prompt
            
            state.result = f"Successfully applied personality to prompt for {state.writer_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error applying personality to prompt: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _apply_to_debate(self, state: PersonalityState) -> PersonalityState:
        """Apply personality to debate response"""
        try:
            if not state.writer_id or not state.base_response or not state.context:
                state.error = "Writer ID, base response, and context are required"
                state.success = False
                return state
            
            profile = state.profiles.get(state.writer_id)
            if not profile:
                state.error = f"No profile found for writer {state.writer_id}"
                state.success = False
                return state
            
            # Adjust response based on personality
            modified_response = self._modify_debate_response(
                state.base_response, profile, state.context
            )
            
            state.modified_response = modified_response
            
            # Generate debate strategy
            debate_strategy = self._get_debate_strategy(profile)
            state.debate_strategy = debate_strategy
            
            state.result = f"Successfully applied personality to debate response for {state.writer_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error applying personality to debate: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _apply_to_critique(self, state: PersonalityState) -> PersonalityState:
        """Apply personality to critique style"""
        try:
            if not state.writer_id or not state.base_critique or not state.target_writer_id:
                state.error = "Writer ID, base critique, and target writer ID are required"
                state.success = False
                return state
            
            profile = state.profiles.get(state.writer_id)
            if not profile:
                state.error = f"No profile found for writer {state.writer_id}"
                state.success = False
                return state
            
            # Modify critique based on personality
            modified_critique = self._modify_critique_style(
                state.base_critique, profile, state.target_writer_id
            )
            
            state.modified_critique = modified_critique
            
            state.result = f"Successfully applied personality to critique for {state.writer_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error applying personality to critique: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _update_history(self, state: PersonalityState) -> PersonalityState:
        """Update interaction history for learning"""
        try:
            if not state.writer_id or not state.context:
                state.error = "Writer ID and context are required for history update"
                state.success = False
                return state
            
            profile = state.profiles.get(state.writer_id)
            if not profile:
                state.error = f"No profile found for writer {state.writer_id}"
                state.success = False
                return state
            
            # Extract interaction data from context
            interaction_type = state.context.get("interaction_type", "unknown")
            outcome = state.context.get("outcome", "unknown")
            metrics = state.context.get("metrics", {})
            
            interaction = {
                "timestamp": time.time(),
                "type": interaction_type,
                "outcome": outcome,
                "metrics": metrics,
            }
            
            profile.interaction_history.append(interaction)
            profile.last_updated = time.time()
            
            # Update performance metrics
            for metric, value in metrics.items():
                if metric in profile.performance_metrics:
                    # Update running average
                    old_value = profile.performance_metrics[metric]
                    count = len(
                        [
                            i
                            for i in profile.interaction_history
                            if metric in i.get("metrics", {})
                        ]
                    )
                    new_value = (old_value * (count - 1) + value) / count
                    profile.performance_metrics[metric] = new_value
                else:
                    profile.performance_metrics[metric] = value
            
            state.result = f"Successfully updated interaction history for {state.writer_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error updating history: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _get_compatibility(self, state: PersonalityState) -> PersonalityState:
        """Calculate compatibility score between two writers"""
        try:
            if not state.writer_id or not state.target_writer_id:
                state.error = "Writer ID and target writer ID are required"
                state.success = False
                return state
            
            profile1 = state.profiles.get(state.writer_id)
            profile2 = state.profiles.get(state.target_writer_id)
            
            if not profile1 or not profile2:
                state.error = "One or both profiles not found"
                state.success = False
                return state
            
            compatibility_score = self._calculate_compatibility_score(profile1, profile2)
            state.compatibility_score = compatibility_score
            
            state.result = f"Successfully calculated compatibility between {state.writer_id} and {state.target_writer_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error calculating compatibility: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _suggest_team(self, state: PersonalityState) -> PersonalityState:
        """Suggest optimal team composition based on personalities"""
        try:
            if not state.available_writers:
                state.error = "Available writers list is required"
                state.success = False
                return state
            
            if len(state.available_writers) < 2:
                state.team_suggestion = {
                    "suggestion": state.available_writers,
                    "reasoning": "Insufficient writers for team analysis",
                }
                state.result = "Insufficient writers for team analysis"
                state.success = True
                return state
            
            # Calculate all compatibility scores
            compatibility_matrix = {}
            for i, writer1 in enumerate(state.available_writers):
                for writer2 in state.available_writers[i + 1 :]:
                    profile1 = state.profiles.get(writer1)
                    profile2 = state.profiles.get(writer2)
                    
                    if profile1 and profile2:
                        score = self._calculate_compatibility_score(profile1, profile2)
                        compatibility_matrix[(writer1, writer2)] = score
            
            # Suggest team based on complexity
            task_complexity = state.task_complexity or "moderate"
            team_size = 3 if task_complexity in ["complex", "expert"] else 2
            
            # Build team with highest overall compatibility
            best_team = []
            best_score = 0.0
            
            for combination in self._generate_combinations(state.available_writers, team_size):
                team_score = self._calculate_team_compatibility(
                    combination, compatibility_matrix
                )
                if team_score > best_score:
                    best_score = team_score
                    best_team = combination
            
            state.team_suggestion = {
                "suggested_team": best_team,
                "team_compatibility": best_score,
                "reasoning": f"Selected for optimal compatibility score of {best_score:.2f}",
                "compatibility_matrix": compatibility_matrix,
            }
            
            state.result = f"Successfully suggested team composition"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error suggesting team: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    def _create_personality_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined personality templates"""
        return {
            "confident_expert": {
                "traits": PersonalityTraits(
                    confidence=0.9,
                    creativity=0.4,
                    analytical_depth=0.8,
                    collaboration_tendency=0.6,
                    debate_aggressiveness=0.7,
                    openness_to_feedback=0.5,
                    perfectionism=0.8,
                    adaptability=0.4,
                    leadership_tendency=0.8,
                    risk_tolerance=0.3,
                ),
                "voice": VoiceCharacteristics(
                    persuasion_style=PersuasionStyle.AUTHORITY,
                    writing_tone=WritingTone.PROFESSIONAL,
                    debate_style=DebateStyle.ANALYTICAL,
                    vocabulary_complexity=0.8,
                    sentence_length_preference=0.7,
                    use_of_metaphors=0.2,
                    humor_level=0.1,
                    formality_level=0.9,
                ),
                "behavior": BehavioralPatterns(
                    response_speed=0.6,
                    critique_style="direct",
                    collaboration_preference="leader",
                    conflict_resolution="direct",
                    learning_style="fixed",
                    stress_response="focused",
                ),
            },
            "creative_innovator": {
                "traits": PersonalityTraits(
                    confidence=0.7,
                    creativity=0.9,
                    analytical_depth=0.4,
                    collaboration_tendency=0.8,
                    debate_aggressiveness=0.3,
                    openness_to_feedback=0.8,
                    perfectionism=0.4,
                    adaptability=0.9,
                    leadership_tendency=0.5,
                    risk_tolerance=0.8,
                ),
                "voice": VoiceCharacteristics(
                    persuasion_style=PersuasionStyle.EMOTIONAL,
                    writing_tone=WritingTone.CREATIVE,
                    debate_style=DebateStyle.INSPIRATIONAL,
                    vocabulary_complexity=0.6,
                    sentence_length_preference=0.4,
                    use_of_metaphors=0.8,
                    humor_level=0.6,
                    formality_level=0.4,
                ),
                "behavior": BehavioralPatterns(
                    response_speed=0.4,
                    critique_style="constructive",
                    collaboration_preference="balanced",
                    conflict_resolution="diplomatic",
                    learning_style="curious",
                    stress_response="creative",
                ),
            },
            "balanced_collaborator": {
                "traits": PersonalityTraits(
                    confidence=0.6,
                    creativity=0.6,
                    analytical_depth=0.6,
                    collaboration_tendency=0.9,
                    debate_aggressiveness=0.2,
                    openness_to_feedback=0.9,
                    perfectionism=0.6,
                    adaptability=0.8,
                    leadership_tendency=0.4,
                    risk_tolerance=0.5,
                ),
                "voice": VoiceCharacteristics(
                    persuasion_style=PersuasionStyle.COLLABORATIVE,
                    writing_tone=WritingTone.CONVERSATIONAL,
                    debate_style=DebateStyle.CONSTRUCTIVE,
                    vocabulary_complexity=0.5,
                    sentence_length_preference=0.5,
                    use_of_metaphors=0.4,
                    humor_level=0.4,
                    formality_level=0.6,
                ),
                "behavior": BehavioralPatterns(
                    response_speed=0.5,
                    critique_style="constructive",
                    collaboration_preference="balanced",
                    conflict_resolution="diplomatic",
                    learning_style="adaptive",
                    stress_response="focused",
                ),
            },
            "analytical_skeptic": {
                "traits": PersonalityTraits(
                    confidence=0.7,
                    creativity=0.3,
                    analytical_depth=0.9,
                    collaboration_tendency=0.4,
                    debate_aggressiveness=0.6,
                    openness_to_feedback=0.4,
                    perfectionism=0.9,
                    adaptability=0.3,
                    leadership_tendency=0.3,
                    risk_tolerance=0.2,
                ),
                "voice": VoiceCharacteristics(
                    persuasion_style=PersuasionStyle.LOGICAL,
                    writing_tone=WritingTone.ACADEMIC,
                    debate_style=DebateStyle.ANALYTICAL,
                    vocabulary_complexity=0.9,
                    sentence_length_preference=0.8,
                    use_of_metaphors=0.1,
                    humor_level=0.1,
                    formality_level=0.8,
                ),
                "behavior": BehavioralPatterns(
                    response_speed=0.7,
                    critique_style="direct",
                    collaboration_preference="follower",
                    conflict_resolution="direct",
                    learning_style="fixed",
                    stress_response="focused",
                ),
            },
            "enthusiastic_generalist": {
                "traits": PersonalityTraits(
                    confidence=0.8,
                    creativity=0.7,
                    analytical_depth=0.5,
                    collaboration_tendency=0.7,
                    debate_aggressiveness=0.4,
                    openness_to_feedback=0.7,
                    perfectionism=0.5,
                    adaptability=0.7,
                    leadership_tendency=0.6,
                    risk_tolerance=0.6,
                ),
                "voice": VoiceCharacteristics(
                    persuasion_style=PersuasionStyle.COLLABORATIVE,
                    writing_tone=WritingTone.CASUAL,
                    debate_style=DebateStyle.CONSTRUCTIVE,
                    vocabulary_complexity=0.4,
                    sentence_length_preference=0.3,
                    use_of_metaphors=0.5,
                    humor_level=0.7,
                    formality_level=0.3,
                ),
                "behavior": BehavioralPatterns(
                    response_speed=0.3,
                    critique_style="constructive",
                    collaboration_preference="balanced",
                    conflict_resolution="diplomatic",
                    learning_style="curious",
                    stress_response="creative",
                ),
            },
        }
    
    def _apply_trait_customizations(
        self, traits: PersonalityTraits, customizations: Dict[str, float]
    ) -> PersonalityTraits:
        """Apply custom trait values"""
        for trait, value in customizations.items():
            if hasattr(traits, trait) and 0.0 <= value <= 1.0:
                setattr(traits, trait, value)
        return traits
    
    def _apply_voice_customizations(
        self, voice: VoiceCharacteristics, customizations: Dict[str, Any]
    ) -> VoiceCharacteristics:
        """Apply custom voice characteristics"""
        for characteristic, value in customizations.items():
            if hasattr(voice, characteristic):
                if characteristic in [
                    "persuasion_style",
                    "writing_tone",
                    "debate_style",
                ]:
                    # Handle enum values
                    try:
                        if characteristic == "persuasion_style":
                            setattr(voice, characteristic, PersuasionStyle(value))
                        elif characteristic == "writing_tone":
                            setattr(voice, characteristic, WritingTone(value))
                        elif characteristic == "debate_style":
                            setattr(voice, characteristic, DebateStyle(value))
                    except ValueError:
                        pass  # Keep default if invalid
                elif isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    setattr(voice, characteristic, value)
        return voice
    
    def _apply_behavior_customizations(
        self, behavior: BehavioralPatterns, customizations: Dict[str, str]
    ) -> BehavioralPatterns:
        """Apply custom behavioral patterns"""
        for pattern, value in customizations.items():
            if hasattr(behavior, pattern):
                setattr(behavior, pattern, value)
        return behavior
    
    def _add_personality_variance(self, traits: PersonalityTraits) -> PersonalityTraits:
        """Add small random variations for uniqueness"""
        variance = 0.1
        for trait_name in traits.__dict__:
            current_value = getattr(traits, trait_name)
            if isinstance(current_value, float):
                # Use secrets.randbelow for cryptographically secure random numbers
                variation = (secrets.randbelow(20001) / 10000.0) - 1.0  # -1.0 to 1.0
                variation = variation * variance  # Scale to -variance to variance
                new_value = max(0.0, min(1.0, current_value + variation))
                setattr(traits, trait_name, new_value)
        return traits
    
    def _generate_personality_instructions(
        self, profile: PersonalityProfile, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate personality-specific instructions"""
        instructions = []
        
        # Confidence instructions
        if profile.traits.confidence > 0.7:
            instructions.append("Write with confidence and authority.")
        elif profile.traits.confidence < 0.3:
            instructions.append("Write humbly and acknowledge limitations.")
        
        # Creativity instructions
        if profile.traits.creativity > 0.7:
            instructions.append("Be creative and innovative in your approach.")
        elif profile.traits.creativity < 0.3:
            instructions.append("Stick to conventional and proven approaches.")
        
        # Analytical depth instructions
        if profile.traits.analytical_depth > 0.7:
            instructions.append("Provide deep analysis and detailed reasoning.")
        elif profile.traits.analytical_depth < 0.3:
            instructions.append("Focus on high-level concepts rather than details.")
        
        # Voice instructions
        instructions.append(f"Use a {profile.voice.writing_tone.value} tone.")
        instructions.append(
            f"Write with {profile.voice.persuasion_style.value} persuasion."
        )
        
        # Vocabulary complexity
        if profile.voice.vocabulary_complexity > 0.7:
            instructions.append(
                "Use sophisticated vocabulary and complex sentence structures."
            )
        elif profile.voice.vocabulary_complexity < 0.3:
            instructions.append("Use simple, clear language accessible to all readers.")
        
        # Humor level
        if profile.voice.humor_level > 0.6:
            instructions.append("Include appropriate humor to engage the reader.")
        elif profile.voice.humor_level < 0.2:
            instructions.append("Maintain a serious and professional tone throughout.")
        
        # Formality
        if profile.voice.formality_level > 0.7:
            instructions.append("Maintain high formality and professional standards.")
        elif profile.voice.formality_level < 0.4:
            instructions.append("Write in a more casual and approachable style.")
        
        # Context-specific instructions
        if context:
            if context.get("debate_mode"):
                instructions.append(
                    f"Debate with a {profile.voice.debate_style.value} style."
                )
                if profile.traits.debate_aggressiveness > 0.6:
                    instructions.append("Be assertive and challenge opposing views.")
                elif profile.traits.debate_aggressiveness < 0.4:
                    instructions.append("Be diplomatic and seek common ground.")
            
            if context.get("collaboration_mode"):
                if profile.traits.collaboration_tendency > 0.7:
                    instructions.append(
                        "Actively build upon others' ideas and seek consensus."
                    )
                elif profile.traits.collaboration_tendency < 0.4:
                    instructions.append(
                        "Focus on contributing your unique perspective."
                    )
        
        return "\n".join(instructions)
    
    def _modify_debate_response(
        self,
        base_response: str,
        profile: PersonalityProfile,
        debate_context: Dict[str, Any],
    ) -> str:
        """Modify debate response based on personality"""
        modified_response = base_response
        
        # Adjust confidence level
        if profile.traits.confidence > 0.7:
            if not any(
                phrase in modified_response.lower()
                for phrase in ["i believe", "i think", "in my opinion"]
            ):
                modified_response = modified_response.replace("I suggest", "I assert")
                modified_response = modified_response.replace("might be", "is clearly")
        elif profile.traits.confidence < 0.3:
            modified_response = modified_response.replace("This is", "This might be")
            modified_response = modified_response.replace("Clearly", "Perhaps")
        
        # Adjust aggressiveness
        if profile.traits.debate_aggressiveness > 0.7:
            # Add challenging language
            if not any(
                phrase in modified_response
                for phrase in ["However", "On the contrary", "This overlooks"]
            ):
                modified_response += (
                    "\n\nHowever, this perspective overlooks crucial considerations."
                )
        elif profile.traits.debate_aggressiveness < 0.4:
            # Add diplomatic language
            modified_response = modified_response.replace("wrong", "different")
            modified_response = modified_response.replace("incorrect", "alternative")
        
        return modified_response
    
    def _modify_critique_style(
        self, base_critique: str, profile: PersonalityProfile, target_writer_id: str
    ) -> str:
        """Modify critique based on personality"""
        modified_critique = base_critique
        
        # Adjust critique style
        if profile.behavior.critique_style == "direct":
            # Make more direct
            modified_critique = modified_critique.replace("might consider", "should")
            modified_critique = modified_critique.replace("perhaps", "")
        elif profile.behavior.critique_style == "gentle":
            # Make more gentle
            modified_critique = modified_critique.replace("should", "might consider")
            modified_critique = modified_critique.replace("wrong", "could be improved")
        
        # Adjust based on openness to feedback
        if profile.traits.openness_to_feedback > 0.7:
            # Add collaborative elements
            modified_critique += "\n\nI'm open to discussing these points further and finding common ground."
        elif profile.traits.openness_to_feedback < 0.3:
            # Add defensive elements
            modified_critique += "\n\nWhile these are my observations, I recognize different approaches may also be valid."
        
        return modified_critique
    
    def _get_debate_strategy(self, profile: PersonalityProfile) -> Dict[str, Any]:
        """Get debate strategy based on personality"""
        strategy = {
            "strategy": profile.voice.debate_style.value,
            "aggressiveness": profile.traits.debate_aggressiveness,
            "persuasion_style": profile.voice.persuasion_style.value,
            "collaboration_tendency": profile.traits.collaboration_tendency,
            "confidence": profile.traits.confidence,
            "openness_to_feedback": profile.traits.openness_to_feedback,
            "preferred_approach": self._get_preferred_debate_approach(profile),
        }
        
        return strategy
    
    def _get_preferred_debate_approach(self, profile: PersonalityProfile) -> str:
        """Get preferred debate approach based on personality"""
        if profile.traits.analytical_depth > 0.7:
            return "Focus on logical analysis and evidence"
        elif profile.traits.creativity > 0.7:
            return "Focus on innovative solutions and new perspectives"
        elif profile.traits.collaboration_tendency > 0.7:
            return "Focus on building consensus and finding common ground"
        elif profile.traits.debate_aggressiveness > 0.7:
            return "Focus on challenging assumptions and defending positions"
        else:
            return "Balanced approach considering multiple perspectives"
    
    def _calculate_compatibility_score(
        self, profile1: PersonalityProfile, profile2: PersonalityProfile
    ) -> float:
        """Calculate compatibility score between two writers"""
        # Calculate trait compatibility
        trait_compatibility = 0.0
        trait_count = 0
        
        for trait_name in profile1.traits.__dict__:
            if isinstance(getattr(profile1.traits, trait_name), float):
                value1 = getattr(profile1.traits, trait_name)
                value2 = getattr(profile2.traits, trait_name)
                
                # Calculate compatibility (inverse of absolute difference)
                compatibility = 1.0 - abs(value1 - value2)
                trait_compatibility += compatibility
                trait_count += 1
        
        if trait_count > 0:
            trait_compatibility /= trait_count
        
        # Calculate style compatibility
        style_compatibility = 0.0
        
        if profile1.voice.persuasion_style == profile2.voice.persuasion_style:
            style_compatibility += 0.3
        if profile1.voice.writing_tone == profile2.voice.writing_tone:
            style_compatibility += 0.3
        if profile1.voice.debate_style == profile2.voice.debate_style:
            style_compatibility += 0.4
        
        # Calculate collaboration compatibility
        collab_compatibility = 1.0 - abs(
            profile1.traits.collaboration_tendency
            - profile2.traits.collaboration_tendency
        )
        
        # Weighted average
        overall_compatibility = (
            trait_compatibility * 0.4
            + style_compatibility * 0.3
            + collab_compatibility * 0.3
        )
        
        return overall_compatibility
    
    def _generate_combinations(self, items: List[str], size: int) -> List[List[str]]:
        """Generate all combinations of specified size"""
        if size == 0:
            return [[]]
        if size > len(items):
            return []
        
        combinations = []
        for i in range(len(items)):
            for combo in self._generate_combinations(items[i + 1 :], size - 1):
                combinations.append([items[i]] + combo)
        
        return combinations
    
    def _calculate_team_compatibility(
        self, team: List[str], compatibility_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate overall team compatibility score"""
        if len(team) < 2:
            return 1.0
        
        total_score = 0.0
        pair_count = 0
        
        for i, writer1 in enumerate(team):
            for writer2 in team[i + 1 :]:
                pair = (
                    (writer1, writer2)
                    if (writer1, writer2) in compatibility_matrix
                    else (writer2, writer1)
                )
                if pair in compatibility_matrix:
                    total_score += compatibility_matrix[pair]
                    pair_count += 1
        
        return total_score / pair_count if pair_count > 0 else 0.0
    
    async def process_request(self, state: PersonalityState) -> PersonalityState:
        """Process a personality system request"""
        # Update internal state from workflow state
        self.profiles = state.profiles
        self.personality_templates = state.personality_templates
        
        # Process request with fallback for when compilation fails
        try:
            # Compile workflow if not already done
            if self.compiled_workflow is None:
                if CompiledGraph is not None:
                    self.compiled_workflow = self.workflow.compile()
                else:
                    # Fallback to direct execution
                    return await self._fallback_process_request(state)
            
            # Process request
            result_state = await self.compiled_workflow.ainvoke(state.dict())
        except Exception as e:
            logger.error(f"Error executing compiled workflow: {str(e)}")
            # Fallback to direct execution
            return await self._fallback_process_request(state)
        
        # Update internal state from result
        self.profiles = result_state.get("profiles", {})
        self.personality_templates = result_state.get("personality_templates", {})
        
        return PersonalityState(**result_state)
    
    async def _fallback_process_request(self, state: PersonalityState) -> PersonalityState:
        """Fallback execution method when workflow compilation is not available"""
        try:
            # Route action
            state = await self._route_action(state)
            
            # Execute based on action
            if state.action == "create":
                state = await self._create_profile(state)
            elif state.action == "get":
                state = await self._get_profile(state)
            elif state.action == "prompt":
                state = await self._apply_to_prompt(state)
            elif state.action == "debate":
                state = await self._apply_to_debate(state)
            elif state.action == "critique":
                state = await self._apply_to_critique(state)
            elif state.action == "history":
                state = await self._update_history(state)
            elif state.action == "compatibility":
                state = await self._get_compatibility(state)
            elif state.action == "team":
                state = await self._suggest_team(state)
            else:
                state.error = f"Unknown action: {state.action}"
                state.success = False
            
            return state
        except Exception as e:
            logger.error(f"Error in fallback execution: {str(e)}")
            state.error = str(e)
            state.success = False
            return state