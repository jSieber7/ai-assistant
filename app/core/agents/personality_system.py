"""
Personality and voice modeling system for writer agents.

This module implements personality profiles that affect writing style,
debate behavior, and collaboration patterns.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets

logger = logging.getLogger(__name__)


class PersuasionStyle(Enum):
    """Persuasion styles for debate and writing"""

    LOGICAL = "logical"
    EMOTIONAL = "emotional"
    AUTHORITY = "authority"
    COLLABORATIVE = "collaborative"
    CHALLENGING = "challenging"


class WritingTone(Enum):
    """Writing tones"""

    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"


class DebateStyle(Enum):
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


class PersonalitySystem:
    """Manages personality profiles and applies them to agent behavior"""

    def __init__(self):
        self.profiles: Dict[str, PersonalityProfile] = {}
        self.personality_templates = self._create_personality_templates()

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

    def create_profile(
        self,
        writer_id: str,
        specialty: str,
        template_name: Optional[str] = None,
        customizations: Optional[Dict[str, Any]] = None,
    ) -> PersonalityProfile:
        """Create a personality profile for a writer"""

        # Select template or create default
        if template_name and template_name in self.personality_templates:
            template = self.personality_templates[template_name]
            traits = PersonalityTraits(**template["traits"].__dict__)
            voice = VoiceCharacteristics(**template["voice"].__dict__)
            behavior = BehavioralPatterns(**template["behavior"].__dict__)
        else:
            # Create balanced default profile
            traits = PersonalityTraits()
            voice = VoiceCharacteristics()
            behavior = BehavioralPatterns()

        # Apply customizations
        if customizations:
            traits = self._apply_trait_customizations(
                traits, customizations.get("traits", {})
            )
            voice = self._apply_voice_customizations(
                voice, customizations.get("voice", {})
            )
            behavior = self._apply_behavior_customizations(
                behavior, customizations.get("behavior", {})
            )

        # Add some randomness for uniqueness
        traits = self._add_personality_variance(traits)

        profile = PersonalityProfile(
            writer_id=writer_id,
            specialty=specialty,
            traits=traits,
            voice=voice,
            behavior=behavior,
        )

        self.profiles[writer_id] = profile
        logger.info(
            f"Created personality profile for {writer_id} using template: {template_name or 'default'}"
        )

        return profile

    def get_profile(self, writer_id: str) -> Optional[PersonalityProfile]:
        """Get personality profile for a writer"""
        return self.profiles.get(writer_id)

    def apply_to_prompt(
        self, writer_id: str, base_prompt: str, context: Dict[str, Any] = None
    ) -> str:
        """Apply personality to a prompt"""
        profile = self.get_profile(writer_id)
        if not profile:
            return base_prompt

        personality_instructions = self._generate_personality_instructions(
            profile, context
        )

        return f"{base_prompt}\n\n{personality_instructions}"

    def apply_to_debate_response(
        self, writer_id: str, base_response: str, debate_context: Dict[str, Any]
    ) -> str:
        """Apply personality to debate response"""
        profile = self.get_profile(writer_id)
        if not profile:
            return base_response

        # Adjust response based on personality
        modified_response = self._modify_debate_response(
            base_response, profile, debate_context
        )

        return modified_response

    def apply_to_critique(
        self, writer_id: str, base_critique: str, target_writer_id: str
    ) -> str:
        """Apply personality to critique style"""
        profile = self.get_profile(writer_id)
        if not profile:
            return base_critique

        # Modify critique based on personality
        modified_critique = self._modify_critique_style(
            base_critique, profile, target_writer_id
        )

        return modified_critique

    def get_debate_strategy(self, writer_id: str) -> Dict[str, Any]:
        """Get debate strategy based on personality"""
        profile = self.get_profile(writer_id)
        if not profile:
            return {"strategy": "balanced", "aggressiveness": 0.5}

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

    def update_interaction_history(
        self,
        writer_id: str,
        interaction_type: str,
        outcome: str,
        metrics: Dict[str, float],
    ):
        """Update interaction history for learning"""
        profile = self.get_profile(writer_id)
        if not profile:
            return

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
        self, profile: PersonalityProfile, context: Dict[str, Any] = None
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

    def get_compatibility_score(self, writer1_id: str, writer2_id: str) -> float:
        """Calculate compatibility score between two writers"""
        profile1 = self.get_profile(writer1_id)
        profile2 = self.get_profile(writer2_id)

        if not profile1 or not profile2:
            return 0.5

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

    def suggest_team_composition(
        self, available_writers: List[str], task_complexity: str = "moderate"
    ) -> Dict[str, Any]:
        """Suggest optimal team composition based on personalities"""
        if len(available_writers) < 2:
            return {
                "suggestion": available_writers,
                "reasoning": "Insufficient writers for team analysis",
            }

        # Calculate all compatibility scores
        compatibility_matrix = {}
        for i, writer1 in enumerate(available_writers):
            for writer2 in available_writers[i + 1 :]:
                score = self.get_compatibility_score(writer1, writer2)
                compatibility_matrix[(writer1, writer2)] = score

        # Find most compatible pairs
        # Suggest team based on complexity
        team_size = 3 if task_complexity in ["complex", "expert"] else 2

        # Build team with highest overall compatibility
        best_team = []
        best_score = 0.0

        for combination in self._generate_combinations(available_writers, team_size):
            team_score = self._calculate_team_compatibility(
                combination, compatibility_matrix
            )
            if team_score > best_score:
                best_score = team_score
                best_team = combination

        return {
            "suggested_team": best_team,
            "team_compatibility": best_score,
            "reasoning": f"Selected for optimal compatibility score of {best_score:.2f}",
            "compatibility_matrix": compatibility_matrix,
        }

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

    def export_profile(self, writer_id: str) -> Optional[Dict[str, Any]]:
        """Export personality profile as dictionary"""
        profile = self.get_profile(writer_id)
        if not profile:
            return None

        return {
            "writer_id": profile.writer_id,
            "specialty": profile.specialty,
            "traits": profile.traits.__dict__,
            "voice": {
                **profile.voice.__dict__,
                "persuasion_style": profile.voice.persuasion_style.value,
                "writing_tone": profile.voice.writing_tone.value,
                "debate_style": profile.voice.debate_style.value,
            },
            "behavior": profile.behavior.__dict__,
            "created_at": profile.created_at,
            "last_updated": profile.last_updated,
            "performance_metrics": profile.performance_metrics,
        }

    def import_profile(self, profile_data: Dict[str, Any]) -> PersonalityProfile:
        """Import personality profile from dictionary"""
        # Reconstruct enums
        voice_data = profile_data["voice"].copy()
        voice_data["persuasion_style"] = PersuasionStyle(voice_data["persuasion_style"])
        voice_data["writing_tone"] = WritingTone(voice_data["writing_tone"])
        voice_data["debate_style"] = DebateStyle(voice_data["debate_style"])

        profile = PersonalityProfile(
            writer_id=profile_data["writer_id"],
            specialty=profile_data["specialty"],
            traits=PersonalityTraits(**profile_data["traits"]),
            voice=VoiceCharacteristics(**voice_data),
            behavior=BehavioralPatterns(**profile_data["behavior"]),
            created_at=profile_data.get("created_at", time.time()),
            last_updated=profile_data.get("last_updated", time.time()),
            performance_metrics=profile_data.get("performance_metrics", {}),
        )

        self.profiles[profile.writer_id] = profile
        return profile
