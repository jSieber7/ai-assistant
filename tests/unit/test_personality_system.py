"""
Unit tests for personality system
"""

import pytest
from app.core.agents.personality_system import (
    PersonalitySystem,
    PersonalityProfile,
    PersonalityTraits,
    VoiceCharacteristics,
    BehavioralPatterns,
    PersuasionStyle,
    WritingTone,
    DebateStyle,
)


@pytest.mark.unit
class TestPersonalitySystem:
    """Test personality system functionality"""

    @pytest.fixture
    def personality_system(self):
        """Create personality system for testing"""
        return PersonalitySystem()

    def test_create_profile_with_template(self, personality_system):
        """Test creating profile with template"""
        writer_id = "test_writer"
        specialty = "technical"
        template_name = "confident_expert"

        profile = personality_system.create_profile(
            writer_id=writer_id, specialty=specialty, template_name=template_name
        )

        assert isinstance(profile, PersonalityProfile)
        assert profile.writer_id == writer_id
        assert profile.specialty == specialty
        assert profile.traits.confidence > 0.7  # High confidence for confident_expert
        assert profile.voice.persuasion_style == PersuasionStyle.AUTHORITY
        assert profile.voice.writing_tone == WritingTone.PROFESSIONAL
        assert profile.behavior.critique_style == "direct"
        assert writer_id in personality_system.profiles

    def test_create_profile_with_customizations(self, personality_system):
        """Test creating profile with customizations"""
        writer_id = "test_writer"
        specialty = "creative"

        customizations = {
            "traits": {"creativity": 0.9, "confidence": 0.3},
            "voice": {"writing_tone": "casual", "humor_level": 0.8},
            "behavior": {"collaboration_preference": "leader"},
        }

        profile = personality_system.create_profile(
            writer_id=writer_id, specialty=specialty, customizations=customizations
        )

        assert profile.traits.creativity == 0.9
        assert profile.traits.confidence == 0.3
        assert profile.voice.writing_tone == WritingTone.CASUAL
        assert profile.voice.humor_level == 0.8
        assert profile.behavior.collaboration_preference == "leader"

    def test_get_profile(self, personality_system):
        """Test getting profile"""
        writer_id = "test_writer"
        specialty = "technical"

        # Create profile first
        created_profile = personality_system.create_profile(
            writer_id=writer_id, specialty=specialty
        )

        # Get profile
        retrieved_profile = personality_system.get_profile(writer_id)

        assert retrieved_profile == created_profile
        assert retrieved_profile.writer_id == writer_id
        assert retrieved_profile.specialty == specialty

    def test_get_nonexistent_profile(self, personality_system):
        """Test getting non-existent profile"""
        profile = personality_system.get_profile("nonexistent_writer")
        assert profile is None

    def test_apply_to_prompt(self, personality_system):
        """Test applying personality to prompt"""
        writer_id = "test_writer"
        specialty = "technical"
        base_prompt = "Write about AI technology"

        # Create profile
        personality_system.create_profile(
            writer_id=writer_id, specialty=specialty, template_name="confident_expert"
        )

        # Apply to prompt
        modified_prompt = personality_system.apply_to_prompt(writer_id, base_prompt)

        assert base_prompt in modified_prompt
        assert "Write with confidence and authority" in modified_prompt
        assert "Use a professional tone" in modified_prompt

    def test_apply_to_prompt_no_profile(self, personality_system):
        """Test applying personality to prompt with no profile"""
        base_prompt = "Write about AI technology"

        # Apply to prompt without creating profile
        modified_prompt = personality_system.apply_to_prompt("nonexistent", base_prompt)

        assert modified_prompt == base_prompt

    def test_apply_to_debate_response(self, personality_system):
        """Test applying personality to debate response"""
        writer_id = "test_writer"
        base_response = "I think this approach is good."
        debate_context = {"aggressive_mode": True}

        # Create profile with high aggressiveness
        customizations = {"traits": {"debate_aggressiveness": 0.9}}
        personality_system.create_profile(
            writer_id=writer_id, specialty="technical", customizations=customizations
        )

        # Apply to response
        modified_response = personality_system.apply_to_debate_response(
            writer_id, base_response, debate_context
        )

        assert base_response in modified_response
        # Should be more assertive
        assert "I assert" in modified_response or "This is clearly" in modified_response

    def test_apply_to_critique(self, personality_system):
        """Test applying personality to critique"""
        writer_id = "test_writer"
        base_critique = "This could be improved."
        target_writer_id = "target_writer"

        # Create profile with gentle critique style
        customizations = {"behavior": {"critique_style": "gentle"}}
        personality_system.create_profile(
            writer_id=writer_id, specialty="technical", customizations=customizations
        )

        # Apply to critique
        modified_critique = personality_system.apply_to_critique(
            writer_id, base_critique, target_writer_id
        )

        assert base_critique in modified_critique
        # Should be more gentle
        assert (
            "could be improved" in modified_critique
            or "might consider" in modified_critique
        )

    def test_get_debate_strategy(self, personality_system):
        """Test getting debate strategy"""
        writer_id = "test_writer"

        # Create profile
        personality_system.create_profile(
            writer_id=writer_id,
            specialty="technical",
            template_name="analytical_skeptic",
        )

        # Get strategy
        strategy = personality_system.get_debate_strategy(writer_id)

        assert "strategy" in strategy
        assert "aggressiveness" in strategy
        assert "persuasion_style" in strategy
        assert "collaboration_tendency" in strategy
        assert "confidence" in strategy
        assert strategy["strategy"] == "analytical"
        assert strategy["persuasion_style"] == "logical"

    def test_update_interaction_history(self, personality_system):
        """Test updating interaction history"""
        writer_id = "test_writer"
        interaction_type = "debate"
        outcome = "successful"
        metrics = {"quality_score": 0.85, "collaboration_score": 0.9}

        # Create profile
        personality_system.create_profile(writer_id=writer_id, specialty="technical")

        # Update interaction history
        personality_system.update_interaction_history(
            writer_id, interaction_type, outcome, metrics
        )

        # Check profile was updated
        profile = personality_system.get_profile(writer_id)
        assert len(profile.interaction_history) == 1
        assert profile.interaction_history[0]["type"] == interaction_type
        assert profile.interaction_history[0]["outcome"] == outcome
        assert profile.interaction_history[0]["metrics"] == metrics
        assert "quality_score" in profile.performance_metrics
        assert profile.performance_metrics["quality_score"] == 0.85

    def test_get_compatibility_score(self, personality_system):
        """Test getting compatibility score between writers"""
        writer1_id = "writer1"
        writer2_id = "writer2"

        # Create similar profiles
        personality_system.create_profile(
            writer1_id, "technical", template_name="balanced_collaborator"
        )
        personality_system.create_profile(
            writer2_id, "technical", template_name="balanced_collaborator"
        )

        # Get compatibility score
        score = personality_system.get_compatibility_score(writer1_id, writer2_id)

        assert 0.0 <= score <= 1.0
        # Should be high for similar templates
        assert score > 0.7

    def test_get_compatibility_score_different_profiles(self, personality_system):
        """Test compatibility score with different profiles"""
        writer1_id = "writer1"
        writer2_id = "writer2"

        # Create different profiles
        personality_system.create_profile(
            writer1_id, "technical", template_name="confident_expert"
        )
        personality_system.create_profile(
            writer2_id, "creative", template_name="creative_innovator"
        )

        # Get compatibility score
        score = personality_system.get_compatibility_score(writer1_id, writer2_id)

        assert 0.0 <= score <= 1.0
        # Should be lower for different templates
        assert score < 0.7

    def test_suggest_team_composition(self, personality_system):
        """Test suggesting team composition"""
        # Create multiple profiles
        writers = ["writer1", "writer2", "writer3", "writer4"]
        for i, writer_id in enumerate(writers):
            template = [
                "confident_expert",
                "creative_innovator",
                "balanced_collaborator",
                "analytical_skeptic",
            ][i]
            personality_system.create_profile(
                writer_id, "technical", template_name=template
            )

        # Suggest team
        suggestion = personality_system.suggest_team_composition(writers, "complex")

        assert "suggested_team" in suggestion
        assert "team_compatibility" in suggestion
        assert "reasoning" in suggestion
        assert "compatibility_matrix" in suggestion
        assert len(suggestion["suggested_team"]) == 3  # For complex task

    def test_export_profile(self, personality_system):
        """Test exporting profile"""
        writer_id = "test_writer"

        # Create profile
        personality_system.create_profile(
            writer_id, "technical", template_name="confident_expert"
        )

        # Export profile
        exported = personality_system.export_profile(writer_id)

        assert exported is not None
        assert exported["writer_id"] == writer_id
        assert exported["specialty"] == "technical"
        assert "traits" in exported
        assert "voice" in exported
        assert "behavior" in exported
        assert "created_at" in exported
        assert "last_updated" in exported

    def test_export_nonexistent_profile(self, personality_system):
        """Test exporting non-existent profile"""
        exported = personality_system.export_profile("nonexistent")
        assert exported is None

    def test_import_profile(self, personality_system):
        """Test importing profile"""
        profile_data = {
            "writer_id": "imported_writer",
            "specialty": "technical",
            "traits": {
                "confidence": 0.8,
                "creativity": 0.4,
                "analytical_depth": 0.7,
                "collaboration_tendency": 0.6,
                "debate_aggressiveness": 0.5,
                "openness_to_feedback": 0.7,
                "perfectionism": 0.6,
                "adaptability": 0.5,
                "leadership_tendency": 0.4,
                "risk_tolerance": 0.3,
            },
            "voice": {
                "persuasion_style": "logical",
                "writing_tone": "professional",
                "debate_style": "analytical",
                "vocabulary_complexity": 0.7,
                "sentence_length_preference": 0.6,
                "use_of_metaphors": 0.2,
                "humor_level": 0.1,
                "formality_level": 0.8,
            },
            "behavior": {
                "response_speed": 0.5,
                "critique_style": "direct",
                "collaboration_preference": "balanced",
                "conflict_resolution": "direct",
                "learning_style": "fixed",
                "stress_response": "focused",
            },
            "created_at": 1234567890,
            "last_updated": 1234567890,
            "performance_metrics": {},
        }

        # Import profile
        imported_profile = personality_system.import_profile(profile_data)

        assert isinstance(imported_profile, PersonalityProfile)
        assert imported_profile.writer_id == "imported_writer"
        assert imported_profile.specialty == "technical"
        assert imported_profile.traits.confidence == 0.8
        assert imported_profile.voice.persuasion_style == PersuasionStyle.LOGICAL
        assert imported_profile.behavior.critique_style == "direct"
        assert "imported_writer" in personality_system.profiles


@pytest.mark.unit
class TestPersonalityProfile:
    """Test personality profile data class"""

    def test_personality_profile_creation(self):
        """Test personality profile creation"""
        traits = PersonalityTraits(
            confidence=0.8,
            creativity=0.6,
            analytical_depth=0.7,
            collaboration_tendency=0.5,
            debate_aggressiveness=0.4,
            openness_to_feedback=0.9,
            perfectionism=0.6,
            adaptability=0.7,
            leadership_tendency=0.3,
            risk_tolerance=0.5,
        )

        voice = VoiceCharacteristics(
            persuasion_style=PersuasionStyle.LOGICAL,
            writing_tone=WritingTone.PROFESSIONAL,
            debate_style=DebateStyle.ANALYTICAL,
            vocabulary_complexity=0.7,
            sentence_length_preference=0.6,
            use_of_metaphors=0.2,
            humor_level=0.1,
            formality_level=0.8,
        )

        behavior = BehavioralPatterns(
            response_speed=0.5,
            critique_style="constructive",
            collaboration_preference="balanced",
            conflict_resolution="diplomatic",
            learning_style="adaptive",
            stress_response="focused",
        )

        profile = PersonalityProfile(
            writer_id="test_writer",
            specialty="technical",
            traits=traits,
            voice=voice,
            behavior=behavior,
        )

        assert profile.writer_id == "test_writer"
        assert profile.specialty == "technical"
        assert profile.traits.confidence == 0.8
        assert profile.voice.persuasion_style == PersuasionStyle.LOGICAL
        assert profile.behavior.critique_style == "constructive"
        assert profile.interaction_history == []
        assert profile.performance_metrics == {}


@pytest.mark.unit
class TestPersonalityTraits:
    """Test personality traits data class"""

    def test_personality_traits_creation(self):
        """Test personality traits creation"""
        traits = PersonalityTraits(
            confidence=0.8,
            creativity=0.6,
            analytical_depth=0.7,
            collaboration_tendency=0.5,
            debate_aggressiveness=0.4,
            openness_to_feedback=0.9,
            perfectionism=0.6,
            adaptability=0.7,
            leadership_tendency=0.3,
            risk_tolerance=0.5,
        )

        assert traits.confidence == 0.8
        assert traits.creativity == 0.6
        assert traits.analytical_depth == 0.7
        assert traits.collaboration_tendency == 0.5
        assert traits.debate_aggressiveness == 0.4
        assert traits.openness_to_feedback == 0.9
        assert traits.perfectionism == 0.6
        assert traits.adaptability == 0.7
        assert traits.leadership_tendency == 0.3
        assert traits.risk_tolerance == 0.5

    def test_personality_traits_defaults(self):
        """Test personality traits default values"""
        traits = PersonalityTraits()

        assert traits.confidence == 0.5
        assert traits.creativity == 0.5
        assert traits.analytical_depth == 0.5
        assert traits.collaboration_tendency == 0.5
        assert traits.debate_aggressiveness == 0.5
        assert traits.openness_to_feedback == 0.5
        assert traits.perfectionism == 0.5
        assert traits.adaptability == 0.5
        assert traits.leadership_tendency == 0.5
        assert traits.risk_tolerance == 0.5


@pytest.mark.unit
class TestVoiceCharacteristics:
    """Test voice characteristics data class"""

    def test_voice_characteristics_creation(self):
        """Test voice characteristics creation"""
        voice = VoiceCharacteristics(
            persuasion_style=PersuasionStyle.LOGICAL,
            writing_tone=WritingTone.PROFESSIONAL,
            debate_style=DebateStyle.ANALYTICAL,
            vocabulary_complexity=0.7,
            sentence_length_preference=0.6,
            use_of_metaphors=0.2,
            humor_level=0.1,
            formality_level=0.8,
        )

        assert voice.persuasion_style == PersuasionStyle.LOGICAL
        assert voice.writing_tone == WritingTone.PROFESSIONAL
        assert voice.debate_style == DebateStyle.ANALYTICAL
        assert voice.vocabulary_complexity == 0.7
        assert voice.sentence_length_preference == 0.6
        assert voice.use_of_metaphors == 0.2
        assert voice.humor_level == 0.1
        assert voice.formality_level == 0.8

    def test_voice_characteristics_defaults(self):
        """Test voice characteristics default values"""
        voice = VoiceCharacteristics()

        assert voice.persuasion_style == PersuasionStyle.LOGICAL
        assert voice.writing_tone == WritingTone.PROFESSIONAL
        assert voice.debate_style == DebateStyle.CONSTRUCTIVE
        assert voice.vocabulary_complexity == 0.5
        assert voice.sentence_length_preference == 0.5
        assert voice.use_of_metaphors == 0.3
        assert voice.humor_level == 0.2
        assert voice.formality_level == 0.7


@pytest.mark.unit
class TestBehavioralPatterns:
    """Test behavioral patterns data class"""

    def test_behavioral_patterns_creation(self):
        """Test behavioral patterns creation"""
        behavior = BehavioralPatterns(
            response_speed=0.7,
            critique_style="direct",
            collaboration_preference="leader",
            conflict_resolution="direct",
            learning_style="fixed",
            stress_response="focused",
        )

        assert behavior.response_speed == 0.7
        assert behavior.critique_style == "direct"
        assert behavior.collaboration_preference == "leader"
        assert behavior.conflict_resolution == "direct"
        assert behavior.learning_style == "fixed"
        assert behavior.stress_response == "focused"

    def test_behavioral_patterns_defaults(self):
        """Test behavioral patterns default values"""
        behavior = BehavioralPatterns()

        assert behavior.response_speed == 0.5
        assert behavior.critique_style == "constructive"
        assert behavior.collaboration_preference == "balanced"
        assert behavior.conflict_resolution == "diplomatic"
        assert behavior.learning_style == "adaptive"
        assert behavior.stress_response == "focused"
