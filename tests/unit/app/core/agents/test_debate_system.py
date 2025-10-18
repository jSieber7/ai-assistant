"""
Unit tests for debate system
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.agents.collaboration.debate_system import (
    DebateOrchestrator,
    DebatePhase,
    DebatePosition,
    DebateCritique,
    DebateState,
)
from app.core.agents.content.writer_agent import WriterAgent


@pytest.mark.unit
class TestDebateOrchestrator:
    """Test debate orchestrator functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.debate_system.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(content="Mock response")
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def debate_orchestrator(self):
        """Create debate orchestrator for testing"""
        return DebateOrchestrator(max_critique_rounds=1)

    @pytest.fixture
    def mock_writers(self):
        """Create mock writer agents"""
        writers = []
        for i in range(3):
            writer = AsyncMock(spec=WriterAgent)
            writer.writer_id = f"writer_{i}"
            writer.specialty = ["technical", "creative", "analytical"][i]
            writer.generate_content = AsyncMock(
                return_value={
                    "content": f"Content from writer {i}",
                    "confidence_score": 0.8,
                    "error": None,
                }
            )
            writers.append(writer)
        return writers

    @pytest.mark.asyncio
    async def test_conduct_debate(self, debate_orchestrator, mock_writers, mock_llm):
        """Test conducting a debate"""
        prompt = "Test debate prompt"
        context = {"test": "context"}

        result = await debate_orchestrator.conduct_debate(prompt, mock_writers, context)

        assert isinstance(result, DebateState)
        assert result.prompt == prompt
        assert result.phase == DebatePhase.FINAL_COLLABORATION
        assert len(result.positions) == 3
        assert result.final_content is not None

    @pytest.mark.asyncio
    async def test_phase_1_initial_positions(
        self, debate_orchestrator, mock_writers, mock_llm
    ):
        """Test initial positions phase"""
        debate_state = DebateState(
            debate_id="test_debate",
            prompt="Test prompt",
            phase=DebatePhase.INITIAL_POSITIONS,
            participants=[w.writer_id for w in mock_writers],
        )

        result = await debate_orchestrator._phase_1_initial_positions(
            debate_state, mock_writers, {}
        )

        assert result.phase == DebatePhase.INITIAL_POSITIONS
        assert len(result.positions) == 3
        for writer_id, position in result.positions.items():
            assert isinstance(position, DebatePosition)
            assert position.content is not None

    @pytest.mark.asyncio
    async def test_phase_2_critique_and_rebuttal(
        self, debate_orchestrator, mock_writers, mock_llm
    ):
        """Test critique and rebuttal phase"""
        # Create positions first
        positions = {}
        for writer in mock_writers:
            positions[writer.writer_id] = DebatePosition(
                writer_id=writer.writer_id,
                content=f"Content from {writer.writer_id}",
                confidence=0.8,
                reasoning="Test reasoning",
            )

        debate_state = DebateState(
            debate_id="test_debate",
            prompt="Test prompt",
            phase=DebatePhase.CRITIQUE_AND_REBUTTAL,
            participants=[w.writer_id for w in mock_writers],
            positions=positions,
        )

        result = await debate_orchestrator._phase_2_critique_and_rebuttal(
            debate_state, mock_writers, {}
        )

        assert result.phase == DebatePhase.CRITIQUE_AND_REBUTTAL
        # Should have critiques between writers
        total_critiques = sum(len(critiques) for critiques in result.critiques.values())
        assert total_critiques > 0

    @pytest.mark.asyncio
    async def test_phase_3_synthesis(self, debate_orchestrator, mock_writers, mock_llm):
        """Test synthesis phase"""
        positions = {}
        for writer in mock_writers:
            positions[writer.writer_id] = DebatePosition(
                writer_id=writer.writer_id,
                content=f"Content from {writer.writer_id}",
                confidence=0.8,
                reasoning="Test reasoning",
            )

        debate_state = DebateState(
            debate_id="test_debate",
            prompt="Test prompt",
            phase=DebatePhase.SYNTHESIS,
            participants=[w.writer_id for w in mock_writers],
            positions=positions,
            critiques={},
        )

        result = await debate_orchestrator._phase_3_synthesis(
            debate_state, mock_writers, {}
        )

        assert result.phase == DebatePhase.SYNTHESIS
        assert result.synthesis is not None

    @pytest.mark.asyncio
    async def test_phase_4_final_collaboration(
        self, debate_orchestrator, mock_writers, mock_llm
    ):
        """Test final collaboration phase"""
        debate_state = DebateState(
            debate_id="test_debate",
            prompt="Test prompt",
            phase=DebatePhase.FINAL_COLLABORATION,
            participants=[w.writer_id for w in mock_writers],
            synthesis="Test synthesis content",
        )

        result = await debate_orchestrator._phase_4_final_collaboration(
            debate_state, mock_writers, {}
        )

        assert result.phase == DebatePhase.FINAL_COLLABORATION
        assert result.final_content is not None

    def test_create_position_prompt(self, debate_orchestrator):
        """Test position prompt creation"""
        prompt = "Test prompt"
        context = {"key": "value"}

        result = debate_orchestrator._create_position_prompt(prompt, context)

        assert prompt in result
        assert "ORIGINAL PROMPT:" in result
        assert "CONTEXT:" in result

    def test_create_critique_prompt(self, debate_orchestrator):
        """Test critique prompt creation"""
        critic_specialty = "technical"
        target_specialty = "creative"
        target_content = "Target content"
        original_prompt = "Original prompt"
        round_num = 1

        result = debate_orchestrator._create_critique_prompt(
            critic_specialty,
            target_specialty,
            target_content,
            original_prompt,
            round_num,
        )

        assert critic_specialty in result
        assert target_specialty in result
        assert target_content in result
        assert original_prompt in result
        assert f"Round {round_num + 1}" in result

    def test_parse_critique_response(self, debate_orchestrator):
        """Test parsing critique response"""
        response = """
        CRITIQUE POINTS:
        - Point 1
        - Point 2
        
        STRENGTHS:
        - Strength 1
        - Strength 2
        
        WEAKNESSES:
        - Weakness 1
        
        SUGGESTIONS:
        - Suggestion 1
        
        OVERALL ASSESSMENT:
        Overall assessment here
        """

        result = debate_orchestrator._parse_critique_response(response)

        assert "critique_points" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "suggestions" in result
        assert "overall_assessment" in result
        assert len(result["critique_points"]) == 2
        assert len(result["strengths"]) == 2
        assert result["overall_assessment"] == "Overall assessment here"

    def test_calculate_collaboration_score(self, debate_orchestrator):
        """Test collaboration score calculation"""
        debate_state = DebateState(
            debate_id="test",
            prompt="Test",
            phase=DebatePhase.FINAL_COLLABORATION,
            participants=["writer1", "writer2"],
            synthesis="Test synthesis",
        )

        # Test with synthesis included
        content = "This includes test synthesis content"
        score = debate_orchestrator._score_collaboration(
            content, debate_state, "writer1"
        )
        assert 0.0 <= score <= 1.0

        # Test without synthesis
        content = "Different content"
        score = debate_orchestrator._score_collaboration(
            content, debate_state, "writer1"
        )
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestDebateState:
    """Test debate state data class"""

    def test_debate_state_creation(self):
        """Test debate state creation"""
        state = DebateState(
            debate_id="test_debate",
            prompt="Test prompt",
            phase=DebatePhase.INITIAL_POSITIONS,
            participants=["writer1", "writer2"],
        )

        assert state.debate_id == "test_debate"
        assert state.prompt == "Test prompt"
        assert state.phase == DebatePhase.INITIAL_POSITIONS
        assert state.participants == ["writer1", "writer2"]
        assert state.positions == {}
        assert state.critiques == {}
        assert state.synthesis is None
        assert state.final_content is None


@pytest.mark.unit
class TestDebatePosition:
    """Test debate position data class"""

    def test_debate_position_creation(self):
        """Test debate position creation"""
        position = DebatePosition(
            writer_id="writer1",
            content="Test content",
            confidence=0.8,
            reasoning="Test reasoning",
        )

        assert position.writer_id == "writer1"
        assert position.content == "Test content"
        assert position.confidence == 0.8
        assert position.reasoning == "Test reasoning"
        assert position.timestamp > 0


@pytest.mark.unit
class TestDebateCritique:
    """Test debate critique data class"""

    def test_debate_critique_creation(self):
        """Test debate critique creation"""
        critique = DebateCritique(
            critic_id="writer1",
            target_id="writer2",
            critique_points=["Point 1", "Point 2"],
            strengths=["Strength 1"],
            weaknesses=["Weakness 1"],
            suggestions=["Suggestion 1"],
            overall_assessment="Good overall",
        )

        assert critique.critic_id == "writer1"
        assert critique.target_id == "writer2"
        assert critique.critique_points == ["Point 1", "Point 2"]
        assert critique.strengths == ["Strength 1"]
        assert critique.weaknesses == ["Weakness 1"]
        assert critique.suggestions == ["Suggestion 1"]
        assert critique.overall_assessment == "Good overall"
        assert critique.timestamp > 0
