"""
Unit tests for dynamic agent selector
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.agents.collaboration.dynamic_selector import (
    DynamicAgentSelector,
    TaskAnalysis,
    SelectionResult,
    TaskComplexity,
    ContentType,
    AgentPerformance,
)
from app.core.agents.content.writer_agent import WriterAgent
from app.core.agents.validation.checker_agent import CheckerAgent


@pytest.mark.unit
class TestDynamicAgentSelector:
    """Test dynamic agent selector functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.dynamic_selector.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(
                content='{"complexity": "moderate", "content_type": "technical"}'
            )
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def dynamic_selector(self):
        """Create dynamic selector for testing"""
        return DynamicAgentSelector()

    @pytest.fixture
    def mock_writers(self):
        """Create mock writer agents"""
        writers = {}
        specialties = ["technical", "creative", "analytical"]
        for i, specialty in enumerate(specialties):
            writer = AsyncMock(spec=WriterAgent)
            writer.writer_id = f"writer_{specialty}"
            writer.specialty = specialty
            writers[writer.writer_id] = writer
        return writers

    @pytest.fixture
    def mock_checkers(self):
        """Create mock checker agents"""
        checkers = {}
        focus_areas = ["factual", "style", "structure", "seo"]
        for i, focus_area in enumerate(focus_areas):
            checker = AsyncMock(spec=CheckerAgent)
            checker.checker_id = f"checker_{focus_area}"
            checker.focus_area = focus_area
            checkers[checker.checker_id] = checker
        return checkers

    @pytest.mark.asyncio
    async def test_select_optimal_team(
        self, dynamic_selector, mock_writers, mock_checkers, mock_llm
    ):
        """Test optimal team selection"""
        prompt = "Test prompt for technical documentation"
        context = {"content_type": "technical"}

        result = await dynamic_selector.select_optimal_team(
            prompt, mock_writers, mock_checkers, context
        )

        assert isinstance(result, SelectionResult)
        assert isinstance(result.task_analysis, TaskAnalysis)
        assert len(result.selected_writers) > 0
        assert len(result.selected_checkers) > 0
        assert 0.0 <= result.selection_confidence <= 1.0
        assert "alternative_writers" in result.alternative_options
        assert "alternative_checkers" in result.alternative_options

    @pytest.mark.asyncio
    async def test_analyze_task(self, dynamic_selector, mock_llm):
        """Test task analysis"""
        task_id = "test_task"
        prompt = "Write technical documentation for API"
        context = {"audience": "developers"}

        result = await dynamic_selector._analyze_task(task_id, prompt, context)

        assert isinstance(result, TaskAnalysis)
        assert result.task_id == task_id
        assert result.prompt == prompt
        assert isinstance(result.complexity, TaskComplexity)
        assert isinstance(result.content_type, ContentType)
        assert isinstance(result.required_specialties, list)
        assert isinstance(result.quality_requirements, list)
        assert 0.0 <= result.estimated_effort <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_select_writers(self, dynamic_selector, mock_writers):
        """Test writer selection"""
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test prompt",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical", "creative"],
            quality_requirements=["accuracy", "clarity"],
            estimated_effort=0.6,
            context_keywords=["api", "documentation"],
            selection_rationale="Test analysis",
            confidence=0.8,
        )

        selected_writers, reasoning = await dynamic_selector._select_writers(
            task_analysis, mock_writers
        )

        assert len(selected_writers) > 0
        assert len(selected_writers) <= len(mock_writers)
        assert all(writer_id in mock_writers for writer_id in selected_writers)
        assert isinstance(reasoning, dict)
        assert all(key.startswith("writer_") for key in reasoning.keys())

    @pytest.mark.asyncio
    async def test_select_checkers(self, dynamic_selector, mock_checkers):
        """Test checker selection"""
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test prompt",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual", "accuracy"],
            estimated_effort=0.6,
            context_keywords=["technical"],
            selection_rationale="Test analysis",
            confidence=0.8,
        )

        selected_checkers, reasoning = await dynamic_selector._select_checkers(
            task_analysis, mock_checkers
        )

        assert len(selected_checkers) > 0
        assert len(selected_checkers) <= len(mock_checkers)
        assert all(checker_id in mock_checkers for checker_id in selected_checkers)
        assert isinstance(reasoning, dict)
        assert all(key.startswith("checker_") for key in reasoning.keys())

    @pytest.mark.asyncio
    async def test_score_writer_for_task(self, dynamic_selector, mock_writers):
        """Test writer scoring for task"""
        writer = mock_writers["writer_technical"]
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test prompt",
            complexity=TaskComplexity.COMPLEX,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["accuracy"],
            estimated_effort=0.8,
            context_keywords=["technical"],
            selection_rationale="Test analysis",
            confidence=0.8,
        )

        score = await dynamic_selector._score_writer_for_task(writer, task_analysis)

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_checker_for_task(self, dynamic_selector, mock_checkers):
        """Test checker scoring for task"""
        checker = mock_checkers["checker_factual"]
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test prompt",
            complexity=TaskComplexity.COMPLEX,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual", "accuracy"],
            estimated_effort=0.8,
            context_keywords=["technical"],
            selection_rationale="Test analysis",
            confidence=0.8,
        )

        score = await dynamic_selector._score_checker_for_task(checker, task_analysis)

        assert 0.0 <= score <= 1.0

    def test_get_agent_performance(self, dynamic_selector):
        """Test getting agent performance"""
        agent_id = "test_writer"
        agent_type = "writer"
        specialty = "technical"

        performance = dynamic_selector._get_agent_performance(
            agent_id, agent_type, specialty
        )

        assert isinstance(performance, AgentPerformance)
        assert performance.agent_id == agent_id
        assert performance.agent_type == agent_type
        assert performance.specialty == specialty
        assert performance.total_tasks >= 0
        assert 0.0 <= performance.success_rate <= 1.0
        assert 0.0 <= performance.average_quality_score <= 1.0
        assert performance.average_response_time >= 0.0

    def test_determine_optimal_writer_count(self, dynamic_selector):
        """Test determining optimal writer count"""
        assert (
            dynamic_selector._determine_optimal_writer_count(TaskComplexity.SIMPLE) == 2
        )
        assert (
            dynamic_selector._determine_optimal_writer_count(TaskComplexity.MODERATE)
            == 3
        )
        assert (
            dynamic_selector._determine_optimal_writer_count(TaskComplexity.COMPLEX)
            == 4
        )
        assert (
            dynamic_selector._determine_optimal_writer_count(TaskComplexity.EXPERT) == 4
        )

    def test_determine_optimal_checker_count(self, dynamic_selector):
        """Test determining optimal checker count"""
        assert dynamic_selector._determine_optimal_checker_count(["factual"]) == 2
        assert (
            dynamic_selector._determine_optimal_checker_count(
                ["factual", "style", "structure", "seo"]
            )
            == 3
        )
        assert (
            dynamic_selector._determine_optimal_checker_count(
                ["comprehensive", "factual"]
            )
            == 4
        )

    def test_calculate_selection_confidence(self, dynamic_selector):
        """Test calculating selection confidence"""
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual"],
            estimated_effort=0.5,
            context_keywords=[],
            selection_rationale="Test",
            confidence=0.7,
        )

        selected_writers = ["writer1", "writer2"]
        selected_checkers = ["checker1", "checker2"]

        confidence = dynamic_selector._calculate_selection_confidence(
            task_analysis, selected_writers, selected_checkers
        )

        assert 0.0 <= confidence <= 1.0

    def test_generate_writer_reasoning(self, dynamic_selector):
        """Test generating writer reasoning"""
        writer_id = "writer_technical"
        score = 0.85
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual"],
            estimated_effort=0.5,
            context_keywords=[],
            selection_rationale="Test",
            confidence=0.7,
        )

        reasoning = dynamic_selector._generate_writer_reasoning(
            writer_id, score, task_analysis
        )

        assert writer_id in reasoning
        assert f"{score:.2f}" in reasoning
        assert "Selected" in reasoning

    def test_generate_checker_reasoning(self, dynamic_selector):
        """Test generating checker reasoning"""
        checker_id = "checker_factual"
        score = 0.9
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual"],
            estimated_effort=0.5,
            context_keywords=[],
            selection_rationale="Test",
            confidence=0.7,
        )

        reasoning = dynamic_selector._generate_checker_reasoning(
            checker_id, score, task_analysis
        )

        assert checker_id in reasoning
        assert f"{score:.2f}" in reasoning
        assert "Selected" in reasoning

    def test_update_workload(self, dynamic_selector):
        """Test updating workload tracking"""
        agent_id = "test_agent"

        # Increment workload
        dynamic_selector.update_workload(agent_id, 1)
        assert dynamic_selector._current_workload[agent_id] == 1

        # Decrement workload
        dynamic_selector.update_workload(agent_id, -1)
        assert dynamic_selector._current_workload[agent_id] == 0

        # Should not go below 0
        dynamic_selector.update_workload(agent_id, -1)
        assert dynamic_selector._current_workload[agent_id] == 0

    def test_update_performance(self, dynamic_selector):
        """Test updating performance metrics"""
        agent_id = "test_agent"
        agent_type = "writer"
        specialty = "technical"
        success = True
        quality_score = 0.85
        response_time = 30.0

        dynamic_selector.update_performance(
            agent_id, agent_type, specialty, success, quality_score, response_time
        )

        performance = dynamic_selector._get_agent_performance(
            agent_id, agent_type, specialty
        )
        assert performance.total_tasks == 1
        assert performance.success_rate == 1.0
        assert performance.average_quality_score == 0.85
        assert performance.average_response_time == 30.0

    def test_parse_task_analysis(self, dynamic_selector):
        """Test parsing task analysis response"""
        response = """
        {
            "complexity": "moderate",
            "content_type": "technical",
            "required_specialties": ["technical", "creative"],
            "quality_requirements": ["factual", "style"],
            "estimated_effort": 0.6,
            "context_keywords": ["api", "documentation"],
            "rationale": "Test analysis",
            "confidence": 0.8,
            "metadata": {}
        }
        """

        result = dynamic_selector._parse_task_analysis(response)

        assert result["complexity"] == "moderate"
        assert result["content_type"] == "technical"
        assert result["required_specialties"] == ["technical", "creative"]
        assert result["quality_requirements"] == ["factual", "style"]
        assert result["estimated_effort"] == 0.6
        assert result["context_keywords"] == ["api", "documentation"]
        assert result["rationale"] == "Test analysis"
        assert result["confidence"] == 0.8

    def test_fallback_selection(self, dynamic_selector, mock_writers, mock_checkers):
        """Test fallback selection when analysis fails"""
        task_id = "test_task"
        prompt = "Test prompt"

        result = dynamic_selector._fallback_selection(
            task_id, prompt, mock_writers, mock_checkers
        )

        assert isinstance(result, SelectionResult)
        assert result.task_analysis.task_id == task_id
        assert result.task_analysis.prompt == prompt
        assert result.task_analysis.complexity == TaskComplexity.MODERATE
        assert result.task_analysis.content_type == ContentType.MIXED
        assert result.selection_confidence == 0.3
        assert len(result.selected_writers) > 0
        assert len(result.selected_checkers) > 0


@pytest.mark.unit
class TestTaskAnalysis:
    """Test task analysis data class"""

    def test_task_analysis_creation(self):
        """Test task analysis creation"""
        analysis = TaskAnalysis(
            task_id="test_task",
            prompt="Test prompt",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual"],
            estimated_effort=0.6,
            context_keywords=["test"],
            selection_rationale="Test rationale",
            confidence=0.8,
        )

        assert analysis.task_id == "test_task"
        assert analysis.prompt == "Test prompt"
        assert analysis.complexity == TaskComplexity.MODERATE
        assert analysis.content_type == ContentType.TECHNICAL
        assert analysis.required_specialties == ["technical"]
        assert analysis.quality_requirements == ["factual"]
        assert analysis.estimated_effort == 0.6
        assert analysis.context_keywords == ["test"]
        assert analysis.selection_rationale == "Test rationale"
        assert analysis.confidence == 0.8


@pytest.mark.unit
class TestSelectionResult:
    """Test selection result data class"""

    def test_selection_result_creation(self):
        """Test selection result creation"""
        task_analysis = TaskAnalysis(
            task_id="test",
            prompt="Test",
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.TECHNICAL,
            required_specialties=["technical"],
            quality_requirements=["factual"],
            estimated_effort=0.5,
            context_keywords=[],
            selection_rationale="Test",
            confidence=0.7,
        )

        result = SelectionResult(
            task_analysis=task_analysis,
            selected_writers=["writer1", "writer2"],
            selected_checkers=["checker1", "checker2"],
            selection_confidence=0.8,
            alternative_options={
                "alternative_writers": ["writer3"],
                "alternative_checkers": ["checker3"],
            },
            reasoning={"writer_writer1": "Good match", "checker_checker1": "Required"},
            estimated_performance={"estimated_team_quality": 0.75},
        )

        assert result.task_analysis == task_analysis
        assert result.selected_writers == ["writer1", "writer2"]
        assert result.selected_checkers == ["checker1", "checker2"]
        assert result.selection_confidence == 0.8
        assert "alternative_writers" in result.alternative_options
        assert "alternative_checkers" in result.alternative_options
        assert "writer_writer1" in result.reasoning
        assert "checker_checker1" in result.reasoning
        assert "estimated_team_quality" in result.estimated_performance


@pytest.mark.unit
class TestAgentPerformance:
    """Test agent performance data class"""

    def test_agent_performance_creation(self):
        """Test agent performance creation"""
        performance = AgentPerformance(
            agent_id="test_agent",
            agent_type="writer",
            specialty="technical",
            total_tasks=10,
            success_rate=0.8,
            average_quality_score=0.75,
            average_response_time=30.0,
            last_used=1234567890,
            strengths=["accuracy", "clarity"],
            weaknesses=["speed"],
            preferred_tasks=["technical"],
            avoided_tasks=["creative"],
        )

        assert performance.agent_id == "test_agent"
        assert performance.agent_type == "writer"
        assert performance.specialty == "technical"
        assert performance.total_tasks == 10
        assert performance.success_rate == 0.8
        assert performance.average_quality_score == 0.75
        assert performance.average_response_time == 30.0
        assert performance.last_used == 1234567890
        assert performance.strengths == ["accuracy", "clarity"]
        assert performance.weaknesses == ["speed"]
        assert performance.preferred_tasks == ["technical"]
        assert performance.avoided_tasks == ["creative"]
