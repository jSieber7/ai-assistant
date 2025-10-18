"""
Unit tests for master checker agent
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.core.agents.master_checker import (
    MasterCheckerAgent,
    MasterCheckerAssessment,
    CheckerEvaluation,
    ConflictResolutionStrategy,
    ValidationLevel,
)
from app.core.agents.checker_agent import CheckerAgent


@pytest.mark.unit
class TestMasterCheckerAgent:
    """Test master checker agent functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.master_checker.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(
                content='{"resolved_score": 85.0, "reasoning": "Test reasoning"}'
            )
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def master_checker(self):
        """Create master checker for testing"""
        return MasterCheckerAgent(validation_level=ValidationLevel.STANDARD)

    @pytest.fixture
    def mock_checker_results(self):
        """Create mock checker results"""
        return [
            {
                "checker_id": "factual_checker",
                "focus_area": "factual",
                "score": 85.0,
                "issues_found": [
                    {
                        "type": "factual",
                        "description": "Minor factual issue",
                        "severity": "low",
                    }
                ],
                "improvements": [
                    {
                        "type": "clarity",
                        "description": "Improve clarity",
                        "priority": "medium",
                    }
                ],
                "improved_content": "Improved content by factual checker",
                "recommendations": ["Verify claims"],
                "error": None,
            },
            {
                "checker_id": "style_checker",
                "focus_area": "style",
                "score": 75.0,
                "issues_found": [
                    {
                        "type": "style",
                        "description": "Style inconsistency",
                        "severity": "medium",
                    }
                ],
                "improvements": [
                    {
                        "type": "style",
                        "description": "Fix style issues",
                        "priority": "high",
                    }
                ],
                "improved_content": "Improved content by style checker",
                "recommendations": ["Improve writing style"],
                "error": None,
            },
        ]

    @pytest.mark.asyncio
    async def test_comprehensive_assessment(
        self, master_checker, mock_checker_results, mock_llm
    ):
        """Test comprehensive assessment"""
        content = "Test content for assessment"
        context = {"content_type": "technical"}

        assessment = await master_checker.comprehensive_assessment(
            content, mock_checker_results, context
        )

        assert isinstance(assessment, MasterCheckerAssessment)
        assert assessment.content == content
        assert 0.0 <= assessment.overall_score <= 100.0
        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.validation_level == ValidationLevel.STANDARD
        assert isinstance(assessment.detailed_analysis, dict)
        assert isinstance(assessment.conflict_resolutions, list)
        assert isinstance(assessment.quality_dimensions, dict)
        assert isinstance(assessment.improvement_recommendations, list)
        assert assessment.final_content is not None
        assert isinstance(assessment.checker_evaluations, dict)
        assert assessment.reasoning is not None
        assert assessment.assessment_id is not None

    @pytest.mark.asyncio
    async def test_analyze_checker_results(self, master_checker, mock_checker_results):
        """Test analyzing checker results"""
        analysis = await master_checker._analyze_checker_results(mock_checker_results)

        assert "total_checkers" in analysis
        assert "checker_scores" in analysis
        assert "checker_issues" in analysis
        assert "checker_improvements" in analysis
        assert "score_variance" in analysis
        assert "common_issues" in analysis
        assert "divergent_assessments" in analysis
        assert "consensus_level" in analysis

        assert analysis["total_checkers"] == 2
        assert len(analysis["checker_scores"]) == 2
        assert analysis["score_variance"] == 10.0  # 85.0 - 75.0
        assert analysis["consensus_level"] == 0.7  # Based on variance

    @pytest.mark.asyncio
    async def test_resolve_conflicts(
        self, master_checker, mock_checker_results, mock_llm
    ):
        """Test resolving conflicts between checkers"""
        # Create analysis with conflicts
        checker_analysis = {
            "total_checkers": 2,
            "checker_scores": {"factual_checker": 85.0, "style_checker": 60.0},
            "score_variance": 25.0,  # High variance
            "consensus_level": 0.3,
        }

        content = "Test content"
        context = {"test": "context"}

        conflicts = await master_checker._resolve_conflicts(
            checker_analysis, content, context
        )

        assert isinstance(conflicts, list)
        # Should have identified score conflict due to high variance
        assert any(c["type"] == "score_disagreement" for c in conflicts)
        assert all("resolution" in c for c in conflicts)

    @pytest.mark.asyncio
    async def test_resolve_score_conflict(self, master_checker, mock_llm):
        """Test resolving score conflict"""
        checker_analysis = {
            "checker_scores": {"factual_checker": 85.0, "style_checker": 60.0},
            "consensus_level": 0.3,
        }

        content = "Test content"

        resolution = await master_checker._resolve_score_conflict(
            checker_analysis, content
        )

        assert isinstance(resolution, dict)
        assert "resolved_score" in resolution
        assert "reasoning" in resolution
        assert "confidence" in resolution
        assert 0.0 <= resolution["resolved_score"] <= 100.0
        assert 0.0 <= resolution["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_identify_content_conflicts(self, master_checker):
        """Test identifying content conflicts"""
        checker_analysis = {
            "checker_improvements": {
                "factual_checker": [
                    {"type": "clarity", "description": "Improve clarity"},
                    {"type": "accuracy", "description": "Improve accuracy"},
                ],
                "style_checker": [
                    {"type": "clarity", "description": "Improve writing clarity"},
                    {"type": "tone", "description": "Adjust tone"},
                ],
            }
        }

        conflicts = await master_checker._identify_content_conflicts(checker_analysis)

        assert isinstance(conflicts, list)
        # Should identify conflict for clarity (different descriptions)
        assert any(c["conflict_type"] == "clarity" for c in conflicts)
        assert all("conflicting_recommendations" in c for c in conflicts)

    @pytest.mark.asyncio
    async def test_apply_resolution_strategy(self, master_checker, mock_llm):
        """Test applying resolution strategy"""
        conflict = {
            "type": "score_disagreement",
            "description": "Score disagreement test",
            "severity": 0.5,
        }

        checker_analysis = {
            "checker_scores": {"factual_checker": 85.0, "style_checker": 75.0},
            "consensus_level": 0.7,
        }

        content = "Test content"
        context = {"test": "context"}

        # Test different strategies
        strategies = [
            ConflictResolutionStrategy.WEIGHTED_AVERAGE,
            ConflictResolutionStrategy.AUTHORITATIVE,
            ConflictResolutionStrategy.CONSENSUS_BUILDING,
        ]

        for strategy in strategies:
            master_checker.conflict_resolution_strategy = strategy
            resolution = await master_checker._apply_resolution_strategy(
                conflict, checker_analysis, content, context
            )

            assert isinstance(resolution, dict)

    @pytest.mark.asyncio
    async def test_weighted_average_resolution(self, master_checker):
        """Test weighted average resolution strategy"""
        conflict = {
            "type": "score_disagreement",
            "description": "Score disagreement test",
            "severity": 0.5,
        }

        checker_analysis = {
            "checker_scores": {"factual_checker": 85.0, "style_checker": 75.0},
            "consensus_level": 0.7,
        }

        resolution = await master_checker._weighted_average_resolution(
            conflict, checker_analysis
        )

        assert isinstance(resolution, dict)
        assert "resolved_score" in resolution
        assert "reasoning" in resolution
        assert "confidence" in resolution
        assert "weights_applied" in resolution
        # Should be between the two scores
        assert 75.0 <= resolution["resolved_score"] <= 85.0

    @pytest.mark.asyncio
    async def test_perform_detailed_analysis(self, master_checker, mock_llm):
        """Test performing detailed content analysis"""
        content = "Test content for detailed analysis"
        context = {"content_type": "technical"}

        analysis = await master_checker._perform_detailed_analysis(content, context)

        assert isinstance(analysis, dict)
        assert "content_summary" in analysis
        assert "key_points" in analysis
        assert "strengths" in analysis
        assert "weaknesses" in analysis
        assert "structure_analysis" in analysis
        assert "language_analysis" in analysis
        assert "content_gaps" in analysis
        assert "improvement_areas" in analysis

    @pytest.mark.asyncio
    async def test_evaluate_quality_dimensions(self, master_checker):
        """Test evaluating quality dimensions"""
        content = "Test content"
        detailed_analysis = {
            "language_analysis": {
                "clarity": "good",
                "tone": "professional",
                "style": "formal",
            }
        }

        checker_analysis = {
            "checker_scores": {"factual_checker": 85.0, "style_checker": 75.0}
        }

        quality_dimensions = await master_checker._evaluate_quality_dimensions(
            content, detailed_analysis, checker_analysis
        )

        assert isinstance(quality_dimensions, dict)
        # Should have all standard quality dimensions
        for dimension in master_checker.quality_dimensions:
            assert dimension in quality_dimensions
            assert 0.0 <= quality_dimensions[dimension] <= 100.0

    @pytest.mark.asyncio
    async def test_generate_improvement_recommendations(self, master_checker):
        """Test generating improvement recommendations"""
        content = "Test content"
        quality_dimensions = {
            "accuracy": 60.0,  # Low score
            "clarity": 80.0,  # High score
            "structure": 70.0,  # Medium score
        }
        detailed_analysis = {
            "weaknesses": ["Lacks detail", "Unclear structure"],
            "improvement_areas": ["Add examples", "Improve flow"],
        }

        recommendations = await master_checker._generate_improvement_recommendations(
            content, quality_dimensions, detailed_analysis
        )

        assert isinstance(recommendations, list)
        # Should have recommendation for low-scoring dimension
        assert any(r["type"] == "accuracy" for r in recommendations)
        # Should have recommendations from detailed analysis
        assert any(
            "detail" in r["description"] or "flow" in r["description"]
            for r in recommendations
        )

    @pytest.mark.asyncio
    async def test_create_final_content(self, master_checker):
        """Test creating final improved content"""
        original_content = "Original content"
        improvement_recommendations = [
            {
                "type": "clarity",
                "description": "Improve clarity",
                "action": "Add explanations",
            },
            {
                "type": "structure",
                "description": "Improve structure",
                "action": "Organize sections",
            },
        ]
        conflict_resolutions = [
            {"type": "score_disagreement", "resolution": "Use balanced approach"}
        ]

        final_content = await master_checker._create_final_content(
            original_content, improvement_recommendations, conflict_resolutions
        )

        assert isinstance(final_content, str)
        assert original_content in final_content
        # Should include improvements summary
        assert "IMPROVEMENTS TO BE APPLIED:" in final_content

    @pytest.mark.asyncio
    async def test_calculate_overall_metrics(self, master_checker):
        """Test calculating overall score and confidence"""
        quality_dimensions = {
            "accuracy": 80.0,
            "clarity": 75.0,
            "structure": 85.0,
            "style": 70.0,
        }

        checker_analysis = {
            "checker_scores": {"factual_checker": 85.0, "style_checker": 75.0},
            "consensus_level": 0.7,
        }

        conflict_resolutions = [{"resolution": "Test resolution", "confidence": 0.8}]

        overall_score, confidence = await master_checker._calculate_overall_metrics(
            quality_dimensions, checker_analysis, conflict_resolutions
        )

        assert 0.0 <= overall_score <= 100.0
        assert 0.0 <= confidence <= 1.0
        # Overall score should be average of dimensions
        expected_score = sum(quality_dimensions.values()) / len(quality_dimensions)
        assert abs(overall_score - expected_score) < 5.0

    @pytest.mark.asyncio
    async def test_evaluate_subordinate_checkers(self, master_checker):
        """Test evaluating subordinate checkers"""
        checker_results = [
            {
                "checker_id": "factual_checker",
                "focus_area": "factual",
                "score": 85.0,
                "issues_found": [{"type": "factual", "description": "Minor issue"}],
                "improved_content": "Improved content",
                "recommendations": ["Good recommendation"],
            },
            {
                "checker_id": "style_checker",
                "focus_area": "style",
                "score": 70.0,
                "issues_found": [
                    {"type": "style", "description": "Style issue"},
                    {"type": "tone", "description": "Tone issue"},
                ],
                "improved_content": "Improved content",
                "recommendations": ["Recommendation"],
            },
        ]

        detailed_analysis = {"content_analysis": "Good content with some issues"}

        final_content = "Final improved content"

        evaluations = await master_checker._evaluate_subordinate_checkers(
            checker_results, detailed_analysis, final_content
        )

        assert isinstance(evaluations, dict)
        assert len(evaluations) == 2
        assert "factual_checker" in evaluations
        assert "style_checker" in evaluations

        for checker_id, score in evaluations.items():
            assert 0.0 <= score <= 100.0

    def test_get_performance_report(self, master_checker):
        """Test getting performance report"""
        # Create some assessment history
        for i in range(5):
            assessment = MasterCheckerAssessment(
                assessment_id=f"assessment_{i}",
                content="Test content",
                overall_score=75.0 + i,
                confidence=0.8,
                validation_level=ValidationLevel.STANDARD,
                detailed_analysis={},
                conflict_resolutions=[],
                quality_dimensions={},
                improvement_recommendations=[],
                final_content="Final content",
                checker_evaluations={},
                reasoning="Test reasoning",
            )
            master_checker.assessment_history.append(assessment)

        report = master_checker.get_performance_report()

        assert "total_assessments" in report
        assert "average_score" in report
        assert "average_confidence" in report
        assert "common_conflicts" in report
        assert "validation_level" in report
        assert "conflict_resolution_strategy" in report

        assert report["total_assessments"] == 5
        assert 75.0 <= report["average_score"] <= 79.0  # Average of 75-79

    def test_create_fallback_assessment(self, master_checker):
        """Test creating fallback assessment"""
        assessment_id = "fallback_test"
        content = "Test content"
        checker_results = [{"checker_id": "test_checker", "score": 75.0, "error": None}]

        assessment = master_checker._create_fallback_assessment(
            assessment_id, content, checker_results
        )

        assert isinstance(assessment, MasterCheckerAssessment)
        assert assessment.assessment_id == assessment_id
        assert assessment.content == content
        assert assessment.validation_level == ValidationLevel.BASIC
        assert assessment.confidence == 0.3  # Low confidence for fallback
        assert assessment.reasoning == "Fallback assessment due to system failure"

    def test_parse_resolution_response(self, master_checker):
        """Test parsing resolution response"""
        response = """
        {
            "resolved_score": 85.0,
            "reasoning": "Test reasoning for resolution",
            "confidence": 0.8,
            "weights_applied": {
                "factual_checker": 0.6,
                "style_checker": 0.4
            }
        }
        """

        result = master_checker._parse_resolution_response(response)

        assert result["resolved_score"] == 85.0
        assert result["reasoning"] == "Test reasoning for resolution"
        assert result["confidence"] == 0.8
        assert "weights_applied" in result
        assert result["weights_applied"]["factual_checker"] == 0.6

    def test_parse_analysis_response(self, master_checker):
        """Test parsing analysis response"""
        response = """
        {
            "content_summary": "Test summary",
            "key_points": ["Point 1", "Point 2"],
            "strengths": ["Strength 1"],
            "weaknesses": ["Weakness 1"],
            "structure_analysis": {
                "introduction": "Good",
                "body": "Needs work",
                "conclusion": "Good"
            },
            "language_analysis": {
                "clarity": "Clear",
                "tone": "Professional",
                "style": "Formal"
            },
            "content_gaps": ["Gap 1"],
            "improvement_areas": ["Area 1"]
        }
        """

        result = master_checker._parse_analysis_response(response)

        assert result["content_summary"] == "Test summary"
        assert result["key_points"] == ["Point 1", "Point 2"]
        assert result["strengths"] == ["Strength 1"]
        assert result["weaknesses"] == ["Weakness 1"]
        assert "structure_analysis" in result
        assert "language_analysis" in result
        assert result["content_gaps"] == ["Gap 1"]
        assert result["improvement_areas"] == ["Area 1"]


@pytest.mark.unit
class TestMasterCheckerAssessment:
    """Test master checker assessment data class"""

    def test_master_checker_assessment_creation(self):
        """Test master checker assessment creation"""
        assessment = MasterCheckerAssessment(
            assessment_id="test_assessment",
            content="Test content",
            overall_score=80.0,
            confidence=0.85,
            validation_level=ValidationLevel.STANDARD,
            detailed_analysis={"analysis": "test"},
            conflict_resolutions=[],
            quality_dimensions={"accuracy": 85.0},
            improvement_recommendations=[],
            final_content="Final content",
            checker_evaluations={"checker1": 80.0},
            reasoning="Test reasoning",
        )

        assert assessment.assessment_id == "test_assessment"
        assert assessment.content == "Test content"
        assert assessment.overall_score == 80.0
        assert assessment.confidence == 0.85
        assert assessment.validation_level == ValidationLevel.STANDARD
        assert assessment.detailed_analysis == {"analysis": "test"}
        assert assessment.conflict_resolutions == []
        assert assessment.quality_dimensions == {"accuracy": 85.0}
        assert assessment.improvement_recommendations == []
        assert assessment.final_content == "Final content"
        assert assessment.checker_evaluations == {"checker1": 80.0}
        assert assessment.reasoning == "Test reasoning"
        assert assessment.timestamp > 0


@pytest.mark.unit
class TestCheckerEvaluation:
    """Test checker evaluation data class"""

    def test_checker_evaluation_creation(self):
        """Test checker evaluation creation"""
        evaluation = CheckerEvaluation(
            checker_id="test_checker",
            focus_area="factual",
            assessment_accuracy=0.85,
            reasoning_quality=0.8,
            thoroughness=0.75,
            consistency=0.9,
            overall_reliability=0.82,
            strengths=["Accuracy", "Thoroughness"],
            weaknesses=["Speed"],
            recommendations=["Improve response time"],
        )

        assert evaluation.checker_id == "test_checker"
        assert evaluation.focus_area == "factual"
        assert evaluation.assessment_accuracy == 0.85
        assert evaluation.reasoning_quality == 0.8
        assert evaluation.thoroughness == 0.75
        assert evaluation.consistency == 0.9
        assert evaluation.overall_reliability == 0.82
        assert evaluation.strengths == ["Accuracy", "Thoroughness"]
        assert evaluation.weaknesses == ["Speed"]
        assert evaluation.recommendations == ["Improve response time"]
