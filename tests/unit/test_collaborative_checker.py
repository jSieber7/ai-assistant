"""
Unit tests for collaborative checker system
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.agents.collaborative_checker import (
    CollaborativeChecker,
    ConsensusLevel,
    DisputeType,
    CheckerAssessment,
    CheckerDispute,
    ConsensusResult,
)
from app.core.agents.checker_agent import CheckerAgent


@pytest.mark.unit
class TestCollaborativeChecker:
    """Test collaborative checker functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.collaborative_checker.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(
                content='{"resolved": true, "resolution": "Test resolution"}'
            )
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def collaborative_checker(self):
        """Create collaborative checker for testing"""
        return CollaborativeChecker(max_discussion_rounds=1)

    @pytest.fixture
    def mock_checkers(self):
        """Create mock checker agents"""
        checkers = []
        focus_areas = ["factual", "style", "structure"]
        for i, focus_area in enumerate(focus_areas):
            checker = AsyncMock(spec=CheckerAgent)
            checker.checker_id = f"checker_{focus_area}"
            checker.focus_area = focus_area
            checker.check_and_improve = AsyncMock(
                return_value={
                    "score": 75 + i * 5,
                    "issues_found": [{"type": "error", "description": f"Issue {i}"}],
                    "improvements": [
                        {"type": "improvement", "description": f"Improvement {i}"}
                    ],
                    "improved_content": f"Improved content by {focus_area} checker",
                    "recommendations": [f"Recommendation {i}"],
                    "error": None,
                }
            )
            checkers.append(checker)
        return checkers

    @pytest.mark.asyncio
    async def test_conduct_collaborative_review(
        self, collaborative_checker, mock_checkers, mock_llm
    ):
        """Test conducting collaborative review"""
        content = "Test content for review"
        context = {"content_type": "technical"}

        result = await collaborative_checker.conduct_collaborative_review(
            content, mock_checkers, context
        )

        assert isinstance(result, ConsensusResult)
        assert isinstance(result.consensus_level, ConsensusLevel)
        assert 0.0 <= result.final_score <= 100.0
        assert result.final_content is not None
        assert isinstance(result.resolved_disputes, list)
        assert isinstance(result.unresolved_disputes, list)
        assert isinstance(result.aggregated_issues, list)
        assert isinstance(result.aggregated_improvements, list)

    @pytest.mark.asyncio
    async def test_gather_individual_assessments(
        self, collaborative_checker, mock_checkers
    ):
        """Test gathering individual assessments"""
        content = "Test content"
        context = {"test": "context"}

        assessments = await collaborative_checker._gather_individual_assessments(
            content, mock_checkers, context
        )

        assert len(assessments) == len(mock_checkers)
        for assessment in assessments:
            assert isinstance(assessment, CheckerAssessment)
            assert assessment.checker_id in [c.checker_id for c in mock_checkers]
            assert 0.0 <= assessment.score <= 100.0
            assert assessment.improved_content is not None

    @pytest.mark.asyncio
    async def test_get_checker_assessment(self, collaborative_checker, mock_checkers):
        """Test getting assessment from a single checker"""
        checker = mock_checkers[0]
        content_dict = {"content": "Test content"}
        context = {"test": "context"}

        assessment = await collaborative_checker._get_checker_assessment(
            checker, content_dict, context
        )

        assert isinstance(assessment, CheckerAssessment)
        assert assessment.checker_id == checker.checker_id
        assert assessment.focus_area == checker.focus_area
        assert 0.0 <= assessment.score <= 100.0
        assert assessment.reasoning is not None

    @pytest.mark.asyncio
    async def test_identify_disputes(self, collaborative_checker):
        """Test identifying disputes between checkers"""
        # Create assessments with different scores
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=90.0,
                issues_found=[],
                improvements=[],
                improved_content="High quality content",
                recommendations=[],
                confidence=0.9,
                reasoning="Good quality",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="style",
                score=60.0,
                issues_found=[],
                improvements=[],
                improved_content="Lower quality content",
                recommendations=[],
                confidence=0.6,
                reasoning="Lower quality",
            ),
        ]

        disputes = await collaborative_checker._identify_disputes(assessments)

        assert isinstance(disputes, list)
        # Should identify score dispute due to large variance
        assert any(d.dispute_type == DisputeType.QUALITY_ASSESSMENT for d in disputes)

    @pytest.mark.asyncio
    async def test_identify_issue_disputes(self, collaborative_checker):
        """Test identifying issue disputes"""
        # Create assessments with conflicting issues
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=75.0,
                issues_found=[
                    {"type": "factual", "description": "Issue A", "severity": "high"}
                ],
                improvements=[],
                improved_content="Content with issues",
                recommendations=[],
                confidence=0.7,
                reasoning="Some issues",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="factual",
                score=75.0,
                issues_found=[
                    {"type": "factual", "description": "Issue A", "severity": "low"}
                ],
                improvements=[],
                improved_content="Content with issues",
                recommendations=[],
                confidence=0.7,
                reasoning="Some issues",
            ),
        ]

        disputes = await collaborative_checker._identify_issue_disputes(assessments)

        assert isinstance(disputes, list)
        # Should identify factual dispute due to different severity
        assert any(d.dispute_type == DisputeType.FACTUAL_DISAGREEMENT for d in disputes)

    @pytest.mark.asyncio
    async def test_resolve_disputes(
        self, collaborative_checker, mock_checkers, mock_llm
    ):
        """Test resolving disputes"""
        # Create a dispute
        dispute = CheckerDispute(
            dispute_id="test_dispute",
            dispute_type=DisputeType.QUALITY_ASSESSMENT,
            involved_checkers=["checker1", "checker2"],
            conflicting_assessments=[],
            description="Test dispute",
            severity=0.5,
        )

        assessments = []
        for checker in mock_checkers:
            assessment = CheckerAssessment(
                checker_id=checker.checker_id,
                focus_area=checker.focus_area,
                score=75.0,
                issues_found=[],
                improvements=[],
                improved_content="Test content",
                recommendations=[],
                confidence=0.7,
                reasoning="Test reasoning",
            )
            assessments.append(assessment)

        resolved, unresolved = await collaborative_checker._resolve_disputes(
            [dispute], assessments, "Test content", {}
        )

        assert isinstance(resolved, list)
        assert isinstance(unresolved, list)

    @pytest.mark.asyncio
    async def test_resolve_single_dispute(self, collaborative_checker, mock_llm):
        """Test resolving a single dispute"""
        dispute = CheckerDispute(
            dispute_id="test_dispute",
            dispute_type=DisputeType.QUALITY_ASSESSMENT,
            involved_checkers=["checker1", "checker2"],
            conflicting_assessments=[],
            description="Test dispute",
            severity=0.5,
        )

        assessments = []
        content = "Test content"

        resolution = await collaborative_checker._resolve_single_dispute(
            dispute, assessments, content, {}
        )

        assert isinstance(resolution, dict)
        assert "resolved" in resolution

    @pytest.mark.asyncio
    async def test_build_consensus(self, collaborative_checker):
        """Test building consensus from assessments"""
        # Create assessments
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=80.0,
                issues_found=[],
                improvements=[],
                improved_content="Good content",
                recommendations=[],
                confidence=0.8,
                reasoning="Good quality",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="style",
                score=75.0,
                issues_found=[],
                improvements=[],
                improved_content="Decent content",
                recommendations=[],
                confidence=0.7,
                reasoning="Decent quality",
            ),
        ]

        consensus = await collaborative_checker._build_consensus(
            assessments, [], [], "Test content"
        )

        assert isinstance(consensus, ConsensusResult)
        assert isinstance(consensus.consensus_level, ConsensusLevel)
        assert 0.0 <= consensus.final_score <= 100.0
        assert consensus.final_content is not None
        assert isinstance(consensus.participant_agreement, dict)

    def test_calculate_content_similarity(self, collaborative_checker):
        """Test calculating content similarity"""
        content1 = "This is test content with some words"
        content2 = "This is test content with different words"
        content3 = "Completely different content here"

        # High similarity
        similarity1 = collaborative_checker._calculate_content_similarity(
            content1, content2
        )
        assert 0.0 <= similarity1 <= 1.0
        assert similarity1 > 0.5

        # Low similarity
        similarity2 = collaborative_checker._calculate_content_similarity(
            content1, content3
        )
        assert 0.0 <= similarity2 <= 1.0
        assert similarity2 < 0.5

        # Identical content
        similarity3 = collaborative_checker._calculate_content_similarity(
            content1, content1
        )
        assert similarity3 == 1.0

    def test_calculate_participant_agreement(self, collaborative_checker):
        """Test calculating participant agreement"""
        # Create assessments
        target_assessment = CheckerAssessment(
            checker_id="target",
            focus_area="factual",
            score=75.0,
            issues_found=[],
            improvements=[],
            improved_content="Target content",
            recommendations=[],
            confidence=0.7,
            reasoning="Target reasoning",
        )

        similar_assessment = CheckerAssessment(
            checker_id="similar",
            focus_area="factual",
            score=80.0,
            issues_found=[],
            improvements=[],
            improved_content="Target content with minor changes",
            recommendations=[],
            confidence=0.8,
            reasoning="Similar reasoning",
        )

        different_assessment = CheckerAssessment(
            checker_id="different",
            focus_area="factual",
            score=50.0,
            issues_found=[],
            improvements=[],
            improved_content="Completely different content",
            recommendations=[],
            confidence=0.5,
            reasoning="Different reasoning",
        )

        all_assessments = [target_assessment, similar_assessment, different_assessment]

        agreement = collaborative_checker._calculate_participant_agreement(
            target_assessment, all_assessments
        )

        assert 0.0 <= agreement <= 1.0
        # Should have moderate agreement due to one similar and one different

    def test_calculate_weighted_score(self, collaborative_checker):
        """Test calculating weighted score"""
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=80.0,
                issues_found=[],
                improvements=[],
                improved_content="Content 1",
                recommendations=[],
                confidence=0.7,
                reasoning="Reasoning 1",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="style",
                score=60.0,
                issues_found=[],
                improvements=[],
                improved_content="Content 2",
                recommendations=[],
                confidence=0.5,
                reasoning="Reasoning 2",
            ),
        ]

        participant_agreement = {"checker1": 0.8, "checker2": 0.4}

        weighted_score = collaborative_checker._calculate_weighted_score(
            assessments, participant_agreement
        )

        assert 0.0 <= weighted_score <= 100.0
        # Should be closer to checker1's score due to higher agreement

    def test_aggregate_issues(self, collaborative_checker):
        """Test aggregating issues from assessments"""
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=75.0,
                issues_found=[
                    {"type": "factual", "description": "Common issue"},
                    {"type": "style", "description": "Unique issue 1"},
                ],
                improvements=[],
                improved_content="Content 1",
                recommendations=[],
                confidence=0.7,
                reasoning="Reasoning 1",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="style",
                score=75.0,
                issues_found=[
                    {"type": "factual", "description": "Common issue"},
                    {"type": "style", "description": "Unique issue 2"},
                ],
                improvements=[],
                improved_content="Content 2",
                recommendations=[],
                confidence=0.7,
                reasoning="Reasoning 2",
            ),
        ]

        aggregated_issues = collaborative_checker._aggregate_issues(assessments)

        assert isinstance(aggregated_issues, list)
        # Should have common issue with frequency 2
        common_issues = [issue for issue in aggregated_issues if issue["frequency"] > 1]
        assert len(common_issues) >= 1
        assert any(issue["description"] == "Common issue" for issue in common_issues)

    def test_aggregate_improvements(self, collaborative_checker):
        """Test aggregating improvements from assessments"""
        assessments = [
            CheckerAssessment(
                checker_id="checker1",
                focus_area="factual",
                score=75.0,
                issues_found=[],
                improvements=[
                    {"type": "clarity", "description": "Common improvement"},
                    {"type": "style", "description": "Unique improvement 1"},
                ],
                improved_content="Content 1",
                recommendations=[],
                confidence=0.7,
                reasoning="Reasoning 1",
            ),
            CheckerAssessment(
                checker_id="checker2",
                focus_area="style",
                score=75.0,
                issues_found=[],
                improvements=[
                    {"type": "clarity", "description": "Common improvement"},
                    {"type": "style", "description": "Unique improvement 2"},
                ],
                improved_content="Content 2",
                recommendations=[],
                confidence=0.7,
                reasoning="Reasoning 2",
            ),
        ]

        aggregated_improvements = collaborative_checker._aggregate_improvements(
            assessments
        )

        assert isinstance(aggregated_improvements, list)
        # Should have common improvement with frequency 2
        common_improvements = [
            imp for imp in aggregated_improvements if imp["frequency"] > 1
        ]
        assert len(common_improvements) >= 1
        assert any(
            imp["description"] == "Common improvement" for imp in common_improvements
        )

    def test_parse_dispute_resolution(self, collaborative_checker):
        """Test parsing dispute resolution response"""
        response = """
        {
            "resolved": true,
            "resolution": "Test resolution description",
            "compromise_position": "Compromise position",
            "reasoning": "Reasoning for resolution"
        }
        """

        result = collaborative_checker._parse_dispute_resolution(response)

        assert result["resolved"] is True
        assert result["resolution"] == "Test resolution description"
        assert result["compromise_position"] == "Compromise position"
        assert result["reasoning"] == "Reasoning for resolution"

    def test_fallback_consensus(self, collaborative_checker, mock_checkers):
        """Test fallback consensus when collaborative review fails"""
        content = "Test content"
        context = {"test": "context"}

        result = collaborative_checker._fallback_consensus(
            content, mock_checkers, context
        )

        assert isinstance(result, ConsensusResult)
        assert result.consensus_level == ConsensusLevel.NO_CONSENSUS
        assert result.final_score == 70.0  # Default score
        assert result.final_content == content  # Original content


@pytest.mark.unit
class TestCheckerAssessment:
    """Test checker assessment data class"""

    def test_checker_assessment_creation(self):
        """Test checker assessment creation"""
        assessment = CheckerAssessment(
            checker_id="test_checker",
            focus_area="factual",
            score=80.0,
            issues_found=[{"type": "error", "description": "Test issue"}],
            improvements=[{"type": "improvement", "description": "Test improvement"}],
            improved_content="Improved content",
            recommendations=["Test recommendation"],
            confidence=0.8,
            reasoning="Test reasoning",
        )

        assert assessment.checker_id == "test_checker"
        assert assessment.focus_area == "factual"
        assert assessment.score == 80.0
        assert assessment.issues_found == [
            {"type": "error", "description": "Test issue"}
        ]
        assert assessment.improvements == [
            {"type": "improvement", "description": "Test improvement"}
        ]
        assert assessment.improved_content == "Improved content"
        assert assessment.recommendations == ["Test recommendation"]
        assert assessment.confidence == 0.8
        assert assessment.reasoning == "Test reasoning"
        assert assessment.timestamp > 0


@pytest.mark.unit
class TestCheckerDispute:
    """Test checker dispute data class"""

    def test_checker_dispute_creation(self):
        """Test checker dispute creation"""
        dispute = CheckerDispute(
            dispute_id="test_dispute",
            dispute_type=DisputeType.FACTUAL_DISAGREEMENT,
            involved_checkers=["checker1", "checker2"],
            conflicting_assessments=[],
            description="Test dispute description",
            severity=0.7,
        )

        assert dispute.dispute_id == "test_dispute"
        assert dispute.dispute_type == DisputeType.FACTUAL_DISAGREEMENT
        assert dispute.involved_checkers == ["checker1", "checker2"]
        assert dispute.conflicting_assessments == []
        assert dispute.description == "Test dispute description"
        assert dispute.severity == 0.7
        assert dispute.timestamp > 0


@pytest.mark.unit
class TestConsensusResult:
    """Test consensus result data class"""

    def test_consensus_result_creation(self):
        """Test consensus result creation"""
        result = ConsensusResult(
            consensus_level=ConsensusLevel.MAJORITY,
            final_score=75.0,
            final_content="Final content",
            resolved_disputes=[],
            unresolved_disputes=[],
            aggregated_issues=[],
            aggregated_improvements=[],
            consensus_reasoning="Test reasoning",
            participant_agreement={"checker1": 0.8, "checker2": 0.6},
        )

        assert result.consensus_level == ConsensusLevel.MAJORITY
        assert result.final_score == 75.0
        assert result.final_content == "Final content"
        assert result.resolved_disputes == []
        assert result.unresolved_disputes == []
        assert result.aggregated_issues == []
        assert result.aggregated_improvements == []
        assert result.consensus_reasoning == "Test reasoning"
        assert result.participant_agreement == {"checker1": 0.8, "checker2": 0.6}
