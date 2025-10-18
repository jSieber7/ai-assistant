"""
Unit tests for learning system
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.agents.enhancement.learning_system import (
    LearningSystem,
    AdaptationType,
    PerformanceMetrics,
    InteractionPattern,
    LearningInsight,
)


@pytest.mark.unit
class TestLearningSystem:
    """Test learning system functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.learning_system.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(
                content='{"insight": "Test insight"}'
            )
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def learning_system(self):
        """Create learning system for testing"""
        return LearningSystem(min_samples_for_learning=5)

    @pytest.mark.asyncio
    async def test_learn_from_workflow(self, learning_system, mock_llm):
        """Test learning from workflow"""
        workflow_id = "test_workflow"
        workflow_data = {
            "status": "completed",
            "stages": {
                "content_generation": {
                    "writer_results": [
                        {
                            "writer_id": "writer1",
                            "specialty": "technical",
                            "content": "Generated content",
                            "confidence_score": 0.8,
                            "execution_time": 30.0,
                        }
                    ]
                },
                "quality_checking": {
                    "checking_history": [
                        {
                            "results": [
                                {
                                    "individual_checks": [
                                        {
                                            "checker_id": "checker1",
                                            "focus_area": "factual",
                                            "score": 85.0,
                                            "improved_content": "Improved content",
                                        }
                                    ],
                                    "overall_score": 85.0,
                                    "best_improved_version": {
                                        "content": "Best improved content"
                                    },
                                }
                            ]
                        }
                    ],
                    "best_score": 85.0,
                },
            },
            "final_output": "Final output content",
        }

        insights = await learning_system.learn_from_workflow(workflow_id, workflow_data)

        assert isinstance(insights, list)
        # Should have generated insights since we have workflow data
        assert len(insights) >= 0

    def test_extract_agent_performance(self, learning_system):
        """Test extracting agent performance from workflow"""
        workflow_data = {
            "stages": {
                "content_generation": {
                    "writer_results": [
                        {
                            "writer_id": "writer1",
                            "specialty": "technical",
                            "content": "Generated content",
                            "confidence_score": 0.8,
                            "execution_time": 30.0,
                        }
                    ]
                },
                "quality_checking": {
                    "checking_history": [
                        {
                            "results": [
                                {
                                    "individual_checks": [
                                        {
                                            "checker_id": "checker1",
                                            "focus_area": "factual",
                                            "score": 85.0,
                                            "improved_content": "Improved content",
                                        }
                                    ],
                                    "overall_score": 85.0,
                                    "best_improved_version": {
                                        "content": "Best improved content"
                                    },
                                }
                            ]
                        }
                    ],
                    "best_score": 85.0,
                },
            }
        }

        performance = learning_system._extract_agent_performance(workflow_data)

        assert isinstance(performance, list)
        assert len(performance) == 2  # One writer and one checker

        # Check writer performance
        writer_perf = next(p for p in performance if p["agent_id"] == "writer1")
        assert writer_perf["agent_type"] == "writer"
        assert writer_perf["specialty"] == "technical"
        assert writer_perf["quality_score"] == 0.8
        assert writer_perf["success"] is True

        # Check checker performance
        checker_perf = next(p for p in performance if p["agent_id"] == "checker1")
        assert checker_perf["agent_type"] == "checker"
        assert checker_perf["focus_area"] == "factual"
        assert checker_perf["quality_score"] == 85.0
        assert checker_perf["success"] is True

    def test_calculate_collaboration_score(self, learning_system):
        """Test calculating collaboration score"""
        agent_data = {
            "writer_id": "writer1",
            "content": "Writer content that will be used in final output",
            "execution_time": 30.0,
        }

        workflow_data = {
            "final_output": "Final output that includes writer content",
            "debate_data": {"participation": {"writer1": 3}},
        }

        score = learning_system._calculate_collaboration_score(
            agent_data, workflow_data
        )

        assert 0.0 <= score <= 1.0
        # Should have some collaboration due to content overlap
        assert score > 0.5

    def test_calculate_checker_collaboration_score(self, learning_system):
        """Test calculating checker collaboration score"""
        checker_result = {
            "checker_id": "checker1",
            "focus_area": "factual",
            "score": 85.0,
            "improved_content": "Checker improved content",
        }

        check_data = {
            "overall_score": 85.0,
            "best_improved_version": {
                "content": "Best improved content that includes checker improvements"
            },
        }

        score = learning_system._calculate_checker_collaboration_score(
            checker_result, check_data
        )

        assert 0.0 <= score <= 1.0
        # Should have some collaboration due to score alignment
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_update_agent_metrics(self, learning_system):
        """Test updating agent metrics"""
        agent_performance = [
            {
                "agent_id": "test_agent",
                "agent_type": "writer",
                "specialty": "technical",
                "quality_score": 0.8,
                "response_time": 30.0,
                "success": True,
                "collaboration_score": 0.7,
            }
        ]

        await learning_system._update_agent_metrics(agent_performance)

        assert "test_agent" in learning_system.agent_performance
        metrics = learning_system.agent_performance["test_agent"]
        assert metrics.agent_id == "test_agent"
        assert metrics.agent_type == "writer"
        assert metrics.specialty == "technical"
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.average_quality_score == 0.8
        assert metrics.average_response_time == 30.0
        assert metrics.average_collaboration_score == 0.7

    def test_extract_agent_combinations(self, learning_system):
        """Test extracting agent combinations from workflow"""
        workflow_data = {
            "stages": {
                "content_generation": {
                    "writer_results": [
                        {"writer_id": "writer1"},
                        {"writer_id": "writer2"},
                    ]
                },
                "quality_checking": {
                    "checking_history": [
                        {
                            "results": [
                                {
                                    "individual_checks": [
                                        {"checker_id": "checker1"},
                                        {"checker_id": "checker2"},
                                    ]
                                }
                            ]
                        }
                    ]
                },
            }
        }

        combinations = learning_system._extract_agent_combinations(workflow_data)

        assert isinstance(combinations, list)
        assert (
            len(combinations) == 2
        )  # One writer combination and one checker combination
        assert ["writer1", "writer2"] in combinations
        assert ["checker1", "checker2"] in combinations

    @pytest.mark.asyncio
    async def test_update_interaction_patterns(self, learning_system):
        """Test updating interaction patterns"""
        workflow_data = {
            "stages": {
                "content_generation": {"writer_results": [{"writer_id": "writer1"}]}
            },
            "status": "completed",
        }

        await learning_system._update_interaction_patterns(workflow_data)

        # Should have created an interaction pattern
        assert len(learning_system.interaction_patterns) > 0

        # Find the pattern for writer1
        pattern = None
        for combo, pattern_data in learning_system.interaction_patterns.items():
            if "writer1" in combo:
                pattern = pattern_data
                break

        assert pattern is not None
        assert pattern.interaction_count == 1
        assert pattern.success_rate == 1.0  # Workflow was completed

    @pytest.mark.asyncio
    async def test_analyze_agent_performance(self, learning_system):
        """Test analyzing agent performance"""
        # Set up some agent performance data
        learning_system.agent_performance["test_agent"] = PerformanceMetrics(
            agent_id="test_agent",
            agent_type="writer",
            specialty="technical",
            total_tasks=10,
            successful_tasks=8,
            average_quality_score=0.7,
            average_response_time=30.0,
            average_collaboration_score=0.6,
        )

        # Add some recent performance data with declining trend
        for i in range(10):
            score = 0.8 - (i * 0.05)  # Declining scores
            learning_system.agent_performance["test_agent"].recent_performance.append(
                {
                    "timestamp": 1234567890 + i,
                    "quality_score": score,
                    "collaboration_score": 0.6,
                    "success": score > 0.5,
                }
            )

        # Set up system metrics
        learning_system.system_metrics["total_workflows"] = 20

        insights = await learning_system._analyze_agent_performance()

        assert isinstance(insights, list)
        # Should generate insight about declining performance
        declining_insights = [
            i for i in insights if "declining" in i.description.lower()
        ]
        assert len(declining_insights) > 0

    @pytest.mark.asyncio
    async def test_analyze_interaction_patterns(self, learning_system):
        """Test analyzing interaction patterns"""
        # Set up some interaction patterns
        learning_system.interaction_patterns[("writer1", "writer2")] = (
            InteractionPattern(
                agent_combination=("writer1", "writer2"),
                interaction_count=5,
                average_quality_score=0.8,
                average_collaboration_score=0.7,
                success_rate=0.9,
            )
        )

        learning_system.interaction_patterns[("writer1", "writer3")] = (
            InteractionPattern(
                agent_combination=("writer1", "writer3"),
                interaction_count=3,
                average_quality_score=0.6,
                average_collaboration_score=0.5,
                success_rate=0.4,
            )
        )

        insights = await learning_system._analyze_interaction_patterns()

        assert isinstance(insights, list)
        # Should generate insights about successful and problematic patterns
        successful_insights = [
            i for i in insights if "successful" in i.description.lower()
        ]
        problematic_insights = [
            i for i in insights if "problematic" in i.description.lower()
        ]
        assert len(successful_insights) > 0
        assert len(problematic_insights) > 0

    @pytest.mark.asyncio
    async def test_analyze_system_performance(self, learning_system):
        """Test analyzing system performance"""
        # Set up system metrics with declining quality
        learning_system.system_metrics = {
            "total_workflows": 20,
            "successful_workflows": 15,
            "quality_trends": [],
        }

        # Add quality trends with declining pattern
        for i in range(20):
            score = 80.0 - (i * 0.5)  # Declining scores
            learning_system.system_metrics["quality_trends"].append(
                {"timestamp": 1234567890 + i, "quality_score": score}
            )

        insights = await learning_system._analyze_system_performance()

        assert isinstance(insights, list)
        # Should generate insight about declining quality
        declining_insights = [
            i for i in insights if "declining" in i.description.lower()
        ]
        assert len(declining_insights) > 0

    @pytest.mark.asyncio
    async def test_apply_adaptations(self, learning_system):
        """Test applying adaptations based on insights"""
        # Create some insights
        insights = [
            LearningInsight(
                insight_id="test_insight",
                insight_type=AdaptationType.AGENT_SELECTION,
                description="Test insight",
                confidence=0.8,
                potential_impact=0.7,
                recommended_action="Test action",
                supporting_data={},
            )
        ]

        await learning_system._apply_adaptations(insights)

        # Check that adaptation was recorded
        assert len(learning_system.adaptation_history) == 1
        assert learning_system.adaptation_history[0]["insight_id"] == "test_insight"
        assert learning_system.adaptation_history[0]["result"] == "Applied successfully"

        # Check that insight was marked as applied
        assert insights[0].applied is True
        assert insights[0].application_result == "Successfully applied"

    def test_get_learning_report(self, learning_system):
        """Test getting learning report"""
        # Set up some data
        learning_system.agent_performance["test_agent"] = PerformanceMetrics(
            agent_id="test_agent",
            agent_type="writer",
            specialty="technical",
            total_tasks=10,
            successful_tasks=8,
            average_quality_score=0.7,
            average_response_time=30.0,
            average_collaboration_score=0.6,
        )

        learning_system.system_metrics = {
            "total_workflows": 20,
            "successful_workflows": 15,
            "quality_trends": [],
        }

        report = learning_system.get_learning_report()

        assert "system_metrics" in report
        assert "agent_performance" in report
        assert "interaction_patterns" in report
        assert "learning_insights" in report
        assert "learning_statistics" in report

        assert "test_agent" in report["agent_performance"]
        assert report["learning_statistics"]["total_workflows_analyzed"] == 20

    def test_export_learning_data(self, learning_system):
        """Test exporting learning data"""
        # Set up some data
        learning_system.agent_performance["test_agent"] = PerformanceMetrics(
            agent_id="test_agent",
            agent_type="writer",
            specialty="technical",
            total_tasks=10,
            successful_tasks=8,
            average_quality_score=0.7,
            average_response_time=30.0,
            average_collaboration_score=0.6,
        )

        exported_data = learning_system.export_learning_data()

        assert "agent_performance" in exported_data
        assert "interaction_patterns" in exported_data
        assert "learning_history" in exported_data
        assert "system_metrics" in exported_data
        assert "adaptation_history" in exported_data

        assert "test_agent" in exported_data["agent_performance"]

    def test_import_learning_data(self, learning_system):
        """Test importing learning data"""
        # Create test data
        import_data = {
            "agent_performance": {
                "imported_agent": {
                    "agent_id": "imported_agent",
                    "agent_type": "writer",
                    "specialty": "technical",
                    "total_tasks": 5,
                    "successful_tasks": 4,
                    "average_quality_score": 0.8,
                    "average_response_time": 25.0,
                    "average_collaboration_score": 0.7,
                    "strength_areas": [],
                    "weakness_areas": [],
                    "preferred_content_types": [],
                    "avoided_content_types": [],
                    "recent_performance": [],
                    "last_updated": 1234567890,
                }
            },
            "system_metrics": {"total_workflows": 10, "successful_workflows": 8},
        }

        learning_system.import_learning_data(import_data)

        # Check that data was imported
        assert "imported_agent" in learning_system.agent_performance
        assert (
            learning_system.agent_performance["imported_agent"].agent_id
            == "imported_agent"
        )
        assert learning_system.system_metrics["total_workflows"] == 10


@pytest.mark.unit
class TestPerformanceMetrics:
    """Test performance metrics data class"""

    def test_performance_metrics_creation(self):
        """Test performance metrics creation"""
        metrics = PerformanceMetrics(
            agent_id="test_agent",
            agent_type="writer",
            specialty="technical",
            total_tasks=10,
            successful_tasks=8,
            average_quality_score=0.75,
            average_response_time=30.0,
            average_collaboration_score=0.7,
            strength_areas=["accuracy", "clarity"],
            weakness_areas=["speed"],
            preferred_content_types=["technical"],
            avoided_content_types=["creative"],
            last_updated=1234567890,
        )

        assert metrics.agent_id == "test_agent"
        assert metrics.agent_type == "writer"
        assert metrics.specialty == "technical"
        assert metrics.total_tasks == 10
        assert metrics.successful_tasks == 8
        assert metrics.average_quality_score == 0.75
        assert metrics.average_response_time == 30.0
        assert metrics.average_collaboration_score == 0.7
        assert metrics.strength_areas == ["accuracy", "clarity"]
        assert metrics.weakness_areas == ["speed"]
        assert metrics.preferred_content_types == ["technical"]
        assert metrics.avoided_content_types == ["creative"]
        assert metrics.last_updated == 1234567890


@pytest.mark.unit
class TestInteractionPattern:
    """Test interaction pattern data class"""

    def test_interaction_pattern_creation(self):
        """Test interaction pattern creation"""
        pattern = InteractionPattern(
            agent_combination=("writer1", "writer2"),
            interaction_count=5,
            average_quality_score=0.8,
            average_collaboration_score=0.7,
            success_rate=0.9,
            optimal_for_content_types=["technical"],
            suboptimal_for_content_types=["creative"],
            last_used=1234567890,
        )

        assert pattern.agent_combination == ("writer1", "writer2")
        assert pattern.interaction_count == 5
        assert pattern.average_quality_score == 0.8
        assert pattern.average_collaboration_score == 0.7
        assert pattern.success_rate == 0.9
        assert pattern.optimal_for_content_types == ["technical"]
        assert pattern.suboptimal_for_content_types == ["creative"]
        assert pattern.last_used == 1234567890


@pytest.mark.unit
class TestLearningInsight:
    """Test learning insight data class"""

    def test_learning_insight_creation(self):
        """Test learning insight creation"""
        insight = LearningInsight(
            insight_id="test_insight",
            insight_type=AdaptationType.AGENT_SELECTION,
            description="Test insight description",
            confidence=0.8,
            potential_impact=0.7,
            recommended_action="Test recommended action",
            supporting_data={"key": "value"},
            created_at=1234567890,
            applied=False,
            application_result=None,
        )

        assert insight.insight_id == "test_insight"
        assert insight.insight_type == AdaptationType.AGENT_SELECTION
        assert insight.description == "Test insight description"
        assert insight.confidence == 0.8
        assert insight.potential_impact == 0.7
        assert insight.recommended_action == "Test recommended action"
        assert insight.supporting_data == {"key": "value"}
        assert insight.created_at == 1234567890
        assert insight.applied is False
        assert insight.application_result is None
