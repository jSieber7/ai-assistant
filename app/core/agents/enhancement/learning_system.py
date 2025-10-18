"""
Learning and adaptation system for multi-writer/checker system.

This module implements a learning system that tracks performance, improves
agent selection, and adapts prompting strategies over time.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque

from app.core.config import get_llm

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the system"""

    OFFLINE = "offline"  # Learn from historical data
    ONLINE = "online"  # Learn in real-time
    HYBRID = "hybrid"  # Combine offline and online learning


class AdaptationType(Enum):
    """Types of adaptations the system can make"""

    AGENT_SELECTION = "agent_selection"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    PERSONALITY_ADJUSTMENT = "personality_adjustment"
    QUALITY_THRESHOLDS = "quality_thresholds"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent"""

    agent_id: str
    agent_type: str  # "writer" or "checker"
    specialty: str
    total_tasks: int = 0
    successful_tasks: int = 0
    average_quality_score: float = 0.0
    average_response_time: float = 0.0
    average_collaboration_score: float = 0.0
    strength_areas: List[str] = field(default_factory=list)
    weakness_areas: List[str] = field(default_factory=list)
    preferred_content_types: List[str] = field(default_factory=list)
    avoided_content_types: List[str] = field(default_factory=list)
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)


@dataclass
class InteractionPattern:
    """Pattern of agent interactions"""

    agent_combination: Tuple[str, ...]  # Tuple of agent IDs
    interaction_count: int = 0
    average_quality_score: float = 0.0
    average_collaboration_score: float = 0.0
    success_rate: float = 0.0
    optimal_for_content_types: List[str] = field(default_factory=list)
    suboptimal_for_content_types: List[str] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


@dataclass
class LearningInsight:
    """Insight generated from learning analysis"""

    insight_id: str
    insight_type: AdaptationType
    description: str
    confidence: float  # 0.0-1.0
    potential_impact: float  # 0.0-1.0
    recommended_action: str
    supporting_data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    applied: bool = False
    application_result: Optional[str] = None


class LearningSystem:
    """Learning and adaptation system for multi-writer/checker"""

    def __init__(
        self,
        learning_mode: LearningMode = LearningMode.HYBRID,
        adaptation_threshold: float = 0.7,
        min_samples_for_learning: int = 10,
        analysis_model: str = "claude-3.5-sonnet",
    ):
        self.learning_mode = learning_mode
        self.adaptation_threshold = adaptation_threshold
        self.min_samples_for_learning = min_samples_for_learning
        self.analysis_model = analysis_model
        self.llm = None  # Will be initialized when needed

        # Performance tracking
        self.agent_performance: Dict[str, PerformanceMetrics] = {}
        self.interaction_patterns: Dict[Tuple[str, ...], InteractionPattern] = {}

        # Learning history
        self.learning_history: List[LearningInsight] = []
        self.adaptation_history: List[Dict[str, Any]] = []

        # System metrics
        self.system_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "average_quality_score": 0.0,
            "average_response_time": 0.0,
            "most_used_agent_combinations": [],
            "quality_trends": deque(maxlen=100),
        }

    async def _get_llm(self):
        """Initialize LLM if not already done"""
        if self.llm is None:
            self.llm = await get_llm(self.analysis_model)
        return self.llm

    async def learn_from_workflow(
        self, workflow_id: str, workflow_data: Dict[str, Any]
    ) -> List[LearningInsight]:
        """
        Learn from a completed workflow.

        Args:
            workflow_id: ID of the completed workflow
            workflow_data: Complete workflow data

        Returns:
            List of learning insights generated
        """
        logger.info(f"Learning from workflow {workflow_id}")

        try:
            # Extract performance data
            agent_performance = self._extract_agent_performance(workflow_data)

            # Update agent metrics
            await self._update_agent_metrics(agent_performance)

            # Update interaction patterns
            await self._update_interaction_patterns(workflow_data)

            # Update system metrics
            self._update_system_metrics(workflow_data)

            # Generate insights
            insights = await self._generate_learning_insights(workflow_data)

            # Store insights
            self.learning_history.extend(insights)

            # Apply adaptations if confidence is high enough
            high_confidence_insights = [
                insight
                for insight in insights
                if insight.confidence >= self.adaptation_threshold
            ]

            if high_confidence_insights:
                await self._apply_adaptations(high_confidence_insights)

            logger.info(
                f"Generated {len(insights)} insights from workflow {workflow_id}"
            )
            return insights

        except Exception as e:
            logger.error(f"Failed to learn from workflow {workflow_id}: {str(e)}")
            return []

    def _extract_agent_performance(
        self, workflow_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract agent performance data from workflow"""
        agent_performance = []

        # Extract writer performance
        if "content_generation" in workflow_data.get("stages", {}):
            writer_results = workflow_data["stages"]["content_generation"].get(
                "writer_results", []
            )
            for writer_result in writer_results:
                agent_performance.append(
                    {
                        "agent_id": writer_result.get("writer_id"),
                        "agent_type": "writer",
                        "specialty": writer_result.get("specialty"),
                        "quality_score": writer_result.get("confidence_score", 0.5),
                        "response_time": writer_result.get("execution_time", 0.0),
                        "success": writer_result.get("content") is not None,
                        "collaboration_score": self._calculate_collaboration_score(
                            writer_result, workflow_data
                        ),
                    }
                )

        # Extract checker performance
        if "quality_checking" in workflow_data.get("stages", {}):
            checking_history = workflow_data["stages"]["quality_checking"].get(
                "checking_history", []
            )
            for round_data in checking_history:
                for check_result in round_data.get("results", []):
                    for checker_result in check_result.get("individual_checks", []):
                        agent_performance.append(
                            {
                                "agent_id": checker_result.get("checker_id"),
                                "agent_type": "checker",
                                "specialty": checker_result.get("focus_area"),
                                "quality_score": checker_result.get("score", 0.0),
                                "response_time": 0.0,  # Not tracked for checkers
                                "success": checker_result.get("score", 0.0) > 70.0,
                                "collaboration_score": self._calculate_checker_collaboration_score(
                                    checker_result, check_result
                                ),
                            }
                        )

        return agent_performance

    def _calculate_collaboration_score(
        self, agent_data: Dict[str, Any], workflow_data: Dict[str, Any]
    ) -> float:
        """Calculate collaboration score for a writer"""
        # Base score
        collaboration_score = 0.5

        # Check if content was used in final output
        final_output = workflow_data.get("final_output", "")
        agent_content = agent_data.get("content", "")

        if agent_content and final_output:
            # Simple content overlap check
            overlap = len(set(agent_content.split()) & set(final_output.split())) / max(
                1, len(set(final_output.split()))
            )
            collaboration_score += overlap * 0.3

        # Check participation in debates
        if "debate_data" in workflow_data:
            debate_participation = (
                workflow_data["debate_data"]
                .get("participation", {})
                .get(agent_data.get("writer_id"), 0)
            )
            collaboration_score += min(0.2, debate_participation * 0.1)

        return min(1.0, collaboration_score)

    def _calculate_checker_collaboration_score(
        self, checker_result: Dict[str, Any], check_data: Dict[str, Any]
    ) -> float:
        """Calculate collaboration score for a checker"""
        # Base score
        collaboration_score = 0.5

        # Check if checker's assessment aligned with consensus
        consensus_score = check_data.get("overall_score", 0.0)
        checker_score = checker_result.get("score", 0.0)

        alignment = 1.0 - abs(consensus_score - checker_score) / 100.0
        collaboration_score += alignment * 0.3

        # Check if checker's improvements were adopted
        final_content = check_data.get("best_improved_version", {}).get("content", "")
        checker_improvements = checker_result.get("improved_content", "")

        if final_content and checker_improvements:
            overlap = len(
                set(checker_improvements.split()) & set(final_content.split())
            ) / max(1, len(set(final_content.split())))
            collaboration_score += overlap * 0.2

        return min(1.0, collaboration_score)

    async def _update_agent_metrics(self, agent_performance: List[Dict[str, Any]]):
        """Update performance metrics for agents"""
        for perf_data in agent_performance:
            agent_id = perf_data["agent_id"]

            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = PerformanceMetrics(
                    agent_id=agent_id,
                    agent_type=perf_data["agent_type"],
                    specialty=perf_data["specialty"],
                )

            metrics = self.agent_performance[agent_id]

            # Update basic metrics
            metrics.total_tasks += 1
            if perf_data["success"]:
                metrics.successful_tasks += 1

            # Update averages
            old_avg_quality = metrics.average_quality_score
            metrics.average_quality_score = (
                old_avg_quality * (metrics.total_tasks - 1) + perf_data["quality_score"]
            ) / metrics.total_tasks

            old_avg_time = metrics.average_response_time
            metrics.average_response_time = (
                old_avg_time * (metrics.total_tasks - 1) + perf_data["response_time"]
            ) / metrics.total_tasks

            old_avg_collab = metrics.average_collaboration_score
            metrics.average_collaboration_score = (
                old_avg_collab * (metrics.total_tasks - 1)
                + perf_data["collaboration_score"]
            ) / metrics.total_tasks

            # Update recent performance
            metrics.recent_performance.append(
                {
                    "timestamp": time.time(),
                    "quality_score": perf_data["quality_score"],
                    "collaboration_score": perf_data["collaboration_score"],
                    "success": perf_data["success"],
                }
            )

            metrics.last_updated = time.time()

    async def _update_interaction_patterns(self, workflow_data: Dict[str, Any]):
        """Update interaction patterns between agents"""
        # Extract agent combinations
        agent_combinations = self._extract_agent_combinations(workflow_data)

        for combination in agent_combinations:
            combination_key = tuple(sorted(combination))

            if combination_key not in self.interaction_patterns:
                self.interaction_patterns[combination_key] = InteractionPattern(
                    agent_combination=combination_key
                )

            pattern = self.interaction_patterns[combination_key]
            pattern.interaction_count += 1
            pattern.last_used = time.time()

            # Update performance metrics for pattern
            workflow_quality = (
                workflow_data.get("stages", {})
                .get("quality_checking", {})
                .get("best_score", 0.0)
            )
            pattern.average_quality_score = (
                pattern.average_quality_score * (pattern.interaction_count - 1)
                + workflow_quality
            ) / pattern.interaction_count

            # Update success rate
            workflow_success = workflow_data.get("status") == "completed"
            pattern.success_rate = (
                pattern.success_rate * (pattern.interaction_count - 1)
                + (1.0 if workflow_success else 0.0)
            ) / pattern.interaction_count

    def _extract_agent_combinations(
        self, workflow_data: Dict[str, Any]
    ) -> List[List[str]]:
        """Extract all agent combinations from workflow"""
        combinations = []

        # Extract writer combinations
        if "content_generation" in workflow_data.get("stages", {}):
            writer_results = workflow_data["stages"]["content_generation"].get(
                "writer_results", []
            )
            writer_ids = [
                w.get("writer_id") for w in writer_results if w.get("writer_id")
            ]
            if writer_ids:
                combinations.append(writer_ids)

        # Extract checker combinations
        if "quality_checking" in workflow_data.get("stages", {}):
            checking_history = workflow_data["stages"]["quality_checking"].get(
                "checking_history", []
            )
            for round_data in checking_history:
                checker_ids = []
                for check_result in round_data.get("results", []):
                    for checker_result in check_result.get("individual_checks", []):
                        checker_id = checker_result.get("checker_id")
                        if checker_id and checker_id not in checker_ids:
                            checker_ids.append(checker_id)

                if checker_ids:
                    combinations.append(checker_ids)

        return combinations

    def _update_system_metrics(self, workflow_data: Dict[str, Any]):
        """Update system-wide metrics"""
        self.system_metrics["total_workflows"] += 1

        if workflow_data.get("status") == "completed":
            self.system_metrics["successful_workflows"] += 1

        # Update quality trends
        quality_score = (
            workflow_data.get("stages", {})
            .get("quality_checking", {})
            .get("best_score", 0.0)
        )
        self.system_metrics["quality_trends"].append(
            {"timestamp": time.time(), "quality_score": quality_score}
        )

        # Update averages
        if self.system_metrics["total_workflows"] > 0:
            self.system_metrics["average_quality_score"] = sum(
                trend["quality_score"]
                for trend in self.system_metrics["quality_trends"]
            ) / len(self.system_metrics["quality_trends"])

    async def _generate_learning_insights(
        self, workflow_data: Dict[str, Any]
    ) -> List[LearningInsight]:
        """Generate learning insights from workflow data"""
        insights = []

        # Only generate insights if we have enough data
        if self.system_metrics["total_workflows"] < self.min_samples_for_learning:
            return insights

        # Generate agent performance insights
        agent_insights = await self._analyze_agent_performance()
        insights.extend(agent_insights)

        # Generate interaction pattern insights
        pattern_insights = await self._analyze_interaction_patterns()
        insights.extend(pattern_insights)

        # Generate system optimization insights
        system_insights = await self._analyze_system_performance()
        insights.extend(system_insights)

        return insights

    async def _analyze_agent_performance(self) -> List[LearningInsight]:
        """Analyze agent performance and generate insights"""
        insights = []

        for agent_id, metrics in self.agent_performance.items():
            if metrics.total_tasks < 5:  # Not enough data
                continue

            # Check for declining performance
            if len(metrics.recent_performance) >= 10:
                recent_scores = [
                    p["quality_score"] for p in list(metrics.recent_performance)[-10:]
                ]
                older_scores = [
                    p["quality_score"]
                    for p in list(metrics.recent_performance)[-20:-10]
                ]

                if (
                    older_scores
                    and statistics.mean(recent_scores)
                    < statistics.mean(older_scores) - 10
                ):
                    insight = LearningInsight(
                        insight_id=f"declining_performance_{agent_id}_{int(time.time())}",
                        insight_type=AdaptationType.PERSONALITY_ADJUSTMENT,
                        description=f"Agent {agent_id} shows declining performance",
                        confidence=0.8,
                        potential_impact=0.6,
                        recommended_action=f"Review and adjust {agent_id}'s personality or prompting",
                        supporting_data={
                            "agent_id": agent_id,
                            "recent_avg": statistics.mean(recent_scores),
                            "older_avg": statistics.mean(older_scores),
                            "decline": statistics.mean(older_scores)
                            - statistics.mean(recent_scores),
                        },
                    )
                    insights.append(insight)

            # Check for underutilization
            if metrics.total_tasks < self.system_metrics["total_workflows"] * 0.1:
                insight = LearningInsight(
                    insight_id=f"underutilized_agent_{agent_id}_{int(time.time())}",
                    insight_type=AdaptationType.AGENT_SELECTION,
                    description=f"Agent {agent_id} is underutilized",
                    confidence=0.7,
                    potential_impact=0.4,
                    recommended_action=f"Increase selection frequency for {agent_id} in appropriate tasks",
                    supporting_data={
                        "agent_id": agent_id,
                        "usage_rate": metrics.total_tasks
                        / self.system_metrics["total_workflows"],
                        "avg_performance": metrics.average_quality_score,
                    },
                )
                insights.append(insight)

        return insights

    async def _analyze_interaction_patterns(self) -> List[LearningInsight]:
        """Analyze interaction patterns and generate insights"""
        insights = []

        # Find most successful patterns
        successful_patterns = [
            (combo, pattern)
            for combo, pattern in self.interaction_patterns.items()
            if pattern.success_rate > 0.8 and pattern.interaction_count >= 3
        ]

        if successful_patterns:
            successful_patterns.sort(key=lambda x: x[1].success_rate, reverse=True)
            best_pattern = successful_patterns[0]

            insight = LearningInsight(
                insight_id=f"successful_pattern_{int(time.time())}",
                insight_type=AdaptationType.AGENT_SELECTION,
                description=f"Agent combination {best_pattern[0]} shows high success rate",
                confidence=0.9,
                potential_impact=0.7,
                recommended_action=f"Prioritize agent combination {best_pattern[0]} for similar tasks",
                supporting_data={
                    "agent_combination": best_pattern[0],
                    "success_rate": best_pattern[1].success_rate,
                    "interaction_count": best_pattern[1].interaction_count,
                    "avg_quality": best_pattern[1].average_quality_score,
                },
            )
            insights.append(insight)

        # Find problematic patterns
        problematic_patterns = [
            (combo, pattern)
            for combo, pattern in self.interaction_patterns.items()
            if pattern.success_rate < 0.5 and pattern.interaction_count >= 3
        ]

        if problematic_patterns:
            problematic_patterns.sort(key=lambda x: x[1].success_rate)
            worst_pattern = problematic_patterns[0]

            insight = LearningInsight(
                insight_id=f"problematic_pattern_{int(time.time())}",
                insight_type=AdaptationType.AGENT_SELECTION,
                description=f"Agent combination {worst_pattern[0]} shows low success rate",
                confidence=0.8,
                potential_impact=0.6,
                recommended_action=f"Avoid agent combination {worst_pattern[0]} or adjust interaction approach",
                supporting_data={
                    "agent_combination": worst_pattern[0],
                    "success_rate": worst_pattern[1].success_rate,
                    "interaction_count": worst_pattern[1].interaction_count,
                    "avg_quality": worst_pattern[1].average_quality_score,
                },
            )
            insights.append(insight)

        return insights

    async def _analyze_system_performance(self) -> List[LearningInsight]:
        """Analyze system performance and generate insights"""
        insights = []

        # Check quality trends
        if len(self.system_metrics["quality_trends"]) >= 20:
            recent_trends = list(self.system_metrics["quality_trends"])[-10:]
            older_trends = list(self.system_metrics["quality_trends"])[-20:-10]

            recent_avg = statistics.mean(t["quality_score"] for t in recent_trends)
            older_avg = statistics.mean(t["quality_score"] for t in older_trends)

            if recent_avg < older_avg - 5:
                insight = LearningInsight(
                    insight_id=f"declining_quality_{int(time.time())}",
                    insight_type=AdaptationType.QUALITY_THRESHOLDS,
                    description="System quality output is declining",
                    confidence=0.7,
                    potential_impact=0.8,
                    recommended_action="Review quality thresholds and agent selection criteria",
                    supporting_data={
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "decline": older_avg - recent_avg,
                    },
                )
                insights.append(insight)

        # Check success rate
        if self.system_metrics["total_workflows"] >= 10:
            success_rate = (
                self.system_metrics["successful_workflows"]
                / self.system_metrics["total_workflows"]
            )

            if success_rate < 0.7:
                insight = LearningInsight(
                    insight_id=f"low_success_rate_{int(time.time())}",
                    insight_type=AdaptationType.WORKFLOW_OPTIMIZATION,
                    description=f"System success rate is low: {success_rate:.2%}",
                    confidence=0.8,
                    potential_impact=0.9,
                    recommended_action="Optimize workflow processes and agent coordination",
                    supporting_data={
                        "success_rate": success_rate,
                        "total_workflows": self.system_metrics["total_workflows"],
                        "successful_workflows": self.system_metrics[
                            "successful_workflows"
                        ],
                    },
                )
                insights.append(insight)

        return insights

    async def _apply_adaptations(self, insights: List[LearningInsight]):
        """Apply adaptations based on high-confidence insights"""
        for insight in insights:
            try:
                if insight.insight_type == AdaptationType.AGENT_SELECTION:
                    await self._apply_agent_selection_adaptation(insight)
                elif insight.insight_type == AdaptationType.PERSONALITY_ADJUSTMENT:
                    await self._apply_personality_adaptation(insight)
                elif insight.insight_type == AdaptationType.QUALITY_THRESHOLDS:
                    await self._apply_quality_threshold_adaptation(insight)
                elif insight.insight_type == AdaptationType.WORKFLOW_OPTIMIZATION:
                    await self._apply_workflow_optimization_adaptation(insight)

                insight.applied = True
                insight.application_result = "Successfully applied"

                # Record adaptation
                self.adaptation_history.append(
                    {
                        "timestamp": time.time(),
                        "insight_id": insight.insight_id,
                        "adaptation_type": insight.insight_type.value,
                        "action": insight.recommended_action,
                        "result": "Applied successfully",
                    }
                )

                logger.info(f"Applied adaptation for insight {insight.insight_id}")

            except Exception as e:
                logger.error(
                    f"Failed to apply adaptation for insight {insight.insight_id}: {str(e)}"
                )
                insight.application_result = f"Failed to apply: {str(e)}"

    async def _apply_agent_selection_adaptation(self, insight: LearningInsight):
        """Apply agent selection adaptation"""
        # This would integrate with the dynamic selector
        # For now, just log the recommendation
        logger.info(f"Agent selection adaptation: {insight.recommended_action}")

    async def _apply_personality_adaptation(self, insight: LearningInsight):
        """Apply personality adaptation"""
        # This would integrate with the personality system
        # For now, just log the recommendation
        logger.info(f"Personality adaptation: {insight.recommended_action}")

    async def _apply_quality_threshold_adaptation(self, insight: LearningInsight):
        """Apply quality threshold adaptation"""
        # This would update system configuration
        # For now, just log the recommendation
        logger.info(f"Quality threshold adaptation: {insight.recommended_action}")

    async def _apply_workflow_optimization_adaptation(self, insight: LearningInsight):
        """Apply workflow optimization adaptation"""
        # This would update workflow processes
        # For now, just log the recommendation
        logger.info(f"Workflow optimization adaptation: {insight.recommended_action}")

    def get_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        report = {
            "system_metrics": self.system_metrics,
            "agent_performance": {
                agent_id: {
                    "total_tasks": metrics.total_tasks,
                    "success_rate": metrics.successful_tasks
                    / max(1, metrics.total_tasks),
                    "average_quality_score": metrics.average_quality_score,
                    "average_collaboration_score": metrics.average_collaboration_score,
                    "specialty": metrics.specialty,
                    "agent_type": metrics.agent_type,
                }
                for agent_id, metrics in self.agent_performance.items()
            },
            "interaction_patterns": {
                str(combo): {
                    "interaction_count": pattern.interaction_count,
                    "success_rate": pattern.success_rate,
                    "average_quality_score": pattern.average_quality_score,
                }
                for combo, pattern in self.interaction_patterns.items()
                if pattern.interaction_count >= 2
            },
            "learning_insights": {
                "total_insights": len(self.learning_history),
                "applied_insights": len(
                    [i for i in self.learning_history if i.applied]
                ),
                "high_confidence_insights": len(
                    [i for i in self.learning_history if i.confidence >= 0.8]
                ),
                "recent_insights": [
                    {
                        "id": insight.insight_id,
                        "type": insight.insight_type.value,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "applied": insight.applied,
                    }
                    for insight in self.learning_history[-10:]
                ],
            },
            "adaptation_history": self.adaptation_history[-10:],  # Last 10 adaptations
            "learning_statistics": {
                "total_workflows_analyzed": self.system_metrics["total_workflows"],
                "learning_mode": self.learning_mode.value,
                "adaptation_threshold": self.adaptation_threshold,
                "last_updated": time.time(),
            },
        }

        return report

    def export_learning_data(self) -> Dict[str, Any]:
        """Export all learning data for backup or analysis"""
        return {
            "agent_performance": {
                agent_id: {
                    "total_tasks": metrics.total_tasks,
                    "successful_tasks": metrics.successful_tasks,
                    "average_quality_score": metrics.average_quality_score,
                    "average_response_time": metrics.average_response_time,
                    "average_collaboration_score": metrics.average_collaboration_score,
                    "strength_areas": metrics.strength_areas,
                    "weakness_areas": metrics.weakness_areas,
                    "preferred_content_types": metrics.preferred_content_types,
                    "avoided_content_types": metrics.avoided_content_types,
                    "recent_performance": list(metrics.recent_performance),
                    "last_updated": metrics.last_updated,
                }
                for agent_id, metrics in self.agent_performance.items()
            },
            "interaction_patterns": {
                str(combo): {
                    "agent_combination": list(pattern.agent_combination),
                    "interaction_count": pattern.interaction_count,
                    "average_quality_score": pattern.average_quality_score,
                    "average_collaboration_score": pattern.average_collaboration_score,
                    "success_rate": pattern.success_rate,
                    "optimal_for_content_types": pattern.optimal_for_content_types,
                    "suboptimal_for_content_types": pattern.suboptimal_for_content_types,
                    "last_used": pattern.last_used,
                }
                for combo, pattern in self.interaction_patterns.items()
            },
            "learning_history": [
                {
                    "insight_id": insight.insight_id,
                    "insight_type": insight.insight_type.value,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "potential_impact": insight.potential_impact,
                    "recommended_action": insight.recommended_action,
                    "supporting_data": insight.supporting_data,
                    "created_at": insight.created_at,
                    "applied": insight.applied,
                    "application_result": insight.application_result,
                }
                for insight in self.learning_history
            ],
            "system_metrics": self.system_metrics,
            "adaptation_history": self.adaptation_history,
        }

    def import_learning_data(self, data: Dict[str, Any]):
        """Import learning data from backup"""
        # Import agent performance
        if "agent_performance" in data:
            for agent_id, perf_data in data["agent_performance"].items():
                self.agent_performance[agent_id] = PerformanceMetrics(
                    agent_id=agent_id,
                    agent_type=perf_data.get("agent_type", "unknown"),
                    specialty=perf_data.get("specialty", "unknown"),
                    total_tasks=perf_data.get("total_tasks", 0),
                    successful_tasks=perf_data.get("successful_tasks", 0),
                    average_quality_score=perf_data.get("average_quality_score", 0.0),
                    average_response_time=perf_data.get("average_response_time", 0.0),
                    average_collaboration_score=perf_data.get(
                        "average_collaboration_score", 0.0
                    ),
                    strength_areas=perf_data.get("strength_areas", []),
                    weakness_areas=perf_data.get("weakness_areas", []),
                    preferred_content_types=perf_data.get(
                        "preferred_content_types", []
                    ),
                    avoided_content_types=perf_data.get("avoided_content_types", []),
                    recent_performance=deque(
                        perf_data.get("recent_performance", []), maxlen=50
                    ),
                    last_updated=perf_data.get("last_updated", time.time()),
                )

        # Import interaction patterns
        if "interaction_patterns" in data:
            for combo_str, pattern_data in data["interaction_patterns"].items():
                combo = tuple(pattern_data["agent_combination"])
                self.interaction_patterns[combo] = InteractionPattern(
                    agent_combination=combo,
                    interaction_count=pattern_data.get("interaction_count", 0),
                    average_quality_score=pattern_data.get(
                        "average_quality_score", 0.0
                    ),
                    average_collaboration_score=pattern_data.get(
                        "average_collaboration_score", 0.0
                    ),
                    success_rate=pattern_data.get("success_rate", 0.0),
                    optimal_for_content_types=pattern_data.get(
                        "optimal_for_content_types", []
                    ),
                    suboptimal_for_content_types=pattern_data.get(
                        "suboptimal_for_content_types", []
                    ),
                    last_used=pattern_data.get("last_used", time.time()),
                )

        # Import other data
        if "system_metrics" in data:
            self.system_metrics.update(data["system_metrics"])

        if "adaptation_history" in data:
            self.adaptation_history.extend(data["adaptation_history"])

        logger.info("Learning data imported successfully")
