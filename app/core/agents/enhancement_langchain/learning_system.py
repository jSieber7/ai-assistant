"""
LangGraph-based learning and adaptation system for multi-agent workflows.

This module implements a learning system using LangGraph workflows that tracks
performance, improves agent selection, and adapts prompting strategies over time.
"""

import logging
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from app.core.langchain.integration import get_langchain_integration

logger = logging.getLogger(__name__)


class LearningMode(str, Enum):
    """Learning modes for the system"""
    OFFLINE = "offline"  # Learn from historical data
    ONLINE = "online"  # Learn in real-time
    HYBRID = "hybrid"  # Combine offline and online learning


class AdaptationType(str, Enum):
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


class LearningState(BaseModel):
    """State for learning workflow"""
    # Input parameters
    action: str  # "learn_from_workflow", "generate_insights", "apply_adaptations", "get_report"
    workflow_data: Optional[Dict[str, Any]] = None
    workflow_id: Optional[str] = None
    
    # Learning data
    agent_performance: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    interaction_patterns: Dict[Tuple[str, ...], InteractionPattern] = field(default_factory=dict)
    learning_history: List[LearningInsight] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # System metrics
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    insights: List[LearningInsight] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class LangGraphLearningSystem:
    """LangGraph-based learning and adaptation system for multi-agent workflows"""
    
    def __init__(
        self,
        learning_mode: LearningMode = LearningMode.HYBRID,
        adaptation_threshold: float = 0.7,
        min_samples_for_learning: int = 10,
    ):
        self.learning_mode = learning_mode
        self.adaptation_threshold = adaptation_threshold
        self.min_samples_for_learning = min_samples_for_learning
        
        # Initialize workflow
        self.workflow = self._create_workflow()
        self.compiled_workflow: Optional[CompiledGraph] = None
        
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
        
        logger.info("Initialized LangGraph Learning System")
    
    def _create_workflow(self) -> StateGraph:
        """Create learning workflow"""
        workflow = StateGraph(LearningState)
        
        # Add nodes
        workflow.add_node("route_action", self._route_action)
        workflow.add_node("learn_from_workflow", self._learn_from_workflow)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("apply_adaptations", self._apply_adaptations)
        workflow.add_node("generate_report", self._generate_report)
        
        # Set entry point
        workflow.set_entry_point("route_action")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "route_action",
            self._determine_action,
            {
                "learn": "learn_from_workflow",
                "insights": "generate_insights",
                "adapt": "apply_adaptations",
                "report": "generate_report",
                "error": END
            }
        )
        
        # Add edges
        workflow.add_edge("learn_from_workflow", END)
        workflow.add_edge("generate_insights", END)
        workflow.add_edge("apply_adaptations", END)
        workflow.add_edge("generate_report", END)
        
        return workflow
    
    async def _route_action(self, state: LearningState) -> LearningState:
        """Route action based on input"""
        try:
            action = state.action.lower()
            
            if action in ["learn", "learn_from_workflow", "analyze"]:
                state.action = "learn"
            elif action in ["insights", "generate_insights", "analyze_patterns"]:
                state.action = "insights"
            elif action in ["adapt", "apply_adaptations", "optimize"]:
                state.action = "adapt"
            elif action in ["report", "get_report", "statistics"]:
                state.action = "report"
            else:
                state.action = "error"
                state.error = f"Unknown action: {action}"
            
            # Copy current state to workflow state
            state.agent_performance = self.agent_performance.copy()
            state.interaction_patterns = self.interaction_patterns.copy()
            state.learning_history = self.learning_history.copy()
            state.adaptation_history = self.adaptation_history.copy()
            state.system_metrics = self.system_metrics.copy()
            
        except Exception as e:
            logger.error(f"Error routing action: {str(e)}")
            state.action = "error"
            state.error = str(e)
        
        return state
    
    def _determine_action(self, state: LearningState) -> str:
        """Determine which action to take"""
        return state.action
    
    async def _learn_from_workflow(self, state: LearningState) -> LearningState:
        """Learn from a completed workflow"""
        try:
            if not state.workflow_data:
                state.error = "No workflow data provided for learning"
                state.success = False
                return state
            
            workflow_id = state.workflow_id or f"workflow_{int(time.time())}"
            workflow_data = state.workflow_data
            
            logger.info(f"Learning from workflow {workflow_id}")
            
            # Extract performance data
            agent_performance = self._extract_agent_performance(workflow_data)
            
            # Update agent metrics
            await self._update_agent_metrics(agent_performance, state)
            
            # Update interaction patterns
            await self._update_interaction_patterns(workflow_data, state)
            
            # Update system metrics
            self._update_system_metrics(workflow_data, state)
            
            # Generate insights
            insights = await self._generate_learning_insights(workflow_data, state)
            
            # Store insights
            state.learning_history.extend(insights)
            
            # Apply adaptations if confidence is high enough
            high_confidence_insights = [
                insight
                for insight in insights
                if insight.confidence >= self.adaptation_threshold
            ]
            
            if high_confidence_insights:
                await self._apply_adaptations_to_system(high_confidence_insights, state)
            
            state.insights = insights
            state.result = f"Successfully learned from workflow {workflow_id}"
            state.success = True
            
            logger.info(f"Generated {len(insights)} insights from workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error learning from workflow: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _generate_insights(self, state: LearningState) -> LearningState:
        """Generate learning insights from current data"""
        try:
            # Only generate insights if we have enough data
            if state.system_metrics.get("total_workflows", 0) < self.min_samples_for_learning:
                state.result = "Insufficient data for insight generation"
                state.success = True
                return state
            
            insights = []
            
            # Generate agent performance insights
            agent_insights = await self._analyze_agent_performance(state)
            insights.extend(agent_insights)
            
            # Generate interaction pattern insights
            pattern_insights = await self._analyze_interaction_patterns(state)
            insights.extend(pattern_insights)
            
            # Generate system optimization insights
            system_insights = await self._analyze_system_performance(state)
            insights.extend(system_insights)
            
            state.insights = insights
            state.result = f"Generated {len(insights)} learning insights"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _apply_adaptations(self, state: LearningState) -> LearningState:
        """Apply adaptations based on insights"""
        try:
            if not state.insights:
                state.error = "No insights provided for adaptation"
                state.success = False
                return state
            
            applied_count = 0
            
            for insight in state.insights:
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
                    applied_count += 1
                    
                    # Record adaptation
                    adaptation_record = {
                        "timestamp": time.time(),
                        "insight_id": insight.insight_id,
                        "adaptation_type": insight.insight_type.value,
                        "action": insight.recommended_action,
                        "result": "Applied successfully",
                    }
                    state.adaptation_history.append(adaptation_record)
                    
                    logger.info(f"Applied adaptation for insight {insight.insight_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to apply adaptation for insight {insight.insight_id}: {str(e)}")
                    insight.application_result = f"Failed to apply: {str(e)}"
            
            state.result = f"Successfully applied {applied_count} adaptations"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error applying adaptations: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _generate_report(self, state: LearningState) -> LearningState:
        """Generate comprehensive learning report"""
        try:
            report = {
                "system_metrics": state.system_metrics,
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
                    for agent_id, metrics in state.agent_performance.items()
                },
                "interaction_patterns": {
                    str(combo): {
                        "interaction_count": pattern.interaction_count,
                        "success_rate": pattern.success_rate,
                        "average_quality_score": pattern.average_quality_score,
                    }
                    for combo, pattern in state.interaction_patterns.items()
                    if pattern.interaction_count >= 2
                },
                "learning_insights": {
                    "total_insights": len(state.learning_history),
                    "applied_insights": len(
                        [i for i in state.learning_history if i.applied]
                    ),
                    "high_confidence_insights": len(
                        [i for i in state.learning_history if i.confidence >= 0.8]
                    ),
                    "recent_insights": [
                        {
                            "id": insight.insight_id,
                            "type": insight.insight_type.value,
                            "description": insight.description,
                            "confidence": insight.confidence,
                            "applied": insight.applied,
                        }
                        for insight in state.learning_history[-10:]
                    ],
                },
                "adaptation_history": state.adaptation_history[-10:],  # Last 10 adaptations
                "learning_statistics": {
                    "total_workflows_analyzed": state.system_metrics.get("total_workflows", 0),
                    "learning_mode": self.learning_mode.value,
                    "adaptation_threshold": self.adaptation_threshold,
                    "last_updated": time.time(),
                },
            }
            
            state.report = report
            state.result = "Successfully generated learning report"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    def _extract_agent_performance(self, workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    
    async def _update_agent_metrics(
        self, agent_performance: List[Dict[str, Any]], state: LearningState
    ):
        """Update performance metrics for agents"""
        for perf_data in agent_performance:
            agent_id = perf_data["agent_id"]
            
            if agent_id not in state.agent_performance:
                state.agent_performance[agent_id] = PerformanceMetrics(
                    agent_id=agent_id,
                    agent_type=perf_data["agent_type"],
                    specialty=perf_data["specialty"],
                )
            
            metrics = state.agent_performance[agent_id]
            
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
    
    async def _update_interaction_patterns(
        self, workflow_data: Dict[str, Any], state: LearningState
    ):
        """Update interaction patterns between agents"""
        # Extract agent combinations
        agent_combinations = self._extract_agent_combinations(workflow_data)
        
        for combination in agent_combinations:
            combination_key = tuple(sorted(combination))
            
            if combination_key not in state.interaction_patterns:
                state.interaction_patterns[combination_key] = InteractionPattern(
                    agent_combination=combination_key
                )
            
            pattern = state.interaction_patterns[combination_key]
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
    
    def _update_system_metrics(self, workflow_data: Dict[str, Any], state: LearningState):
        """Update system-wide metrics"""
        state.system_metrics["total_workflows"] = (
            state.system_metrics.get("total_workflows", 0) + 1
        )
        
        if workflow_data.get("status") == "completed":
            state.system_metrics["successful_workflows"] = (
                state.system_metrics.get("successful_workflows", 0) + 1
            )
        
        # Update quality trends
        quality_score = (
            workflow_data.get("stages", {})
            .get("quality_checking", {})
            .get("best_score", 0.0)
        )
        state.system_metrics.setdefault("quality_trends", deque(maxlen=100)).append(
            {"timestamp": time.time(), "quality_score": quality_score}
        )
        
        # Update averages
        total_workflows = state.system_metrics["total_workflows"]
        if total_workflows > 0:
            quality_trends = state.system_metrics.get("quality_trends", deque(maxlen=100))
            if quality_trends:
                state.system_metrics["average_quality_score"] = sum(
                    trend["quality_score"] for trend in quality_trends
                ) / len(quality_trends)
    
    async def _generate_learning_insights(
        self, workflow_data: Dict[str, Any], state: LearningState
    ) -> List[LearningInsight]:
        """Generate learning insights from workflow data"""
        insights = []
        
        # Only generate insights if we have enough data
        if state.system_metrics.get("total_workflows", 0) < self.min_samples_for_learning:
            return insights
        
        # Generate agent performance insights
        agent_insights = await self._analyze_agent_performance(state)
        insights.extend(agent_insights)
        
        # Generate interaction pattern insights
        pattern_insights = await self._analyze_interaction_patterns(state)
        insights.extend(pattern_insights)
        
        # Generate system optimization insights
        system_insights = await self._analyze_system_performance(state)
        insights.extend(system_insights)
        
        return insights
    
    async def _analyze_agent_performance(self, state: LearningState) -> List[LearningInsight]:
        """Analyze agent performance and generate insights"""
        insights = []
        
        for agent_id, metrics in state.agent_performance.items():
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
            total_workflows = state.system_metrics.get("total_workflows", 1)
            if metrics.total_tasks < total_workflows * 0.1:
                insight = LearningInsight(
                    insight_id=f"underutilized_agent_{agent_id}_{int(time.time())}",
                    insight_type=AdaptationType.AGENT_SELECTION,
                    description=f"Agent {agent_id} is underutilized",
                    confidence=0.7,
                    potential_impact=0.4,
                    recommended_action=f"Increase selection frequency for {agent_id} in appropriate tasks",
                    supporting_data={
                        "agent_id": agent_id,
                        "usage_rate": metrics.total_tasks / total_workflows,
                        "avg_performance": metrics.average_quality_score,
                    },
                )
                insights.append(insight)
        
        return insights
    
    async def _analyze_interaction_patterns(self, state: LearningState) -> List[LearningInsight]:
        """Analyze interaction patterns and generate insights"""
        insights = []
        
        # Find most successful patterns
        successful_patterns = [
            (combo, pattern)
            for combo, pattern in state.interaction_patterns.items()
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
            for combo, pattern in state.interaction_patterns.items()
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
    
    async def _analyze_system_performance(self, state: LearningState) -> List[LearningInsight]:
        """Analyze system performance and generate insights"""
        insights = []
        
        # Check quality trends
        quality_trends = state.system_metrics.get("quality_trends", deque(maxlen=100))
        if len(quality_trends) >= 20:
            recent_trends = list(quality_trends)[-10:]
            older_trends = list(quality_trends)[-20:-10]
            
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
        total_workflows = state.system_metrics.get("total_workflows", 0)
        successful_workflows = state.system_metrics.get("successful_workflows", 0)
        
        if total_workflows >= 10:
            success_rate = successful_workflows / total_workflows
            
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
                        "total_workflows": total_workflows,
                        "successful_workflows": successful_workflows,
                    },
                )
                insights.append(insight)
        
        return insights
    
    async def _apply_adaptations_to_system(
        self, insights: List[LearningInsight], state: LearningState
    ):
        """Apply adaptations to the system"""
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
                adaptation_record = {
                    "timestamp": time.time(),
                    "insight_id": insight.insight_id,
                    "adaptation_type": insight.insight_type.value,
                    "action": insight.recommended_action,
                    "result": "Applied successfully",
                }
                state.adaptation_history.append(adaptation_record)
                
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
    
    async def process_request(self, state: LearningState) -> LearningState:
        """Process a learning request"""
        # Compile workflow if not already done
        if self.compiled_workflow is None:
            self.compiled_workflow = self.workflow.compile()
        
        # Update internal state from workflow state
        self.agent_performance = state.agent_performance
        self.interaction_patterns = state.interaction_patterns
        self.learning_history = state.learning_history
        self.adaptation_history = state.adaptation_history
        self.system_metrics = state.system_metrics
        
        # Process request
        result_state = await self.compiled_workflow.ainvoke(state.dict())
        
        # Update internal state from result
        self.agent_performance = result_state.get("agent_performance", {})
        self.interaction_patterns = result_state.get("interaction_patterns", {})
        self.learning_history = result_state.get("learning_history", [])
        self.adaptation_history = result_state.get("adaptation_history", [])
        self.system_metrics = result_state.get("system_metrics", {})
        
        return LearningState(**result_state)