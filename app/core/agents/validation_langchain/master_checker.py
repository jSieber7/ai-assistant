"""
LangGraph-based advanced master checker agent for multi-writer/checker system.

This module implements a large, comprehensive checker model that can assess
and coordinate smaller checkers, resolve conflicts, and provide authoritative
final assessments.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from .checker_agent import LangGraphCheckerAgent
from .collaborative_checker import CheckerAssessment, CheckerDispute, ConsensusLevel, ConsensusResult
from app.core.langchain.integration import get_integration

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between checkers"""
    WEIGHTED_AVERAGE = "weighted_average"
    EVIDENCE_BASED = "evidence_based"
    CONSENSUS_BUILDING = "consensus_building"
    AUTHORITATIVE_OVERRIDE = "authoritative_override"
    CONTEXT_DEPENDENT = "context_dependent"


class ValidationLevel(Enum):
    """Validation levels for master checker"""
    BASIC = "basic"  # Simple validation
    STANDARD = "standard"  # Standard validation
    COMPREHENSIVE = "comprehensive"  # Thorough validation
    EXHAUSTIVE = "exhaustive"  # Maximum validation


@dataclass
class MasterCheckerAssessment:
    """Comprehensive assessment from master checker"""
    assessment_id: str
    content: str
    overall_score: float
    confidence: float
    validation_level: ValidationLevel
    detailed_analysis: Dict[str, Any]
    conflict_resolutions: List[Dict[str, Any]]
    quality_dimensions: Dict[str, float]
    improvement_recommendations: List[Dict[str, Any]]
    final_content: str
    checker_evaluations: Dict[str, float]  # checker_id -> evaluation_score
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckerEvaluation:
    """Evaluation of a subordinate checker"""
    checker_id: str
    focus_area: str
    assessment_accuracy: float  # How accurate their assessment was
    reasoning_quality: float  # Quality of their reasoning
    thoroughness: float  # How thorough they were
    consistency: float  # Consistency with other checkers
    overall_reliability: float  # Overall reliability score
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class MasterCheckerState(TypedDict):
    """State for the LangGraph master checker"""
    content: str
    checker_results: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]]
    assessment_id: str
    checker_analysis: Optional[Dict[str, Any]]
    conflict_resolutions: List[Dict[str, Any]]
    detailed_analysis: Optional[Dict[str, Any]]
    quality_dimensions: Optional[Dict[str, float]]
    improvement_recommendations: List[Dict[str, Any]]
    final_content: Optional[str]
    overall_score: Optional[float]
    confidence: Optional[float]
    checker_evaluations: Optional[Dict[str, float]]
    reasoning: Optional[str]
    final_assessment: Optional[MasterCheckerAssessment]
    error: Optional[str]


class LangGraphMasterCheckerAgent:
    """LangGraph-based advanced master checker agent with comprehensive assessment capabilities"""

    def __init__(
        self,
        model: str = "claude-3.5-sonnet",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_AVERAGE,
        enable_learning: bool = True,
    ):
        self.model = model
        self.validation_level = validation_level
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.enable_learning = enable_learning
        self.workflow = self._create_workflow()

        # Performance tracking
        self.assessment_history: List[MasterCheckerAssessment] = []
        self.checker_performance: Dict[str, CheckerEvaluation] = {}

        # Quality dimensions
        self.quality_dimensions = [
            "accuracy",
            "clarity",
            "coherence",
            "completeness",
            "engagement",
            "originality",
            "structure",
            "style",
        ]

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the master checker"""
        workflow = StateGraph(MasterCheckerState)

        # Add nodes
        workflow.add_node("analyze_checker_results", self._analyze_checker_results)
        workflow.add_node("resolve_conflicts", self._resolve_conflicts)
        workflow.add_node("perform_detailed_analysis", self._perform_detailed_analysis)
        workflow.add_node("evaluate_quality_dimensions", self._evaluate_quality_dimensions)
        workflow.add_node("generate_improvement_recommendations", self._generate_improvement_recommendations)
        workflow.add_node("create_final_content", self._create_final_content)
        workflow.add_node("calculate_overall_metrics", self._calculate_overall_metrics)
        workflow.add_node("evaluate_subordinate_checkers", self._evaluate_subordinate_checkers)
        workflow.add_node("generate_assessment_reasoning", self._generate_assessment_reasoning)
        workflow.add_node("finalize_assessment", self._finalize_assessment)

        # Add edges
        workflow.set_entry_point("analyze_checker_results")
        workflow.add_edge("analyze_checker_results", "resolve_conflicts")
        workflow.add_edge("resolve_conflicts", "perform_detailed_analysis")
        workflow.add_edge("perform_detailed_analysis", "evaluate_quality_dimensions")
        workflow.add_edge("evaluate_quality_dimensions", "generate_improvement_recommendations")
        workflow.add_edge("generate_improvement_recommendations", "create_final_content")
        workflow.add_edge("create_final_content", "calculate_overall_metrics")
        workflow.add_edge("calculate_overall_metrics", "evaluate_subordinate_checkers")
        workflow.add_edge("evaluate_subordinate_checkers", "generate_assessment_reasoning")
        workflow.add_edge("generate_assessment_reasoning", "finalize_assessment")
        workflow.add_edge("finalize_assessment", END)

        return workflow.compile()

    async def comprehensive_assessment(
        self,
        content: str,
        checker_results: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
    ) -> MasterCheckerAssessment:
        """
        Provide comprehensive assessment using master checker capabilities.

        Args:
            content: Content to assess
            checker_results: Results from subordinate checkers
            context: Additional context

        Returns:
            MasterCheckerAssessment with comprehensive evaluation
        """
        assessment_id = f"master_assessment_{int(time.time())}"
        logger.info(f"Starting comprehensive assessment {assessment_id}")

        initial_state = {
            "content": content,
            "checker_results": checker_results,
            "context": context,
            "assessment_id": assessment_id,
            "checker_analysis": None,
            "conflict_resolutions": [],
            "detailed_analysis": None,
            "quality_dimensions": None,
            "improvement_recommendations": [],
            "final_content": None,
            "overall_score": None,
            "confidence": None,
            "checker_evaluations": None,
            "reasoning": None,
            "final_assessment": None,
            "error": None,
        }

        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            if result.get("error"):
                logger.error(f"Comprehensive assessment {assessment_id} failed: {result['error']}")
                return self._create_fallback_assessment(
                    assessment_id, content, checker_results
                )
            
            final_assessment = result.get("final_assessment")
            if final_assessment:
                # Store assessment
                self.assessment_history.append(final_assessment)

                # Update checker performance if learning is enabled
                if self.enable_learning:
                    await self._update_checker_performance(final_assessment.checker_evaluations)

                logger.info(
                    f"Completed comprehensive assessment {assessment_id} with score {final_assessment.overall_score:.1f}"
                )
                return final_assessment
            else:
                logger.error(f"No final assessment produced for {assessment_id}")
                return self._create_fallback_assessment(
                    assessment_id, content, checker_results
                )

        except Exception as e:
            logger.error(f"Comprehensive assessment {assessment_id} failed: {str(e)}")
            return self._create_fallback_assessment(
                assessment_id, content, checker_results
            )

    async def _analyze_checker_results(self, state: MasterCheckerState) -> MasterCheckerState:
        """Analyze results from subordinate checkers"""
        checker_results = state["checker_results"]
        
        analysis = {
            "total_checkers": len(checker_results),
            "checker_scores": {},
            "checker_issues": {},
            "checker_improvements": {},
            "score_variance": 0.0,
            "common_issues": [],
            "divergent_assessments": [],
            "consensus_level": 0.0,
        }

        if not checker_results:
            state["checker_analysis"] = analysis
            return state

        # Extract scores and issues
        checker_analysis_scores = []
        all_issues = []

        for result in checker_results:
            checker_id = result.get("checker_id", "unknown")
            score = result.get("score", 0.0)
            issues = result.get("issues_found", [])
            improvements = result.get("improvements", [])

            analysis["checker_scores"][checker_id] = score
            analysis["checker_issues"][checker_id] = issues
            analysis["checker_improvements"][checker_id] = improvements

            checker_analysis_scores.append(score)
            all_issues.extend(issues)

        # Calculate statistics
        if checker_analysis_scores:
            analysis["score_variance"] = max(checker_analysis_scores) - min(
                checker_analysis_scores
            )
            analysis["average_score"] = sum(checker_analysis_scores) / len(
                checker_analysis_scores
            )

        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_desc = issue.get("description", "")
            issue_type = issue.get("type", "general")
            issue_key = f"{issue_type}:{issue_desc}"

            if issue_key not in issue_counts:
                issue_counts[issue_key] = {
                    "count": 0,
                    "type": issue_type,
                    "description": issue_desc,
                }
            issue_counts[issue_key]["count"] += 1

        # Identify common issues (mentioned by multiple checkers)
        analysis["common_issues"] = [
            issue_data
            for issue_data in issue_counts.values()
            if issue_data["count"] >= 2
        ]

        # Calculate consensus level
        if analysis["score_variance"] < 10:
            analysis["consensus_level"] = 0.9
        elif analysis["score_variance"] < 20:
            analysis["consensus_level"] = 0.7
        elif analysis["score_variance"] < 30:
            analysis["consensus_level"] = 0.5
        else:
            analysis["consensus_level"] = 0.3

        state["checker_analysis"] = analysis
        return state

    async def _resolve_conflicts(self, state: MasterCheckerState) -> MasterCheckerState:
        """Resolve conflicts between checker assessments"""
        checker_analysis = state["checker_analysis"]
        content = state["content"]
        context = state.get("context", {})
        
        conflicts = []

        # Identify score conflicts
        if checker_analysis["score_variance"] > 15:
            conflict = {
                "type": "score_disagreement",
                "description": f"Significant score variance: {checker_analysis['score_variance']:.1f}",
                "severity": min(1.0, checker_analysis["score_variance"] / 50.0),
                "resolution": await self._resolve_score_conflict(
                    checker_analysis, content
                ),
            }
            conflicts.append(conflict)

        # Identify content conflicts
        content_conflicts = await self._identify_content_conflicts(checker_analysis)
        conflicts.extend(content_conflicts)

        # Resolve identified conflicts
        for conflict in conflicts:
            if not conflict.get("resolution"):
                conflict["resolution"] = await self._apply_resolution_strategy(
                    conflict, checker_analysis, content, context
                )

        state["conflict_resolutions"] = conflicts
        return state

    async def _resolve_score_conflict(
        self, checker_analysis: Dict[str, Any], content: str
    ) -> Dict[str, Any]:
        """Resolve score conflicts between checkers"""
        resolution_prompt = f"""
Resolve score conflict between checkers for this content:

CONTENT:
{content[:500]}...

CHECKER SCORES:
{json.dumps(checker_analysis["checker_scores"], indent=2)}

CONSENSUS LEVEL: {checker_analysis["consensus_level"]}

Provide a resolution as JSON:
{{
    "resolved_score": <0-100>,
    "reasoning": "explanation of resolution",
    "confidence": <0.0-1.0>,
    "weights_applied": {{"checker_id": <weight>}}
}}
"""

        try:
            integration = get_integration()
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(self.model)
            
            response = await llm.ainvoke([
                {
                    "role": "system",
                    "content": self._get_conflict_resolution_system_prompt(),
                },
                {"role": "user", "content": resolution_prompt},
            ])

            return self._parse_resolution_response(response.content if hasattr(response, 'content') else str(response))

        except Exception as e:
            logger.error(f"Score conflict resolution failed: {str(e)}")
            # Fallback to weighted average
            checker_analysis_scores = list(checker_analysis["checker_scores"].values())
            return {
                "resolved_score": sum(checker_analysis_scores)
                / len(checker_analysis_scores),
                "reasoning": "Used weighted average due to resolution failure",
                "confidence": 0.5,
                "weights_applied": {
                    checker_id: 1.0 for checker_id in checker_analysis["checker_scores"]
                },
            }

    async def _identify_content_conflicts(
        self, checker_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify conflicts in content recommendations"""
        conflicts = []

        # Check for conflicting improvements
        all_improvements = []
        for checker_id, improvements in checker_analysis[
            "checker_improvements"
        ].items():
            for improvement in improvements:
                all_improvements.append(
                    {
                        "checker_id": checker_id,
                        "type": improvement.get("type", "general"),
                        "description": improvement.get("description", ""),
                    }
                )

        # Group improvements by type
        improvement_groups = {}
        for improvement in all_improvements:
            imp_type = improvement["type"]
            if imp_type not in improvement_groups:
                improvement_groups[imp_type] = []
            improvement_groups[imp_type].append(improvement)

        # Look for conflicting recommendations
        for imp_type, improvements in improvement_groups.items():
            if len(improvements) > 1:
                # Check if descriptions suggest different approaches
                descriptions = [imp["description"] for imp in improvements]
                if len(set(descriptions)) > 1:  # Different recommendations
                    conflict = {
                        "type": "content_conflict",
                        "conflict_type": imp_type,
                        "description": f"Conflicting {imp_type} recommendations",
                        "severity": 0.6,
                        "conflicting_recommendations": improvements,
                        "resolution": None,  # Will be resolved later
                    }
                    conflicts.append(conflict)

        return conflicts

    async def _apply_resolution_strategy(
        self,
        conflict: Dict[str, Any],
        checker_analysis: Dict[str, Any],
        content: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the configured conflict resolution strategy"""
        if (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.WEIGHTED_AVERAGE
        ):
            return await self._weighted_average_resolution(conflict, checker_analysis)
        elif (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.EVIDENCE_BASED
        ):
            return await self._evidence_based_resolution(conflict, content, context)
        elif (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.CONSENSUS_BUILDING
        ):
            return await self._consensus_building_resolution(conflict, checker_analysis)
        elif (
            self.conflict_resolution_strategy
            == ConflictResolutionStrategy.AUTHORITATIVE_OVERRIDE
        ):
            return await self._authoritative_resolution(conflict, checker_analysis)
        else:  # CONTEXT_DEPENDENT
            return await self._context_dependent_resolution(
                conflict, checker_analysis, content, context
            )

    async def _weighted_average_resolution(
        self, conflict: Dict[str, Any], checker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply weighted average resolution strategy"""
        if conflict["type"] == "score_disagreement":
            weights = {
                checker_id: 1.0 for checker_id in checker_analysis["checker_scores"]
            }

            # Apply consensus-based weighting
            consensus_weight = checker_analysis["consensus_level"]
            for checker_id in weights:
                weights[checker_id] *= consensus_weight

            # Calculate weighted average
            weighted_sum = sum(
                checker_analysis["checker_scores"][checker_id] * weight
                for checker_id, weight in weights.items()
            )
            total_weight = sum(weights.values())

            resolved_score = weighted_sum / total_weight if total_weight > 0 else 70.0

            return {
                "resolved_score": resolved_score,
                "reasoning": f"Applied weighted average resolution (consensus: {consensus_weight:.2f})",
                "confidence": min(0.8, consensus_weight),
                "weights_applied": weights,
            }

        elif conflict["type"] == "content_conflict":
            # For content conflicts, select the most frequently recommended approach
            recommendations = conflict["conflicting_recommendations"]
            recommendation_counts = {}

            for rec in recommendations:
                desc = rec["description"]
                if desc not in recommendation_counts:
                    recommendation_counts[desc] = []
                recommendation_counts[desc].append(rec["checker_id"])

            # Find most recommended
            most_recommended = max(
                recommendation_counts.items(), key=lambda x: len(x[1])
            )

            return {
                "resolved_recommendation": most_recommended[0],
                "supporting_checkers": most_recommended[1],
                "reasoning": f"Selected most recommended approach ({len(most_recommended[1])} checkers)",
                "confidence": min(0.9, len(most_recommended[1]) / len(recommendations)),
            }

        return {"resolution": "No resolution available", "confidence": 0.0}

    async def _evidence_based_resolution(
        self, conflict: Dict[str, Any], content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply evidence-based resolution strategy"""
        # Analyze content evidence to resolve conflicts
        evidence_prompt = f"""
Analyze this content to resolve a conflict:

CONTENT:
{content[:500]}...

CONTEXT:
{json.dumps(context or {}, indent=2)}

CONFLICT:
{json.dumps(conflict, indent=2)}

Provide evidence-based resolution as JSON:
{{
    "resolution": "resolution based on content evidence",
    "evidence": ["evidence1", "evidence2"],
    "reasoning": "explanation",
    "confidence": <0.0-1.0>
}}
"""

        try:
            integration = get_integration()
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(self.model)
            
            response = await llm.ainvoke([
                {
                    "role": "system",
                    "content": "You are an evidence-based conflict resolver. Analyze content evidence to determine the best resolution.",
                },
                {"role": "user", "content": evidence_prompt},
            ])

            return self._parse_resolution_response(response.content if hasattr(response, 'content') else str(response))

        except Exception as e:
            logger.error(f"Evidence-based resolution failed: {str(e)}")
            # Fallback to weighted average
            return await self._weighted_average_resolution(conflict, {})

    async def _consensus_building_resolution(
        self, conflict: Dict[str, Any], checker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply consensus building resolution strategy"""
        # For now, implement as weighted average favoring consensus
        consensus_factor = checker_analysis["consensus_level"]

        if consensus_factor > 0.7:
            # High consensus - use average
            return await self._weighted_average_resolution(conflict, checker_analysis)
        else:
            # Low consensus - favor more conservative approach
            if conflict["type"] == "score_disagreement":
                checker_analysis_scores = list(
                    checker_analysis["checker_scores"].values()
                )
                return {
                    "resolved_score": max(checker_analysis_scores)
                    - 10,  # Conservative scoring
                    "reasoning": "Applied conservative resolution due to low consensus",
                    "confidence": consensus_factor,
                    "weights_applied": {},
                }

        return {"resolution": "No consensus resolution available", "confidence": 0.0}

    async def _authoritative_resolution(
        self, conflict: Dict[str, Any], checker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply authoritative resolution strategy (master checker decides)"""
        if conflict["type"] == "score_disagreement":
            # Use master checker's own assessment
            authoritative_prompt = f"""
As the master checker, provide authoritative resolution for this score conflict:

CHECKER SCORES:
{json.dumps(checker_analysis["checker_scores"], indent=2)}

CONSENSUS LEVEL: {checker_analysis["consensus_level"]}

Provide your authoritative assessment as JSON:
{{
    "resolved_score": <0-100>,
    "reasoning": "authoritative reasoning",
    "confidence": <0.0-1.0>
}}
"""

            try:
                integration = get_integration()
                llm_manager = integration.get_llm_manager()
                llm = await llm_manager.get_llm(self.model)
                
                response = await llm.ainvoke([
                    {
                        "role": "system",
                        "content": "You are the authoritative master checker. Provide definitive resolutions based on your expertise.",
                    },
                    {"role": "user", "content": authoritative_prompt},
                ])

                return self._parse_resolution_response(response.content if hasattr(response, 'content') else str(response))

            except Exception as e:
                logger.error(f"Authoritative resolution failed: {str(e)}")
                # Fallback
                checker_analysis_scores = list(
                    checker_analysis["checker_scores"].values()
                )
                return {
                    "resolved_score": sum(checker_analysis_scores)
                    / len(checker_analysis_scores),
                    "reasoning": "Used average due to resolution failure",
                    "confidence": 0.6,
                }

        return {
            "resolution": "No authoritative resolution available",
            "confidence": 0.0,
        }

    async def _context_dependent_resolution(
        self,
        conflict: Dict[str, Any],
        checker_analysis: Dict[str, Any],
        content: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply context-dependent resolution strategy"""
        # Analyze context to determine best resolution approach
        quality_requirements = context.get("quality_requirements", [])

        if (
            "factual" in quality_requirements
            and conflict["type"] == "score_disagreement"
        ):
            # For factual content, prefer more conservative scoring
            checker_analysis_scores = list(checker_analysis["checker_scores"].values())
            return {
                "resolved_score": min(
                    checker_analysis_scores
                ),  # Use most conservative score
                "reasoning": "Applied conservative resolution for factual content",
                "confidence": 0.8,
            }
        elif "creative" in quality_requirements:
            # For creative content, prefer higher scores
            checker_analysis_scores = list(checker_analysis["checker_scores"].values())
            return {
                "resolved_score": max(
                    checker_analysis_scores
                ),  # Use most liberal score
                "reasoning": "Applied liberal resolution for creative content",
                "confidence": 0.8,
            }
        else:
            # Default to weighted average
            return await self._weighted_average_resolution(conflict, checker_analysis)

    async def _perform_detailed_analysis(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Perform detailed content analysis"""
        content = state["content"]
        context = state.get("context", {})
        
        analysis_prompt = f"""
Perform detailed analysis of this content:

CONTENT:
{content[:1000]}...

CONTEXT:
{json.dumps(context or {}, indent=2)}

Provide comprehensive analysis as JSON:
{{
    "content_summary": "brief summary",
    "key_points": ["point1", "point2"],
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "structure_analysis": {{
        "introduction": "assessment",
        "body": "assessment",
        "conclusion": "assessment"
    }},
    "language_analysis": {{
        "clarity": "assessment",
        "tone": "assessment",
        "style": "assessment"
    }},
    "content_gaps": ["gap1", "gap2"],
    "improvement_areas": ["area1", "area2"]
}}
"""

        try:
            integration = get_integration()
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(self.model)
            
            response = await llm.ainvoke([
                {"role": "system", "content": self._get_analysis_system_prompt()},
                {"role": "user", "content": analysis_prompt},
            ])

            state["detailed_analysis"] = self._parse_analysis_response(response.content if hasattr(response, 'content') else str(response))
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {str(e)}")
            state["detailed_analysis"] = {
                "content_summary": "Analysis failed",
                "key_points": [],
                "strengths": [],
                "weaknesses": [],
                "structure_analysis": {},
                "language_analysis": {},
                "content_gaps": [],
                "improvement_areas": [],
            }
            
        return state

    async def _evaluate_quality_dimensions(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Evaluate content across multiple quality dimensions"""
        content = state["content"]
        detailed_analysis = state.get("detailed_analysis", {})
        checker_analysis = state.get("checker_analysis", {})
        
        dimensions = {}

        # Get checker scores by focus area
        checker_scores_by_area = {}
        for checker_id, score in checker_analysis.get("checker_scores", {}).items():
            # Determine checker focus area from checker_id if possible
            if "factual" in checker_id.lower():
                checker_scores_by_area["factual"] = score
            elif "style" in checker_id.lower():
                checker_scores_by_area["style"] = score
            elif "structure" in checker_id.lower():
                checker_scores_by_area["structure"] = score
            elif "seo" in checker_id.lower():
                checker_scores_by_area["seo"] = score

        # Evaluate each dimension
        for dimension in self.quality_dimensions:
            score = 0.0

            if dimension == "accuracy" and "factual" in checker_scores_by_area:
                score = checker_scores_by_area["factual"]
            elif dimension == "clarity":
                # Analyze from detailed analysis
                language_analysis = detailed_analysis.get("language_analysis", {})
                if language_analysis.get("clarity") == "good":
                    score = 80.0
                elif language_analysis.get("clarity") == "excellent":
                    score = 90.0
                else:
                    score = 70.0
            elif dimension == "structure" and "structure" in checker_scores_by_area:
                score = checker_scores_by_area["structure"]
            elif dimension == "style" and "style" in checker_scores_by_area:
                score = checker_scores_by_area["style"]
            else:
                # Use average checker score as fallback
                checker_analysis_scores = list(
                    checker_analysis.get("checker_scores", {}).values()
                )
                score = (
                    sum(checker_analysis_scores) / len(checker_analysis_scores)
                    if checker_analysis_scores
                    else 70.0
                )

            dimensions[dimension] = score

        state["quality_dimensions"] = dimensions
        return state

    async def _generate_improvement_recommendations(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Generate specific improvement recommendations"""
        quality_dimensions = state.get("quality_dimensions", {})
        detailed_analysis = state.get("detailed_analysis", {})
        
        recommendations = []

        # Identify low-scoring dimensions
        low_scoring_dimensions = [
            (dim, score) for dim, score in quality_dimensions.items() if score < 75.0
        ]

        # Generate recommendations for each low-scoring dimension
        for dimension, score in low_scoring_dimensions:
            recommendation = await self._generate_dimension_recommendation(
                dimension, score, detailed_analysis
            )
            recommendations.append(recommendation)

        # Add recommendations from detailed analysis
        improvement_areas = detailed_analysis.get("improvement_areas", [])
        for area in improvement_areas:
            recommendations.append(
                {
                    "type": "general",
                    "priority": "medium",
                    "description": f"Improve: {area}",
                    "action": f"Address the identified issues with {area}",
                }
            )

        state["improvement_recommendations"] = recommendations
        return state

    async def _generate_dimension_recommendation(
        self, dimension: str, score: float, detailed_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendation for a specific quality dimension"""
        recommendation_prompts = {
            "accuracy": "Improve factual accuracy and verify claims",
            "clarity": "Enhance clarity and readability",
            "coherence": "Improve logical flow and coherence",
            "completeness": "Add missing information and complete thoughts",
            "engagement": "Increase reader engagement and interest",
            "originality": "Enhance originality and unique insights",
            "structure": "Improve content structure and organization",
            "style": "Refine writing style and tone",
        }

        base_description = recommendation_prompts.get(dimension, f"Improve {dimension}")

        return {
            "type": dimension,
            "priority": "high" if score < 60 else "medium",
            "description": base_description,
            "action": f"Focus on improving {dimension} to increase quality score from {score:.1f}",
            "current_score": score,
        }

    async def _create_final_content(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Create final improved content"""
        original_content = state["content"]
        improvement_recommendations = state.get("improvement_recommendations", [])
        conflict_resolutions = state.get("conflict_resolutions", [])
        
        # For now, return original content with note that improvements would be applied
        # In a full implementation, this would use the recommendations to actually improve the content

        if not improvement_recommendations and not conflict_resolutions:
            state["final_content"] = original_content
            return state

        # Create a summary of improvements that would be made
        improvement_summary = "\n\nIMPROVEMENTS TO BE APPLIED:\n"

        for rec in improvement_recommendations[:3]:  # Limit to top 3
            improvement_summary += f"- {rec['description']}\n"

        for resolution in conflict_resolutions[:2]:  # Limit to top 2
            if resolution.get("resolution"):
                improvement_summary += (
                    f"- Resolved {resolution['type']}: {resolution['resolution']}\n"
                )

        state["final_content"] = f"{original_content}{improvement_summary}"
        return state

    async def _calculate_overall_metrics(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Calculate overall score and confidence"""
        quality_dimensions = state.get("quality_dimensions", {})
        checker_analysis = state.get("checker_analysis", {})
        conflict_resolutions = state.get("conflict_resolutions", [])
        
        # Calculate overall score from quality dimensions
        if quality_dimensions:
            overall_score = sum(quality_dimensions.values()) / len(quality_dimensions)
        else:
            checker_analysis_scores = list(checker_analysis.get("checker_scores", {}).values())
            overall_score = (
                sum(checker_analysis_scores) / len(checker_analysis_scores)
                if checker_analysis_scores
                else 70.0
            )

        # Calculate confidence based on consensus and conflicts
        base_confidence = checker_analysis.get("consensus_level", 0.5)

        # Adjust confidence based on conflicts
        if conflict_resolutions:
            conflict_penalty = min(0.3, len(conflict_resolutions) * 0.1)
            base_confidence = max(0.3, base_confidence - conflict_penalty)

        # Adjust confidence based on resolution success
        successful_resolutions = [
            r
            for r in conflict_resolutions
            if r.get("resolution") and r.get("confidence", 0) > 0.5
        ]

        if conflict_resolutions:
            resolution_bonus = min(
                0.2, len(successful_resolutions) / len(conflict_resolutions) * 0.2
            )
            base_confidence = min(1.0, base_confidence + resolution_bonus)

        state["overall_score"] = overall_score
        state["confidence"] = base_confidence
        return state

    async def _evaluate_subordinate_checkers(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Evaluate the performance of subordinate checkers"""
        checker_results = state["checker_results"]
        detailed_analysis = state.get("detailed_analysis", {})
        final_content = state.get("final_content", "")
        
        evaluations = {}

        for result in checker_results:
            checker_id = result.get("checker_id", "unknown")

            # Calculate evaluation score based on multiple factors
            accuracy_score = 0.0  # How accurate their assessment was
            reasoning_score = 0.0  # Quality of their reasoning
            thoroughness_score = 0.0  # How thorough they were

            # For now, use a simple evaluation based on their score
            checker_score = result.get("score", 70.0)

            # Calculate evaluation metrics
            accuracy_score = min(
                100, 100 - abs(checker_score - 75)
            )  # Assume 75 is "correct"
            reasoning_score = min(
                100, checker_score + 10
            )  # Higher score suggests better reasoning
            thoroughness_score = min(100, len(result.get("issues_found", [])) * 10 + 50)

            # Overall evaluation
            evaluation_score = (
                accuracy_score + reasoning_score + thoroughness_score
            ) / 3.0
            evaluations[checker_id] = evaluation_score

            # Store detailed evaluation if learning is enabled
            if self.enable_learning:
                self.checker_performance[checker_id] = CheckerEvaluation(
                    checker_id=checker_id,
                    focus_area=result.get("focus_area", "unknown"),
                    assessment_accuracy=accuracy_score,
                    reasoning_quality=reasoning_score,
                    thoroughness=thoroughness_score,
                    consistency=0.8,  # Would be calculated based on consistency with others
                    overall_reliability=evaluation_score,
                    strengths=[],
                    weaknesses=[],
                    recommendations=[],
                )

        state["checker_evaluations"] = evaluations
        return state

    async def _generate_assessment_reasoning(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Generate reasoning for the assessment"""
        overall_score = state.get("overall_score", 0.0)
        quality_dimensions = state.get("quality_dimensions", {})
        conflict_resolutions = state.get("conflict_resolutions", [])
        checker_evaluations = state.get("checker_evaluations", {})
        
        reasoning_parts = []

        reasoning_parts.append(f"Overall quality score: {overall_score:.1f}/100")

        # Add dimension analysis
        high_scoring_dims = [
            dim for dim, score in quality_dimensions.items() if score >= 80
        ]
        low_scoring_dims = [
            dim for dim, score in quality_dimensions.items() if score < 70
        ]

        if high_scoring_dims:
            reasoning_parts.append(f"Strengths in: {', '.join(high_scoring_dims)}")

        if low_scoring_dims:
            reasoning_parts.append(
                f"Areas for improvement: {', '.join(low_scoring_dims)}"
            )

        # Add conflict resolution information
        if conflict_resolutions:
            reasoning_parts.append(
                f"Resolved {len(conflict_resolutions)} conflicts between checkers"
            )

        # Add checker evaluation information
        if checker_evaluations:
            avg_checker_performance = sum(checker_evaluations.values()) / len(
                checker_evaluations
            )
            reasoning_parts.append(
                f"Average checker performance: {avg_checker_performance:.1f}/100"
            )

        state["reasoning"] = " | ".join(reasoning_parts)
        return state

    async def _finalize_assessment(
        self, state: MasterCheckerState
    ) -> MasterCheckerState:
        """Finalize the assessment"""
        assessment_id = state["assessment_id"]
        content = state["content"]
        overall_score = state.get("overall_score", 0.0)
        confidence = state.get("confidence", 0.0)
        detailed_analysis = state.get("detailed_analysis", {})
        conflict_resolutions = state.get("conflict_resolutions", [])
        quality_dimensions = state.get("quality_dimensions", {})
        improvement_recommendations = state.get("improvement_recommendations", [])
        final_content = state.get("final_content", content)
        checker_evaluations = state.get("checker_evaluations", {})
        reasoning = state.get("reasoning", "")

        # Create assessment
        assessment = MasterCheckerAssessment(
            assessment_id=assessment_id,
            content=content,
            overall_score=overall_score,
            confidence=confidence,
            validation_level=self.validation_level,
            detailed_analysis=detailed_analysis,
            conflict_resolutions=conflict_resolutions,
            quality_dimensions=quality_dimensions,
            improvement_recommendations=improvement_recommendations,
            final_content=final_content,
            checker_evaluations=checker_evaluations,
            reasoning=reasoning,
            metadata={
                "checker_count": len(state.get("checker_results", [])),
                "conflict_count": len(conflict_resolutions),
                "context": state.get("context"),
            },
        )

        state["final_assessment"] = assessment
        return state

    def _get_conflict_resolution_system_prompt(self) -> str:
        """Get system prompt for conflict resolution"""
        return """
You are an expert conflict resolver specializing in mediating between AI checkers.
Your role is to find fair, evidence-based resolutions that maintain quality standards.
Consider all perspectives and prioritize to best outcome for the content.
"""

    def _get_analysis_system_prompt(self) -> str:
        """Get system prompt for content analysis"""
        return """
You are an expert content analyst with deep understanding of writing quality,
structure, and effectiveness. Provide comprehensive, objective analysis.
"""

    def _parse_resolution_response(self, response: str) -> Dict[str, Any]:
        """Parse resolution response from LLM"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # Look for JSON object in the text
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]

            return json.loads(json_str)
        except (json.JSONDecodeError, KeyError, AttributeError):
            # Fallback resolution
            return {
                "resolution": "Resolution parsing failed",
                "reasoning": "Unable to parse structured response",
                "confidence": 0.3,
            }

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse analysis response from LLM"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # Look for JSON object in the text
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]

            return json.loads(json_str)
        except (json.JSONDecodeError, KeyError, AttributeError):
            # Fallback analysis
            return {
                "content_summary": "Analysis parsing failed",
                "key_points": [],
                "strengths": [],
                "weaknesses": [],
                "structure_analysis": {},
                "language_analysis": {},
                "content_gaps": [],
                "improvement_areas": [],
            }

    def _create_fallback_assessment(
        self, assessment_id: str, content: str, checker_results: List[Dict[str, Any]]
    ) -> MasterCheckerAssessment:
        """Create fallback assessment when comprehensive assessment fails"""
        # Calculate basic metrics from checker results
        scores = [r.get("score", 70.0) for r in checker_results]
        avg_score = sum(scores) / len(scores) if scores else 70.0

        return MasterCheckerAssessment(
            assessment_id=assessment_id,
            content=content,
            overall_score=avg_score,
            confidence=0.3,  # Low confidence for fallback
            validation_level=ValidationLevel.BASIC,
            detailed_analysis={"error": "Comprehensive assessment failed"},
            conflict_resolutions=[],
            quality_dimensions={dim: avg_score for dim in self.quality_dimensions},
            improvement_recommendations=[
                {"type": "general", "description": "Review content manually"}
            ],
            final_content=content,
            checker_evaluations={
                r.get("checker_id", "unknown"): 50.0 for r in checker_results
            },
            reasoning="Fallback assessment due to system failure",
        )

    async def _update_checker_performance(self, checker_evaluations: Dict[str, float]):
        """Update checker performance metrics"""
        # This would integrate with a learning system in a full implementation
        for checker_id, evaluation_score in checker_evaluations.items():
            if checker_id in self.checker_performance:
                # Update performance metrics
                current = self.checker_performance[checker_id]
                # Simple update - in full implementation would use more sophisticated learning
                current.overall_reliability = (
                    current.overall_reliability + evaluation_score
                ) / 2.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for master checker"""
        if not self.assessment_history:
            return {"message": "No assessments performed yet"}

        recent_assessments = self.assessment_history[-10:]  # Last 10 assessments

        return {
            "total_assessments": len(self.assessment_history),
            "average_score": sum(a.overall_score for a in recent_assessments)
            / len(recent_assessments),
            "average_confidence": sum(a.confidence for a in recent_assessments)
            / len(recent_assessments),
            "common_conflicts": self._get_common_conflicts(recent_assessments),
            "checker_performance": {
                checker_id: eval.overall_reliability
                for checker_id, eval in self.checker_performance.items()
            },
            "validation_level": self.validation_level.value,
            "conflict_resolution_strategy": self.conflict_resolution_strategy.value,
        }

    def _get_common_conflicts(
        self, assessments: List[MasterCheckerAssessment]
    ) -> List[Dict[str, Any]]:
        """Identify most common conflict types"""
        conflict_counts = {}

        for assessment in assessments:
            for conflict in assessment.conflict_resolutions:
                conflict_type = conflict.get("type", "unknown")
                if conflict_type not in conflict_counts:
                    conflict_counts[conflict_type] = 0
                conflict_counts[conflict_type] += 1

        # Sort by frequency
        sorted_conflicts = sorted(
            conflict_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"type": conflict_type, "count": count}
            for conflict_type, count in sorted_conflicts[:5]
        ]