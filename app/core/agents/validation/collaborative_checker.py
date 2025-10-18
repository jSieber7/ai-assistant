"""
Collaborative checking system for multi-writer/checker system.

This module implements a collaborative checking system where checker agents
can discuss, reconcile differences, and build consensus on content quality.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .checker_agent import CheckerAgent
from app.core.config import get_llm

logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Levels of consensus between checkers"""

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    PLURALITY = "plurality"
    NO_CONSENSUS = "no_consensus"


class DisputeType(Enum):
    """Types of disputes between checkers"""

    FACTUAL_DISAGREEMENT = "factual_disagreement"
    STYLE_PREFERENCE = "style_preference"
    STRUCTURE_DIFFERENCE = "structure_difference"
    QUALITY_ASSESSMENT = "quality_assessment"
    SCOPE_INTERPRETATION = "scope_interpretation"


@dataclass
class CheckerAssessment:
    """Assessment from a single checker"""

    checker_id: str
    focus_area: str
    score: float
    issues_found: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]
    improved_content: str
    recommendations: List[str]
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CheckerDispute:
    """Dispute between checkers"""

    dispute_id: str
    dispute_type: DisputeType
    involved_checkers: List[str]
    conflicting_assessments: List[CheckerAssessment]
    description: str
    severity: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsensusResult:
    """Result of consensus building"""

    consensus_level: ConsensusLevel
    final_score: float
    final_content: str
    resolved_disputes: List[CheckerDispute]
    unresolved_disputes: List[CheckerDispute]
    aggregated_issues: List[Dict[str, Any]]
    aggregated_improvements: List[Dict[str, Any]]
    consensus_reasoning: str
    participant_agreement: Dict[str, float]  # checker_id -> agreement_level


class CollaborativeChecker:
    """Orchestrates collaborative checking between checker agents"""

    def __init__(
        self,
        consensus_threshold: float = 0.7,
        max_discussion_rounds: int = 3,
        reconciliation_model: str = "claude-3.5-sonnet",
    ):
        self.consensus_threshold = consensus_threshold
        self.max_discussion_rounds = max_discussion_rounds
        self.reconciliation_model = reconciliation_model
        self.llm = None  # Will be initialized when needed

    async def _get_llm(self):
        """Initialize LLM if not already done"""
        if self.llm is None:
            self.llm = await get_llm(self.reconciliation_model)
        return self.llm

    async def conduct_collaborative_review(
        self, content: str, checkers: List[CheckerAgent], context: Dict[str, Any] = None
    ) -> ConsensusResult:
        """
        Conduct collaborative review with multiple checkers.

        Args:
            content: Content to be reviewed
            checkers: List of checker agents
            context: Additional context

        Returns:
            ConsensusResult with collaborative assessment
        """
        logger.info(f"Starting collaborative review with {len(checkers)} checkers")

        try:
            # Step 1: Individual assessments
            individual_assessments = await self._gather_individual_assessments(
                content, checkers, context
            )

            # Step 2: Identify disputes
            disputes = await self._identify_disputes(individual_assessments)

            # Step 3: Resolve disputes through discussion
            resolved_disputes, unresolved_disputes = await self._resolve_disputes(
                disputes, individual_assessments, content, context
            )

            # Step 4: Build consensus
            consensus_result = await self._build_consensus(
                individual_assessments, resolved_disputes, unresolved_disputes, content
            )

            logger.info(
                f"Collaborative review completed with consensus: {consensus_result.consensus_level.value}"
            )
            return consensus_result

        except Exception as e:
            logger.error(f"Collaborative review failed: {str(e)}")
            # Fallback to simple aggregation
            return self._fallback_consensus(content, checkers, context)

    async def _gather_individual_assessments(
        self, content: str, checkers: List[CheckerAgent], context: Dict[str, Any]
    ) -> List[CheckerAssessment]:
        """Gather individual assessments from all checkers"""
        assessments = []

        # Prepare content for checking
        content_dict = {
            "content": content,
            "writer_id": context.get("writer_id", "unknown"),
            "specialty": context.get("specialty", "unknown"),
        }

        # Run assessments in parallel
        tasks = []
        for checker in checkers:
            task = self._get_checker_assessment(checker, content_dict, context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Checker {checkers[i].checker_id} failed: {str(result)}")
                continue

            assessments.append(result)

        logger.info(f"Gathered {len(assessments)} individual assessments")
        return assessments

    async def _get_checker_assessment(
        self,
        checker: CheckerAgent,
        content_dict: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CheckerAssessment:
        """Get assessment from a single checker"""
        try:
            # Get standard check result
            check_result = await checker.check_and_improve(content_dict, context)

            if check_result.get("error"):
                raise Exception(check_result["error"])

            # Generate additional reasoning
            reasoning = await self._generate_assessment_reasoning(
                checker, check_result, content_dict
            )

            return CheckerAssessment(
                checker_id=checker.checker_id,
                focus_area=checker.focus_area,
                score=check_result.get("score", 0.0),
                issues_found=check_result.get("issues_found", []),
                improvements=check_result.get("improvements", []),
                improved_content=check_result.get("improved_content", ""),
                recommendations=check_result.get("recommendations", []),
                confidence=0.8,  # Could be calculated based on checker performance
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(
                f"Failed to get assessment from {checker.checker_id}: {str(e)}"
            )
            raise

    async def _identify_disputes(
        self, assessments: List[CheckerAssessment]
    ) -> List[CheckerDispute]:
        """Identify disputes between checker assessments"""
        disputes = []

        if len(assessments) < 2:
            return disputes

        # Check for score disagreements
        scores = [a.score for a in assessments]
        score_variance = max(scores) - min(scores)

        if score_variance > 20.0:  # Significant score difference
            dispute = CheckerDispute(
                dispute_id=f"score_dispute_{int(time.time())}",
                dispute_type=DisputeType.QUALITY_ASSESSMENT,
                involved_checkers=[a.checker_id for a in assessments],
                conflicting_assessments=assessments,
                description=f"Significant score disagreement: {min(scores):.1f} vs {max(scores):.1f}",
                severity=min(1.0, score_variance / 50.0),
            )
            disputes.append(dispute)

        # Check for content disagreements
        content_similarities = []
        for i, assessment1 in enumerate(assessments):
            for assessment2 in assessments[i + 1 :]:
                similarity = self._calculate_content_similarity(
                    assessment1.improved_content, assessment2.improved_content
                )
                content_similarities.append(similarity)

        if content_similarities and min(content_similarities) < 0.5:
            # Find the most different assessments
            min_similarity_idx = content_similarities.index(min(content_similarities))
            dispute = CheckerDispute(
                dispute_id=f"content_dispute_{int(time.time())}",
                dispute_type=DisputeType.STYLE_PREFERENCE,
                involved_checkers=[assessments[min_similarity_idx].checker_id],
                conflicting_assessments=[assessments[min_similarity_idx]],
                description="Significant content improvement differences",
                severity=1.0 - min(content_similarities),
            )
            disputes.append(dispute)

        # Check for issue conflicts
        issue_disputes = await self._identify_issue_disputes(assessments)
        disputes.extend(issue_disputes)

        logger.info(f"Identified {len(disputes)} disputes")
        return disputes

    async def _identify_issue_disputes(
        self, assessments: List[CheckerAssessment]
    ) -> List[CheckerDispute]:
        """Identify disputes based on issues found"""
        disputes = []

        # Group issues by type/severity
        issue_types = {}
        for assessment in assessments:
            for issue in assessment.issues_found:
                issue_type = issue.get("type", "general")
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append((assessment.checker_id, issue))

        # Look for conflicting assessments of same issue type
        for issue_type, issue_list in issue_types.items():
            if len(issue_list) > 1:
                # Check if checkers disagree on severity
                severities = [
                    issue[1].get("severity", "medium") for issue in issue_list
                ]
                if len(set(severities)) > 1:
                    involved_assessments = [
                        a
                        for a in assessments
                        if a.checker_id in [issue[0] for issue in issue_list]
                    ]

                    dispute = CheckerDispute(
                        dispute_id=f"issue_dispute_{issue_type}_{int(time.time())}",
                        dispute_type=DisputeType.FACTUAL_DISAGREEMENT,
                        involved_checkers=[issue[0] for issue in issue_list],
                        conflicting_assessments=involved_assessments,
                        description=f"Disagreement on {issue_type} issues: {severities}",
                        severity=0.6,
                    )
                    disputes.append(dispute)

        return disputes

    async def _resolve_disputes(
        self,
        disputes: List[CheckerDispute],
        assessments: List[CheckerAssessment],
        content: str,
        context: Dict[str, Any],
    ) -> Tuple[List[CheckerDispute], List[CheckerDispute]]:
        """Resolve disputes through discussion"""
        resolved_disputes = []
        unresolved_disputes = []

        for dispute in disputes:
            try:
                resolution = await self._resolve_single_dispute(
                    dispute, assessments, content, context
                )

                if resolution["resolved"]:
                    resolved_disputes.append(dispute)
                    logger.info(f"Resolved dispute {dispute.dispute_id}")
                else:
                    unresolved_disputes.append(dispute)
                    logger.warning(f"Could not resolve dispute {dispute.dispute_id}")

            except Exception as e:
                logger.error(f"Error resolving dispute {dispute.dispute_id}: {str(e)}")
                unresolved_disputes.append(dispute)

        return resolved_disputes, unresolved_disputes

    async def _resolve_single_dispute(
        self,
        dispute: CheckerDispute,
        assessments: List[CheckerAssessment],
        content: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve a single dispute through discussion"""
        # Create discussion prompt
        discussion_prompt = self._create_discussion_prompt(
            dispute, assessments, content
        )

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._get_moderator_system_prompt()},
                    {"role": "user", "content": discussion_prompt},
                ]
            )

            # Parse resolution
            resolution = self._parse_dispute_resolution(response.content)

            return resolution

        except Exception as e:
            logger.error(f"Dispute resolution failed: {str(e)}")
            return {"resolved": False, "reasoning": "Resolution process failed"}

    async def _build_consensus(
        self,
        assessments: List[CheckerAssessment],
        resolved_disputes: List[CheckerDispute],
        unresolved_disputes: List[CheckerDispute],
        content: str,
    ) -> ConsensusResult:
        """Build consensus from all assessments and resolutions"""

        # Calculate agreement levels
        participant_agreement = {}
        for assessment in assessments:
            # Base agreement on similarity to others
            agreement = self._calculate_participant_agreement(assessment, assessments)
            participant_agreement[assessment.checker_id] = agreement

        # Determine consensus level
        avg_agreement = sum(participant_agreement.values()) / len(participant_agreement)

        if avg_agreement >= 0.9:
            consensus_level = ConsensusLevel.UNANIMOUS
        elif avg_agreement >= 0.7:
            consensus_level = ConsensusLevel.MAJORITY
        elif avg_agreement >= 0.5:
            consensus_level = ConsensusLevel.PLURALITY
        else:
            consensus_level = ConsensusLevel.NO_CONSENSUS

        # Aggregate assessments
        final_score = self._calculate_weighted_score(assessments, participant_agreement)
        final_content = self._select_best_content(assessments, participant_agreement)

        # Aggregate issues and improvements
        aggregated_issues = self._aggregate_issues(assessments)
        aggregated_improvements = self._aggregate_improvements(assessments)

        # Generate consensus reasoning
        consensus_reasoning = await self._generate_consensus_reasoning(
            consensus_level, assessments, resolved_disputes, unresolved_disputes
        )

        return ConsensusResult(
            consensus_level=consensus_level,
            final_score=final_score,
            final_content=final_content,
            resolved_disputes=resolved_disputes,
            unresolved_disputes=unresolved_disputes,
            aggregated_issues=aggregated_issues,
            aggregated_improvements=aggregated_improvements,
            consensus_reasoning=consensus_reasoning,
            participant_agreement=participant_agreement,
        )

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content pieces"""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 1.0

        return len(intersection) / len(union)

    def _calculate_participant_agreement(
        self,
        target_assessment: CheckerAssessment,
        all_assessments: List[CheckerAssessment],
    ) -> float:
        """Calculate how much a participant agrees with others"""
        if len(all_assessments) <= 1:
            return 1.0

        similarities = []
        for assessment in all_assessments:
            if assessment.checker_id == target_assessment.checker_id:
                continue

            # Score similarity
            score_diff = abs(target_assessment.score - assessment.score)
            score_similarity = max(0, 1.0 - score_diff / 100.0)

            # Content similarity
            content_similarity = self._calculate_content_similarity(
                target_assessment.improved_content, assessment.improved_content
            )

            # Overall similarity
            overall_similarity = (score_similarity + content_similarity) / 2.0
            similarities.append(overall_similarity)

        return sum(similarities) / len(similarities) if similarities else 1.0

    def _calculate_weighted_score(
        self,
        assessments: List[CheckerAssessment],
        participant_agreement: Dict[str, float],
    ) -> float:
        """Calculate weighted score based on participant agreement"""
        if not assessments:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for assessment in assessments:
            weight = participant_agreement.get(assessment.checker_id, 0.5)
            weighted_sum += assessment.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _select_best_content(
        self,
        assessments: List[CheckerAssessment],
        participant_agreement: Dict[str, float],
    ) -> str:
        """Select the best content based on weighted agreement"""
        if not assessments:
            return ""

        best_content = ""
        best_score = -1.0

        for assessment in assessments:
            weight = participant_agreement.get(assessment.checker_id, 0.5)
            weighted_score = assessment.score * weight

            if weighted_score > best_score:
                best_score = weighted_score
                best_content = assessment.improved_content

        return best_content

    def _aggregate_issues(
        self, assessments: List[CheckerAssessment]
    ) -> List[Dict[str, Any]]:
        """Aggregate issues from all assessments"""
        aggregated_issues = []
        issue_tracker = {}

        for assessment in assessments:
            for issue in assessment.issues_found:
                issue_key = f"{issue.get('type', 'unknown')}_{issue.get('description', '')[:50]}"

                if issue_key not in issue_tracker:
                    issue_tracker[issue_key] = {
                        "type": issue.get("type", "unknown"),
                        "description": issue.get("description", ""),
                        "severity": issue.get("severity", "medium"),
                        "frequency": 0,
                        "sources": [],
                    }

                issue_tracker[issue_key]["frequency"] += 1
                issue_tracker[issue_key]["sources"].append(assessment.checker_id)

        # Convert to list and sort by frequency
        for issue_data in issue_tracker.values():
            aggregated_issues.append(issue_data)

        aggregated_issues.sort(key=lambda x: x["frequency"], reverse=True)
        return aggregated_issues

    def _aggregate_improvements(
        self, assessments: List[CheckerAssessment]
    ) -> List[Dict[str, Any]]:
        """Aggregate improvements from all assessments"""
        aggregated_improvements = []
        improvement_tracker = {}

        for assessment in assessments:
            for improvement in assessment.improvements:
                improvement_key = f"{improvement.get('type', 'unknown')}_{improvement.get('description', '')[:50]}"

                if improvement_key not in improvement_tracker:
                    improvement_tracker[improvement_key] = {
                        "type": improvement.get("type", "unknown"),
                        "description": improvement.get("description", ""),
                        "frequency": 0,
                        "sources": [],
                    }

                improvement_tracker[improvement_key]["frequency"] += 1
                improvement_tracker[improvement_key]["sources"].append(
                    assessment.checker_id
                )

        # Convert to list and sort by frequency
        for improvement_data in improvement_tracker.values():
            aggregated_improvements.append(improvement_data)

        aggregated_improvements.sort(key=lambda x: x["frequency"], reverse=True)
        return aggregated_improvements

    async def _generate_assessment_reasoning(
        self,
        checker: CheckerAgent,
        check_result: Dict[str, Any],
        content_dict: Dict[str, Any],
    ) -> str:
        """Generate reasoning for checker assessment"""
        reasoning_prompt = f"""
As a {checker.focus_area} checker, explain your reasoning for this assessment:

CONTENT: {content_dict.get("content", "")[:500]}...

YOUR ASSESSMENT:
- Score: {check_result.get("score", 0)}
- Issues: {len(check_result.get("issues_found", []))}
- Improvements: {len(check_result.get("improvements", []))}

Provide a brief explanation of your reasoning (50-100 words):
"""

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(reasoning_prompt)
            return response.content.strip()
        except Exception:
            return f"Assessment based on {checker.focus_area} expertise and standard quality criteria."

    def _create_discussion_prompt(
        self,
        dispute: CheckerDispute,
        assessments: List[CheckerAssessment],
        content: str,
    ) -> str:
        """Create prompt for dispute discussion"""
        assessments_text = "\n\n".join(
            [
                f"Assessment from {a.checker_id} ({a.focus_area}):\n"
                f"Score: {a.score}\n"
                f"Reasoning: {a.reasoning}\n"
                f"Improved Content: {a.improved_content[:300]}..."
                for a in dispute.conflicting_assessments
            ]
        )

        return f"""
Moderate a discussion between checkers to resolve this dispute:

DISPUTE TYPE: {dispute.dispute_type.value}
DESCRIPTION: {dispute.description}
SEVERITY: {dispute.severity:.2f}

CONTENT UNDER REVIEW:
{content[:500]}...

CONFLICTING ASSESSMENTS:
{assessments_text}

Please facilitate a resolution by:
1. Identifying the core disagreement
2. Finding common ground
3. Suggesting a compromise or consensus position
4. Determining if the dispute can be resolved

Provide your assessment as JSON:
{{
    "resolved": true/false,
    "resolution": "description of resolution",
    "compromise_position": "compromise assessment",
    "reasoning": "explanation of your decision"
}}
"""

    def _get_moderator_system_prompt(self) -> str:
        """Get system prompt for dispute moderation"""
        return """
You are an expert moderator facilitating discussion between AI checkers.
Your role is to find common ground and build consensus while maintaining quality standards.
Be fair, balanced, and focused on achieving the best possible outcome for the content.
"""

    def _parse_dispute_resolution(self, response: str) -> Dict[str, Any]:
        """Parse dispute resolution response"""
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
                "resolved": False,
                "resolution": "Unable to parse resolution",
                "compromise_position": "",
                "reasoning": "Parsing failed",
            }

    async def _generate_consensus_reasoning(
        self,
        consensus_level: ConsensusLevel,
        assessments: List[CheckerAssessment],
        resolved_disputes: List[CheckerDispute],
        unresolved_disputes: List[CheckerDispute],
    ) -> str:
        """Generate reasoning for consensus result"""
        reasoning_parts = []

        reasoning_parts.append(f"Consensus Level: {consensus_level.value}")

        if consensus_level == ConsensusLevel.UNANIMOUS:
            reasoning_parts.append("All checkers agreed on the assessment.")
        elif consensus_level == ConsensusLevel.MAJORITY:
            reasoning_parts.append("Majority of checkers agreed on key points.")
        elif consensus_level == ConsensusLevel.PLURALITY:
            reasoning_parts.append("Largest group of checkers formed consensus.")
        else:
            reasoning_parts.append("No clear consensus could be reached.")

        if resolved_disputes:
            reasoning_parts.append(
                f"Resolved {len(resolved_disputes)} disputes through discussion."
            )

        if unresolved_disputes:
            reasoning_parts.append(
                f"{len(unresolved_disputes)} disputes remain unresolved."
            )

        # Add assessment summary
        scores = [a.score for a in assessments]
        reasoning_parts.append(f"Score range: {min(scores):.1f} - {max(scores):.1f}")

        return " ".join(reasoning_parts)

    def _fallback_consensus(
        self, content: str, checkers: List[CheckerAgent], context: Dict[str, Any]
    ) -> ConsensusResult:
        """Fallback consensus when collaborative review fails"""
        # Create minimal assessments
        fallback_assessments = []
        for checker in checkers:
            assessment = CheckerAssessment(
                checker_id=checker.checker_id,
                focus_area=checker.focus_area,
                score=70.0,  # Default score
                issues_found=[],
                improvements=[],
                improved_content=content,
                recommendations=["Collaborative review failed, using original content"],
                confidence=0.3,
                reasoning="Fallback assessment due to system failure",
            )
            fallback_assessments.append(assessment)

        return ConsensusResult(
            consensus_level=ConsensusLevel.NO_CONSENSUS,
            final_score=70.0,
            final_content=content,
            resolved_disputes=[],
            unresolved_disputes=[],
            aggregated_issues=[],
            aggregated_improvements=[],
            consensus_reasoning="Collaborative review failed, using fallback assessment",
            participant_agreement={a.checker_id: 0.3 for a in fallback_assessments},
        )
