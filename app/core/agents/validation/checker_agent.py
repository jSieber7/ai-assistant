"""
Checker agent implementation for multi-writer system
"""

from typing import List, Dict, Any
from app.core.agents.base.base import BaseAgent
from app.core.config import get_llm
import json
import asyncio


class CheckerAgent(BaseAgent):
    """AI checker agent for content validation and improvement"""

    def __init__(
        self, checker_id: str, focus_area: str, model: str = "claude-3.5-sonnet"
    ):
        # Initialize with empty tool registry since checkers don't use tools directly
        from app.core.tools.execution.registry import ToolRegistry

        super().__init__(ToolRegistry(), max_iterations=1)
        self.checker_id = checker_id
        self.focus_area = focus_area  # e.g., "factual", "style", "structure", "seo"
        self.model = model

    @property
    def name(self) -> str:
        return f"checker_{self.checker_id}"

    @property
    def description(self) -> str:
        return f"{self.focus_area} checker agent for content validation"

    async def _process_message_impl(
        self,
        message: str,
        conversation_id: str | None = None,
        context: Dict[str, Any] = None,
    ) -> Any:
        """Process message for checker agent - not used in this implementation"""
        pass

    async def check_and_improve(
        self, content: Dict[str, Any], criteria: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Check content and provide improvements"""

        check_prompt = self._create_check_prompt(content, criteria)

        try:
            client = await get_llm(self.model)

            response = await client.ainvoke(
                [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": check_prompt},
                ]
            )

            result = response.content

            # Parse the structured response
            check_result = self._parse_check_result(result)

            return {
                "checker_id": self.checker_id,
                "focus_area": self.focus_area,
                "original_content": content["content"],
                "score": check_result["score"],
                "issues_found": check_result["issues"],
                "improvements": check_result["improvements"],
                "improved_content": check_result["improved_content"],
                "recommendations": check_result["recommendations"],
            }

        except Exception as e:
            return {"checker_id": self.checker_id, "error": str(e), "score": 0.0}

    def _get_system_prompt(self) -> str:
        """Get system prompt based on checker focus area"""
        if self.focus_area == "factual":
            return """
            You are a fact-checker specializing in content accuracy.
            Verify claims, check for consistency, and flag potential inaccuracies.
            """
        elif self.focus_area == "style":
            return """
            You are a style editor focusing on writing quality, tone, and readability.
            Improve clarity, flow, and engagement while maintaining the author's voice.
            """
        elif self.focus_area == "structure":
            return """
            You are a structural editor analyzing content organization.
            Ensure logical flow, proper formatting, and effective structure.
            """
        elif self.focus_area == "seo":
            return """
            You are an SEO specialist optimizing content for search engines.
            Improve keyword usage, meta elements, and search visibility.
            """
        else:
            return "You are a content quality checker providing comprehensive feedback."

    def _create_check_prompt(
        self, content: Dict[str, Any], criteria: Dict[str, Any]
    ) -> str:
        """Create prompt for content checking"""

        prompt = f"""
        Content to Check:
        Writer ID: {content.get("writer_id", "unknown")}
        Specialty: {content.get("specialty", "unknown")}
        
        Content:
        {content.get("content", "")}
        
        Please analyze this content and provide:
        1. Overall quality score (0-100)
        2. Specific issues found
        3. Suggested improvements
        4. Improved version of the content
        5. General recommendations
        
        Format your response as JSON:
        {{
            "score": <0-100>,
            "issues": [
                {{"type": "issue_type", "description": "description", "severity": "low/medium/high"}}
            ],
            "improvements": [
                {{"type": "improvement_type", "description": "description"}}
            ],
            "improved_content": "improved version here",
            "recommendations": [
                "general recommendations here"
            ]
        }}
        """

        if criteria:
            prompt += f"\n\nAdditional Criteria: {criteria}"

        return prompt

    def _parse_check_result(self, result: str) -> Dict[str, Any]:
        """Parse the structured result from checker"""
        try:
            # Try to extract JSON from the response
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
            else:
                # Look for JSON object in the text
                start = result.find("{")
                end = result.rfind("}") + 1
                json_str = result[start:end]

            return json.loads(json_str)
        except (json.JSONDecodeError, KeyError):
            # Fallback if JSON parsing fails
            return {
                "score": 70,  # Default score
                "issues": [],
                "improvements": [],
                "improved_content": result,
                "recommendations": ["Unable to parse structured feedback"],
            }


class MultiCheckerOrchestrator:
    """Orchestrates multiple checker agents"""

    def __init__(self):
        self.checkers = {
            "factual_1": CheckerAgent("factual_1", "factual", "claude-3.5-sonnet"),
            "style_1": CheckerAgent("style_1", "style", "claude-3.5-sonnet"),
            "structure_1": CheckerAgent("structure_1", "structure", "gpt-4-turbo"),
            "seo_1": CheckerAgent("seo_1", "seo", "gpt-4-turbo"),
        }

    async def comprehensive_check(
        self,
        content: Dict[str, Any],
        checker_ids: List[str] = None,
        min_score_threshold: float = 70.0,
    ) -> Dict[str, Any]:
        """Run comprehensive content validation"""

        if checker_ids is None:
            checker_ids = list(self.checkers.keys())

        # Run all checks
        tasks = []
        for checker_id in checker_ids:
            if checker_id in self.checkers:
                task = self.checkers[checker_id].check_and_improve(content)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_checks = [
            result
            for result in results
            if isinstance(result, dict) and not result.get("error")
        ]

        # Calculate overall score
        scores = [check["score"] for check in successful_checks]
        overall_score = sum(scores) / len(scores) if scores else 0

        # Select best improved version
        best_version = self._select_best_version(successful_checks)

        # Aggregate feedback
        all_issues = []
        all_improvements = []
        all_recommendations = []

        for check in successful_checks:
            all_issues.extend(check.get("issues_found", []))
            all_improvements.extend(check.get("improvements", []))
            all_recommendations.extend(check.get("recommendations", []))

        return {
            "original_content": content,
            "overall_score": overall_score,
            "passes_threshold": overall_score >= min_score_threshold,
            "best_improved_version": best_version,
            "individual_checks": successful_checks,
            "aggregated_feedback": {
                "issues": all_issues,
                "improvements": all_improvements,
                "recommendations": all_recommendations,
            },
            "checker_summary": {
                "total_checkers": len(checker_ids),
                "successful_checks": len(successful_checks),
                "average_score": overall_score,
            },
        }

    def _select_best_version(self, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best improved version from all checks"""
        if not checks:
            return {"content": "", "score": 0}

        # Sort by score and return the highest scoring version
        best_check = max(checks, key=lambda x: x.get("score", 0))

        return {
            "content": best_check.get("improved_content", ""),
            "score": best_check.get("score", 0),
            "checker_id": best_check.get("checker_id", ""),
            "focus_area": best_check.get("focus_area", ""),
        }
