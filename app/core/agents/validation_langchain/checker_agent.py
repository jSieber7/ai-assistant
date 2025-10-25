"""
LangGraph-based checker agent implementation for multi-writer system
"""

from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import json
import time

# Legacy integration layer removed - direct LLM provider access


class CheckerState(TypedDict):
    """State for the LangGraph checker agent"""
    content: Dict[str, Any]
    criteria: Optional[Dict[str, Any]]
    system_prompt: str
    check_prompt: str
    check_result: Optional[str]
    parsed_result: Optional[Dict[str, Any]]
    final_result: Optional[Dict[str, Any]]
    error: Optional[str]


class CheckerResult(BaseModel):
    """Structured result from checker agent"""
    score: float = Field(description="Overall quality score (0-100)")
    issues: List[Dict[str, Any]] = Field(description="List of issues found")
    improvements: List[Dict[str, Any]] = Field(description="List of suggested improvements")
    improved_content: str = Field(description="Improved version of the content")
    recommendations: List[str] = Field(description="General recommendations")


class LangGraphCheckerAgent:
    """LangGraph-based checker agent for content validation and improvement"""

    def __init__(self, checker_id: str, focus_area: str, model: str = "claude-3.5-sonnet"):
        self.checker_id = checker_id
        self.focus_area = focus_area  # e.g., "factual", "style", "structure", "seo"
        self.model = model
        self.workflow = self._create_workflow()

    @property
    def name(self) -> str:
        return f"checker_{self.checker_id}"

    @property
    def description(self) -> str:
        return f"{self.focus_area} checker agent for content validation"

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the checker agent"""
        workflow = StateGraph(CheckerState)

        # Add nodes
        workflow.add_node("prepare_prompts", self._prepare_prompts)
        workflow.add_node("check_content", self._check_content)
        workflow.add_node("parse_result", self._parse_result)
        workflow.add_node("finalize_result", self._finalize_result)

        # Add edges
        workflow.set_entry_point("prepare_prompts")
        workflow.add_edge("prepare_prompts", "check_content")
        workflow.add_edge("check_content", "parse_result")
        workflow.add_edge("parse_result", "finalize_result")
        workflow.add_edge("finalize_result", END)

        return workflow.compile()

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

    async def _prepare_prompts(self, state: CheckerState) -> CheckerState:
        """Prepare system and check prompts"""
        system_prompt = self._get_system_prompt()
        
        check_prompt = f"""
        Content to Check:
        Writer ID: {state['content'].get("writer_id", "unknown")}
        Specialty: {state['content'].get("specialty", "unknown")}
        
        Content:
        {state['content'].get("content", "")}
        
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

        if state.get("criteria"):
            check_prompt += f"\n\nAdditional Criteria: {state['criteria']}"

        state["system_prompt"] = system_prompt
        state["check_prompt"] = check_prompt
        return state

    async def _check_content(self, state: CheckerState) -> CheckerState:
        """Check content using LLM"""
        try:
            from app.core.llm_providers import get_llm
            llm = await get_llm(self.model)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": state["system_prompt"]},
                {"role": "user", "content": state["check_prompt"]},
            ]
            
            # Get response from LLM
            response = await llm.ainvoke(messages)
            state["check_result"] = response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            state["error"] = str(e)
            
        return state

    async def _parse_result(self, state: CheckerState) -> CheckerState:
        """Parse the structured result from checker"""
        if state.get("error"):
            return state
            
        try:
            result = state["check_result"]
            
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

            parsed = json.loads(json_str)
            state["parsed_result"] = parsed
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            # Fallback if JSON parsing fails
            state["parsed_result"] = {
                "score": 70,  # Default score
                "issues": [],
                "improvements": [],
                "improved_content": state.get("check_result", ""),
                "recommendations": ["Unable to parse structured feedback"],
            }
            
        return state

    async def _finalize_result(self, state: CheckerState) -> CheckerState:
        """Finalize the checker result"""
        if state.get("error"):
            state["final_result"] = {
                "checker_id": self.checker_id,
                "error": state["error"],
                "score": 0.0
            }
            return state
            
        parsed = state.get("parsed_result", {})
        content = state["content"]
        
        state["final_result"] = {
            "checker_id": self.checker_id,
            "focus_area": self.focus_area,
            "original_content": content.get("content", ""),
            "score": parsed.get("score", 0.0),
            "issues_found": parsed.get("issues", []),
            "improvements": parsed.get("improvements", []),
            "improved_content": parsed.get("improved_content", ""),
            "recommendations": parsed.get("recommendations", []),
        }
        
        return state

    async def check_and_improve(
        self, content: Dict[str, Any], criteria: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Check content and provide improvements"""
        initial_state = {
            "content": content,
            "criteria": criteria,
            "system_prompt": "",
            "check_prompt": "",
            "check_result": None,
            "parsed_result": None,
            "final_result": None,
            "error": None,
        }
        
        # Run the workflow
        result = await self.workflow.ainvoke(initial_state)
        
        return result.get("final_result", {
            "checker_id": self.checker_id,
            "error": "Workflow failed to produce result",
            "score": 0.0
        })


class LangGraphMultiCheckerOrchestrator:
    """Orchestrates multiple LangGraph checker agents"""

    def __init__(self):
        self.checkers = {
            "factual_1": LangGraphCheckerAgent("factual_1", "factual", "claude-3.5-sonnet"),
            "style_1": LangGraphCheckerAgent("style_1", "style", "claude-3.5-sonnet"),
            "structure_1": LangGraphCheckerAgent("structure_1", "structure", "gpt-4-turbo"),
            "seo_1": LangGraphCheckerAgent("seo_1", "seo", "gpt-4-turbo"),
        }

    async def comprehensive_check(
        self,
        content: Dict[str, Any],
        checker_ids: List[str] = None,
        min_score_threshold: float = 70.0,
    ) -> Dict[str, Any]:
        """Run comprehensive content validation"""
        import asyncio

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