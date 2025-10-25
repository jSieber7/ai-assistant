"""
LangGraph-based dynamic agent selection system for multi-writer/checker system.

This module implements intelligent agent selection using LangGraph's StateGraph
based on task analysis, historical performance, and current system state.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.core.config import get_llm
from app.core.agents.content_langchain.writer_agent import WriterAgent
from app.core.agents.validation_langchain.checker_agent import LangGraphCheckerAgent

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ContentType(Enum):
    """Content types for classification"""
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    MIXED = "mixed"
    MARKETING = "marketing"
    EDUCATIONAL = "educational"


@dataclass
class TaskAnalysis:
    """Results of task analysis"""
    task_id: str
    prompt: str
    complexity: TaskComplexity
    content_type: ContentType
    required_specialties: List[str]
    quality_requirements: List[str]
    estimated_effort: float
    context_keywords: List[str]
    selection_rationale: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Historical performance data for an agent"""
    agent_id: str
    agent_type: str  # "writer" or "checker"
    specialty: str
    total_tasks: int
    success_rate: float
    average_quality_score: float
    average_response_time: float
    last_used: float
    strengths: List[str]
    weaknesses: List[str]
    preferred_tasks: List[str]
    avoided_tasks: List[str]


@dataclass
class SelectionResult:
    """Result of agent selection"""
    task_analysis: TaskAnalysis
    selected_writers: List[str]
    selected_checkers: List[str]
    selection_confidence: float
    alternative_options: Dict[str, List[str]]
    reasoning: Dict[str, str]
    estimated_performance: Dict[str, float]


class SelectorState(BaseModel):
    """State for the agent selection workflow"""
    task_id: str
    prompt: str
    available_writers: Dict[str, Dict[str, Any]] = {}
    available_checkers: Dict[str, Dict[str, Any]] = {}
    context: Dict[str, Any] = {}
    
    # Analysis results
    task_analysis: Optional[TaskAnalysis] = None
    
    # Selection results
    selected_writers: List[str] = []
    selected_checkers: List[str] = []
    writer_scores: Dict[str, float] = {}
    checker_scores: Dict[str, float] = {}
    selection_confidence: float = 0.0
    alternative_options: Dict[str, List[str]] = {}
    reasoning: Dict[str, str] = {}
    estimated_performance: Dict[str, float] = {}
    
    # Configuration
    analysis_model: str = "claude-3.5-sonnet"
    performance_weight: float = 0.4
    diversity_weight: float = 0.3
    workload_weight: float = 0.3
    
    # Error handling
    error: Optional[str] = None
    
    # Performance cache (in production, use database)
    performance_cache: Dict[str, AgentPerformance] = {}
    
    # Current workload tracking
    current_workload: Dict[str, int] = {}


class LangGraphDynamicSelector:
    """LangGraph-based dynamic agent selector for intelligent team composition"""

    def __init__(
        self,
        analysis_model: str = "claude-3.5-sonnet",
        performance_weight: float = 0.4,
        diversity_weight: float = 0.3,
        workload_weight: float = 0.3,
        memory: Optional[MemorySaver] = None,
    ):
        self.analysis_model = analysis_model
        self.performance_weight = performance_weight
        self.diversity_weight = diversity_weight
        self.workload_weight = workload_weight
        self.memory = memory or MemorySaver()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        # Compile the workflow
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _create_workflow(self) -> StateGraph:
        """Create the agent selection workflow graph"""
        workflow = StateGraph(SelectorState)
        
        # Add nodes for each phase
        workflow.add_node("initialize_selection", self._initialize_selection)
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("score_writers", self._score_writers)
        workflow.add_node("score_checkers", self._score_checkers)
        workflow.add_node("select_optimal_team", self._select_optimal_team)
        workflow.add_node("generate_alternatives", self._generate_alternatives)
        workflow.add_node("estimate_performance", self._estimate_performance)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("initialize_selection")
        
        # Add edges
        workflow.add_edge("initialize_selection", "analyze_task")
        workflow.add_edge("analyze_task", "score_writers")
        workflow.add_edge("score_writers", "score_checkers")
        workflow.add_edge("score_checkers", "select_optimal_team")
        workflow.add_edge("select_optimal_team", "generate_alternatives")
        workflow.add_edge("generate_alternatives", "estimate_performance")
        workflow.add_edge("estimate_performance", END)
        workflow.add_edge("handle_error", END)
        
        return workflow

    async def _initialize_selection(self, state: SelectorState) -> SelectorState:
        """Initialize the selection process"""
        try:
            logger.info(f"Initializing agent selection for task {state.task_id}")
            
            # Set configuration
            state.analysis_model = self.analysis_model
            state.performance_weight = self.performance_weight
            state.diversity_weight = self.diversity_weight
            state.workload_weight = self.workload_weight
            
            # Initialize empty structures
            state.selected_writers = []
            state.selected_checkers = []
            state.writer_scores = {}
            state.checker_scores = {}
            state.alternative_options = {}
            state.reasoning = {}
            state.estimated_performance = {}
            
            logger.info(f"Selection initialized with {len(state.available_writers)} writers and {len(state.available_checkers)} checkers")
            return state
            
        except Exception as e:
            logger.error(f"Failed to initialize selection: {str(e)}")
            state.error = str(e)
            return state

    async def _analyze_task(self, state: SelectorState) -> SelectorState:
        """Analyze the task to understand requirements"""
        try:
            logger.info(f"Analyzing task {state.task_id}")
            
            # Create analysis prompt
            analysis_prompt = self._create_task_analysis_prompt(state.prompt, state.context)
            
            # Get LLM directly
            llm = await get_llm(state.analysis_model)
            
            # Generate analysis
            messages = [
                {
                    "role": "system",
                    "content": self._get_task_analysis_system_prompt(),
                },
                {"role": "user", "content": analysis_prompt},
            ]
            
            response = await llm.ainvoke(messages)
            
            # Parse the analysis
            analysis_data = self._parse_task_analysis(response.content)
            
            # Create task analysis object
            state.task_analysis = TaskAnalysis(
                task_id=state.task_id,
                prompt=state.prompt,
                complexity=TaskComplexity(analysis_data.get("complexity", "moderate")),
                content_type=ContentType(analysis_data.get("content_type", "mixed")),
                required_specialties=analysis_data.get("required_specialties", []),
                quality_requirements=analysis_data.get("quality_requirements", []),
                estimated_effort=float(analysis_data.get("estimated_effort", 0.5)),
                context_keywords=analysis_data.get("context_keywords", []),
                selection_rationale=analysis_data.get("rationale", ""),
                confidence=float(analysis_data.get("confidence", 0.7)),
                metadata=analysis_data.get("metadata", {}),
            )
            
            logger.info(f"Task analysis completed: {state.task_analysis.complexity.value} complexity, {state.task_analysis.content_type.value} content type")
            return state
            
        except Exception as e:
            logger.error(f"Task analysis failed: {str(e)}")
            state.error = str(e)
            
            # Fallback analysis
            state.task_analysis = TaskAnalysis(
                task_id=state.task_id,
                prompt=state.prompt,
                complexity=TaskComplexity.MODERATE,
                content_type=ContentType.MIXED,
                required_specialties=["technical", "creative"],
                quality_requirements=["clarity", "accuracy"],
                estimated_effort=0.5,
                context_keywords=[],
                selection_rationale="Analysis failed, using defaults",
                confidence=0.5,
            )
            
            return state

    async def _score_writers(self, state: SelectorState) -> SelectorState:
        """Score all available writers for the task"""
        try:
            logger.info(f"Scoring writers for task {state.task_id}")
            
            state.writer_scores = {}
            
            for writer_id, writer_data in state.available_writers.items():
                score = await self._score_writer_for_task(writer_id, writer_data, state.task_analysis)
                state.writer_scores[writer_id] = score
            
            logger.info(f"Scored {len(state.writer_scores)} writers")
            return state
            
        except Exception as e:
            logger.error(f"Failed to score writers: {str(e)}")
            state.error = str(e)
            return state

    async def _score_checkers(self, state: SelectorState) -> SelectorState:
        """Score all available checkers for the task"""
        try:
            logger.info(f"Scoring checkers for task {state.task_id}")
            
            state.checker_scores = {}
            
            for checker_id, checker_data in state.available_checkers.items():
                score = await self._score_checker_for_task(checker_id, checker_data, state.task_analysis)
                state.checker_scores[checker_id] = score
            
            logger.info(f"Scored {len(state.checker_scores)} checkers")
            return state
            
        except Exception as e:
            logger.error(f"Failed to score checkers: {str(e)}")
            state.error = str(e)
            return state

    async def _select_optimal_team(self, state: SelectorState) -> SelectorState:
        """Select the optimal team of writers and checkers"""
        try:
            logger.info(f"Selecting optimal team for task {state.task_id}")
            
            # Sort writers by score
            sorted_writers = sorted(state.writer_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top writers based on task complexity
            num_writers = self._determine_optimal_writer_count(state.task_analysis.complexity)
            state.selected_writers = [writer_id for writer_id, _ in sorted_writers[:num_writers]]
            
            # Generate writer reasoning
            for writer_id in state.selected_writers:
                state.reasoning[f"writer_{writer_id}"] = self._generate_writer_reasoning(
                    writer_id, state.writer_scores[writer_id], state.task_analysis
                )
            
            # Sort checkers by score
            sorted_checkers = sorted(state.checker_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top checkers based on quality requirements
            num_checkers = self._determine_optimal_checker_count(state.task_analysis.quality_requirements)
            state.selected_checkers = [checker_id for checker_id, _ in sorted_checkers[:num_checkers]]
            
            # Generate checker reasoning
            for checker_id in state.selected_checkers:
                state.reasoning[f"checker_{checker_id}"] = self._generate_checker_reasoning(
                    checker_id, state.checker_scores[checker_id], state.task_analysis
                )
            
            # Calculate selection confidence
            state.selection_confidence = self._calculate_selection_confidence(
                state.task_analysis, state.selected_writers, state.selected_checkers
            )
            
            logger.info(f"Selected {len(state.selected_writers)} writers and {len(state.selected_checkers)} checkers")
            return state
            
        except Exception as e:
            logger.error(f"Failed to select optimal team: {str(e)}")
            state.error = str(e)
            return state

    async def _generate_alternatives(self, state: SelectorState) -> SelectorState:
        """Generate alternative agent selections"""
        try:
            logger.info(f"Generating alternatives for task {state.task_id}")
            
            # Alternative writers
            unused_writers = [
                w for w in state.available_writers.keys() if w not in state.selected_writers
            ]
            if unused_writers:
                state.alternative_options["alternative_writers"] = unused_writers[:2]
            
            # Alternative checkers
            unused_checkers = [
                c for c in state.available_checkers.keys() if c not in state.selected_checkers
            ]
            if unused_checkers:
                state.alternative_options["alternative_checkers"] = unused_checkers[:2]
            
            logger.info(f"Generated {len(state.alternative_options)} alternative options")
            return state
            
        except Exception as e:
            logger.error(f"Failed to generate alternatives: {str(e)}")
            state.error = str(e)
            return state

    async def _estimate_performance(self, state: SelectorState) -> SelectorState:
        """Estimate team performance"""
        try:
            logger.info(f"Estimating team performance for task {state.task_id}")
            
            # Writer performance
            writer_scores = []
            for writer_id in state.selected_writers:
                perf = self._get_agent_performance(writer_id, "writer", "")
                if perf:
                    writer_scores.append(perf.average_quality_score)
            
            if writer_scores:
                state.estimated_performance["estimated_writer_quality"] = sum(writer_scores) / len(writer_scores)
            
            # Checker performance
            checker_scores = []
            for checker_id in state.selected_checkers:
                perf = self._get_agent_performance(checker_id, "checker", "")
                if perf:
                    checker_scores.append(perf.average_quality_score)
            
            if checker_scores:
                state.estimated_performance["estimated_checker_quality"] = sum(checker_scores) / len(checker_scores)
            
            # Overall team performance
            if writer_scores and checker_scores:
                state.estimated_performance["estimated_team_quality"] = (
                    sum(writer_scores) / len(writer_scores) * 0.6
                    + sum(checker_scores) / len(checker_scores) * 0.4
                )
            
            logger.info(f"Performance estimation completed: {state.estimated_performance}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to estimate performance: {str(e)}")
            state.error = str(e)
            return state

    async def _handle_error(self, state: SelectorState) -> SelectorState:
        """Handle errors in the selection workflow"""
        logger.error(f"Handling error in selection for task {state.task_id}: {state.error}")
        return state

    async def _score_writer_for_task(
        self, writer_id: str, writer_data: Dict[str, Any], task_analysis: TaskAnalysis
    ) -> float:
        """Score a writer for a specific task"""
        score = 0.0
        specialty = writer_data.get("specialty", "analytical")
        
        # Base specialty match
        if specialty in task_analysis.required_specialties:
            score += 0.3
        elif specialty == task_analysis.content_type.value:
            score += 0.25
        else:
            score += 0.1
        
        # Performance weighting
        performance = self._get_agent_performance(writer_id, "writer", specialty)
        if performance:
            score += performance.success_rate * self.performance_weight
            score += performance.average_quality_score * 0.2
        
        # Workload balancing
        current_load = self.current_workload.get(writer_id, 0)
        workload_score = max(0, 1.0 - (current_load * 0.1))
        score += workload_score * self.workload_weight
        
        # Content type preference
        if task_analysis.content_type == ContentType.TECHNICAL and specialty == "technical":
            score += 0.2
        elif task_analysis.content_type == ContentType.CREATIVE and specialty == "creative":
            score += 0.2
        elif task_analysis.content_type == ContentType.ANALYTICAL and specialty == "analytical":
            score += 0.2
        
        # Complexity matching
        if task_analysis.complexity == TaskComplexity.EXPERT and specialty in ["technical", "analytical"]:
            score += 0.1
        elif task_analysis.complexity == TaskComplexity.SIMPLE and specialty == "creative":
            score += 0.1
        
        return min(1.0, score)

    async def _score_checker_for_task(
        self, checker_id: str, checker_data: Dict[str, Any], task_analysis: TaskAnalysis
    ) -> float:
        """Score a checker for a specific task"""
        score = 0.0
        focus_area = checker_data.get("focus_area", "factual")
        
        # Quality requirement matching
        if focus_area in task_analysis.quality_requirements:
            score += 0.4
        else:
            score += 0.2
        
        # Content type matching
        if task_analysis.content_type == ContentType.TECHNICAL and focus_area == "factual":
            score += 0.2
        elif task_analysis.content_type == ContentType.CREATIVE and focus_area == "style":
            score += 0.2
        elif task_analysis.content_type == ContentType.MARKETING and focus_area == "seo":
            score += 0.2
        
        # Performance weighting
        performance = self._get_agent_performance(checker_id, "checker", focus_area)
        if performance:
            score += performance.success_rate * self.performance_weight
        
        # Complexity matching
        if task_analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            if focus_area in ["factual", "structure"]:
                score += 0.1
        
        return min(1.0, score)

    def _get_agent_performance(
        self, agent_id: str, agent_type: str, specialty: str
    ) -> Optional[AgentPerformance]:
        """Get performance data for an agent"""
        # In production, this would query a database
        # For now, return mock data or cached data
        if agent_id in self.performance_cache:
            return self.performance_cache[agent_id]
        
        # Create mock performance data
        mock_performance = AgentPerformance(
            agent_id=agent_id,
            agent_type=agent_type,
            specialty=specialty,
            total_tasks=10,
            success_rate=0.8,
            average_quality_score=0.75,
            average_response_time=30.0,
            last_used=time.time() - 3600,
            strengths=[f"Strong {specialty} skills"],
            weaknesses=["Occasionally slow"],
            preferred_tasks=[specialty],
            avoided_tasks=[],
        )
        
        self.performance_cache[agent_id] = mock_performance
        return mock_performance

    def _determine_optimal_writer_count(self, complexity: TaskComplexity) -> int:
        """Determine optimal number of writers based on complexity"""
        counts = {
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 3,
            TaskComplexity.COMPLEX: 4,
            TaskComplexity.EXPERT: 4,
        }
        return counts.get(complexity, 3)

    def _determine_optimal_checker_count(self, quality_requirements: List[str]) -> int:
        """Determine optimal number of checkers based on quality requirements"""
        base_count = 2
        if len(quality_requirements) > 3:
            base_count = 3
        if "comprehensive" in quality_requirements:
            base_count = 4
        return base_count

    def _calculate_selection_confidence(
        self,
        task_analysis: TaskAnalysis,
        selected_writers: List[str],
        selected_checkers: List[str],
    ) -> float:
        """Calculate confidence in the selection"""
        confidence = task_analysis.confidence
        
        # Adjust based on team size
        if len(selected_writers) >= 3 and len(selected_checkers) >= 2:
            confidence += 0.1
        
        # Adjust based on coverage
        if set(task_analysis.required_specialties).issubset(set(selected_writers)):
            confidence += 0.1
        
        return min(1.0, confidence)

    def _create_task_analysis_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Create prompt for task analysis"""
        context_text = json.dumps(context, indent=2) if context else "No additional context"
        
        return f"""
Analyze this writing task to determine optimal agent selection:

TASK PROMPT:
{prompt}

CONTEXT:
{context_text}

Please analyze and provide:
1. Complexity level (simple, moderate, complex, expert)
2. Content type (technical, creative, analytical, mixed, marketing, educational)
3. Required writer specialties
4. Quality requirements (factual, style, structure, seo)
5. Estimated effort (0.0-1.0)
6. Key context keywords
7. Brief rationale for agent selection
8. Confidence in analysis (0.0-1.0)

Format as JSON:
{{
    "complexity": "moderate",
    "content_type": "mixed",
    "required_specialties": ["technical", "creative"],
    "quality_requirements": ["factual", "style"],
    "estimated_effort": 0.6,
    "context_keywords": ["keyword1", "keyword2"],
    "rationale": "Analysis reasoning",
    "confidence": 0.8,
    "metadata": {{}}
}}
"""

    def _get_task_analysis_system_prompt(self) -> str:
        """Get system prompt for task analysis"""
        return """
You are an expert at analyzing writing tasks and determining optimal agent requirements.
Consider the complexity, content type, and quality needs to provide accurate analysis.
"""

    def _parse_task_analysis(self, response: str) -> Dict[str, Any]:
        """Parse task analysis response"""
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
            # Fallback parsing
            return {
                "complexity": "moderate",
                "content_type": "mixed",
                "required_specialties": ["technical", "creative"],
                "quality_requirements": ["factual", "style"],
                "estimated_effort": 0.5,
                "context_keywords": [],
                "rationale": "Parsing failed, using defaults",
                "confidence": 0.5,
                "metadata": {},
            }

    def _generate_writer_reasoning(
        self, writer_id: str, score: float, task_analysis: TaskAnalysis
    ) -> str:
        """Generate reasoning for writer selection"""
        reasons = []
        
        if score > 0.7:
            reasons.append("High specialty match")
        if score > 0.6:
            reasons.append("Strong performance history")
        if score > 0.5:
            reasons.append("Good availability")
        
        return f"Selected {writer_id} (score: {score:.2f}): " + ", ".join(reasons)

    def _generate_checker_reasoning(
        self, checker_id: str, score: float, task_analysis: TaskAnalysis
    ) -> str:
        """Generate reasoning for checker selection"""
        reasons = []
        
        if score > 0.7:
            reasons.append("Excellent quality requirement match")
        if score > 0.6:
            reasons.append("Proven accuracy")
        if score > 0.5:
            reasons.append("Reliable performance")
        
        return f"Selected {checker_id} (score: {score:.2f}): " + ", ".join(reasons)

    async def select_optimal_team(
        self,
        prompt: str,
        available_writers: Dict[str, WriterAgent],
        available_checkers: Dict[str, LangGraphCheckerAgent],
        context: Dict[str, Any] = None,
        thread_id: Optional[str] = None,
    ) -> SelectionResult:
        """
        Select optimal team of writers and checkers for a task.

        Args:
            prompt: The task prompt
            available_writers: Available writer agents
            available_checkers: Available checker agents
            context: Additional context
            thread_id: Optional thread ID for conversation persistence

        Returns:
            SelectionResult with selected agents and reasoning
        """
        task_id = thread_id or f"task_{int(time.time())}"
        logger.info(f"Starting dynamic agent selection for task {task_id}")
        
        # Convert agents to dictionaries for state
        writers_dict = {
            writer_id: {
                "specialty": writer.specialty,
                "writer_id": writer.writer_id,
            }
            for writer_id, writer in available_writers.items()
        }
        
        checkers_dict = {
            checker_id: {
                "focus_area": checker.focus_area,
                "checker_id": checker.checker_id,
            }
            for checker_id, checker in available_checkers.items()
        }
        
        # Initialize state
        initial_state = SelectorState(
            task_id=task_id,
            prompt=prompt,
            available_writers=writers_dict,
            available_checkers=checkers_dict,
            context=context or {},
        )
        
        # Configure the run
        config = {"configurable": {"thread_id": task_id}}
        
        try:
            # Run the workflow
            result = await self.app.ainvoke(initial_state, config=config)
            
            # Create selection result
            selection_result = SelectionResult(
                task_analysis=result.task_analysis,
                selected_writers=result.selected_writers,
                selected_checkers=result.selected_checkers,
                selection_confidence=result.selection_confidence,
                alternative_options=result.alternative_options,
                reasoning=result.reasoning,
                estimated_performance=result.estimated_performance,
            )
            
            logger.info(f"Selected {len(result.selected_writers)} writers and {len(result.selected_checkers)} checkers")
            return selection_result
            
        except Exception as e:
            logger.error(f"Agent selection failed: {str(e)}")
            # Fallback to default selection
            return self._fallback_selection(task_id, prompt, available_writers, available_checkers)

    def _fallback_selection(
        self,
        task_id: str,
        prompt: str,
        available_writers: Dict[str, WriterAgent],
        available_checkers: Dict[str, LangGraphCheckerAgent],
    ) -> SelectionResult:
        """Fallback selection when analysis fails"""
        # Default task analysis
        task_analysis = TaskAnalysis(
            task_id=task_id,
            prompt=prompt,
            complexity=TaskComplexity.MODERATE,
            content_type=ContentType.MIXED,
            required_specialties=["technical", "creative"],
            quality_requirements=["clarity", "accuracy"],
            estimated_effort=0.5,
            context_keywords=[],
            selection_rationale="Fallback selection due to analysis failure",
            confidence=0.3,
        )
        
        # Default selections
        selected_writers = list(available_writers.keys())[:3]
        selected_checkers = list(available_checkers.keys())[:2]
        
        return SelectionResult(
            task_analysis=task_analysis,
            selected_writers=selected_writers,
            selected_checkers=selected_checkers,
            selection_confidence=0.3,
            alternative_options={},
            reasoning={"fallback": "Used default selection due to analysis failure"},
            estimated_performance={"estimated_team_quality": 0.6},
        )

    def update_workload(self, agent_id: str, delta: int):
        """Update workload tracking for an agent"""
        current = self.current_workload.get(agent_id, 0)
        self.current_workload[agent_id] = max(0, current + delta)

    def update_performance(
        self,
        agent_id: str,
        agent_type: str,
        specialty: str,
        success: bool,
        quality_score: float,
        response_time: float,
    ):
        """Update performance data for an agent"""
        if agent_id not in self.performance_cache:
            self.performance_cache[agent_id] = AgentPerformance(
                agent_id=agent_id,
                agent_type=agent_type,
                specialty=specialty,
                total_tasks=0,
                success_rate=0.0,
                average_quality_score=0.0,
                average_response_time=0.0,
                last_used=time.time(),
                strengths=[],
                weaknesses=[],
                preferred_tasks=[],
                avoided_tasks=[],
            )
        
        perf = self.performance_cache[agent_id]
        perf.total_tasks += 1
        
        # Update success rate
        if success:
            perf.success_rate = (
                perf.success_rate * (perf.total_tasks - 1) + 1.0
            ) / perf.total_tasks
        else:
            perf.success_rate = (
                perf.success_rate * (perf.total_tasks - 1)
            ) / perf.total_tasks
        
        # Update quality score
        perf.average_quality_score = (
            perf.average_quality_score * (perf.total_tasks - 1) + quality_score
        ) / perf.total_tasks
        
        # Update response time
        perf.average_response_time = (
            perf.average_response_time * (perf.total_tasks - 1) + response_time
        ) / perf.total_tasks
        
        perf.last_used = time.time()

    async def get_selection_state(self, thread_id: str) -> Optional[SelectorState]:
        """Get the current state of a selection process"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = await self.app.aget_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Failed to get selection state for {thread_id}: {str(e)}")
            return None

    def get_selection_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get the history of a selection process"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            history = []
            
            for state in self.app.get_state_history(config):
                history.append({
                    "step": state.config.get("step", 0),
                    "selected_writers": state.values.get("selected_writers", []),
                    "selected_checkers": state.values.get("selected_checkers", []),
                    "timestamp": state.metadata.get("timestamp"),
                })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get selection history for {thread_id}: {str(e)}")
            return []