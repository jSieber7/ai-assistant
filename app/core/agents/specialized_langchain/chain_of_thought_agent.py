"""
ChainOfThoughtAgent - A specialized agent for structured thinking using LangChain.

This agent implements chain-of-thought reasoning to break down complex
problems into steps, analyze them systematically, and provide structured
solutions with clear reasoning paths.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class ThoughtStep(BaseModel):
    """Individual step in chain of thought"""
    step_number: int = Field(description="Step number in the reasoning chain")
    description: str = Field(description="Description of the reasoning step")
    analysis: str = Field(description="Analysis performed in this step")
    conclusion: str = Field(description="Conclusion or result from this step")
    confidence: float = Field(description="Confidence level in this step (0.0-1.0)")
    alternatives: List[str] = Field(default_factory=list, description="Alternative approaches considered")


class ChainOfThoughtState(BaseModel):
    """State for chain-of-thought reasoning workflow"""
    problem: str = Field(description="Problem or question to solve")
    context: Optional[str] = Field(default=None, description="Additional context for the problem")
    domain: str = Field(default="general", description="Domain of the problem: general, math, logic, scientific, etc.")
    reasoning_depth: int = Field(default=3, description="Depth of reasoning steps to generate")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for conclusions")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    thought_steps: List[ThoughtStep] = Field(default_factory=list, description="Reasoning steps")
    final_conclusion: Optional[str] = Field(default=None, description="Final conclusion from reasoning")
    overall_confidence: float = Field(default=0.0, description="Overall confidence in final conclusion")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether reasoning was successful")


class ChainOfThoughtAgent:
    """
    A specialized agent for structured chain-of-thought reasoning using LangChain.
    
    This agent breaks down complex problems into systematic steps,
    analyzes each step, and provides structured solutions.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChainOfThoughtAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "chain_of_thought_agent"
        self.description = "Specialized agent for structured chain-of-thought reasoning"
        
        # LangGraph workflow
        self.workflow = None
        self.checkpointer = MemorySaver()
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize components asynchronously"""
        try:
            if not self.llm:
                self.llm = await get_llm()
            
            # Create LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("ChainOfThoughtAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChainOfThoughtAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for chain-of-thought reasoning"""
        workflow = StateGraph(ChainOfThoughtState)
        
        # Define nodes
        def decompose_problem(state: ChainOfThoughtState) -> ChainOfThoughtState:
            """Decompose the problem into smaller components"""
            try:
                # Create decomposition prompt
                prompt = f"""
                Analyze and decompose the following problem into its fundamental components:
                
                Problem: "{state.problem}"
                Context: {state.context or 'None'}
                Domain: {state.domain}
                
                Please identify:
                1. Key components or sub-problems
                2. Relationships between components
                3. Required information or assumptions
                4. Potential solution approaches
                5. Challenges or constraints
                
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    decomposition = json.loads(response.content)
                    state.messages.append(SystemMessage(content="Problem decomposition completed"))
                except json.JSONDecodeError:
                    decomposition = {
                        "components": ["Unable to parse decomposition"],
                        "relationships": [],
                        "information_needed": [],
                        "approaches": [],
                        "challenges": ["Parsing error"],
                        "raw_response": response.content
                    }
                    state.messages.append(SystemMessage(content="Problem decomposition completed (fallback)"))
                
                # Store decomposition for reference
                state.decomposition = decomposition
                return state
                
            except Exception as e:
                state.error = f"Error decomposing problem: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
                return state
        
        def generate_reasoning_steps(state: ChainOfThoughtState) -> ChainOfThoughtState:
            """Generate systematic reasoning steps"""
            try:
                # Create reasoning prompt
                decomposition_text = json.dumps(getattr(state, 'decomposition', {}))
                
                prompt = f"""
                Generate a chain of thought with {state.reasoning_depth} systematic reasoning steps for:
                
                Problem: "{state.problem}"
                Context: {state.context or 'None'}
                Domain: {state.domain}
                Decomposition: {decomposition_text}
                
                For each step, provide:
                1. Clear description of the reasoning step
                2. Analysis or logic applied
                3. Conclusion or result from the step
                4. Confidence level (0.0-1.0)
                5. Alternative approaches considered
                
                Ensure logical flow between steps.
                Respond with JSON array of steps.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    steps_data = json.loads(response.content)
                    thought_steps = []
                    
                    for i, step_data in enumerate(steps_data):
                        thought_step = ThoughtStep(
                            step_number=i + 1,
                            description=step_data.get("description", ""),
                            analysis=step_data.get("analysis", ""),
                            conclusion=step_data.get("conclusion", ""),
                            confidence=step_data.get("confidence", 0.5),
                            alternatives=step_data.get("alternatives", [])
                        )
                        thought_steps.append(thought_step)
                    
                    state.thought_steps = thought_steps
                    state.messages.append(SystemMessage(content=f"Generated {len(thought_steps)} reasoning steps"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback to single step
                    thought_step = ThoughtStep(
                        step_number=1,
                        description="Analyze the problem",
                        analysis="Basic analysis of the given problem",
                        conclusion=response.content,
                        confidence=0.6,
                        alternatives=[]
                    )
                    state.thought_steps = [thought_step]
                    state.messages.append(SystemMessage(content="Generated reasoning steps (fallback)"))
                
            except Exception as e:
                state.error = f"Error generating reasoning steps: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
                return state
        
        def validate_reasoning(state: ChainOfThoughtState) -> ChainOfThoughtState:
            """Validate the reasoning chain for consistency and logic"""
            try:
                if not state.thought_steps:
                    state.error = "No reasoning steps to validate"
                    return state
                
                # Create validation prompt
                steps_text = "\n".join([
                    f"Step {step.step_number}: {step.description}\nAnalysis: {step.analysis}\nConclusion: {step.conclusion}"
                    for step in state.thought_steps
                ])
                
                prompt = f"""
                Validate the following reasoning chain for logical consistency and completeness:
                
                Original Problem: "{state.problem}"
                {steps_text}
                
                Check for:
                1. Logical flow between steps
                2. Contradictions or inconsistencies
                3. Missing steps or gaps in reasoning
                4. Assumptions that need verification
                5. Overall coherence of the argument
                
                Provide validation feedback and suggest improvements if needed.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    validation = json.loads(response.content)
                    state.validation = validation
                    state.messages.append(SystemMessage(content="Reasoning validation completed"))
                except json.JSONDecodeError:
                    state.validation = {
                        "is_consistent": True,
                        "has_gaps": False,
                        "contradictions": [],
                        "missing_assumptions": [],
                        "suggestions": [],
                        "raw_response": response.content
                    }
                    state.messages.append(SystemMessage(content="Reasoning validation completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error validating reasoning: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
                return state
        
        def synthesize_conclusion(state: ChainOfThoughtState) -> ChainOfThoughtState:
            """Synthesize final conclusion from reasoning steps"""
            try:
                # Create synthesis prompt
                steps_summary = "\n".join([
                    f"Step {step.step_number}: {step.conclusion} (confidence: {step.confidence})"
                    for step in state.thought_steps
                ])
                
                validation_text = json.dumps(getattr(state, 'validation', {}))
                
                prompt = f"""
                Synthesize a final conclusion from the following reasoning chain:
                
                Problem: "{state.problem}"
                Context: {state.context or 'None'}
                
                Reasoning Steps:
                {steps_summary}
                
                Validation Feedback:
                {validation_text}
                
                Provide:
                1. Clear, concise final conclusion
                2. Overall confidence level (0.0-1.0)
                3. Key supporting points
                4. Limitations or uncertainties
                5. Recommendations for further investigation
                
                Ensure conclusion directly addresses the original problem.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    synthesis = json.loads(response.content)
                    state.final_conclusion = synthesis.get("conclusion", "")
                    state.overall_confidence = synthesis.get("confidence", 0.5)
                    state.supporting_points = synthesis.get("supporting_points", [])
                    state.limitations = synthesis.get("limitations", [])
                    state.recommendations = synthesis.get("recommendations", [])
                    state.messages.append(SystemMessage(content="Conclusion synthesis completed"))
                    
                except json.JSONDecodeError:
                    state.final_conclusion = response.content
                    state.overall_confidence = 0.5
                    state.supporting_points = []
                    state.limitations = ["Unable to parse synthesis"]
                    state.recommendations = []
                    state.messages.append(SystemMessage(content="Conclusion synthesis completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error synthesizing conclusion: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
                return state
        
        def finalize_reasoning(state: ChainOfThoughtState) -> ChainOfThoughtState:
            """Finalize the reasoning process"""
            state.success = (
                state.final_conclusion is not None and
                state.overall_confidence >= state.confidence_threshold
            )
            
            if state.success:
                state.messages.append(SystemMessage(content=f"Reasoning completed successfully with confidence {state.overall_confidence}"))
            else:
                state.messages.append(SystemMessage(content=f"Reasoning completed with low confidence {state.overall_confidence}"))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("decompose_problem", decompose_problem)
        workflow.add_node("generate_reasoning_steps", generate_reasoning_steps)
        workflow.add_node("validate_reasoning", validate_reasoning)
        workflow.add_node("synthesize_conclusion", synthesize_conclusion)
        workflow.add_node("finalize_reasoning", finalize_reasoning)
        
        # Set up workflow
        workflow.set_entry_point("decompose_problem")
        workflow.add_edge("decompose_problem", "generate_reasoning_steps")
        workflow.add_edge("generate_reasoning_steps", "validate_reasoning")
        workflow.add_edge("validate_reasoning", "synthesize_conclusion")
        workflow.add_edge("synthesize_conclusion", "finalize_reasoning")
        workflow.add_edge("finalize_reasoning", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def reason(
        self,
        problem: str,
        context: Optional[str] = None,
        domain: str = "general",
        reasoning_depth: int = 3,
        confidence_threshold: float = 0.7,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning on a problem.
        
        Args:
            problem: Problem or question to solve
            context: Additional context for the problem
            domain: Domain of the problem (general, math, logic, scientific, etc.)
            reasoning_depth: Depth of reasoning steps to generate
            confidence_threshold: Minimum confidence for conclusions
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing reasoning results and metadata
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize ChainOfThoughtAgent workflow")
        
        try:
            # Create initial state
            state = ChainOfThoughtState(
                problem=problem,
                context=context,
                domain=domain,
                reasoning_depth=reasoning_depth,
                confidence_threshold=confidence_threshold,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "problem": result.problem,
                "final_conclusion": result.final_conclusion,
                "overall_confidence": result.overall_confidence,
                "thought_steps": [
                    {
                        "step_number": step.step_number,
                        "description": step.description,
                        "analysis": step.analysis,
                        "conclusion": step.conclusion,
                        "confidence": step.confidence,
                        "alternatives": step.alternatives,
                    }
                    for step in result.thought_steps
                ],
                "supporting_points": getattr(result, 'supporting_points', []),
                "limitations": getattr(result, 'limitations', []),
                "recommendations": getattr(result, 'recommendations', []),
                "validation": getattr(result, 'validation', {}),
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in chain-of-thought reasoning: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "thought_steps": [],
            }
    
    async def stream_reason(
        self,
        problem: str,
        context: Optional[str] = None,
        domain: str = "general",
        reasoning_depth: int = 3,
        confidence_threshold: float = 0.7,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chain-of-thought reasoning process for real-time updates.
        
        Args:
            problem: Problem or question to solve
            context: Additional context for the problem
            domain: Domain of the problem
            reasoning_depth: Depth of reasoning steps to generate
            confidence_threshold: Minimum confidence for conclusions
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate reasoning steps and final conclusion
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize ChainOfThoughtAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = ChainOfThoughtState(
                problem=problem,
                context=context,
                domain=domain,
                reasoning_depth=reasoning_depth,
                confidence_threshold=confidence_threshold,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield decomposition updates
                if "decompose_problem" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'decomposition'):
                        yield {
                            "type": "decomposition_complete",
                            "decomposition": node_state.decomposition,
                        }
                
                # Yield reasoning step updates
                if "generate_reasoning_steps" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'thought_steps') and node_state.thought_steps:
                        yield {
                            "type": "reasoning_step",
                            "step": node_state.thought_steps[-1].dict() if node_state.thought_steps else None,
                            "total_steps": len(node_state.thought_steps),
                        }
                
                # Yield validation updates
                if "validate_reasoning" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'validation'):
                        yield {
                            "type": "validation_complete",
                            "validation": node_state.validation,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "reasoning_complete",
                        "success": final_state.success,
                        "final_conclusion": final_state.final_conclusion,
                        "overall_confidence": final_state.overall_confidence,
                        "thought_steps": [
                            step.dict() for step in final_state.thought_steps
                        ],
                        "supporting_points": getattr(final_state, 'supporting_points', []),
                        "limitations": getattr(final_state, 'limitations', []),
                        "recommendations": getattr(final_state, 'recommendations', []),
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming chain-of-thought reasoning: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }