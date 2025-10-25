"""
ToolSelectionAgent - A specialized agent for analyzing context and selecting appropriate tools using LangChain.

This agent can analyze a given task or context, understand requirements,
and recommend the most suitable tools from the available toolkit.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class ToolCapability(BaseModel):
    """Tool capability definition"""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    category: str = Field(description="Tool category: web, content, execution, visual, etc.")
    inputs: List[str] = Field(description="Required inputs for the tool")
    outputs: List[str] = Field(description="Expected outputs from the tool")
    use_cases: List[str] = Field(description="Common use cases for the tool")
    limitations: List[str] = Field(description="Known limitations of the tool")


class TaskRequirement(BaseModel):
    """Task requirement definition"""
    requirement: str = Field(description="Specific requirement of the task")
    priority: str = Field(description="Priority level: high, medium, low")
    category: str = Field(description="Requirement category: input, processing, output, etc.")


class ToolRecommendation(BaseModel):
    """Tool recommendation"""
    tool_name: str = Field(description="Name of recommended tool")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    reasoning: str = Field(description="Reasoning for the recommendation")
    alternative_tools: List[str] = Field(description="Alternative tools if primary is unavailable")
    integration_notes: str = Field(description="Notes on integration with other tools")


class ToolSelectionState(BaseModel):
    """State for tool selection workflow"""
    task_description: str = Field(description="Description of the task to be performed")
    context: Optional[str] = Field(default=None, description="Additional context about the task")
    available_tools: List[ToolCapability] = Field(default_factory=list, description="List of available tools")
    task_requirements: List[TaskRequirement] = Field(default_factory=list, description="Extracted task requirements")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    recommendations: List[ToolRecommendation] = Field(default_factory=list, description="Tool recommendations")
    workflow_plan: Optional[str] = Field(default=None, description="Suggested workflow plan")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether tool selection was successful")


class ToolSelectionAgent:
    """
    A specialized agent for analyzing context and selecting appropriate tools using LangChain.
    
    This agent can understand task requirements, evaluate available tools,
    and recommend the best tool combinations for specific tasks.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ToolSelectionAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "tool_selection_agent"
        self.description = "Specialized agent for analyzing context and selecting appropriate tools"
        
        # Predefined tool capabilities
        self.tool_capabilities = [
            ToolCapability(
                name="firecrawl",
                description="Web scraping and crawling tool with advanced features",
                category="web",
                inputs=["url", "scrape_options"],
                outputs=["scraped_content", "metadata"],
                use_cases=["web scraping", "content extraction", "site crawling"],
                limitations=["requires internet access", "may be blocked by some sites"]
            ),
            ToolCapability(
                name="playwright",
                description="Browser automation tool for dynamic content",
                category="web",
                inputs=["url", "actions"],
                outputs=["page_content", "screenshots", "interactions"],
                use_cases=["dynamic content scraping", "form filling", "browser testing"],
                limitations=["resource intensive", "slower than static scraping"]
            ),
            ToolCapability(
                name="searxng",
                description="Search engine tool for web queries",
                category="web",
                inputs=["query", "search_options"],
                outputs=["search_results", "metadata"],
                use_cases=["web search", "information gathering", "research"],
                limitations=["dependent on search engines", "may return outdated results"]
            ),
            ToolCapability(
                name="jina_reranker",
                description="Content reranking tool for improved relevance",
                category="content",
                inputs=["content", "query"],
                outputs=["reranked_content", "relevance_scores"],
                use_cases=["content filtering", "search result optimization", "content ranking"],
                limitations=["requires content to rerank", "depends on model quality"]
            ),
            ToolCapability(
                name="ollama_reranker",
                description="Local content reranking tool",
                category="content",
                inputs=["content", "query"],
                outputs=["reranked_content", "relevance_scores"],
                use_cases=["local content filtering", "privacy-focused ranking", "offline processing"],
                limitations=["requires local model", "limited by model capabilities"]
            ),
            ToolCapability(
                name="visual_analyzer",
                description="Image and visual content analysis tool",
                category="visual",
                inputs=["image", "analysis_type"],
                outputs=["analysis_results", "extracted_data"],
                use_cases=["image analysis", "visual data extraction", "content understanding"],
                limitations=["image quality dependent", "may not recognize all objects"]
            ),
            ToolCapability(
                name="dynamic_executor",
                description="Dynamic code execution tool",
                category="execution",
                inputs=["code", "execution_context"],
                outputs=["execution_results", "output", "errors"],
                use_cases=["code testing", "dynamic computation", "script execution"],
                limitations=["security concerns", "resource intensive"]
            ),
            ToolCapability(
                name="summarize",
                description="Text summarization tool",
                category="content",
                inputs=["text", "summary_options"],
                outputs=["summary", "key_points"],
                use_cases=["content summarization", "information extraction", "document analysis"],
                limitations=["may miss nuances", "depends on text quality"]
            ),
            ToolCapability(
                name="chain_of_thought",
                description="Structured reasoning tool",
                category="content",
                inputs=["problem", "context"],
                outputs=["reasoning_steps", "conclusion"],
                use_cases=["problem solving", "analysis", "decision making"],
                limitations=["requires clear problem definition", "may be verbose"]
            ),
        ]
        
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
            
            logger.info("ToolSelectionAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ToolSelectionAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for tool selection"""
        workflow = StateGraph(ToolSelectionState)
        
        # Define nodes
        def analyze_task(state: ToolSelectionState) -> ToolSelectionState:
            """Analyze the task to extract requirements"""
            try:
                # Create task analysis prompt
                prompt = f"""
                Analyze the following task to extract key requirements:
                
                Task Description: "{state.task_description}"
                Additional Context: {state.context or "None provided"}
                
                Please identify:
                1. Primary goal of the task
                2. Required inputs (data, URLs, etc.)
                3. Expected outputs
                4. Processing steps needed
                5. Any constraints or special requirements
                
                For each requirement, assign a priority (high, medium, low) and category:
                - input: Data or resources needed
                - processing: Operations to perform
                - output: Desired results
                - quality: Quality requirements
                - constraints: Limitations or constraints
                
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    analysis = json.loads(response.content)
                    
                    # Extract requirements
                    requirements = []
                    for req_data in analysis.get("requirements", []):
                        requirement = TaskRequirement(
                            requirement=req_data.get("requirement", ""),
                            priority=req_data.get("priority", "medium"),
                            category=req_data.get("category", "processing")
                        )
                        requirements.append(requirement)
                    
                    state.task_requirements = requirements
                    state.messages.append(SystemMessage(content="Task analysis completed"))
                    
                except (json.JSONDecodeError, KeyError):
                    # Fallback requirement extraction
                    fallback_requirements = [
                        TaskRequirement(
                            requirement="Complete the described task",
                            priority="high",
                            category="processing"
                        )
                    ]
                    state.task_requirements = fallback_requirements
                    state.messages.append(SystemMessage(content="Task analysis completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error analyzing task: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def evaluate_tools(state: ToolSelectionState) -> ToolSelectionState:
            """Evaluate available tools against task requirements"""
            try:
                # Set available tools
                state.available_tools = self.tool_capabilities
                
                # Create tool evaluation prompt
                requirements_text = "\n".join([
                    f"- {req.requirement} (Priority: {req.priority}, Category: {req.category})"
                    for req in state.task_requirements
                ])
                
                tools_text = "\n".join([
                    f"- {tool.name}: {tool.description} (Category: {tool.category})"
                    for tool in state.available_tools
                ])
                
                prompt = f"""
                Evaluate the following tools against the task requirements:
                
                Task Requirements:
                {requirements_text}
                
                Available Tools:
                {tools_text}
                
                For each tool, evaluate:
                1. How well it matches the requirements (0.0-1.0)
                2. Which specific requirements it addresses
                3. Potential limitations for this task
                4. How it might work with other tools
                
                Focus on tools that can directly contribute to completing the task.
                Consider tool combinations and workflows.
                
                Respond with JSON format listing tool evaluations.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    evaluations = json.loads(response.content)
                    
                    # Store evaluations for later use
                    state.tool_evaluations = evaluations
                    state.messages.append(SystemMessage(content="Tool evaluation completed"))
                    
                except json.JSONDecodeError:
                    state.tool_evaluations = {}
                    state.messages.append(SystemMessage(content="Tool evaluation completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error evaluating tools: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def recommend_tools(state: ToolSelectionState) -> ToolSelectionState:
            """Recommend the best tools for the task"""
            try:
                # Create tool recommendation prompt
                requirements_text = "\n".join([
                    f"- {req.requirement} (Priority: {req.priority})"
                    for req in state.task_requirements
                ])
                
                # Use evaluations if available, otherwise use tool descriptions
                if hasattr(state, 'tool_evaluations') and state.tool_evaluations:
                    evaluations_text = json.dumps(state.tool_evaluations, indent=2)
                else:
                    evaluations_text = "\n".join([
                        f"- {tool.name}: {tool.description}"
                        for tool in state.available_tools
                    ])
                
                prompt = f"""
                Based on the task requirements and tool evaluations, recommend the best tools:
                
                Task Requirements:
                {requirements_text}
                
                Tool Evaluations:
                {evaluations_text}
                
                Please recommend:
                1. Primary tools (most essential)
                2. Supporting tools (helpful but not essential)
                3. Tool workflow (how tools should work together)
                
                For each recommendation, provide:
                - Tool name
                - Confidence score (0.0-1.0)
                - Reasoning for selection
                - Alternative tools if primary is unavailable
                - Integration notes
                
                Focus on creating an effective workflow that addresses all requirements.
                Consider tool dependencies and execution order.
                
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    recommendations_data = json.loads(response.content)
                    
                    # Extract recommendations
                    recommendations = []
                    for rec_data in recommendations_data.get("recommendations", []):
                        recommendation = ToolRecommendation(
                            tool_name=rec_data.get("tool_name", ""),
                            confidence=rec_data.get("confidence", 0.5),
                            reasoning=rec_data.get("reasoning", ""),
                            alternative_tools=rec_data.get("alternative_tools", []),
                            integration_notes=rec_data.get("integration_notes", "")
                        )
                        recommendations.append(recommendation)
                    
                    state.recommendations = recommendations
                    state.workflow_plan = recommendations_data.get("workflow_plan", "")
                    state.messages.append(SystemMessage(content="Tool recommendations generated"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback recommendation
                    fallback_rec = ToolRecommendation(
                        tool_name="dynamic_executor",
                        confidence=0.5,
                        reasoning="General purpose tool for task execution",
                        alternative_tools=["firecrawl", "searxng"],
                        integration_notes="Can be combined with other tools as needed"
                    )
                    state.recommendations = [fallback_rec]
                    state.workflow_plan = "Use dynamic_executor as primary tool"
                    state.messages.append(SystemMessage(content="Tool recommendations generated (fallback)"))
                
            except Exception as e:
                state.error = f"Error recommending tools: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def finalize_recommendations(state: ToolSelectionState) -> ToolSelectionState:
            """Finalize and validate recommendations"""
            try:
                # Validate recommendations
                if not state.recommendations:
                    state.error = "No tool recommendations generated"
                    state.success = False
                    return state
                
                # Sort by confidence
                state.recommendations.sort(key=lambda x: x.confidence, reverse=True)
                
                # Validate tool names exist in available tools
                available_tool_names = {tool.name for tool in state.available_tools}
                valid_recommendations = []
                
                for rec in state.recommendations:
                    if rec.tool_name in available_tool_names:
                        valid_recommendations.append(rec)
                    else:
                        # Try to find alternative
                        for alt_tool in rec.alternative_tools:
                            if alt_tool in available_tool_names:
                                rec.tool_name = alt_tool
                                valid_recommendations.append(rec)
                                break
                
                state.recommendations = valid_recommendations
                state.success = True
                
                validation_msg = f"Finalized {len(valid_recommendations)} tool recommendations"
                state.messages.append(SystemMessage(content=validation_msg))
                
            except Exception as e:
                state.error = f"Error finalizing recommendations: {str(e)}"
                state.success = False
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("analyze_task", analyze_task)
        workflow.add_node("evaluate_tools", evaluate_tools)
        workflow.add_node("recommend_tools", recommend_tools)
        workflow.add_node("finalize_recommendations", finalize_recommendations)
        
        # Set up workflow
        workflow.set_entry_point("analyze_task")
        workflow.add_edge("analyze_task", "evaluate_tools")
        workflow.add_edge("evaluate_tools", "recommend_tools")
        workflow.add_edge("recommend_tools", "finalize_recommendations")
        workflow.add_edge("finalize_recommendations", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def select_tools(
        self,
        task_description: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Select appropriate tools for a given task.
        
        Args:
            task_description: Description of the task to be performed
            context: Additional context about the task
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing tool recommendations and analysis
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize ToolSelectionAgent workflow")
        
        try:
            # Create initial state
            state = ToolSelectionState(
                task_description=task_description,
                context=context,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "task_requirements": [req.dict() for req in result.task_requirements],
                "recommendations": [rec.dict() for rec in result.recommendations],
                "workflow_plan": result.workflow_plan,
                "available_tools": [tool.dict() for tool in result.available_tools],
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in tool selection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "task_requirements": [],
            }
    
    async def stream_select_tools(
        self,
        task_description: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tool selection process for real-time updates.
        
        Args:
            task_description: Description of the task to be performed
            context: Additional context about the task
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize ToolSelectionAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = ToolSelectionState(
                task_description=task_description,
                context=context,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield task analysis updates
                if "analyze_task" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'task_requirements') and node_state.task_requirements:
                        yield {
                            "type": "task_analyzed",
                            "requirements": [req.dict() for req in node_state.task_requirements],
                            "count": len(node_state.task_requirements),
                        }
                
                # Yield tool evaluation updates
                if "evaluate_tools" in event:
                    node_state = list(event.values())[0]
                    yield {
                        "type": "tools_evaluated",
                        "available_tools": len(self.tool_capabilities),
                    }
                
                # Yield recommendation updates
                if "recommend_tools" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'recommendations') and node_state.recommendations:
                        yield {
                            "type": "tools_recommended",
                            "recommendations": [rec.dict() for rec in node_state.recommendations],
                            "workflow_plan": node_state.workflow_plan,
                            "count": len(node_state.recommendations),
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "selection_complete",
                        "success": final_state.success,
                        "task_requirements": [req.dict() for req in final_state.task_requirements],
                        "recommendations": [rec.dict() for rec in final_state.recommendations],
                        "workflow_plan": final_state.workflow_plan,
                        "available_tools": [tool.dict() for tool in final_state.available_tools],
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming tool selection: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }
    
    def get_tool_by_name(self, name: str) -> Optional[ToolCapability]:
        """Get tool capability by name"""
        for tool in self.tool_capabilities:
            if tool.name == name:
                return tool
        return None
    
    def get_tools_by_category(self, category: str) -> List[ToolCapability]:
        """Get tools by category"""
        return [tool for tool in self.tool_capabilities if tool.category == category]