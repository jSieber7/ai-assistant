"""
SummarizeAgent - A specialized agent for text summarization using LangChain and LangGraph.

This agent can summarize various types of content including articles, documents,
conversations, and other text sources with different summarization strategies.
"""

import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import asyncio

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class SummarizeState(BaseModel):
    """State for the summarization workflow"""
    content: str = Field(description="Content to summarize")
    summary_type: str = Field(default="general", description="Type of summarization: general, bullet_points, key_points, abstract")
    max_length: Optional[int] = Field(default=None, description="Maximum length of summary")
    target_audience: Optional[str] = Field(default=None, description="Target audience for summary")
    language: str = Field(default="english", description="Language of summary")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    current_summary: Optional[str] = Field(default=None, description="Current generated summary")
    iteration_count: int = Field(default=0, description="Number of refinement iterations")
    max_iterations: int = Field(default=3, description="Maximum refinement iterations")


class SummarizeAgent:
    """
    A specialized agent for text summarization using LangChain and LangGraph.
    
    This agent provides various summarization strategies and can refine summaries
    based on user requirements and feedback.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SummarizeAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for the agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "summarize_agent"
        self.description = "Specialized agent for text summarization with various strategies"
        
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
            
            # Create the LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("SummarizeAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SummarizeAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for summarization"""
        workflow = StateGraph(SummarizeState)
        
        # Define nodes
        def analyze_content(state: SummarizeState) -> SummarizeState:
            """Analyze the content and determine summarization strategy"""
            content_length = len(state.content.split())
            
            # Determine strategy based on content length and type
            if content_length > 2000:
                strategy = "multi_pass"
            elif content_length > 500:
                strategy = "extractive"
            else:
                strategy = "direct"
            
            # Update state with strategy information
            state.messages.append(SystemMessage(content=f"Using {strategy} summarization strategy"))
            return state
        
        def generate_summary(state: SummarizeState) -> SummarizeState:
            """Generate initial summary"""
            # Create prompt based on summary type and requirements
            prompt = self._create_summary_prompt(state)
            
            # Generate summary
            messages = [SystemMessage(content=prompt), HumanMessage(content=state.content)]
            
            try:
                response = self.llm.invoke(messages)
                state.current_summary = response.content
                state.messages.append(AIMessage(content=response.content))
                state.iteration_count += 1
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                state.messages.append(SystemMessage(content=f"Error: {str(e)}"))
            
            return state
        
        def refine_summary(state: SummarizeState) -> SummarizeState:
            """Refine the summary if needed"""
            if state.iteration_count >= state.max_iterations:
                return state
            
            # Check if refinement is needed
            if not self._needs_refinement(state):
                return state
            
            # Create refinement prompt
            refinement_prompt = self._create_refinement_prompt(state)
            
            try:
                messages = [
                    SystemMessage(content=refinement_prompt),
                    HumanMessage(content=f"Original content: {state.content}\n\nCurrent summary: {state.current_summary}")
                ]
                
                response = self.llm.invoke(messages)
                state.current_summary = response.content
                state.messages.append(AIMessage(content=response.content))
                state.iteration_count += 1
            except Exception as e:
                logger.error(f"Error refining summary: {str(e)}")
                state.messages.append(SystemMessage(content=f"Error: {str(e)}"))
            
            return state
        
        def finalize_summary(state: SummarizeState) -> SummarizeState:
            """Finalize the summary"""
            # Add final message
            state.messages.append(SystemMessage(content="Summary generation completed"))
            return state
        
        # Add nodes to workflow
        workflow.add_node("analyze_content", analyze_content)
        workflow.add_node("generate_summary", generate_summary)
        workflow.add_node("refine_summary", refine_summary)
        workflow.add_node("finalize_summary", finalize_summary)
        
        # Define edges
        workflow.set_entry_point("analyze_content")
        workflow.add_edge("analyze_content", "generate_summary")
        workflow.add_edge("generate_summary", "refine_summary")
        workflow.add_edge("refine_summary", "finalize_summary")
        workflow.add_edge("finalize_summary", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _create_summary_prompt(self, state: SummarizeState) -> str:
        """Create a prompt for summarization based on state"""
        base_prompt = f"""
        You are a professional summarizer. Please summarize the following content in {state.language}.
        
        Summary type: {state.summary_type}
        Target audience: {state.target_audience or 'general audience'}
        """
        
        if state.max_length:
            base_prompt += f"\nMaximum length: {state.max_length} words"
        
        # Add specific instructions based on summary type
        if state.summary_type == "bullet_points":
            base_prompt += "\n\nFormat the summary as bullet points."
        elif state.summary_type == "key_points":
            base_prompt += "\n\nExtract and present the key points."
        elif state.summary_type == "abstract":
            base_prompt += "\n\nCreate an academic-style abstract."
        else:
            base_prompt += "\n\nProvide a comprehensive yet concise summary."
        
        return base_prompt
    
    def _create_refinement_prompt(self, state: SummarizeState) -> str:
        """Create a prompt for refining the summary"""
        return f"""
        Please refine the following summary to improve its quality, clarity, and completeness.
        
        Requirements:
        - Summary type: {state.summary_type}
        - Target audience: {state.target_audience or 'general audience'}
        - Language: {state.language}
        {"- Maximum length: " + str(state.max_length) + " words" if state.max_length else ""}
        
        Ensure the summary is accurate, coherent, and meets all requirements.
        """
    
    def _needs_refinement(self, state: SummarizeState) -> bool:
        """Determine if the summary needs refinement"""
        if not state.current_summary:
            return True
        
        # Check length constraints
        if state.max_length and len(state.current_summary.split()) > state.max_length:
            return True
        
        # Check for common issues that need refinement
        issues = [
            "I apologize",
            "I cannot",
            "As an AI",
            "This content appears to be",
        ]
        
        for issue in issues:
            if issue.lower() in state.current_summary.lower():
                return True
        
        return False
    
    async def summarize(
        self,
        content: str,
        summary_type: str = "general",
        max_length: Optional[int] = None,
        target_audience: Optional[str] = None,
        language: str = "english",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Summarize the given content.
        
        Args:
            content: Content to summarize
            summary_type: Type of summarization (general, bullet_points, key_points, abstract)
            max_length: Maximum length of summary in words
            target_audience: Target audience for the summary
            language: Language of the summary
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing the summary and metadata
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize SummarizeAgent workflow")
        
        try:
            # Create initial state
            state = SummarizeState(
                content=content,
                summary_type=summary_type,
                max_length=max_length,
                target_audience=target_audience,
                language=language,
                max_iterations=self.config.get("max_iterations", 3),
            )
            
            # Run the workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "summary": result.current_summary,
                "summary_type": summary_type,
                "content_length": len(content.split()),
                "summary_length": len(result.current_summary.split()) if result.current_summary else 0,
                "iterations": result.iteration_count,
                "messages": [msg.content for msg in result.messages],
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return {
                "summary": None,
                "error": str(e),
                "success": False,
            }
    
    async def stream_summarize(
        self,
        content: str,
        summary_type: str = "general",
        max_length: Optional[int] = None,
        target_audience: Optional[str] = None,
        language: str = "english",
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the summarization process.
        
        Args:
            content: Content to summarize
            summary_type: Type of summarization
            max_length: Maximum length of summary in words
            target_audience: Target audience for the summary
            language: Language of the summary
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and final summary
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize SummarizeAgent workflow")
        
        try:
            # Create initial state
            state = SummarizeState(
                content=content,
                summary_type=summary_type,
                max_length=max_length,
                target_audience=target_audience,
                language=language,
                max_iterations=self.config.get("max_iterations", 3),
            )
            
            # Stream the workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                if "generate_summary" in event or "refine_summary" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'current_summary') and node_state.current_summary:
                        yield {
                            "type": "summary_update",
                            "content": node_state.current_summary,
                            "iteration": node_state.iteration_count,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "final_summary",
                        "summary": final_state.current_summary,
                        "iterations": final_state.iteration_count,
                        "success": True,
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming summarization: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }