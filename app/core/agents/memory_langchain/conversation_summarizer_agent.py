"""
LangGraph-based Conversation Summarizer Agent for AI Assistant.

This module provides a specialized agent for summarizing conversations
with different summarization strategies and levels of detail.
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ...langchain.memory_workflow import memory_workflow, MemoryOperation
from ...langchain.llm_manager import llm_manager

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of conversation summaries"""
    
    BRIEF = "brief"  # 1-2 sentences
    CONCISE = "concise"  # 1 paragraph
    DETAILED = "detailed"  # Multiple paragraphs
    TOPICAL = "topical"  # Organized by topics
    ACTION_ITEMS = "action_items"  # Focus on action items
    DECISIONS = "decisions"  # Focus on decisions made


class SummaryLevel(Enum):
    """Levels of summarization detail"""
    
    HIGH_LEVEL = "high_level"  # General overview
    MID_LEVEL = "mid_level"  # Key points and details
    LOW_LEVEL = "low_level"  # Comprehensive summary


class ConversationSummarizerState(TypedDict):
    """State for conversation summarizer workflow"""
    
    # Conversation information
    conversation_id: str
    agent_name: Optional[str]
    
    # Summarization configuration
    summary_type: SummaryType
    summary_level: SummaryLevel
    include_metadata: bool
    time_range: Optional[Dict[str, Any]]
    
    # Conversation data
    conversation_messages: List[Dict[str, Any]]
    conversation_info: Optional[Dict[str, Any]]
    
    # Summarization operations
    summarization_operations: List[Dict[str, Any]]
    
    # Results
    success: bool
    result: Any
    error: Optional[str]
    
    # Generated summaries
    brief_summary: Optional[str]
    concise_summary: Optional[str]
    detailed_summary: Optional[str]
    topical_summary: Optional[Dict[str, Any]]
    action_items: Optional[List[str]]
    decisions: Optional[List[str]]
    
    # Final summary
    final_summary: Optional[str]
    summary_metadata: Dict[str, Any]
    
    # Internal state
    timestamp: datetime
    step_count: int


class LangGraphConversationSummarizerAgent:
    """
    LangGraph-based conversation summarizer agent.
    
    This agent provides:
    - Multiple summarization strategies
    - Different levels of detail
    - Topic-based summarization
    - Action item extraction
    - Decision tracking
    - Metadata preservation
    """
    
    def __init__(self):
        self._workflow = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the conversation summarizer agent"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Conversation Summarizer Agent...")
        
        # Initialize memory workflow
        await memory_workflow.initialize()
        
        # Create the agent workflow
        self._create_workflow()
        
        self._initialized = True
        logger.info("LangGraph Conversation Summarizer Agent initialized successfully")
        
    def _create_workflow(self):
        """Create the conversation summarizer workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(ConversationSummarizerState)
        
        # Add nodes
        workflow.add_node("analyze_conversation", self._analyze_conversation)
        workflow.add_node("fetch_conversation_data", self._fetch_conversation_data)
        workflow.add_node("prepare_summarization", self._prepare_summarization)
        workflow.add_node("generate_summaries", self._generate_summaries)
        workflow.add_node("extract_key_elements", self._extract_key_elements)
        workflow.add_node("finalize_summary", self._finalize_summary)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("analyze_conversation")
        workflow.add_edge("analyze_conversation", "fetch_conversation_data")
        workflow.add_edge("fetch_conversation_data", "prepare_summarization")
        workflow.add_edge("prepare_summarization", "generate_summaries")
        workflow.add_edge("generate_summaries", "extract_key_elements")
        workflow.add_edge("extract_key_elements", "finalize_summary")
        workflow.add_edge("finalize_summary", "process_results")
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_conversation",
            self._should_continue,
            {
                "continue": "fetch_conversation_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "fetch_conversation_data",
            self._should_prepare,
            {
                "prepare": "prepare_summarization",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "prepare_summarization",
            self._should_generate,
            {
                "generate": "generate_summaries",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_summaries",
            self._should_extract,
            {
                "extract": "extract_key_elements",
                "finalize": "finalize_summary"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_key_elements",
            self._should_finalize,
            {
                "finalize": "finalize_summary",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "finalize_summary",
            self._should_process,
            {
                "process": "process_results",
                "error": "handle_error"
            }
        )
        
        # Compile the workflow
        self._workflow = workflow.compile()
        
    async def _analyze_conversation(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Analyze the conversation and determine summarization approach"""
        try:
            conversation_id = state.get("conversation_id")
            summary_type = state.get("summary_type", SummaryType.CONCISE)
            summary_level = state.get("summary_level", SummaryLevel.MID_LEVEL)
            
            # Initialize state
            state["success"] = False
            state["result"] = None
            state["error"] = None
            state["timestamp"] = datetime.now()
            state["step_count"] = 0
            state["summarization_operations"] = []
            state["conversation_messages"] = []
            state["conversation_info"] = None
            state["brief_summary"] = None
            state["concise_summary"] = None
            state["detailed_summary"] = None
            state["topical_summary"] = None
            state["action_items"] = []
            state["decisions"] = []
            state["final_summary"] = None
            state["summary_metadata"] = {}
            
            # Validate conversation ID
            if not conversation_id:
                state["error"] = "Conversation ID is required for summarization"
                return state
                
            # Set default values
            if not state.get("include_metadata"):
                state["include_metadata"] = True
                
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")
            state["error"] = f"Conversation analysis error: {str(e)}"
            return state
            
    async def _fetch_conversation_data(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Fetch conversation data from memory"""
        try:
            conversation_id = state.get("conversation_id")
            state["step_count"] += 1
            
            # Prepare operations to fetch conversation data
            operations = []
            
            # Get conversation messages
            messages_op = {
                "operation": MemoryOperation.GET_MESSAGES,
                "conversation_id": conversation_id
            }
            operations.append(messages_op)
            
            # Get conversation info
            info_op = {
                "operation": MemoryOperation.GET_CONVERSATION_INFO,
                "conversation_id": conversation_id
            }
            operations.append(info_op)
            
            # Execute operations
            for operation in operations:
                result = await memory_workflow.run(operation)
                
                if result.get("success"):
                    op_type = operation.get("operation")
                    
                    if op_type == MemoryOperation.GET_MESSAGES:
                        state["conversation_messages"] = result.get("result", {}).get("messages", [])
                        
                    elif op_type == MemoryOperation.GET_CONVERSATION_INFO:
                        state["conversation_info"] = result.get("result", {}).get("info")
                        
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Failed to fetch conversation data: {error_msg}")
                    
                    if not state.get("error"):
                        state["error"] = error_msg
                        
            # Check if we have messages to summarize
            if not state.get("conversation_messages"):
                state["error"] = "No messages found for conversation"
                return state
                
            return state
            
        except Exception as e:
            logger.error(f"Error fetching conversation data: {str(e)}")
            state["error"] = f"Data fetching error: {str(e)}"
            return state
            
    async def _prepare_summarization(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Prepare for summarization based on type and level"""
        try:
            summary_type = state.get("summary_type")
            summary_level = state.get("summary_level")
            messages = state.get("conversation_messages", [])
            state["step_count"] += 1
            
            # Filter messages if time range is specified
            time_range = state.get("time_range")
            if time_range:
                filtered_messages = []
                for msg in messages:
                    msg_timestamp = msg.get("timestamp")
                    if msg_timestamp:
                        # Apply time range filtering (simplified)
                        # In practice, this would need proper datetime comparison
                        filtered_messages.append(msg)
                state["conversation_messages"] = filtered_messages
                
            # Determine summarization approach based on type and level
            if summary_type == SummaryType.BRIEF:
                # Brief summary - only need basic summarization
                pass
                
            elif summary_type == SummaryType.CONCISE:
                # Concise summary - standard approach
                pass
                
            elif summary_type == SummaryType.DETAILED:
                # Detailed summary - need comprehensive approach
                pass
                
            elif summary_type == SummaryType.TOPICAL:
                # Topical summary - need topic analysis
                pass
                
            elif summary_type == SummaryType.ACTION_ITEMS:
                # Action items - need extraction focus
                pass
                
            elif summary_type == SummaryType.DECISIONS:
                # Decisions - need decision tracking
                pass
                
            return state
            
        except Exception as e:
            logger.error(f"Error preparing summarization: {str(e)}")
            state["error"] = f"Summarization preparation error: {str(e)}"
            return state
            
    async def _generate_summaries(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Generate summaries based on the configuration"""
        try:
            summary_type = state.get("summary_type")
            summary_level = state.get("summary_level")
            messages = state.get("conversation_messages", [])
            state["step_count"] += 1
            
            # Get LLM for summarization
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            if not llm:
                state["error"] = "No LLM available for summarization"
                return state
                
            # Create conversation text
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in messages
            ])
            
            # Generate summary based on type
            if summary_type == SummaryType.BRIEF:
                brief_prompt = f"""Please provide a brief 1-2 sentence summary of this conversation:

{conversation_text}

Focus only on the main topic or purpose."""
                
                response = await llm.ainvoke(brief_prompt)
                state["brief_summary"] = response.content
                state["final_summary"] = state["brief_summary"]
                
            elif summary_type == SummaryType.CONCISE:
                concise_prompt = f"""Please provide a concise 1-paragraph summary of this conversation:

{conversation_text}

Include the main topics discussed and key points."""
                
                response = await llm.ainvoke(concise_prompt)
                state["concise_summary"] = response.content
                state["final_summary"] = state["concise_summary"]
                
            elif summary_type == SummaryType.DETAILED:
                if summary_level == SummaryLevel.HIGH_LEVEL:
                    detailed_prompt = f"""Please provide a detailed summary of this conversation:

{conversation_text}

Include main topics, key points, and important details in multiple paragraphs."""
                else:
                    detailed_prompt = f"""Please provide a comprehensive detailed summary of this conversation:

{conversation_text}

Include all main topics, key points, important details, and the flow of conversation."""
                
                response = await llm.ainvoke(detailed_prompt)
                state["detailed_summary"] = response.content
                state["final_summary"] = state["detailed_summary"]
                
            elif summary_type == SummaryType.TOPICAL:
                topical_prompt = f"""Please analyze this conversation and provide a topical summary:

{conversation_text}

Organize the summary by topics discussed. For each topic, include:
- Topic name
- Key points discussed
- Any conclusions reached

Format as JSON with topic structure."""
                
                response = await llm.ainvoke(topical_prompt)
                try:
                    import json
                    state["topical_summary"] = json.loads(response.content)
                    state["final_summary"] = response.content
                except json.JSONDecodeError:
                    # Fallback to plain text
                    state["topical_summary"] = {"summary": response.content}
                    state["final_summary"] = response.content
                    
            elif summary_type == SummaryType.ACTION_ITEMS:
                action_prompt = f"""Please analyze this conversation and extract action items:

{conversation_text}

List all action items, tasks, or next steps mentioned. For each item include:
- The action item
- Who is responsible (if mentioned)
- Any deadlines or timelines (if mentioned)

Format as a JSON list of action items."""
                
                response = await llm.ainvoke(action_prompt)
                try:
                    import json
                    state["action_items"] = json.loads(response.content)
                    state["final_summary"] = f"Action Items: {response.content}"
                except json.JSONDecodeError:
                    # Fallback to plain text
                    state["action_items"] = [{"item": response.content}]
                    state["final_summary"] = f"Action Items: {response.content}"
                    
            elif summary_type == SummaryType.DECISIONS:
                decisions_prompt = f"""Please analyze this conversation and extract decisions made:

{conversation_text}

List all decisions, conclusions, or agreements reached. For each decision include:
- The decision made
- The reasoning behind it (if available)
- Who was involved (if mentioned)

Format as a JSON list of decisions."""
                
                response = await llm.ainvoke(decisions_prompt)
                try:
                    import json
                    state["decisions"] = json.loads(response.content)
                    state["final_summary"] = f"Decisions: {response.content}"
                except json.JSONDecodeError:
                    # Fallback to plain text
                    state["decisions"] = [{"decision": response.content}]
                    state["final_summary"] = f"Decisions: {response.content}"
                    
            return state
            
        except Exception as e:
            logger.error(f"Error generating summaries: {str(e)}")
            state["error"] = f"Summary generation error: {str(e)}"
            return state
            
    async def _extract_key_elements(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Extract key elements like action items and decisions"""
        try:
            summary_type = state.get("summary_type")
            messages = state.get("conversation_messages", [])
            state["step_count"] += 1
            
            # Skip if we already extracted for specific types
            if summary_type in [SummaryType.ACTION_ITEMS, SummaryType.DECISIONS]:
                return state
                
            # Get LLM for extraction
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            if not llm:
                return state  # Don't fail for extraction
                
            # Create conversation text
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in messages
            ])
            
            # Extract action items
            action_prompt = f"""Please extract action items from this conversation:

{conversation_text}

List only clear action items, tasks, or next steps. Format as a JSON list."""
            
            try:
                response = await llm.ainvoke(action_prompt)
                import json
                state["action_items"] = json.loads(response.content)
            except (json.JSONDecodeError, Exception):
                # Ignore extraction errors
                pass
                
            # Extract decisions
            decisions_prompt = f"""Please extract decisions from this conversation:

{conversation_text}

List only clear decisions or conclusions reached. Format as a JSON list."""
            
            try:
                response = await llm.ainvoke(decisions_prompt)
                import json
                state["decisions"] = json.loads(response.content)
            except (json.JSONDecodeError, Exception):
                # Ignore extraction errors
                pass
                
            return state
            
        except Exception as e:
            logger.error(f"Error extracting key elements: {str(e)}")
            # Don't fail the workflow for extraction errors
            return state
            
    async def _finalize_summary(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Finalize the summary with metadata"""
        try:
            conversation_id = state.get("conversation_id")
            conversation_info = state.get("conversation_info")
            include_metadata = state.get("include_metadata", True)
            state["step_count"] += 1
            
            # Create summary metadata
            metadata = {
                "conversation_id": conversation_id,
                "summary_type": state.get("summary_type").value,
                "summary_level": state.get("summary_level").value,
                "generated_at": state.get("timestamp").isoformat(),
                "message_count": len(state.get("conversation_messages", []))
            }
            
            if include_metadata and conversation_info:
                metadata.update({
                    "agent_name": conversation_info.get("agent_name"),
                    "title": conversation_info.get("title"),
                    "created_at": conversation_info.get("created_at"),
                    "updated_at": conversation_info.get("updated_at")
                })
                
            # Add extracted elements to metadata
            if state.get("action_items"):
                metadata["action_items_count"] = len(state["action_items"])
                
            if state.get("decisions"):
                metadata["decisions_count"] = len(state["decisions"])
                
            state["summary_metadata"] = metadata
            state["success"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error finalizing summary: {str(e)}")
            state["error"] = f"Summary finalization error: {str(e)}"
            return state
            
    async def _process_results(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Process the final results"""
        try:
            state["step_count"] += 1
            
            # Create comprehensive result
            state["result"] = {
                "conversation_id": state.get("conversation_id"),
                "summary": state.get("final_summary"),
                "summary_type": state.get("summary_type").value,
                "summary_level": state.get("summary_level").value,
                "metadata": state.get("summary_metadata"),
                "extracted_elements": {
                    "action_items": state.get("action_items", []),
                    "decisions": state.get("decisions", [])
                },
                "alternative_summaries": {
                    "brief": state.get("brief_summary"),
                    "concise": state.get("concise_summary"),
                    "detailed": state.get("detailed_summary"),
                    "topical": state.get("topical_summary")
                }
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            state["error"] = f"Result processing error: {str(e)}"
            return state
            
    async def _handle_error(self, state: ConversationSummarizerState) -> ConversationSummarizerState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        logger.error(f"Conversation summarizer error: {error}")
        
        state["success"] = False
        state["result"] = {"error": error}
        
        return state
        
    def _should_continue(self, state: ConversationSummarizerState) -> str:
        """Determine if workflow should continue"""
        if state.get("error"):
            return "error"
        return "continue"
        
    def _should_prepare(self, state: ConversationSummarizerState) -> str:
        """Determine if summarization should be prepared"""
        if state.get("error"):
            return "error"
        return "prepare"
        
    def _should_generate(self, state: ConversationSummarizerState) -> str:
        """Determine if summaries should be generated"""
        if state.get("error"):
            return "error"
        return "generate"
        
    def _should_extract(self, state: ConversationSummarizerState) -> str:
        """Determine if key elements should be extracted"""
        if state.get("error"):
            return "error"
        
        # Extract for most types except brief summaries
        summary_type = state.get("summary_type")
        if summary_type == SummaryType.BRIEF:
            return "finalize"
        return "extract"
        
    def _should_finalize(self, state: ConversationSummarizerState) -> str:
        """Determine if summary should be finalized"""
        if state.get("error"):
            return "error"
        return "finalize"
        
    def _should_process(self, state: ConversationSummarizerState) -> str:
        """Determine if results should be processed"""
        if state.get("error"):
            return "error"
        return "process"
        
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the conversation summarizer workflow"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create thread ID for conversation tracking
            thread_id = initial_state.get("conversation_id", "default")
            
            # Run the workflow
            result = await self._workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running conversation summarizer: {str(e)}")
            return {
                "success": False,
                "error": f"Agent error: {str(e)}",
                "result": None
            }
            
    async def summarize_conversation(
        self,
        conversation_id: str,
        summary_type: SummaryType = SummaryType.CONCISE,
        summary_level: SummaryLevel = SummaryLevel.MID_LEVEL,
        include_metadata: bool = True,
        time_range: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Summarize a conversation"""
        state = {
            "conversation_id": conversation_id,
            "summary_type": summary_type,
            "summary_level": summary_level,
            "include_metadata": include_metadata,
            "time_range": time_range
        }
        
        return await self.run(state)


# Global conversation summarizer agent instance
conversation_summarizer_agent = LangGraphConversationSummarizerAgent()