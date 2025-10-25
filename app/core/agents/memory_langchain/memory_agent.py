"""
LangGraph-based Memory Agent for AI Assistant.

This module provides a comprehensive memory agent that uses LangGraph workflows
for memory management, context retrieval, and conversation operations.
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


class MemoryTaskType(Enum):
    """Memory task types"""
    
    STORE_MESSAGE = "store_message"
    RETRIEVE_CONTEXT = "retrieve_context"
    SUMMARIZE_CONVERSATION = "summarize_conversation"
    SEARCH_MEMORY = "search_memory"
    MANAGE_CONVERSATION = "manage_conversation"
    GET_CONVERSATION_HISTORY = "get_conversation_history"


class MemoryAgentState(TypedDict):
    """State for memory agent workflow"""
    
    # Task information
    task_type: MemoryTaskType
    conversation_id: Optional[str]
    agent_name: Optional[str]
    
    # Input data
    message: Optional[str]
    role: Optional[str]
    query: Optional[str]
    context_limit: Optional[int]
    metadata: Optional[Dict[str, Any]]
    
    # Memory operations
    memory_operations: List[Dict[str, Any]]
    
    # Results
    success: bool
    result: Any
    error: Optional[str]
    
    # Context and memory
    retrieved_context: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    summary: Optional[str]
    search_results: List[Dict[str, Any]]
    
    # Internal state
    timestamp: datetime
    step_count: int


class LangGraphMemoryAgent:
    """
    LangGraph-based memory agent for comprehensive memory management.
    
    This agent provides:
    - Message storage and retrieval
    - Context-aware memory operations
    - Conversation summarization
    - Semantic search capabilities
    - Conversation lifecycle management
    """
    
    def __init__(self):
        self._workflow = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the memory agent"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Memory Agent...")
        
        # Initialize memory workflow
        await memory_workflow.initialize()
        
        # Create the agent workflow
        self._create_workflow()
        
        self._initialized = True
        logger.info("LangGraph Memory Agent initialized successfully")
        
    def _create_workflow(self):
        """Create the memory agent workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(MemoryAgentState)
        
        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("prepare_memory_operations", self._prepare_memory_operations)
        workflow.add_node("execute_memory_operations", self._execute_memory_operations)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("analyze_task")
        workflow.add_edge("analyze_task", "prepare_memory_operations")
        workflow.add_edge("prepare_memory_operations", "execute_memory_operations")
        workflow.add_edge("execute_memory_operations", "process_results")
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_task",
            self._should_continue,
            {
                "continue": "prepare_memory_operations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "prepare_memory_operations",
            self._should_execute,
            {
                "execute": "execute_memory_operations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_memory_operations",
            self._should_process,
            {
                "process": "process_results",
                "error": "handle_error"
            }
        )
        
        # Compile the workflow
        self._workflow = workflow.compile()
        
    async def _analyze_task(self, state: MemoryAgentState) -> MemoryAgentState:
        """Analyze the memory task and prepare execution plan"""
        try:
            task_type = state.get("task_type")
            
            # Initialize state
            state["success"] = False
            state["result"] = None
            state["error"] = None
            state["timestamp"] = datetime.now()
            state["step_count"] = 0
            state["memory_operations"] = []
            state["retrieved_context"] = []
            state["conversation_history"] = []
            state["summary"] = None
            state["search_results"] = []
            
            # Validate task requirements
            if task_type == MemoryTaskType.STORE_MESSAGE:
                if not state.get("conversation_id") or not state.get("message"):
                    state["error"] = "Conversation ID and message are required for store_message"
                    return state
                    
            elif task_type == MemoryTaskType.RETRIEVE_CONTEXT:
                if not state.get("query"):
                    state["error"] = "Query is required for retrieve_context"
                    return state
                    
            elif task_type == MemoryTaskType.SUMMARIZE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for summarize_conversation"
                    return state
                    
            elif task_type == MemoryTaskType.SEARCH_MEMORY:
                if not state.get("query"):
                    state["error"] = "Query is required for search_memory"
                    return state
                    
            elif task_type == MemoryTaskType.GET_CONVERSATION_HISTORY:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for get_conversation_history"
                    return state
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            state["error"] = f"Task analysis error: {str(e)}"
            return state
            
    async def _prepare_memory_operations(self, state: MemoryAgentState) -> MemoryAgentState:
        """Prepare memory operations based on task type"""
        try:
            task_type = state.get("task_type")
            state["step_count"] += 1
            
            if task_type == MemoryTaskType.STORE_MESSAGE:
                # Prepare to store message
                operation = {
                    "operation": MemoryOperation.ADD_MESSAGE,
                    "conversation_id": state.get("conversation_id"),
                    "role": state.get("role", "human"),
                    "content": state.get("message"),
                    "metadata": state.get("metadata")
                }
                state["memory_operations"].append(operation)
                
            elif task_type == MemoryTaskType.RETRIEVE_CONTEXT:
                # Prepare to retrieve relevant context
                operation = {
                    "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
                    "query": state.get("query"),
                    "conversation_id": state.get("conversation_id"),
                    "limit": state.get("context_limit", 5)
                }
                state["memory_operations"].append(operation)
                
            elif task_type == MemoryTaskType.SUMMARIZE_CONVERSATION:
                # Prepare to summarize conversation
                operation = {
                    "operation": MemoryOperation.SUMMARIZE_CONVERSATION,
                    "conversation_id": state.get("conversation_id")
                }
                state["memory_operations"].append(operation)
                
            elif task_type == MemoryTaskType.SEARCH_MEMORY:
                # Prepare to search memory
                operation = {
                    "operation": MemoryOperation.SEARCH_CONVERSATIONS,
                    "query": state.get("query"),
                    "conversation_id": state.get("conversation_id"),
                    "limit": state.get("context_limit", 10)
                }
                state["memory_operations"].append(operation)
                
            elif task_type == MemoryTaskType.GET_CONVERSATION_HISTORY:
                # Prepare to get conversation history
                operation = {
                    "operation": MemoryOperation.GET_MESSAGES,
                    "conversation_id": state.get("conversation_id"),
                    "limit": state.get("context_limit")
                }
                state["memory_operations"].append(operation)
                
            elif task_type == MemoryTaskType.MANAGE_CONVERSATION:
                # Prepare conversation management operations
                if state.get("conversation_id"):
                    # Get conversation info first
                    info_op = {
                        "operation": MemoryOperation.GET_CONVERSATION_INFO,
                        "conversation_id": state.get("conversation_id")
                    }
                    state["memory_operations"].append(info_op)
                    
            return state
            
        except Exception as e:
            logger.error(f"Error preparing memory operations: {str(e)}")
            state["error"] = f"Operation preparation error: {str(e)}"
            return state
            
    async def _execute_memory_operations(self, state: MemoryAgentState) -> MemoryAgentState:
        """Execute the prepared memory operations"""
        try:
            state["step_count"] += 1
            
            for operation in state.get("memory_operations", []):
                # Execute each operation using memory workflow
                result = await memory_workflow.run(operation)
                
                if result.get("success"):
                    # Store results based on operation type
                    op_type = operation.get("operation")
                    
                    if op_type == MemoryOperation.ADD_MESSAGE:
                        # Message stored successfully
                        pass
                        
                    elif op_type == MemoryOperation.GET_RELEVANT_CONTEXT:
                        state["retrieved_context"] = result.get("result", {}).get("context", [])
                        
                    elif op_type == MemoryOperation.SUMMARIZE_CONVERSATION:
                        state["summary"] = result.get("result", {}).get("summary")
                        
                    elif op_type == MemoryOperation.SEARCH_CONVERSATIONS:
                        state["search_results"] = result.get("result", {}).get("results", [])
                        
                    elif op_type == MemoryOperation.GET_MESSAGES:
                        state["conversation_history"] = result.get("result", {}).get("messages", [])
                        
                    elif op_type == MemoryOperation.GET_CONVERSATION_INFO:
                        conv_info = result.get("result", {}).get("info")
                        if conv_info:
                            state["result"] = {"conversation_info": conv_info}
                            
                else:
                    # Handle operation failure
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Memory operation failed: {error_msg}")
                    
                    # Continue with other operations but note the error
                    if not state.get("error"):
                        state["error"] = error_msg
                        
            # Mark as successful if no critical errors
            if not state.get("error"):
                state["success"] = True
                
            return state
            
        except Exception as e:
            logger.error(f"Error executing memory operations: {str(e)}")
            state["error"] = f"Operation execution error: {str(e)}"
            return state
            
    async def _process_results(self, state: MemoryAgentState) -> MemoryAgentState:
        """Process the memory operation results"""
        try:
            task_type = state.get("task_type")
            state["step_count"] += 1
            
            # Create result based on task type
            if task_type == MemoryTaskType.STORE_MESSAGE:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "message_stored": state.get("success", False),
                    "timestamp": state.get("timestamp").isoformat()
                }
                
            elif task_type == MemoryTaskType.RETRIEVE_CONTEXT:
                state["result"] = {
                    "query": state.get("query"),
                    "context": state.get("retrieved_context", []),
                    "context_count": len(state.get("retrieved_context", []))
                }
                
            elif task_type == MemoryTaskType.SUMMARIZE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "summary": state.get("summary"),
                    "generated_at": state.get("timestamp").isoformat()
                }
                
            elif task_type == MemoryTaskType.SEARCH_MEMORY:
                state["result"] = {
                    "query": state.get("query"),
                    "results": state.get("search_results", []),
                    "result_count": len(state.get("search_results", []))
                }
                
            elif task_type == MemoryTaskType.GET_CONVERSATION_HISTORY:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "messages": state.get("conversation_history", []),
                    "message_count": len(state.get("conversation_history", []))
                }
                
            elif task_type == MemoryTaskType.MANAGE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "managed": state.get("success", False),
                    "info": state.get("result", {}).get("conversation_info")
                }
                
            return state
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            state["error"] = f"Result processing error: {str(e)}"
            return state
            
    async def _handle_error(self, state: MemoryAgentState) -> MemoryAgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        logger.error(f"Memory agent error: {error}")
        
        state["success"] = False
        state["result"] = {"error": error}
        
        return state
        
    def _should_continue(self, state: MemoryAgentState) -> str:
        """Determine if workflow should continue"""
        if state.get("error"):
            return "error"
        return "continue"
        
    def _should_execute(self, state: MemoryAgentState) -> str:
        """Determine if operations should be executed"""
        if state.get("error"):
            return "error"
        return "execute"
        
    def _should_process(self, state: MemoryAgentState) -> str:
        """Determine if results should be processed"""
        if state.get("error"):
            return "error"
        return "process"
        
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the memory agent workflow"""
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
            logger.error(f"Error running memory agent: {str(e)}")
            return {
                "success": False,
                "error": f"Agent error: {str(e)}",
                "result": None
            }
            
    async def store_message(
        self,
        conversation_id: str,
        message: str,
        role: str = "human",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a message in memory"""
        state = {
            "task_type": MemoryTaskType.STORE_MESSAGE,
            "conversation_id": conversation_id,
            "message": message,
            "role": role,
            "metadata": metadata
        }
        
        return await self.run(state)
        
    async def retrieve_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        state = {
            "task_type": MemoryTaskType.RETRIEVE_CONTEXT,
            "query": query,
            "conversation_id": conversation_id,
            "context_limit": context_limit
        }
        
        return await self.run(state)
        
    async def summarize_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Summarize a conversation"""
        state = {
            "task_type": MemoryTaskType.SUMMARIZE_CONVERSATION,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def search_memory(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        context_limit: int = 10
    ) -> Dict[str, Any]:
        """Search memory for relevant information"""
        state = {
            "task_type": MemoryTaskType.SEARCH_MEMORY,
            "query": query,
            "conversation_id": conversation_id,
            "context_limit": context_limit
        }
        
        return await self.run(state)
        
    async def get_conversation_history(
        self,
        conversation_id: str,
        context_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get conversation history"""
        state = {
            "task_type": MemoryTaskType.GET_CONVERSATION_HISTORY,
            "conversation_id": conversation_id,
            "context_limit": context_limit
        }
        
        return await self.run(state)
        
    async def manage_conversation(
        self,
        conversation_id: str,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manage a conversation (get info, etc.)"""
        state = {
            "task_type": MemoryTaskType.MANAGE_CONVERSATION,
            "conversation_id": conversation_id,
            "agent_name": agent_name
        }
        
        return await self.run(state)


# Global memory agent instance
memory_agent = LangGraphMemoryAgent()