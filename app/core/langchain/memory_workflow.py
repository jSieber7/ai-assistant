"""
LangGraph-based Memory Management Workflow for AI Assistant.

This module provides a LangGraph workflow for managing memory operations,
integrating with the LangChain Memory Manager for persistence and retrieval.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .memory_manager import memory_manager, MemoryType, ConversationInfo, MessageInfo
from .llm_manager import llm_manager
from .monitoring import LangChainMonitoring

logger = logging.getLogger(__name__)


class MemoryOperation(Enum):
    """Memory operation types"""
    
    CREATE_CONVERSATION = "create_conversation"
    ADD_MESSAGE = "add_message"
    GET_MESSAGES = "get_messages"
    SEARCH_CONVERSATIONS = "search_conversations"
    GET_CONVERSATION_INFO = "get_conversation_info"
    LIST_CONVERSATIONS = "list_conversations"
    DELETE_CONVERSATION = "delete_conversation"
    SUMMARIZE_CONVERSATION = "summarize_conversation"
    GET_RELEVANT_CONTEXT = "get_relevant_context"


class MemoryState(TypedDict):
    """State for memory management workflow"""
    
    # Operation details
    operation: MemoryOperation
    conversation_id: Optional[str]
    agent_name: Optional[str]
    role: Optional[str]
    content: Optional[str]
    metadata: Optional[Dict[str, Any]]
    limit: Optional[int]
    query: Optional[str]
    
    # Results
    success: bool
    result: Any
    error: Optional[str]
    
    # Context
    messages: List[Dict[str, Any]]
    conversations: List[ConversationInfo]
    conversation_info: Optional[ConversationInfo]
    search_results: List[Dict[str, Any]]
    summary: Optional[str]
    relevant_context: List[Dict[str, Any]]
    
    # Internal state
    timestamp: datetime
    step_count: int


class LangGraphMemoryWorkflow:
    """
    LangGraph workflow for memory management operations.
    
    This workflow provides:
    - Structured memory operations with state management
    - Integration with LangChain Memory Manager
    - Context-aware memory retrieval
    - Conversation summarization
    - Semantic search capabilities
    """
    
    def __init__(self):
        self._workflow = None
        self._initialized = False
        self._monitoring = LangChainMonitoring()
        
    async def initialize(self):
        """Initialize the memory workflow"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Memory Workflow...")
        
        # Initialize monitoring
        await self._monitoring.initialize()
        
        # Ensure memory manager is initialized
        await memory_manager.initialize()
        
        # Create the workflow graph
        self._create_workflow()
        
        self._initialized = True
        logger.info("LangGraph Memory Workflow initialized successfully")
        
    def _create_workflow(self):
        """Create the memory workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(MemoryState)
        
        # Add nodes
        workflow.add_node("validate_operation", self._validate_operation)
        workflow.add_node("execute_operation", self._execute_operation)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("validate_operation")
        workflow.add_edge("validate_operation", "execute_operation")
        workflow.add_edge("execute_operation", "process_results")
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate_operation",
            self._should_continue,
            {
                "continue": "execute_operation",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_operation",
            self._should_process,
            {
                "process": "process_results",
                "error": "handle_error"
            }
        )
        
        # Compile with memory
        self._workflow = workflow.compile(checkpointer=MemorySaver())
        
    async def _validate_operation(self, state: MemoryState) -> MemoryState:
        """Validate the memory operation"""
        try:
            operation = state.get("operation")
            
            # Validate required fields based on operation
            if operation == MemoryOperation.CREATE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for create_conversation"
                    return state
                    
            elif operation == MemoryOperation.ADD_MESSAGE:
                if not state.get("conversation_id") or not state.get("content"):
                    state["error"] = "Conversation ID and content are required for add_message"
                    return state
                    
            elif operation == MemoryOperation.GET_MESSAGES:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for get_messages"
                    return state
                    
            elif operation == MemoryOperation.SEARCH_CONVERSATIONS:
                if not state.get("query"):
                    state["error"] = "Query is required for search_conversations"
                    return state
                    
            elif operation == MemoryOperation.GET_CONVERSATION_INFO:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for get_conversation_info"
                    return state
                    
            elif operation == MemoryOperation.DELETE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for delete_conversation"
                    return state
                    
            elif operation == MemoryOperation.SUMMARIZE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for summarize_conversation"
                    return state
                    
            elif operation == MemoryOperation.GET_RELEVANT_CONTEXT:
                if not state.get("query"):
                    state["error"] = "Query is required for get_relevant_context"
                    return state
            
            # Initialize state
            state["success"] = False
            state["result"] = None
            state["error"] = None
            state["timestamp"] = datetime.now()
            state["step_count"] = 0
            
            return state
            
        except Exception as e:
            async def _execute_operation(self, state: MemoryState) -> MemoryState:
                """Execute the memory operation"""
                start_time = time.time()
                
                try:
                    operation = state.get("operation")
                    state["step_count"] += 1
                    operation_name = operation.value if hasattr(operation, 'value') else str(operation)
                    conversation_id = state.get("conversation_id", "unknown")
                    
                    if operation == MemoryOperation.CREATE_CONVERSATION:
                        success = await memory_manager.create_conversation(
                            conversation_id=state.get("conversation_id"),
                            agent_name=state.get("agent_name"),
                            title=state.get("metadata", {}).get("title") if state.get("metadata") else None,
                            metadata=state.get("metadata")
                        )
                        state["success"] = success
                        
                    elif operation == MemoryOperation.ADD_MESSAGE:
                        success = await memory_manager.add_message(
                            conversation_id=state.get("conversation_id"),
                            role=state.get("role", "human"),
                            content=state.get("content"),
                            metadata=state.get("metadata")
                        )
                        state["success"] = success
                        
                    elif operation == MemoryOperation.GET_MESSAGES:
                        messages = await memory_manager.get_conversation_messages(
                            conversation_id=state.get("conversation_id"),
                            limit=state.get("limit")
                        )
                        state["messages"] = messages
                        state["success"] = True
                        
                    elif operation == MemoryOperation.SEARCH_CONVERSATIONS:
                        search_results = await memory_manager.search_conversations(
                            query=state.get("query"),
                            limit=state.get("limit", 10),
                            conversation_id=state.get("conversation_id")
                        )
                        state["search_results"] = search_results
                        state["success"] = True
                        
                    elif operation == MemoryOperation.GET_CONVERSATION_INFO:
                        conv_info = await memory_manager.get_conversation_info(
                            conversation_id=state.get("conversation_id")
                        )
                        state["conversation_info"] = conv_info
                        state["success"] = conv_info is not None
                        
                    elif operation == MemoryOperation.LIST_CONVERSATIONS:
                        conversations = await memory_manager.list_conversations(
                            agent_name=state.get("agent_name"),
                            limit=state.get("limit")
                        )
                        state["conversations"] = conversations
                        state["success"] = True
                        
                    elif operation == MemoryOperation.DELETE_CONVERSATION:
                        success = await memory_manager.delete_conversation(
                            conversation_id=state.get("conversation_id")
                        )
                        state["success"] = success
                        
                    elif operation == MemoryOperation.SUMMARIZE_CONVERSATION:
                        summary = await self._summarize_conversation(
                            conversation_id=state.get("conversation_id")
                        )
                        state["summary"] = summary
                        state["success"] = summary is not None
                        
                    elif operation == MemoryOperation.GET_RELEVANT_CONTEXT:
                        context = await self._get_relevant_context(
                            query=state.get("query"),
                            conversation_id=state.get("conversation_id"),
                            limit=state.get("limit", 5)
                        )
                        state["relevant_context"] = context
                        state["success"] = True
                        
                    else:
                        state["error"] = f"Unknown operation: {operation}"
                        
                    # Record success metric
                    await self._monitoring.record_metric(
                        component_type="workflow",
                        component_name="memory_operation",
                        metric_name="success",
                        metric_value=1,
                        metadata={
                            "operation": operation_name,
                            "conversation_id": conversation_id
                        }
                    )
                        
                    return state
                    
                except Exception as e:
                    # Record error metric
                    await self._monitoring.record_metric(
                        component_type="workflow",
                        component_name="memory_operation",
                        metric_name="error",
                        metric_value=1,
                        metadata={
                            "operation": operation_name if 'operation_name' in locals() else "unknown",
                            "conversation_id": state.get("conversation_id", "unknown"),
                            "error": str(e)
                        }
                    )
                    logger.error(f"Error executing operation: {str(e)}")
                    state["error"] = f"Execution error: {str(e)}"
                    return state
                    
                finally:
                    # Record duration metric
                    duration = time.time() - start_time
                    await self._monitoring.record_metric(
                        component_type="workflow",
                        component_name="memory_operation",
                        metric_name="duration",
                        metric_value=duration,
                        metadata={
                            "operation": operation_name if 'operation_name' in locals() else "unknown",
                            "conversation_id": state.get("conversation_id", "unknown")
                        }
                    )
        except Exception as e:
            logger.error(f"Error executing operation: {str(e)}")
            state["error"] = f"Execution error: {str(e)}"
            return state
            
    async def _process_results(self, state: MemoryState) -> MemoryState:
        """Process the operation results"""
        try:
            operation = state.get("operation")
            state["step_count"] += 1
            
            # Create result based on operation
            if operation == MemoryOperation.CREATE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "created": state.get("success", False)
                }
                
            elif operation == MemoryOperation.ADD_MESSAGE:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "message_added": state.get("success", False)
                }
                
            elif operation == MemoryOperation.GET_MESSAGES:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "messages": state.get("messages", [])
                }
                
            elif operation == MemoryOperation.SEARCH_CONVERSATIONS:
                state["result"] = {
                    "query": state.get("query"),
                    "results": state.get("search_results", [])
                }
                
            elif operation == MemoryOperation.GET_CONVERSATION_INFO:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "info": state.get("conversation_info")
                }
                
            elif operation == MemoryOperation.LIST_CONVERSATIONS:
                state["result"] = {
                    "conversations": state.get("conversations", []),
                    "count": len(state.get("conversations", []))
                }
                
            elif operation == MemoryOperation.DELETE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "deleted": state.get("success", False)
                }
                
            elif operation == MemoryOperation.SUMMARIZE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "summary": state.get("summary")
                }
                
            elif operation == MemoryOperation.GET_RELEVANT_CONTEXT:
                state["result"] = {
                    "query": state.get("query"),
                    "context": state.get("relevant_context", [])
                }
                
            return state
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            state["error"] = f"Processing error: {str(e)}"
            return state
            
    async def _handle_error(self, state: MemoryState) -> MemoryState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        logger.error(f"Memory workflow error: {error}")
        
        state["success"] = False
        state["result"] = {"error": error}
        
        return state
        
    async def _summarize_conversation(self, conversation_id: str) -> Optional[str]:
        """Summarize a conversation using LLM"""
        try:
            # Get conversation messages
            messages = await memory_manager.get_conversation_messages(conversation_id)
            
            if not messages:
                return "No messages to summarize"
                
            # Get LLM for summarization
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            if not llm:
                logger.warning("No LLM available for summarization")
                return None
                
            # Create prompt for summarization
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in messages
            ])
            
            prompt = f"""Please summarize the following conversation:

{conversation_text}

Provide a concise summary that captures the main topics and key points of the conversation."""
            
            # Generate summary
            response = await llm.ainvoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {str(e)}")
            return None
            
    async def _get_relevant_context(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant context for a query"""
        try:
            # Search conversations
            search_results = await memory_manager.search_conversations(
                query=query,
                limit=limit,
                conversation_id=conversation_id
            )
            
            # Get current conversation messages if conversation_id provided
            if conversation_id:
                current_messages = await memory_manager.get_conversation_messages(
                    conversation_id=conversation_id,
                    limit=10
                )
                
                # Add recent messages to context
                for msg in current_messages[-5:]:  # Last 5 messages
                    search_results.append({
                        "content": msg["content"],
                        "metadata": {
                            "conversation_id": conversation_id,
                            "role": msg["role"],
                            "source": "current_conversation"
                        },
                        "score": 1.0  # High relevance for current conversation
                    })
            
            # Sort by relevance and limit
            search_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return []
            
    def _should_continue(self, state: MemoryState) -> str:
        """Determine if workflow should continue"""
        if state.get("error"):
            return "error"
        return "continue"
        
    def _should_process(self, state: MemoryState) -> str:
        """Determine if results should be processed"""
        if state.get("error"):
            return "error"
        return "process"
        
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the memory workflow"""
        start_time = time.time()
        operation = initial_state.get("operation", "unknown")
        conversation_id = initial_state.get("conversation_id", "unknown")
        
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create thread ID for conversation tracking
            thread_id = conversation_id
            
            # Run the workflow
            result = await self._workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Record success metric
            await self._monitoring.record_metric(
                component_type="workflow",
                component_name="memory_workflow",
                metric_name="success",
                metric_value=1,
                metadata={
                    "operation": operation.value if hasattr(operation, 'value') else str(operation),
                    "conversation_id": conversation_id
                }
            )
            
            return result
            
        except Exception as e:
            # Record error metric
            await self._monitoring.record_metric(
                component_type="workflow",
                component_name="memory_workflow",
                metric_name="error",
                metric_value=1,
                metadata={
                    "operation": operation.value if hasattr(operation, 'value') else str(operation),
                    "conversation_id": conversation_id,
                    "error": str(e)
                }
            )
            logger.error(f"Error running memory workflow: {str(e)}")
            return {
                "success": False,
                "error": f"Workflow error: {str(e)}",
                "result": None
            }
            
        finally:
            # Record duration metric
            duration = time.time() - start_time
            await self._monitoring.record_metric(
                component_type="workflow",
                component_name="memory_workflow",
                metric_name="duration",
                metric_value=duration,
                metadata={
                    "operation": operation.value if hasattr(operation, 'value') else str(operation),
                    "conversation_id": conversation_id
                }
            )
            
    async def create_conversation(
        self,
        conversation_id: str,
        agent_name: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation"""
        state = {
            "operation": MemoryOperation.CREATE_CONVERSATION,
            "conversation_id": conversation_id,
            "agent_name": agent_name,
            "metadata": {
                **(metadata or {}),
                "title": title
            }
        }
        
        return await self.run(state)
        
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a message to a conversation"""
        state = {
            "operation": MemoryOperation.ADD_MESSAGE,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": metadata
        }
        
        return await self.run(state)
        
    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get messages from a conversation"""
        state = {
            "operation": MemoryOperation.GET_MESSAGES,
            "conversation_id": conversation_id,
            "limit": limit
        }
        
        return await self.run(state)
        
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search conversations"""
        state = {
            "operation": MemoryOperation.SEARCH_CONVERSATIONS,
            "query": query,
            "limit": limit,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def get_conversation_info(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation information"""
        state = {
            "operation": MemoryOperation.GET_CONVERSATION_INFO,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def list_conversations(
        self,
        agent_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """List conversations"""
        state = {
            "operation": MemoryOperation.LIST_CONVERSATIONS,
            "agent_name": agent_name,
            "limit": limit
        }
        
        return await self.run(state)
        
    async def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation"""
        state = {
            "operation": MemoryOperation.DELETE_CONVERSATION,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def summarize_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Summarize a conversation"""
        state = {
            "operation": MemoryOperation.SUMMARIZE_CONVERSATION,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def get_relevant_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Get relevant context for a query"""
        state = {
            "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
            "query": query,
            "conversation_id": conversation_id,
            "limit": limit
        }
        
        return await self.run(state)


# Global memory workflow instance
memory_workflow = LangGraphMemoryWorkflow()