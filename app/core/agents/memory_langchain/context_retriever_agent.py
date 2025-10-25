"""
LangGraph-based Context Retriever Agent for AI Assistant.

This module provides a specialized agent for retrieving relevant context
from memory using semantic search and conversation history.
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


class ContextRetrievalStrategy(Enum):
    """Context retrieval strategies"""
    
    SEMANTIC_SEARCH = "semantic_search"
    CONVERSATION_HISTORY = "conversation_history"
    HYBRID = "hybrid"
    TEMPORAL = "temporal"
    ENTITY_BASED = "entity_based"


class ContextRetrieverState(TypedDict):
    """State for context retriever workflow"""
    
    # Query information
    query: str
    conversation_id: Optional[str]
    agent_name: Optional[str]
    
    # Retrieval configuration
    strategy: ContextRetrievalStrategy
    context_limit: int
    time_range: Optional[Dict[str, Any]]
    entities: Optional[List[str]]
    
    # Retrieval operations
    retrieval_operations: List[Dict[str, Any]]
    
    # Results
    success: bool
    result: Any
    error: Optional[str]
    
    # Retrieved context
    semantic_context: List[Dict[str, Any]]
    conversation_context: List[Dict[str, Any]]
    temporal_context: List[Dict[str, Any]]
    entity_context: List[Dict[str, Any]]
    
    # Merged context
    merged_context: List[Dict[str, Any]]
    context_summary: Optional[str]
    
    # Internal state
    timestamp: datetime
    step_count: int


class LangGraphContextRetrieverAgent:
    """
    LangGraph-based context retriever agent for intelligent context retrieval.
    
    This agent provides:
    - Multiple context retrieval strategies
    - Semantic search capabilities
    - Temporal context filtering
    - Entity-based retrieval
    - Context merging and ranking
    - Context summarization
    """
    
    def __init__(self):
        self._workflow = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the context retriever agent"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Context Retriever Agent...")
        
        # Initialize memory workflow
        await memory_workflow.initialize()
        
        # Create the agent workflow
        self._create_workflow()
        
        self._initialized = True
        logger.info("LangGraph Context Retriever Agent initialized successfully")
        
    def _create_workflow(self):
        """Create the context retriever workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(ContextRetrieverState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("prepare_retrieval_operations", self._prepare_retrieval_operations)
        workflow.add_node("execute_retrieval", self._execute_retrieval)
        workflow.add_node("merge_context", self._merge_context)
        workflow.add_node("summarize_context", self._summarize_context)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "prepare_retrieval_operations")
        workflow.add_edge("prepare_retrieval_operations", "execute_retrieval")
        workflow.add_edge("execute_retrieval", "merge_context")
        workflow.add_edge("merge_context", "summarize_context")
        workflow.add_edge("summarize_context", "process_results")
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._should_continue,
            {
                "continue": "prepare_retrieval_operations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "prepare_retrieval_operations",
            self._should_execute,
            {
                "execute": "execute_retrieval",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_retrieval",
            self._should_merge,
            {
                "merge": "merge_context",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "merge_context",
            self._should_summarize,
            {
                "summarize": "summarize_context",
                "process": "process_results"
            }
        )
        
        workflow.add_conditional_edges(
            "summarize_context",
            self._should_process,
            {
                "process": "process_results",
                "error": "handle_error"
            }
        )
        
        # Compile the workflow
        self._workflow = workflow.compile()
        
    async def _analyze_query(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Analyze the query and determine optimal retrieval strategy"""
        try:
            query = state.get("query", "")
            strategy = state.get("strategy", ContextRetrievalStrategy.HYBRID)
            
            # Initialize state
            state["success"] = False
            state["result"] = None
            state["error"] = None
            state["timestamp"] = datetime.now()
            state["step_count"] = 0
            state["retrieval_operations"] = []
            state["semantic_context"] = []
            state["conversation_context"] = []
            state["temporal_context"] = []
            state["entity_context"] = []
            state["merged_context"] = []
            state["context_summary"] = None
            
            # Validate query
            if not query:
                state["error"] = "Query is required for context retrieval"
                return state
                
            # Determine optimal strategy if not specified
            if strategy == ContextRetrievalStrategy.HYBRID:
                # Use hybrid strategy by default
                state["strategy"] = ContextRetrievalStrategy.HYBRID
                
            # Set default context limit if not specified
            if not state.get("context_limit"):
                state["context_limit"] = 5
                
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            state["error"] = f"Query analysis error: {str(e)}"
            return state
            
    async def _prepare_retrieval_operations(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Prepare retrieval operations based on strategy"""
        try:
            strategy = state.get("strategy")
            query = state.get("query")
            conversation_id = state.get("conversation_id")
            context_limit = state.get("context_limit", 5)
            state["step_count"] += 1
            
            if strategy == ContextRetrievalStrategy.SEMANTIC_SEARCH:
                # Semantic search operation
                operation = {
                    "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
                    "query": query,
                    "conversation_id": conversation_id,
                    "limit": context_limit
                }
                state["retrieval_operations"].append(operation)
                
            elif strategy == ContextRetrievalStrategy.CONVERSATION_HISTORY:
                # Conversation history operation
                operation = {
                    "operation": MemoryOperation.GET_MESSAGES,
                    "conversation_id": conversation_id,
                    "limit": context_limit * 2  # Get more messages for context
                }
                state["retrieval_operations"].append(operation)
                
            elif strategy == ContextRetrievalStrategy.HYBRID:
                # Multiple operations for hybrid strategy
                # Semantic search
                semantic_op = {
                    "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
                    "query": query,
                    "conversation_id": conversation_id,
                    "limit": context_limit
                }
                state["retrieval_operations"].append(semantic_op)
                
                # Conversation history
                if conversation_id:
                    history_op = {
                        "operation": MemoryOperation.GET_MESSAGES,
                        "conversation_id": conversation_id,
                        "limit": context_limit
                    }
                    state["retrieval_operations"].append(history_op)
                    
                # Global search
                global_search_op = {
                    "operation": MemoryOperation.SEARCH_CONVERSATIONS,
                    "query": query,
                    "limit": context_limit
                }
                state["retrieval_operations"].append(global_search_op)
                
            elif strategy == ContextRetrievalStrategy.TEMPORAL:
                # Temporal context retrieval
                time_range = state.get("time_range")
                if time_range:
                    # This would require custom implementation for temporal filtering
                    # For now, use semantic search with time metadata
                    operation = {
                        "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
                        "query": query,
                        "conversation_id": conversation_id,
                        "limit": context_limit
                    }
                    state["retrieval_operations"].append(operation)
                    
            elif strategy == ContextRetrievalStrategy.ENTITY_BASED:
                # Entity-based retrieval
                entities = state.get("entities", [])
                if entities:
                    # Create entity-focused query
                    entity_query = f"{query} " + " ".join(entities)
                    operation = {
                        "operation": MemoryOperation.GET_RELEVANT_CONTEXT,
                        "query": entity_query,
                        "conversation_id": conversation_id,
                        "limit": context_limit
                    }
                    state["retrieval_operations"].append(operation)
                    
            return state
            
        except Exception as e:
            logger.error(f"Error preparing retrieval operations: {str(e)}")
            state["error"] = f"Operation preparation error: {str(e)}"
            return state
            
    async def _execute_retrieval(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Execute the retrieval operations"""
        try:
            state["step_count"] += 1
            
            for i, operation in enumerate(state.get("retrieval_operations", [])):
                # Execute each operation using memory workflow
                result = await memory_workflow.run(operation)
                
                if result.get("success"):
                    # Store results based on operation type
                    op_type = operation.get("operation")
                    
                    if op_type == MemoryOperation.GET_RELEVANT_CONTEXT:
                        if i == 0 and state.get("strategy") == ContextRetrievalStrategy.HYBRID:
                            # First semantic search result
                            state["semantic_context"] = result.get("result", {}).get("context", [])
                        else:
                            # Additional semantic results
                            state["semantic_context"].extend(
                                result.get("result", {}).get("context", [])
                            )
                            
                    elif op_type == MemoryOperation.GET_MESSAGES:
                        state["conversation_context"] = result.get("result", {}).get("messages", [])
                        
                    elif op_type == MemoryOperation.SEARCH_CONVERSATIONS:
                        search_results = result.get("result", {}).get("results", [])
                        # Convert search results to context format
                        for search_result in search_results:
                            state["semantic_context"].append({
                                "content": search_result.get("content"),
                                "metadata": search_result.get("metadata", {}),
                                "score": search_result.get("score", 0.0),
                                "source": "global_search"
                            })
                            
                else:
                    # Handle operation failure
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Retrieval operation failed: {error_msg}")
                    
                    # Continue with other operations but note the error
                    if not state.get("error"):
                        state["error"] = error_msg
                        
            # Mark as successful if no critical errors
            if not state.get("error"):
                state["success"] = True
                
            return state
            
        except Exception as e:
            logger.error(f"Error executing retrieval: {str(e)}")
            state["error"] = f"Retrieval execution error: {str(e)}"
            return state
            
    async def _merge_context(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Merge and rank retrieved context"""
        try:
            state["step_count"] += 1
            
            # Collect all context
            all_context = []
            
            # Add semantic context with high priority
            for ctx in state.get("semantic_context", []):
                ctx_copy = ctx.copy()
                ctx_copy["source_type"] = "semantic"
                ctx_copy["priority"] = 1.0
                all_context.append(ctx_copy)
                
            # Add conversation context with medium priority
            for ctx in state.get("conversation_context", []):
                ctx_copy = {
                    "content": ctx.get("content", ""),
                    "metadata": ctx.get("metadata", {}),
                    "source_type": "conversation",
                    "priority": 0.8,
                    "role": ctx.get("role", ""),
                    "timestamp": ctx.get("timestamp")
                }
                all_context.append(ctx_copy)
                
            # Remove duplicates based on content similarity
            unique_context = []
            seen_content = set()
            
            for ctx in all_context:
                content = ctx.get("content", "").lower().strip()
                if content and content not in seen_content:
                    seen_content.add(content)
                    unique_context.append(ctx)
                    
            # Sort by priority and score
            unique_context.sort(
                key=lambda x: (
                    x.get("priority", 0) * x.get("score", 0.5),
                    x.get("score", 0.5)
                ),
                reverse=True
            )
            
            # Apply context limit
            context_limit = state.get("context_limit", 5)
            state["merged_context"] = unique_context[:context_limit]
            
            return state
            
        except Exception as e:
            logger.error(f"Error merging context: {str(e)}")
            state["error"] = f"Context merging error: {str(e)}"
            return state
            
    async def _summarize_context(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Summarize the merged context if needed"""
        try:
            merged_context = state.get("merged_context", [])
            context_limit = state.get("context_limit", 5)
            state["step_count"] += 1
            
            # Only summarize if we have substantial context
            if len(merged_context) > 3:
                # Get LLM for summarization
                llm = await llm_manager.get_llm("gpt-3.5-turbo")
                
                if llm:
                    # Create context text
                    context_text = "\n".join([
                        f"- {ctx.get('content', '')}" 
                        for ctx in merged_context[:context_limit]
                    ])
                    
                    prompt = f"""Please summarize the following context for the query "{state.get('query')}":

{context_text}

Provide a concise summary that captures the most relevant information for answering the query."""
                    
                    # Generate summary
                    response = await llm.ainvoke(prompt)
                    state["context_summary"] = response.content
                else:
                    logger.warning("No LLM available for context summarization")
                    
            return state
            
        except Exception as e:
            logger.error(f"Error summarizing context: {str(e)}")
            # Don't fail the workflow for summarization errors
            return state
            
    async def _process_results(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Process the final results"""
        try:
            state["step_count"] += 1
            
            # Create result
            state["result"] = {
                "query": state.get("query"),
                "strategy": state.get("strategy").value,
                "context": state.get("merged_context", []),
                "context_count": len(state.get("merged_context", [])),
                "summary": state.get("context_summary"),
                "sources": {
                    "semantic_context": len(state.get("semantic_context", [])),
                    "conversation_context": len(state.get("conversation_context", [])),
                    "temporal_context": len(state.get("temporal_context", [])),
                    "entity_context": len(state.get("entity_context", []))
                }
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            state["error"] = f"Result processing error: {str(e)}"
            return state
            
    async def _handle_error(self, state: ContextRetrieverState) -> ContextRetrieverState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        logger.error(f"Context retriever error: {error}")
        
        state["success"] = False
        state["result"] = {"error": error}
        
        return state
        
    def _should_continue(self, state: ContextRetrieverState) -> str:
        """Determine if workflow should continue"""
        if state.get("error"):
            return "error"
        return "continue"
        
    def _should_execute(self, state: ContextRetrieverState) -> str:
        """Determine if operations should be executed"""
        if state.get("error"):
            return "error"
        return "execute"
        
    def _should_merge(self, state: ContextRetrieverState) -> str:
        """Determine if context should be merged"""
        if state.get("error"):
            return "error"
        return "merge"
        
    def _should_summarize(self, state: ContextRetrieverState) -> str:
        """Determine if context should be summarized"""
        if state.get("error"):
            return "error"
        
        # Only summarize if we have enough context
        if len(state.get("merged_context", [])) > 3:
            return "summarize"
        return "process"
        
    def _should_process(self, state: ContextRetrieverState) -> str:
        """Determine if results should be processed"""
        if state.get("error"):
            return "error"
        return "process"
        
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the context retriever workflow"""
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
            logger.error(f"Error running context retriever: {str(e)}")
            return {
                "success": False,
                "error": f"Agent error: {str(e)}",
                "result": None
            }
            
    async def retrieve_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        strategy: ContextRetrievalStrategy = ContextRetrievalStrategy.HYBRID,
        context_limit: int = 5,
        time_range: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve context for a query"""
        state = {
            "query": query,
            "conversation_id": conversation_id,
            "strategy": strategy,
            "context_limit": context_limit,
            "time_range": time_range,
            "entities": entities
        }
        
        return await self.run(state)


# Global context retriever agent instance
context_retriever_agent = LangGraphContextRetrieverAgent()