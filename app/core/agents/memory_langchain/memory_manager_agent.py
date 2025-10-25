"""
LangGraph-based Memory Manager Agent for AI Assistant.

This module provides a specialized agent for high-level memory management
operations including conversation lifecycle, memory maintenance, and analytics.
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ...langchain.memory_workflow import memory_workflow, MemoryOperation
from ...langchain.llm_manager import llm_manager

logger = logging.getLogger(__name__)


class MemoryManagementTask(Enum):
    """Memory management task types"""
    
    CREATE_CONVERSATION = "create_conversation"
    DELETE_CONVERSATION = "delete_conversationation"
    LIST_CONVERSATIONS = "list_conversations"
    MEMORY_CLEANUP = "memory_cleanup"
    MEMORY_ANALYTICS = "memory_analytics"
    MEMORY_OPTIMIZATION = "memory_optimization"
    EXPORT_MEMORY = "export_memory"
    IMPORT_MEMORY = "import_memory"
    MEMORY_BACKUP = "memory_backup"
    MEMORY_RESTORE = "memory_restore"


class MemoryCleanupPolicy(Enum):
    """Memory cleanup policies"""
    
    TIME_BASED = "time_based"  # Clean up conversations older than X days
    SIZE_BASED = "size_based"  # Clean up when memory exceeds X size
    USAGE_BASED = "usage_based"  # Clean up least recently used conversations
    MANUAL = "manual"  # Manual cleanup with specific criteria


class MemoryManagerState(TypedDict):
    """State for memory manager workflow"""
    
    # Task information
    task_type: MemoryManagementTask
    agent_name: Optional[str]
    
    # Task parameters
    conversation_id: Optional[str]
    conversation_title: Optional[str]
    metadata: Optional[Dict[str, Any]]
    cleanup_policy: Optional[MemoryCleanupPolicy]
    cleanup_criteria: Optional[Dict[str, Any]]
    export_format: Optional[str]
    import_data: Optional[Dict[str, Any]]
    
    # Management operations
    management_operations: List[Dict[str, Any]]
    
    # Results
    success: bool
    result: Any
    error: Optional[str]
    
    # Memory data
    conversations: List[Dict[str, Any]]
    conversation_count: int
    memory_stats: Dict[str, Any]
    cleanup_results: Dict[str, Any]
    analytics_data: Dict[str, Any]
    export_data: Optional[Dict[str, Any]]
    import_results: Dict[str, Any]
    
    # Internal state
    timestamp: datetime
    step_count: int


class LangGraphMemoryManagerAgent:
    """
    LangGraph-based memory manager agent for comprehensive memory management.
    
    This agent provides:
    - Conversation lifecycle management
    - Memory cleanup and maintenance
    - Memory analytics and insights
    - Memory optimization
    - Import/export capabilities
    - Backup and restore operations
    """
    
    def __init__(self):
        self._workflow = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the memory manager agent"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Memory Manager Agent...")
        
        # Initialize memory workflow
        await memory_workflow.initialize()
        
        # Create the agent workflow
        self._create_workflow()
        
        self._initialized = True
        logger.info("LangGraph Memory Manager Agent initialized successfully")
        
    def _create_workflow(self):
        """Create the memory manager workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(MemoryManagerState)
        
        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("prepare_operations", self._prepare_operations)
        workflow.add_node("execute_operations", self._execute_operations)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("analyze_task")
        workflow.add_edge("analyze_task", "prepare_operations")
        workflow.add_edge("prepare_operations", "execute_operations")
        workflow.add_edge("execute_operations", "process_results")
        workflow.add_edge("process_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_task",
            self._should_continue,
            {
                "continue": "prepare_operations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "prepare_operations",
            self._should_execute,
            {
                "execute": "execute_operations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_operations",
            self._should_process,
            {
                "process": "process_results",
                "error": "handle_error"
            }
        )
        
        # Compile the workflow
        self._workflow = workflow.compile()
        
    async def _analyze_task(self, state: MemoryManagerState) -> MemoryManagerState:
        """Analyze the memory management task"""
        try:
            task_type = state.get("task_type")
            
            # Initialize state
            state["success"] = False
            state["result"] = None
            state["error"] = None
            state["timestamp"] = datetime.now()
            state["step_count"] = 0
            state["management_operations"] = []
            state["conversations"] = []
            state["conversation_count"] = 0
            state["memory_stats"] = {}
            state["cleanup_results"] = {}
            state["analytics_data"] = {}
            state["export_data"] = None
            state["import_results"] = {}
            
            # Validate task requirements
            if task_type == MemoryManagementTask.CREATE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for create_conversation"
                    return state
                    
            elif task_type == MemoryManagementTask.DELETE_CONVERSATION:
                if not state.get("conversation_id"):
                    state["error"] = "Conversation ID is required for delete_conversation"
                    return state
                    
            elif task_type == MemoryManagementTask.MEMORY_CLEANUP:
                if not state.get("cleanup_policy"):
                    state["cleanup_policy"] = MemoryCleanupPolicy.TIME_BASED
                    
            elif task_type == MemoryManagementTask.EXPORT_MEMORY:
                if not state.get("export_format"):
                    state["export_format"] = "json"
                    
            elif task_type == MemoryManagementTask.IMPORT_MEMORY:
                if not state.get("import_data"):
                    state["error"] = "Import data is required for import_memory"
                    return self._handle_error(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            state["error"] = f"Task analysis error: {str(e)}"
            return state
            
    async def _prepare_operations(self, state: MemoryManagerState) -> MemoryManagerState:
        """Prepare memory management operations"""
        try:
            task_type = state.get("task_type")
            state["step_count"] += 1
            
            if task_type == MemoryManagementTask.CREATE_CONVERSATION:
                # Prepare to create conversation
                operation = {
                    "operation": MemoryOperation.CREATE_CONVERSATION,
                    "conversation_id": state.get("conversation_id"),
                    "agent_name": state.get("agent_name"),
                    "metadata": state.get("metadata", {})
                }
                
                # Add title if provided
                if state.get("conversation_title"):
                    operation["metadata"]["title"] = state.get("conversation_title")
                    
                state["management_operations"].append(operation)
                
            elif task_type == MemoryManagementTask.DELETE_CONVERSATION:
                # Prepare to delete conversation
                operation = {
                    "operation": MemoryOperation.DELETE_CONVERSATION,
                    "conversation_id": state.get("conversation_id")
                }
                state["management_operations"].append(operation)
                
            elif task_type == MemoryManagementTask.LIST_CONVERSATIONS:
                # Prepare to list conversations
                operation = {
                    "operation": MemoryOperation.LIST_CONVERSATIONS,
                    "agent_name": state.get("agent_name")
                }
                state["management_operations"].append(operation)
                
            elif task_type == MemoryManagementTask.MEMORY_CLEANUP:
                # Prepare cleanup operations
                cleanup_policy = state.get("cleanup_policy", MemoryCleanupPolicy.TIME_BASED)
                cleanup_criteria = state.get("cleanup_criteria", {})
                
                # First, get all conversations to analyze
                list_op = {
                    "operation": MemoryOperation.LIST_CONVERSATIONS,
                    "agent_name": state.get("agent_name")
                }
                state["management_operations"].append(list_op)
                
                # Get memory stats
                stats_op = {
                    "operation": "GET_MEMORY_STATS"  # Custom operation
                }
                state["management_operations"].append(stats_op)
                
            elif task_type == MemoryManagementTask.MEMORY_ANALYTICS:
                # Prepare analytics operations
                # Get conversations list
                list_op = {
                    "operation": MemoryOperation.LIST_CONVERSATIONS,
                    "agent_name": state.get("agent_name")
                }
                state["management_operations"].append(list_op)
                
                # Get memory stats
                stats_op = {
                    "operation": "GET_MEMORY_STATS"
                }
                state["management_operations"].append(stats_op)
                
            elif task_type == MemoryManagementTask.MEMORY_OPTIMIZATION:
                # Prepare optimization operations
                # Get current stats
                stats_op = {
                    "operation": "GET_MEMORY_STATS"
                }
                state["management_operations"].append(stats_op)
                
            elif task_type == MemoryManagementTask.EXPORT_MEMORY:
                # Prepare export operations
                # Get all conversations
                list_op = {
                    "operation": MemoryOperation.LIST_CONVERSATIONS,
                    "agent_name": state.get("agent_name")
                }
                state["management_operations"].append(list_op)
                
            elif task_type == MemoryManagementTask.IMPORT_MEMORY:
                # Prepare import operations
                # Import data will be processed in execute phase
                
            elif task_type == MemoryManagementTask.MEMORY_BACKUP:
                # Prepare backup operations
                # Get all conversations for backup
                list_op = {
                    "operation": MemoryOperation.LIST_CONVERSATIONS,
                    "agent_name": state.get("agent_name")
                }
                state["management_operations"].append(list_op)
                
            elif task_type == MemoryManagementTask.MEMORY_RESTORE:
                # Prepare restore operations
                # Restore data will be processed in execute phase
                
            return state
            
        except Exception as e:
            logger.error(f"Error preparing operations: {str(e)}")
            state["error"] = f"Operation preparation error: {str(e)}"
            return state
            
    async def _execute_operations(self, state: MemoryManagerState) -> MemoryManagerState:
        """Execute the memory management operations"""
        try:
            task_type = state.get("task_type")
            state["step_count"] += 1
            
            # Execute standard operations
            for operation in state.get("management_operations", []):
                op_type = operation.get("operation")
                
                if op_type == "GET_MEMORY_STATS":
                    # Get memory stats directly from memory manager
                    stats = await memory_manager.get_memory_stats()
                    state["memory_stats"] = stats
                    
                else:
                    # Execute standard memory workflow operation
                    result = await memory_workflow.run(operation)
                    
                    if result.get("success"):
                        # Process results based on operation type
                        if op_type == MemoryOperation.CREATE_CONVERSATION:
                            # Conversation created successfully
                            pass
                            
                        elif op_type == MemoryOperation.DELETE_CONVERSATION:
                            # Conversation deleted successfully
                            pass
                            
                        elif op_type == MemoryOperation.LIST_CONVERSATIONS:
                            conversations = result.get("result", {}).get("conversations", [])
                            state["conversations"] = [
                                {
                                    "conversation_id": conv.conversation_id,
                                    "agent_name": conv.agent_name,
                                    "title": conv.metadata.get("title") if conv.metadata else None,
                                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                                    "message_count": conv.message_count
                                }
                                for conv in conversations
                            ]
                            state["conversation_count"] = len(state["conversations"])
                            
                    else:
                        # Handle operation failure
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"Memory operation failed: {error_msg}")
                        
                        # Continue with other operations but note the error
                        if not state.get("error"):
                            state["error"] = error_msg
                            
            # Execute task-specific operations
            if task_type == MemoryManagementTask.MEMORY_CLEANUP:
                await self._execute_cleanup(state)
                
            elif task_type == MemoryManagementTask.MEMORY_ANALYTICS:
                await self._execute_analytics(state)
                
            elif task_type == MemoryManagementTask.MEMORY_OPTIMIZATION:
                await self._execute_optimization(state)
                
            elif task_type == MemoryManagementTask.EXPORT_MEMORY:
                await self._execute_export(state)
                
            elif task_type == MemoryManagementTask.IMPORT_MEMORY:
                await self._execute_import(state)
                
            elif task_type == MemoryManagementTask.MEMORY_BACKUP:
                await self._execute_backup(state)
                
            elif task_type == MemoryManagementTask.MEMORY_RESTORE:
                await self._execute_restore(state)
                
            # Mark as successful if no critical errors
            if not state.get("error"):
                state["success"] = True
                
            return state
            
        except Exception as e:
            logger.error(f"Error executing operations: {str(e)}")
            state["error"] = f"Operation execution error: {str(e)}"
            return state
            
    async def _execute_cleanup(self, state: MemoryManagerState):
        """Execute memory cleanup operations"""
        try:
            cleanup_policy = state.get("cleanup_policy", MemoryCleanupPolicy.TIME_BASED)
            cleanup_criteria = state.get("cleanup_criteria", {})
            conversations = state.get("conversations", [])
            
            deleted_count = 0
            deleted_conversations = []
            
            if cleanup_policy == MemoryCleanupPolicy.TIME_BASED:
                # Clean up conversations older than specified days
                days_threshold = cleanup_criteria.get("days", 30)
                cutoff_date = datetime.now() - timedelta(days=days_threshold)
                
                for conv in conversations:
                    if conv.get("created_at"):
                        created_at = datetime.fromisoformat(conv["created_at"])
                        if created_at < cutoff_date:
                            # Delete this conversation
                            delete_op = {
                                "operation": MemoryOperation.DELETE_CONVERSATION,
                                "conversation_id": conv["conversation_id"]
                            }
                            result = await memory_workflow.run(delete_op)
                            
                            if result.get("success"):
                                deleted_count += 1
                                deleted_conversations.append(conv["conversation_id"])
                                
            elif cleanup_policy == MemoryCleanupPolicy.SIZE_BASED:
                # Clean up oldest conversations when exceeding size limit
                max_conversations = cleanup_criteria.get("max_conversations", 1000)
                
                if len(conversations) > max_conversations:
                    # Sort by created_at (oldest first)
                    conversations.sort(key=lambda x: x.get("created_at", ""))
                    
                    # Delete excess conversations
                    for conv in conversations[:len(conversations) - max_conversations]:
                        delete_op = {
                            "operation": MemoryOperation.DELETE_CONVERSATION,
                            "conversation_id": conv["conversation_id"]
                        }
                        result = await memory_workflow.run(delete_op)
                        
                        if result.get("success"):
                            deleted_count += 1
                            deleted_conversations.append(conv["conversation_id"])
                            
            elif cleanup_policy == MemoryCleanupPolicy.USAGE_BASED:
                # Clean up least recently used conversations
                max_conversations = cleanup_criteria.get("max_conversations", 1000)
                
                if len(conversations) > max_conversations:
                    # Sort by updated_at (oldest first)
                    conversations.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")))
                    
                    # Delete excess conversations
                    for conv in conversations[:len(conversations) - max_conversations]:
                        delete_op = {
                            "operation": MemoryOperation.DELETE_CONVERSATION,
                            "conversation_id": conv["conversation_id"]
                        }
                        result = await memory_workflow.run(delete_op)
                        
                        if result.get("success"):
                            deleted_count += 1
                            deleted_conversations.append(conv["conversation_id"])
                            
            elif cleanup_policy == MemoryCleanupPolicy.MANUAL:
                # Manual cleanup with specific conversation IDs
                conversation_ids = cleanup_criteria.get("conversation_ids", [])
                
                for conv_id in conversation_ids:
                    delete_op = {
                        "operation": MemoryOperation.DELETE_CONVERSATION,
                        "conversation_id": conv_id
                    }
                    result = await memory_workflow.run(delete_op)
                    
                    if result.get("success"):
                        deleted_count += 1
                        deleted_conversations.append(conv_id)
                        
            state["cleanup_results"] = {
                "policy": cleanup_policy.value,
                "deleted_count": deleted_count,
                "deleted_conversations": deleted_conversations
            }
            
        except Exception as e:
            logger.error(f"Error executing cleanup: {str(e)}")
            state["error"] = f"Cleanup execution error: {str(e)}"
            
    async def _execute_analytics(self, state: MemoryManagerState):
        """Execute memory analytics operations"""
        try:
            conversations = state.get("conversations", [])
            memory_stats = state.get("memory_stats", {})
            
            # Calculate analytics
            analytics = {
                "conversation_stats": {
                    "total_conversations": len(conversations),
                    "total_messages": sum(conv.get("message_count", 0) for conv in conversations),
                    "average_messages_per_conversation": 0
                },
                "agent_distribution": {},
                "time_distribution": {
                    "last_day": 0,
                    "last_week": 0,
                    "last_month": 0,
                    "older": 0
                },
                "memory_usage": memory_stats
            }
            
            # Calculate average messages per conversation
            if conversations:
                analytics["conversation_stats"]["average_messages_per_conversation"] = (
                    analytics["conversation_stats"]["total_messages"] / len(conversations)
                )
                
            # Calculate agent distribution
            for conv in conversations:
                agent_name = conv.get("agent_name", "unknown")
                analytics["agent_distribution"][agent_name] = (
                    analytics["agent_distribution"].get(agent_name, 0) + 1
                )
                
            # Calculate time distribution
            now = datetime.now()
            for conv in conversations:
                updated_at = conv.get("updated_at")
                if updated_at:
                    updated_dt = datetime.fromisoformat(updated_at)
                    days_diff = (now - updated_dt).days
                    
                    if days_diff <= 1:
                        analytics["time_distribution"]["last_day"] += 1
                    elif days_diff <= 7:
                        analytics["time_distribution"]["last_week"] += 1
                    elif days_diff <= 30:
                        analytics["time_distribution"]["last_month"] += 1
                    else:
                        analytics["time_distribution"]["older"] += 1
                        
            state["analytics_data"] = analytics
            
        except Exception as e:
            logger.error(f"Error executing analytics: {str(e)}")
            state["error"] = f"Analytics execution error: {str(e)}"
            
    async def _execute_optimization(self, state: MemoryManagerState):
        """Execute memory optimization operations"""
        try:
            memory_stats = state.get("memory_stats", {})
            
            # Analyze memory usage and suggest optimizations
            optimizations = []
            
            # Check for optimization opportunities
            if memory_stats.get("total_conversations", 0) > 1000:
                optimizations.append({
                    "type": "cleanup",
                    "description": "Consider cleaning up old conversations",
                    "suggestion": "Run memory cleanup with time-based policy"
                })
                
            if memory_stats.get("total_messages", 0) > 10000:
                optimizations.append({
                    "type": "summarization",
                    "description": "Consider summarizing long conversations",
                    "suggestion": "Generate summaries for conversations with many messages"
                })
                
            # Get LLM for optimization suggestions
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            if llm and optimizations:
                # Generate optimization recommendations
                optimization_text = "\n".join([
                    f"- {opt['description']}: {opt['suggestion']}"
                    for opt in optimizations
                ])
                
                prompt = f"""Based on the following memory statistics and optimization opportunities, provide detailed recommendations:

Memory Stats: {memory_stats}

Optimization Opportunities:
{optimization_text}

Please provide specific, actionable recommendations for optimizing memory usage."""
                
                response = await llm.ainvoke(prompt)
                
                state["optimization_results"] = {
                    "opportunities": optimizations,
                    "recommendations": response.content,
                    "analyzed_at": datetime.now().isoformat()
                }
            else:
                state["optimization_results"] = {
                    "opportunities": optimizations,
                    "recommendations": "No specific optimizations needed at this time.",
                    "analyzed_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error executing optimization: {str(e)}")
            state["error"] = f"Optimization execution error: {str(e)}"
            
    async def _execute_export(self, state: MemoryManagerState):
        """Execute memory export operations"""
        try:
            conversations = state.get("conversations", [])
            export_format = state.get("export_format", "json")
            
            # Prepare export data
            export_data = {
                "export_info": {
                    "format": export_format,
                    "exported_at": datetime.now().isoformat(),
                    "conversation_count": len(conversations)
                },
                "conversations": []
            }
            
            # Get full conversation data for each conversation
            for conv in conversations:
                conv_id = conv["conversation_id"]
                
                # Get conversation messages
                messages_op = {
                    "operation": MemoryOperation.GET_MESSAGES,
                    "conversation_id": conv_id
                }
                messages_result = await memory_workflow.run(messages_op)
                
                # Get conversation info
                info_op = {
                    "operation": MemoryOperation.GET_CONVERSATION_INFO,
                    "conversation_id": conv_id
                }
                info_result = await memory_workflow.run(info_op)
                
                conversation_data = {
                    "conversation_info": conv,
                    "messages": messages_result.get("result", {}).get("messages", []),
                    "detailed_info": info_result.get("result", {}).get("info")
                }
                
                export_data["conversations"].append(conversation_data)
                
            state["export_data"] = export_data
            
        except Exception as e:
            logger.error(f"Error executing export: {str(e)}")
            state["error"] = f"Export execution error: {str(e)}"
            
    async def _execute_import(self, state: MemoryManagerState):
        """Execute memory import operations"""
        try:
            import_data = state.get("import_data", {})
            conversations = import_data.get("conversations", [])
            
            imported_count = 0
            import_errors = []
            
            for conv_data in conversations:
                try:
                    conv_info = conv_data.get("conversation_info", {})
                    messages = conv_data.get("messages", [])
                    
                    # Create conversation
                    conv_id = conv_info.get("conversation_id")
                    if not conv_id:
                        import_errors.append("Missing conversation ID")
                        continue
                        
                    create_op = {
                        "operation": MemoryOperation.CREATE_CONVERSATION,
                        "conversation_id": conv_id,
                        "agent_name": conv_info.get("agent_name"),
                        "metadata": conv_info.get("metadata", {})
                    }
                    
                    # Add title if available
                    if conv_info.get("title"):
                        create_op["metadata"]["title"] = conv_info["title"]
                        
                    create_result = await memory_workflow.run(create_op)
                    
                    if create_result.get("success"):
                        # Add messages
                        for msg in messages:
                            add_msg_op = {
                                "operation": MemoryOperation.ADD_MESSAGE,
                                "conversation_id": conv_id,
                                "role": msg.get("role", "human"),
                                "content": msg.get("content", ""),
                                "metadata": msg.get("metadata", {})
                            }
                            
                            await memory_workflow.run(add_msg_op)
                            
                        imported_count += 1
                    else:
                        import_errors.append(f"Failed to create conversation {conv_id}")
                        
                except Exception as e:
                    import_errors.append(f"Error importing conversation: {str(e)}")
                    
            state["import_results"] = {
                "imported_count": imported_count,
                "import_errors": import_errors
            }
            
        except Exception as e:
            logger.error(f"Error executing import: {str(e)}")
            state["error"] = f"Import execution error: {str(e)}"
            
    async def _execute_backup(self, state: MemoryManagerState):
        """Execute memory backup operations"""
        try:
            conversations = state.get("conversations", [])
            
            # Create backup data
            backup_data = {
                "backup_info": {
                    "backup_type": "full",
                    "created_at": datetime.now().isoformat(),
                    "conversation_count": len(conversations),
                    "version": "1.0"
                },
                "conversations": conversations
            }
            
            state["backup_data"] = backup_data
            
        except Exception as e:
            logger.error(f"Error executing backup: {str(e)}")
            state["error"] = f"Backup execution error: {str(e)}"
            
    async def _execute_restore(self, state: MemoryManagerState):
        """Execute memory restore operations"""
        try:
            backup_data = state.get("backup_data", {})
            conversations = backup_data.get("conversations", [])
            
            restored_count = 0
            restore_errors = []
            
            for conv in conversations:
                try:
                    conv_id = conv.get("conversation_id")
                    if not conv_id:
                        restore_errors.append("Missing conversation ID")
                        continue
                        
                    # Check if conversation already exists
                    info_op = {
                        "operation": MemoryOperation.GET_CONVERSATION_INFO,
                        "conversation_id": conv_id
                    }
                    info_result = await memory_workflow.run(info_op)
                    
                    if info_result.get("success") and info_result.get("result", {}).get("info"):
                        # Conversation already exists, skip or update
                        continue
                        
                    # Create conversation
                    create_op = {
                        "operation": MemoryOperation.CREATE_CONVERSATION,
                        "conversation_id": conv_id,
                        "agent_name": conv.get("agent_name"),
                        "metadata": conv.get("metadata", {})
                    }
                    
                    if conv.get("title"):
                        create_op["metadata"]["title"] = conv["title"]
                        
                    create_result = await memory_workflow.run(create_op)
                    
                    if create_result.get("success"):
                        restored_count += 1
                    else:
                        restore_errors.append(f"Failed to restore conversation {conv_id}")
                        
                except Exception as e:
                    restore_errors.append(f"Error restoring conversation: {str(e)}")
                    
            state["restore_results"] = {
                "restored_count": restored_count,
                "restore_errors": restore_errors
            }
            
        except Exception as e:
            logger.error(f"Error executing restore: {str(e)}")
            state["error"] = f"Restore execution error: {str(e)}"
            
    async def _process_results(self, state: MemoryManagerState) -> MemoryManagerState:
        """Process the final results"""
        try:
            task_type = state.get("task_type")
            state["step_count"] += 1
            
            # Create result based on task type
            if task_type == MemoryManagementTask.CREATE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "created": state.get("success", False),
                    "title": state.get("conversation_title")
                }
                
            elif task_type == MemoryManagementTask.DELETE_CONVERSATION:
                state["result"] = {
                    "conversation_id": state.get("conversation_id"),
                    "deleted": state.get("success", False)
                }
                
            elif task_type == MemoryManagementTask.LIST_CONVERSATIONS:
                state["result"] = {
                    "conversations": state.get("conversations", []),
                    "count": state.get("conversation_count", 0)
                }
                
            elif task_type == MemoryManagementTask.MEMORY_CLEANUP:
                state["result"] = {
                    "cleanup_results": state.get("cleanup_results", {}),
                    "memory_stats_after": state.get("memory_stats", {})
                }
                
            elif task_type == MemoryManagementTask.MEMORY_ANALYTICS:
                state["result"] = {
                    "analytics": state.get("analytics_data", {}),
                    "memory_stats": state.get("memory_stats", {})
                }
                
            elif task_type == MemoryManagementTask.MEMORY_OPTIMIZATION:
                state["result"] = {
                    "optimization_results": state.get("optimization_results", {}),
                    "memory_stats": state.get("memory_stats", {})
                }
                
            elif task_type == MemoryManagementTask.EXPORT_MEMORY:
                state["result"] = {
                    "export_data": state.get("export_data"),
                    "format": state.get("export_format", "json")
                }
                
            elif task_type == MemoryManagementTask.IMPORT_MEMORY:
                state["result"] = {
                    "import_results": state.get("import_results", {})
                }
                
            elif task_type == MemoryManagementTask.MEMORY_BACKUP:
                state["result"] = {
                    "backup_data": state.get("backup_data")
                }
                
            elif task_type == MemoryManagementTask.MEMORY_RESTORE:
                state["result"] = {
                    "restore_results": state.get("restore_results", {})
                }
                
            return state
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            state["error"] = f"Result processing error: {str(e)}"
            return state
            
    async def _handle_error(self, state: MemoryManagerState) -> MemoryManagerState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        logger.error(f"Memory manager error: {error}")
        
        state["success"] = False
        state["result"] = {"error": error}
        
        return state
        
    def _should_continue(self, state: MemoryManagerState) -> str:
        """Determine if workflow should continue"""
        if state.get("error"):
            return "error"
        return "continue"
        
    def _should_execute(self, state: MemoryManagerState) -> str:
        """Determine if operations should be executed"""
        if state.get("error"):
            return "error"
        return "execute"
        
    def _should_process(self, state: MemoryManagerState) -> str:
        """Determine if results should be processed"""
        if state.get("error"):
            return "error"
        return "process"
        
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the memory manager workflow"""
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
            logger.error(f"Error running memory manager: {str(e)}")
            return {
                "success": False,
                "error": f"Agent error: {str(e)}",
                "result": None
            }
            
    async def create_conversation(
        self,
        conversation_id: str,
        agent_name: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation"""
        state = {
            "task_type": MemoryManagementTask.CREATE_CONVERSATION,
            "conversation_id": conversation_id,
            "agent_name": agent_name,
            "conversation_title": title,
            "metadata": metadata
        }
        
        return await self.run(state)
        
    async def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation"""
        state = {
            "task_type": MemoryManagementTask.DELETE_CONVERSATION,
            "conversation_id": conversation_id
        }
        
        return await self.run(state)
        
    async def list_conversations(
        self,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """List conversations"""
        state = {
            "task_type": MemoryManagementTask.LIST_CONVERSATIONS,
            "agent_name": agent_name
        }
        
        return await self.run(state)
        
    async def cleanup_memory(
        self,
        cleanup_policy: MemoryCleanupPolicy = MemoryCleanupPolicy.TIME_BASED,
        cleanup_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clean up memory based on policy"""
        state = {
            "task_type": MemoryManagementTask.MEMORY_CLEANUP,
            "cleanup_policy": cleanup_policy,
            "cleanup_criteria": cleanup_criteria or {}
        }
        
        return await self.run(state)
        
    async def get_memory_analytics(self) -> Dict[str, Any]:
        """Get memory analytics"""
        state = {
            "task_type": MemoryManagementTask.MEMORY_ANALYTICS
        }
        
        return await self.run(state)
        
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        state = {
            "task_type": MemoryManagementTask.MEMORY_OPTIMIZATION
        }
        
        return await self.run(state)
        
    async def export_memory(
        self,
        export_format: str = "json",
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export memory data"""
        state = {
            "task_type": MemoryManagementTask.EXPORT_MEMORY,
            "export_format": export_format,
            "agent_name": agent_name
        }
        
        return await self.run(state)
        
    async def import_memory(
        self,
        import_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Import memory data"""
        state = {
            "task_type": MemoryManagementTask.IMPORT_MEMORY,
            "import_data": import_data
        }
        
        return await self.run(state)
        
    async def backup_memory(self) -> Dict[str, Any]:
        """Backup memory data"""
        state = {
            "task_type": MemoryManagementTask.MEMORY_BACKUP
        }
        
        return await self.run(state)
        
    async def restore_memory(
        self,
        backup_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Restore memory data from backup"""
        state = {
            "task_type": MemoryManagementTask.MEMORY_RESTORE,
            "backup_data": backup_data
        }
        
        return await self.run(state)


# Global memory manager agent instance
memory_manager_agent = LangGraphMemoryManagerAgent()