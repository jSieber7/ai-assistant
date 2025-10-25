"""
LangChain-based Memory Manager for AI Assistant.

This module provides comprehensive memory management using LangChain's memory components,
supporting multiple backends with PostgreSQL persistence.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from langchain_core.memory import BaseMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import PostgreSQLChatMessageHistory
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.secure_settings import secure_settings

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Supported memory types"""
    
    CONVERSATION = "conversation"
    SUMMARY = "summary"
    VECTOR = "vector"
    BUFFER = "buffer"
    TOKEN_BUFFER = "token_buffer"


@dataclass
class ConversationInfo:
    """Information about a conversation"""
    
    conversation_id: str
    agent_name: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MessageInfo:
    """Information about a message"""
    
    conversation_id: str
    role: str
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LangChainMemoryManager:
    """
    Comprehensive memory manager using LangChain's memory components.
    
    This manager provides:
    - Multiple memory backends (conversation, summary, vector)
    - PostgreSQL persistence
    - LangChain-compatible memory interfaces
    - Conversation lifecycle management
    - Memory search and retrieval
    """
    
    def __init__(self):
        self._memory_backends: Dict[str, BaseMemory] = {}
        self._chat_histories: Dict[str, PostgreSQLChatMessageHistory] = {}
        self._vector_stores: Dict[str, PGVector] = {}
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._conversation_cache: Dict[str, ConversationInfo] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the memory manager"""
        if self._initialized:
            return
            
        logger.info("Initializing LangChain Memory Manager...")
        
        # Initialize embeddings
        await self._initialize_embeddings()
        
        # Initialize conversation memory
        await self._initialize_conversation_memory()
        
        # Initialize summary memory
        await self._initialize_summary_memory()
        
        # Initialize vector memory
        await self._initialize_vector_memory()
        
        self._initialized = True
        logger.info("LangChain Memory Manager initialized successfully")
        
    async def _initialize_embeddings(self):
        """Initialize embeddings for vector memory"""
        try:
            # Try to get OpenAI API key for embeddings
            api_key = secure_settings.get_setting("llm_providers", "openai", "api_key")
            if not api_key:
                api_key = secure_settings.get_setting("llm_providers", "openai_compatible", "api_key")
                
            if api_key:
                self._embeddings = OpenAIEmbeddings(
                    api_key=api_key,
                    model="text-embedding-ada-002"
                )
                logger.info("Initialized OpenAI embeddings")
            else:
                logger.warning("No API key found for embeddings, vector memory will be disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            
    async def _initialize_conversation_memory(self):
        """Initialize conversation memory backend"""
        try:
            # Get PostgreSQL connection string
            connection_string = await self._get_postgres_connection_string()
            
            if connection_string:
                # Create chat message history
                chat_history = PostgreSQLChatMessageHistory(
                    connection_string=connection_string,
                    session_id="default"  # Will be overridden per conversation
                )
                
                # Create conversation buffer memory
                conversation_memory = ConversationBufferMemory(
                    chat_memory=chat_history,
                    return_messages=True,
                    memory_key="chat_history",
                    human_prefix="Human",
                    ai_prefix="Assistant"
                )
                
                self._memory_backends[MemoryType.CONVERSATION.value] = conversation_memory
                self._chat_histories["default"] = chat_history
                
                logger.info("Initialized conversation memory backend")
            else:
                logger.warning("No PostgreSQL connection available, conversation memory disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize conversation memory: {str(e)}")
            
    async def _initialize_summary_memory(self):
        """Initialize summary memory backend"""
        try:
            # Get LLM for summarization
            from .llm_manager import llm_manager
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            if llm:
                # Create summary memory
                summary_memory = ConversationSummaryMemory(
                    llm=llm,
                    return_messages=True,
                    memory_key="chat_history",
                    human_prefix="Human",
                    ai_prefix="Assistant"
                )
                
                self._memory_backends[MemoryType.SUMMARY.value] = summary_memory
                
                logger.info("Initialized summary memory backend")
            else:
                logger.warning("No LLM available, summary memory disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize summary memory: {str(e)}")
            
    async def _initialize_vector_memory(self):
        """Initialize vector memory backend"""
        try:
            if not self._embeddings:
                logger.warning("No embeddings available, vector memory disabled")
                return
                
            # Get PostgreSQL connection string
            connection_string = await self._get_postgres_connection_string()
            
            if connection_string:
                # Create vector store
                vector_store = PGVector(
                    connection_string=connection_string,
                    embedding_function=self._embeddings,
                    collection_name="conversation_embeddings"
                )
                
                self._vector_stores["default"] = vector_store
                
                logger.info("Initialized vector memory backend")
            else:
                logger.warning("No PostgreSQL connection available, vector memory disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {str(e)}")
            
    async def _get_postgres_connection_string(self) -> Optional[str]:
        """Get PostgreSQL connection string from settings"""
        try:
            # Try to get from secure settings first
            db_host = secure_settings.get_setting("database", "host")
            db_port = secure_settings.get_setting("database", "port")
            db_name = secure_settings.get_setting("database", "name")
            db_user = secure_settings.get_setting("database", "user")
            db_password = secure_settings.get_setting("database", "password")
            
            if all([db_host, db_port, db_name, db_user, db_password]):
                return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                
            # Fallback to environment variables
            import os
            return os.getenv("DATABASE_URL")
            
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL connection string: {str(e)}")
            return None
            
    async def create_conversation(
        self,
        conversation_id: str,
        agent_name: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            agent_name: Name of the agent
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store conversation info
            conversation_info = ConversationInfo(
                conversation_id=conversation_id,
                agent_name=agent_name,
                title=title,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata=metadata or {}
            )
            
            self._conversation_cache[conversation_id] = conversation_info
            
            # Create chat history for this conversation
            if MemoryType.CONVERSATION.value in self._memory_backends:
                connection_string = await self._get_postgres_connection_string()
                if connection_string:
                    chat_history = PostgreSQLChatMessageHistory(
                        connection_string=connection_string,
                        session_id=conversation_id
                    )
                    self._chat_histories[conversation_id] = chat_history
                    
                    # Store initial system message if provided
                    if metadata and "system_message" in metadata:
                        await chat_history.add_message(
                            SystemMessage(content=metadata["system_message"])
                        )
                        
            logger.info(f"Created conversation '{conversation_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create conversation '{conversation_id}': {str(e)}")
            return False
            
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            role: Message role (human, ai, system)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update conversation info
            if conversation_id in self._conversation_cache:
                self._conversation_cache[conversation_id].updated_at = datetime.now()
                self._conversation_cache[conversation_id].message_count += 1
                
            # Add to chat history
            if conversation_id in self._chat_histories:
                chat_history = self._chat_histories[conversation_id]
                
                # Create appropriate message type
                if role == "human":
                    message = HumanMessage(content=content)
                elif role == "ai":
                    message = AIMessage(content=content)
                elif role == "system":
                    message = SystemMessage(content=content)
                else:
                    message = HumanMessage(content=content)  # Fallback
                    
                await chat_history.add_message(message)
                
                # Add to vector store if available
                if self._embeddings and "default" in self._vector_stores:
                    vector_store = self._vector_stores["default"]
                    await vector_store.aadd_texts(
                        texts=[content],
                        metadatas=[{
                            "conversation_id": conversation_id,
                            "role": role,
                            "timestamp": datetime.now().isoformat(),
                            **(metadata or {})
                        }]
                    )
                    
            logger.debug(f"Added {role} message to conversation '{conversation_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation '{conversation_id}': {str(e)}")
            return False
            
    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        try:
            if conversation_id in self._chat_histories:
                chat_history = self._chat_histories[conversation_id]
                messages = await chat_history.aget_messages()
                
                # Apply limit if specified
                if limit:
                    messages = messages[-limit:]
                    
                # Convert to dictionaries
                result = []
                for message in messages:
                    result.append({
                        "role": self._get_message_role(message),
                        "content": message.content,
                        "timestamp": getattr(message, 'timestamp', None),
                    })
                    
                return result
            return []
            
        except Exception as e:
            logger.error(f"Failed to get messages from conversation '{conversation_id}': {str(e)}")
            return []
            
    def _get_message_role(self, message: BaseMessage) -> str:
        """Get the role of a message"""
        if isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "unknown"
            
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search conversations using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            conversation_id: Limit search to specific conversation
            
        Returns:
            List of search results
        """
        try:
            if not self._embeddings or "default" not in self._vector_stores:
                logger.warning("Vector search not available")
                return []
                
            vector_store = self._vector_stores["default"]
            
            # Create filter for specific conversation if provided
            filter_dict = {}
            if conversation_id:
                filter_dict["conversation_id"] = conversation_id
                
            # Search vector store
            results = await vector_store.asimilarity_search(
                query=query,
                k=limit,
                filter=filter_dict
            )
            
            # Convert results to dictionaries
            search_results = []
            for doc in results:
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0.0),
                })
                
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {str(e)}")
            return []
            
    async def get_conversation_info(self, conversation_id: str) -> Optional[ConversationInfo]:
        """
        Get information about a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            ConversationInfo or None if not found
        """
        return self._conversation_cache.get(conversation_id)
        
    async def list_conversations(
        self,
        agent_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ConversationInfo]:
        """
        List conversations with optional filtering.
        
        Args:
            agent_name: Filter by agent name
            limit: Maximum number of conversations to return
            
        Returns:
            List of ConversationInfo objects
        """
        conversations = list(self._conversation_cache.values())
        
        # Filter by agent name
        if agent_name:
            conversations = [
                conv for conv in conversations 
                if conv.agent_name == agent_name
            ]
            
        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)
        
        # Apply limit
        if limit:
            conversations = conversations[:limit]
            
        return conversations
        
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache
            if conversation_id in self._conversation_cache:
                del self._conversation_cache[conversation_id]
                
            # Remove chat history
            if conversation_id in self._chat_histories:
                chat_history = self._chat_histories[conversation_id]
                await chat_history.aclear()
                del self._chat_histories[conversation_id]
                
            # Remove from vector store
            if self._embeddings and "default" in self._vector_stores:
                vector_store = self._vector_stores["default"]
                # Note: PGVector doesn't have a direct delete by filter method
                # This would need custom SQL implementation
                
            logger.info(f"Deleted conversation '{conversation_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete conversation '{conversation_id}': {str(e)}")
            return False
            
    async def get_memory_backend(self, memory_type: MemoryType) -> Optional[BaseMemory]:
        """
        Get a specific memory backend.
        
        Args:
            memory_type: Type of memory backend
            
        Returns:
            BaseMemory instance or None if not found
        """
        return self._memory_backends.get(memory_type.value)
        
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_conversations = len(self._conversation_cache)
        total_messages = sum(
            conv.message_count for conv in self._conversation_cache.values()
        )
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "active_backends": list(self._memory_backends.keys()),
            "vector_search_available": bool(self._embeddings and "default" in self._vector_stores),
            "chat_histories": len(self._chat_histories),
            "vector_stores": len(self._vector_stores),
        }
        
    async def clear_cache(self):
        """Clear conversation cache"""
        self._conversation_cache.clear()
        logger.info("Conversation cache cleared")
        
    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Clean up old conversations.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of conversations deleted
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            old_conversations = [
                conv_id for conv_id, conv in self._conversation_cache.items()
                if conv.created_at and conv.created_at.timestamp() < cutoff_date
            ]
            
            deleted_count = 0
            for conv_id in old_conversations:
                if await self.delete_conversation(conv_id):
                    deleted_count += 1
                    
            logger.info(f"Cleaned up {deleted_count} old conversations")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {str(e)}")
            return 0


# Global memory manager instance
memory_manager = LangChainMemoryManager()