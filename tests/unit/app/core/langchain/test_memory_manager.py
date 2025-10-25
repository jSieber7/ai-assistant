"""
Unit tests for LangChain Memory Manager.

This module tests conversation management, message handling,
and integration with LangChain memory components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.memory import BaseMemory

from app.core.langchain.memory_manager import LangChainMemoryManager, MemoryType, ConversationInfo
from app.core.secure_settings import secure_settings


class TestLangChainMemoryManager:
    """Test cases for LangChain Memory Manager"""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create a LangChain Memory Manager instance for testing"""
        manager = LangChainMemoryManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('database', 'host'): 'localhost',
            ('database', 'port'): '5432',
            ('database', 'name'): 'test_db',
            ('database', 'user'): 'test_user',
            ('database', 'password'): 'test_password',
            ('llm_providers', 'openai', 'api_key'): 'test-openai-key',
        }.get((section, key), default)
        return mock_settings
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock database client for testing"""
        db_client = Mock()
        db_client.create_conversation = AsyncMock(return_value="conv_123")
        db_client.add_chat_message = AsyncMock()
        db_client.get_chat_messages = AsyncMock(return_value=[])
        db_client.get_conversation = AsyncMock(return_value=None)
        db_client.list_conversations = AsyncMock(return_value=[])
        db_client.delete_conversation = AsyncMock(return_value=True)
        db_client.update_conversation = AsyncMock()
        return db_client
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of memory manager"""
        with patch('app.core.langchain.memory_manager.secure_settings', mock_settings):
            with patch('app.core.langchain.memory_manager.get_langchain_client') as mock_get_client:
                mock_get_client.return_value = Mock()
                
                manager = LangChainMemoryManager()
                
                # Test initialization
                await manager.initialize()
                
                # Verify initialization
                assert manager._initialized is True
                assert manager._monitoring is not None
                assert manager._db_client is not None
    
    async def test_initialize_already_initialized(self, memory_manager):
        """Test that initialize doesn't re-initialize already initialized manager"""
        # Get initial state
        initial_monitoring = memory_manager._monitoring
        
        # Call initialize again
        await memory_manager.initialize()
        
        # Verify same monitoring instance is used
        assert memory_manager._monitoring is initial_monitoring
    
    async def test_create_conversation_success(self, memory_manager, mock_db_client):
        """Test successful conversation creation"""
        # Mock database client
        memory_manager._db_client = mock_db_client
        
        # Create conversation
        result = await memory_manager.create_conversation(
            conversation_id="test_conv",
            agent_name="test_agent",
            title="Test Conversation",
            metadata={"test": "data"}
        )
        
        # Verify creation
        assert result is True
        mock_db_client.create_conversation.assert_called_once_with(
            conversation_id="test_conv",
            agent_name="test_agent",
            title="Test Conversation",
            metadata={"test": "data"}
        )
        
        # Verify conversation is in cache
        assert "test_conv" in memory_manager._conversation_cache
    
    async def test_create_conversation_with_system_message(self, memory_manager, mock_db_client):
        """Test conversation creation with system message"""
        # Mock database client and chat history
        memory_manager._db_client = mock_db_client
        memory_manager._chat_histories["test_conv"] = Mock()
        memory_manager._chat_histories["test_conv"].add_message = AsyncMock()
        
        # Create conversation with system message
        await memory_manager.create_conversation(
            conversation_id="test_conv",
            metadata={"system_message": "You are a helpful assistant"}
        )
        
        # Verify system message was added
        memory_manager._chat_histories["test_conv"].add_message.assert_called_once()
        call_args = memory_manager._chat_histories["test_conv"].add_message.call_args[0][0]
        assert isinstance(call_args, SystemMessage)
        assert call_args.content == "You are a helpful assistant"
    
    async def test_add_message_success(self, memory_manager, mock_db_client):
        """Test successful message addition"""
        # Create conversation first
        memory_manager._db_client = mock_db_client
        memory_manager._conversation_cache["test_conv"] = ConversationInfo(
            conversation_id="test_conv",
            message_count=0
        )
        
        # Add message
        result = await memory_manager.add_message(
            conversation_id="test_conv",
            role="human",
            content="Hello, world!",
            metadata={"source": "test"}
        )
        
        # Verify addition
        assert result is True
        mock_db_client.add_chat_message.assert_called_once_with(
            conversation_id="test_conv",
            role="human",
            content="Hello, world!",
            metadata={"source": "test"}
        )
        
        # Verify conversation cache was updated
        assert memory_manager._conversation_cache["test_conv"].message_count == 1
    
    async def test_get_conversation_messages_from_db(self, memory_manager, mock_db_client):
        """Test getting messages from database"""
        # Mock database response
        mock_messages = [
            {"role": "human", "content": "Hello"},
            {"role": "ai", "content": "Hi there!"}
        ]
        mock_db_client.get_chat_messages.return_value = mock_messages
        
        memory_manager._db_client = mock_db_client
        
        # Get messages
        messages = await memory_manager.get_conversation_messages("test_conv")
        
        # Verify messages from database
        assert messages == mock_messages
        mock_db_client.get_chat_messages.assert_called_once_with(
            conversation_id="test_conv",
            limit=None
        )
    
    async def test_get_conversation_messages_with_limit(self, memory_manager, mock_db_client):
        """Test getting messages with limit"""
        memory_manager._db_client = mock_db_client
        
        # Get messages with limit
        await memory_manager.get_conversation_messages("test_conv", limit=10)
        
        # Verify limit was passed
        mock_db_client.get_chat_messages.assert_called_once_with(
            conversation_id="test_conv",
            limit=10
        )
    
    async def test_get_conversation_info_from_db(self, memory_manager, mock_db_client):
        """Test getting conversation info from database"""
        # Mock database response
        mock_conv = {
            "conversation_id": "test_conv",
            "agent_name": "test_agent",
            "title": "Test Conversation",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "message_count": 5,
            "metadata": {"test": "data"}
        }
        mock_db_client.get_conversation.return_value = mock_conv
        
        memory_manager._db_client = mock_db_client
        
        # Get conversation info
        info = await memory_manager.get_conversation_info("test_conv")
        
        # Verify info
        assert info is not None
        assert info.conversation_id == "test_conv"
        assert info.agent_name == "test_agent"
        assert info.title == "Test Conversation"
        assert info.message_count == 5
        assert info.metadata == {"test": "data"}
    
    async def test_get_conversation_info_from_cache(self, memory_manager):
        """Test getting conversation info from cache"""
        # Add to cache
        cached_info = ConversationInfo(
            conversation_id="test_conv",
            agent_name="test_agent",
            title="Test Conversation"
        )
        memory_manager._conversation_cache["test_conv"] = cached_info
        
        # Get info (should use cache)
        info = await memory_manager.get_conversation_info("test_conv")
        
        # Verify cached info is returned
        assert info is cached_info
    
    async def test_list_conversations_from_db(self, memory_manager, mock_db_client):
        """Test listing conversations from database"""
        # Mock database response
        mock_convs = [
            {
                "conversation_id": "conv_1",
                "agent_name": "agent_1",
                "title": "Conversation 1",
                "message_count": 3
            },
            {
                "conversation_id": "conv_2",
                "agent_name": "agent_2",
                "title": "Conversation 2",
                "message_count": 5
            }
        ]
        mock_db_client.list_conversations.return_value = mock_convs
        
        memory_manager._db_client = mock_db_client
        
        # List conversations
        conversations = await memory_manager.list_conversations()
        
        # Verify conversations
        assert len(conversations) == 2
        assert conversations[0].conversation_id == "conv_1"
        assert conversations[1].conversation_id == "conv_2"
    
    async def test_list_conversations_with_filter(self, memory_manager, mock_db_client):
        """Test listing conversations with agent filter"""
        memory_manager._db_client = mock_db_client
        
        # List conversations with agent filter
        await memory_manager.list_conversations(agent_name="test_agent", limit=10)
        
        # Verify filter was passed
        mock_db_client.list_conversations.assert_called_once_with(
            agent_name="test_agent",
            limit=10
        )
    
    async def test_delete_conversation_success(self, memory_manager, mock_db_client):
        """Test successful conversation deletion"""
        # Add to cache and chat histories
        memory_manager._conversation_cache["test_conv"] = ConversationInfo(
            conversation_id="test_conv"
        )
        memory_manager._chat_histories["test_conv"] = Mock()
        memory_manager._chat_histories["test_conv"].aclear = AsyncMock()
        
        memory_manager._db_client = mock_db_client
        
        # Delete conversation
        result = await memory_manager.delete_conversation("test_conv")
        
        # Verify deletion
        assert result is True
        mock_db_client.delete_conversation.assert_called_once_with(
            conversation_id="test_conv"
        )
        
        # Verify cache was cleared
        assert "test_conv" not in memory_manager._conversation_cache
        assert "test_conv" not in memory_manager._chat_histories
    
    async def test_get_memory_backend(self, memory_manager):
        """Test getting memory backend"""
        # Test getting conversation memory
        backend = await memory_manager.get_memory_backend(MemoryType.CONVERSATION)
        assert isinstance(backend, BaseMemory)
        
        # Test getting non-existent backend
        backend = await memory_manager.get_memory_backend(MemoryType.VECTOR)
        assert backend is None
    
    async def test_get_memory_stats(self, memory_manager):
        """Test getting memory statistics"""
        # Add some test data
        memory_manager._conversation_cache["conv_1"] = ConversationInfo(
            conversation_id="conv_1",
            message_count=5
        )
        memory_manager._conversation_cache["conv_2"] = ConversationInfo(
            conversation_id="conv_2",
            message_count=3
        )
        memory_manager._memory_backends["conversation"] = Mock()
        memory_manager._chat_histories["conv_1"] = Mock()
        memory_manager._chat_histories["conv_2"] = Mock()
        
        # Get stats
        stats = await memory_manager.get_memory_stats()
        
        # Verify stats
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 8
        assert "conversation" in stats["active_backends"]
        assert stats["chat_histories"] == 2
    
    async def test_clear_cache(self, memory_manager):
        """Test clearing conversation cache"""
        # Add some test data
        memory_manager._conversation_cache["conv_1"] = ConversationInfo(
            conversation_id="conv_1"
        )
        memory_manager._conversation_cache["conv_2"] = ConversationInfo(
            conversation_id="conv_2"
        )
        
        # Clear cache
        await memory_manager.clear_cache()
        
        # Verify cache is cleared
        assert len(memory_manager._conversation_cache) == 0
    
    async def test_cleanup_old_conversations(self, memory_manager):
        """Test cleaning up old conversations"""
        # Add old and new conversations
        old_time = datetime.now().timestamp() - (40 * 24 * 60 * 60)  # 40 days ago
        new_time = datetime.now().timestamp() - (10 * 24 * 60 * 60)  # 10 days ago
        
        old_conv = ConversationInfo(
            conversation_id="old_conv",
            created_at=datetime.fromtimestamp(old_time)
        )
        new_conv = ConversationInfo(
            conversation_id="new_conv",
            created_at=datetime.fromtimestamp(new_time)
        )
        
        memory_manager._conversation_cache["old_conv"] = old_conv
        memory_manager._conversation_cache["new_conv"] = new_conv
        
        # Mock delete_conversation
        memory_manager.delete_conversation = AsyncMock(return_value=True)
        
        # Cleanup conversations older than 30 days
        deleted_count = await memory_manager.cleanup_old_conversations(days=30)
        
        # Verify only old conversation was deleted
        assert deleted_count == 1
        memory_manager.delete_conversation.assert_called_once_with("old_conv")
    
    async def test_monitoring_integration(self, memory_manager, mock_db_client):
        """Test that monitoring is properly integrated"""
        memory_manager._db_client = mock_db_client
        
        # Mock monitoring to track calls
        memory_manager._monitoring.record_metric = AsyncMock()
        
        # Create conversation and add message
        await memory_manager.create_conversation("test_conv")
        await memory_manager.add_message("test_conv", "human", "Hello")
        
        # Verify monitoring was called
        assert memory_manager._monitoring.record_metric.call_count >= 2
    
    async def test_error_handling_during_creation(self, memory_manager, mock_db_client):
        """Test error handling during conversation creation"""
        # Mock database to raise error
        mock_db_client.create_conversation.side_effect = Exception("Database error")
        memory_manager._db_client = mock_db_client
        
        # Create conversation should handle error
        result = await memory_manager.create_conversation("test_conv")
        assert result is False
    
    async def test_error_handling_during_message_addition(self, memory_manager, mock_db_client):
        """Test error handling during message addition"""
        # Mock database to raise error
        mock_db_client.add_chat_message.side_effect = Exception("Database error")
        memory_manager._db_client = mock_db_client
        
        # Add message should handle error
        result = await memory_manager.add_message("test_conv", "human", "Hello")
        assert result is False
    
    async def test_get_message_role(self, memory_manager):
        """Test message role detection"""
        # Test different message types
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        system_msg = SystemMessage(content="System message")
        
        assert memory_manager._get_message_role(human_msg) == "human"
        assert memory_manager._get_message_role(ai_msg) == "ai"
        assert memory_manager._get_message_role(system_msg) == "system"
        
        # Test unknown message type
        unknown_msg = Mock()
        unknown_msg.__class__.__name__ = "UnknownMessage"
        assert memory_manager._get_message_role(unknown_msg) == "unknown"
    
    async def test_shutdown(self, memory_manager):
        """Test shutdown functionality"""
        # Mock monitoring shutdown
        memory_manager._monitoring.shutdown = AsyncMock()
        
        await memory_manager.shutdown()
        
        # Verify shutdown was called
        memory_manager._monitoring.shutdown.assert_called_once()
        
        # Verify manager is marked as not initialized
        assert memory_manager._initialized is False
    
    async def test_concurrent_operations(self, memory_manager, mock_db_client):
        """Test concurrent memory operations"""
        memory_manager._db_client = mock_db_client
        
        # Create conversation
        await memory_manager.create_conversation("test_conv")
        
        # Add messages concurrently
        tasks = [
            memory_manager.add_message("test_conv", "human", f"Message {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all additions succeeded
        assert all(results)
        assert mock_db_client.add_chat_message.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__])