"""
Unit tests for LangChain Memory Manager.

This module tests the LangChain-based memory manager functionality,
including different memory backends and conversation management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid
from typing import List, Dict, Any

from app.core.langchain.memory_manager import (
    LangChainMemoryManager,
    MemoryType,
    MemoryBackend,
    ConversationInfo,
    MessageInfo,
    memory_manager
)


class TestMemoryType:
    """Test MemoryType enum"""

    def test_memory_type_values(self):
        """Test that MemoryType has expected values"""
        expected_types = [
            "conversation",
            "summary",
            "vector"
        ]
        
        actual_types = [memory_type.value for memory_type in MemoryType]
        assert actual_types == expected_types


class TestMemoryBackend:
    """Test MemoryBackend dataclass"""

    def test_memory_backend_defaults(self):
        """Test MemoryBackend default values"""
        backend = MemoryBackend(
            name="test_backend",
            backend_type=MemoryType.CONVERSATION
        )
        
        assert backend.name == "test_backend"
        assert backend.backend_type == MemoryType.CONVERSATION
        assert backend.enabled is True
        assert backend.config == {}
        assert backend.created_at is not None
        assert backend.last_used is None
        assert backend.metadata == {}

    def test_memory_backend_with_values(self):
        """Test MemoryBackend with provided values"""
        created_at = datetime.now()
        last_used = datetime.now()
        metadata = {"version": "1.0"}
        
        backend = MemoryBackend(
            name="advanced_backend",
            backend_type=MemoryType.VECTOR,
            enabled=False,
            config={"dimension": 1536, "index_type": "hnsw"},
            created_at=created_at,
            last_used=last_used,
            metadata=metadata
        )
        
        assert backend.name == "advanced_backend"
        assert backend.backend_type == MemoryType.VECTOR
        assert backend.enabled is False
        assert backend.config["dimension"] == 1536
        assert backend.config["index_type"] == "hnsw"
        assert backend.created_at == created_at
        assert backend.last_used == last_used
        assert backend.metadata == metadata


class TestConversationInfo:
    """Test ConversationInfo dataclass"""

    def test_conversation_info_defaults(self):
        """Test ConversationInfo default values"""
        info = ConversationInfo(
            conversation_id="conv-123",
            agent_name="test_agent"
        )
        
        assert info.conversation_id == "conv-123"
        assert info.agent_name == "test_agent"
        assert info.title is None
        assert info.created_at is not None
        assert info.last_updated is not None
        assert info.message_count == 0
        assert info.total_tokens == 0
        assert info.metadata == {}

    def test_conversation_info_with_values(self):
        """Test ConversationInfo with provided values"""
        created_at = datetime.now()
        last_updated = datetime.now()
        metadata = {"user_id": "user-123"}
        
        info = ConversationInfo(
            conversation_id="conv-456",
            agent_name="advanced_agent",
            title="Test Conversation",
            created_at=created_at,
            last_updated=last_updated,
            message_count=10,
            total_tokens=1500,
            metadata=metadata
        )
        
        assert info.conversation_id == "conv-456"
        assert info.agent_name == "advanced_agent"
        assert info.title == "Test Conversation"
        assert info.created_at == created_at
        assert info.last_updated == last_updated
        assert info.message_count == 10
        assert info.total_tokens == 1500
        assert info.metadata == metadata


class TestMessageInfo:
    """Test MessageInfo dataclass"""

    def test_message_info_defaults(self):
        """Test MessageInfo default values"""
        info = MessageInfo(
            conversation_id="conv-123",
            role="human",
            content="Hello, world!"
        )
        
        assert info.conversation_id == "conv-123"
        assert info.role == "human"
        assert info.content == "Hello, world!"
        assert info.timestamp is not None
        assert info.token_count is None
        assert info.metadata == {}

    def test_message_info_with_values(self):
        """Test MessageInfo with provided values"""
        timestamp = datetime.now()
        metadata = {"source": "api"}
        
        info = MessageInfo(
            conversation_id="conv-456",
            role="ai",
            content="Hi there!",
            timestamp=timestamp,
            token_count=25,
            metadata=metadata
        )
        
        assert info.conversation_id == "conv-456"
        assert info.role == "ai"
        assert info.content == "Hi there!"
        assert info.timestamp == timestamp
        assert info.token_count == 25
        assert info.metadata == metadata


class TestLangChainMemoryManager:
    """Test LangChainMemoryManager class"""

    @pytest.fixture
    def memory_manager_instance(self):
        """Create a fresh memory manager instance for testing"""
        return LangChainMemoryManager()

    @pytest.fixture
    def mock_monitoring(self):
        """Mock monitoring system"""
        with patch('app.core.langchain.memory_manager.LangChainMonitoring') as mock:
            mock_instance = Mock()
            mock_instance.initialize = AsyncMock()
            mock_instance.track_memory_operation = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_postgres_client(self):
        """Mock PostgreSQL client"""
        with patch('app.core.langchain.memory_manager.get_postgresql_client') as mock:
            mock_client = Mock()
            mock_client.create_conversation = AsyncMock(return_value=True)
            mock_client.get_conversation = AsyncMock(return_value=None)
            mock_client.add_message = AsyncMock(return_value=True)
            mock_client.get_messages = AsyncMock(return_value=[])
            mock_client.search_conversations = AsyncMock(return_value=[])
            mock_client.delete_conversation = AsyncMock(return_value=True)
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_milvus_client(self):
        """Mock Milvus client"""
        with patch('app.core.langchain.memory_manager.get_milvus_client') as mock:
            mock_client = Mock()
            mock_client.add_vector = AsyncMock(return_value=True)
            mock_client.search_vectors = AsyncMock(return_value=[])
            mock_client.delete_vectors = AsyncMock(return_value=True)
            mock.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_initialize(self, memory_manager_instance, mock_monitoring):
        """Test memory manager initialization"""
        await memory_manager_instance.initialize()
        
        assert memory_manager_instance._initialized is True
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, memory_manager_instance, mock_monitoring):
        """Test that initialize is idempotent"""
        await memory_manager_instance.initialize()
        await memory_manager_instance.initialize()
        
        assert memory_manager_instance._initialized is True
        # Should only initialize once
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_memory_backend(
        self, 
        memory_manager_instance, 
        mock_postgres_client
    ):
        """Test registering a memory backend"""
        await memory_manager_instance.initialize()
        
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION,
            config={"table_name": "conversations"}
        )
        
        result = await memory_manager_instance.register_memory_backend(backend)
        
        assert result is True
        assert "conversation_backend" in memory_manager_instance._backends
        assert memory_manager_instance._backends["conversation_backend"] == backend

    @pytest.mark.asyncio
    async def test_register_memory_backend_duplicate(self, memory_manager_instance):
        """Test registering duplicate memory backend"""
        await memory_manager_instance.initialize()
        
        backend = MemoryBackend(
            name="duplicate_backend",
            backend_type=MemoryType.CONVERSATION
        )
        
        # Register first time
        result1 = await memory_manager_instance.register_memory_backend(backend)
        assert result1 is True
        
        # Register second time
        result2 = await memory_manager_instance.register_memory_backend(backend)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_create_conversation(
        self, 
        memory_manager_instance, 
        mock_postgres_client, 
        mock_monitoring
    ):
        """Test creating a conversation"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        result = await memory_manager_instance.create_conversation(
            conversation_id="conv-123",
            agent_name="test_agent",
            title="Test Conversation",
            metadata={"user_id": "user-123"}
        )
        
        assert result is True
        mock_postgres_client.return_value.create_conversation.assert_called_once()
        call_args = mock_postgres_client.return_value.create_conversation.call_args[1]
        assert call_args["conversation_id"] == "conv-123"
        assert call_args["agent_name"] == "test_agent"
        assert call_args["title"] == "Test Conversation"
        assert call_args["metadata"]["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_create_conversation_no_backend(
        self, 
        memory_manager_instance, 
        mock_monitoring
    ):
        """Test creating conversation without registered backend"""
        await memory_manager_instance.initialize()
        
        result = await memory_manager_instance.create_conversation(
            conversation_id="conv-123",
            agent_name="test_agent"
        )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_add_message(
        self, 
        memory_manager_instance, 
        mock_postgres_client, 
        mock_monitoring
    ):
        """Test adding a message to conversation"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        result = await memory_manager_instance.add_message(
            conversation_id="conv-123",
            role="human",
            content="Hello, world!",
            metadata={"source": "api"}
        )
        
        assert result is True
        mock_postgres_client.return_value.add_message.assert_called_once()
        call_args = mock_postgres_client.return_value.add_message.call_args[1]
        assert call_args["conversation_id"] == "conv-123"
        assert call_args["role"] == "human"
        assert call_args["content"] == "Hello, world!"
        assert call_args["metadata"]["source"] == "api"

    @pytest.mark.asyncio
    async def test_add_message_with_vector_backend(
        self, 
        memory_manager_instance, 
        mock_postgres_client, 
        mock_milvus_client, 
        mock_monitoring
    ):
        """Test adding message with vector backend"""
        await memory_manager_instance.initialize()
        
        # Register backends
        conv_backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        vector_backend = MemoryBackend(
            name="vector_backend",
            backend_type=MemoryType.VECTOR
        )
        
        await memory_manager_instance.register_memory_backend(conv_backend)
        await memory_manager_instance.register_memory_backend(vector_backend)
        
        # Mock embedding generation
        with patch('app.core.langchain.memory_manager.generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 1536
            
            result = await memory_manager_instance.add_message(
                conversation_id="conv-123",
                role="human",
                content="Hello, world!"
            )
            
            assert result is True
            mock_postgres_client.return_value.add_message.assert_called_once()
            mock_milvus_client.return_value.add_vector.assert_called_once()
            mock_embedding.assert_called_once_with("Hello, world!")

    @pytest.mark.asyncio
    async def test_get_conversation_messages(
        self, 
        memory_manager_instance, 
        mock_postgres_client
    ):
        """Test getting conversation messages"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Mock messages
        mock_messages = [
            {
                "id": "msg-1",
                "conversation_id": "conv-123",
                "role": "human",
                "content": "Hello",
                "timestamp": "2023-01-01T00:00:00",
                "token_count": 5
            },
            {
                "id": "msg-2",
                "conversation_id": "conv-123",
                "role": "ai",
                "content": "Hi there!",
                "timestamp": "2023-01-01T00:01:00",
                "token_count": 8
            }
        ]
        mock_postgres_client.return_value.get_messages.return_value = mock_messages
        
        result = await memory_manager_instance.get_conversation_messages(
            conversation_id="conv-123",
            limit=10
        )
        
        assert len(result) == 2
        assert result[0]["role"] == "human"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "ai"
        assert result[1]["content"] == "Hi there!"
        
        mock_postgres_client.return_value.get_messages.assert_called_once_with(
            "conv-123", limit=10
        )

    @pytest.mark.asyncio
    async def test_get_conversation_info(
        self, 
        memory_manager_instance, 
        mock_postgres_client
    ):
        """Test getting conversation information"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Mock conversation info
        mock_conv_info = {
            "conversation_id": "conv-123",
            "agent_name": "test_agent",
            "title": "Test Conversation",
            "created_at": "2023-01-01T00:00:00",
            "last_updated": "2023-01-01T00:05:00",
            "message_count": 5,
            "total_tokens": 150
        }
        mock_postgres_client.return_value.get_conversation.return_value = mock_conv_info
        
        result = await memory_manager_instance.get_conversation_info("conv-123")
        
        assert result is not None
        assert result.conversation_id == "conv-123"
        assert result.agent_name == "test_agent"
        assert result.title == "Test Conversation"
        assert result.message_count == 5
        assert result.total_tokens == 150
        
        mock_postgres_client.return_value.get_conversation.assert_called_once_with("conv-123")

    @pytest.mark.asyncio
    async def test_search_conversations(
        self, 
        memory_manager_instance, 
        mock_postgres_client, 
        mock_milvus_client
    ):
        """Test searching conversations"""
        await memory_manager_instance.initialize()
        
        # Register backends
        conv_backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        vector_backend = MemoryBackend(
            name="vector_backend",
            backend_type=MemoryType.VECTOR
        )
        
        await memory_manager_instance.register_memory_backend(conv_backend)
        await memory_manager_instance.register_memory_backend(vector_backend)
        
        # Mock search results
        mock_search_results = [
            {
                "conversation_id": "conv-123",
                "content": "This is about machine learning",
                "score": 0.95,
                "metadata": {"role": "human"}
            },
            {
                "conversation_id": "conv-456",
                "content": "AI and deep learning topics",
                "score": 0.87,
                "metadata": {"role": "ai"}
            }
        ]
        mock_milvus_client.return_value.search_vectors.return_value = mock_search_results
        
        result = await memory_manager_instance.search_conversations(
            query="machine learning",
            limit=5,
            conversation_id=None
        )
        
        assert len(result) == 2
        assert result[0]["conversation_id"] == "conv-123"
        assert result[0]["score"] == 0.95
        assert result[1]["conversation_id"] == "conv-456"
        assert result[1]["score"] == 0.87
        
        # Verify embedding generation and search were called
        with patch('app.core.langchain.memory_manager.generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 1536
            
            await memory_manager_instance.search_conversations(
                query="machine learning",
                limit=5
            )
            
            mock_embedding.assert_called_once_with("machine learning")
            mock_milvus_client.return_value.search_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_conversations(
        self, 
        memory_manager_instance, 
        mock_postgres_client
    ):
        """Test listing conversations"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Mock conversation list
        mock_conversations = [
            {
                "conversation_id": "conv-123",
                "agent_name": "test_agent",
                "title": "First Conversation",
                "created_at": "2023-01-01T00:00:00",
                "last_updated": "2023-01-01T00:05:00",
                "message_count": 5
            },
            {
                "conversation_id": "conv-456",
                "agent_name": "test_agent",
                "title": "Second Conversation",
                "created_at": "2023-01-02T00:00:00",
                "last_updated": "2023-01-02T00:03:00",
                "message_count": 3
            }
        ]
        mock_postgres_client.return_value.list_conversations.return_value = mock_conversations
        
        result = await memory_manager_instance.list_conversations(
            agent_name="test_agent",
            limit=10
        )
        
        assert len(result) == 2
        assert result[0].conversation_id == "conv-123"
        assert result[0].title == "First Conversation"
        assert result[1].conversation_id == "conv-456"
        assert result[1].title == "Second Conversation"
        
        mock_postgres_client.return_value.list_conversations.assert_called_once_with(
            agent_name="test_agent", limit=10
        )

    @pytest.mark.asyncio
    async def test_delete_conversation(
        self, 
        memory_manager_instance, 
        mock_postgres_client, 
        mock_milvus_client, 
        mock_monitoring
    ):
        """Test deleting a conversation"""
        await memory_manager_instance.initialize()
        
        # Register backends
        conv_backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        vector_backend = MemoryBackend(
            name="vector_backend",
            backend_type=MemoryType.VECTOR
        )
        
        await memory_manager_instance.register_memory_backend(conv_backend)
        await memory_manager_instance.register_memory_backend(vector_backend)
        
        result = await memory_manager_instance.delete_conversation("conv-123")
        
        assert result is True
        mock_postgres_client.return_value.delete_conversation.assert_called_once_with("conv-123")
        mock_milvus_client.return_value.delete_vectors.assert_called_once_with(
            conversation_id="conv-123"
        )

    @pytest.mark.asyncio
    async def test_get_memory_backend(self, memory_manager_instance):
        """Test getting a memory backend"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="test_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Get existing backend
        result = memory_manager_instance.get_memory_backend("test_backend")
        assert result == backend
        
        # Get non-existent backend
        result = memory_manager_instance.get_memory_backend("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_memory_backends(self, memory_manager_instance):
        """Test listing memory backends"""
        await memory_manager_instance.initialize()
        
        # Register backends
        backends = [
            MemoryBackend(name="conv_backend", backend_type=MemoryType.CONVERSATION),
            MemoryBackend(name="summary_backend", backend_type=MemoryType.SUMMARY),
            MemoryBackend(name="vector_backend", backend_type=MemoryType.VECTOR)
        ]
        
        for backend in backends:
            await memory_manager_instance.register_memory_backend(backend)
        
        result = await memory_manager_instance.list_memory_backends()
        
        assert len(result) == 3
        backend_names = [backend.name for backend in result]
        assert "conv_backend" in backend_names
        assert "summary_backend" in backend_names
        assert "vector_backend" in backend_names

    @pytest.mark.asyncio
    async def test_enable_disable_backend(self, memory_manager_instance):
        """Test enabling and disabling backends"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="test_backend",
            backend_type=MemoryType.CONVERSATION,
            enabled=True
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Disable backend
        result = await memory_manager_instance.disable_backend("test_backend")
        assert result is True
        assert memory_manager_instance._backends["test_backend"].enabled is False
        
        # Enable backend
        result = await memory_manager_instance.enable_backend("test_backend")
        assert result is True
        assert memory_manager_instance._backends["test_backend"].enabled is True

    @pytest.mark.asyncio
    async def test_get_backend_stats(self, memory_manager_instance):
        """Test getting backend statistics"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="test_backend",
            backend_type=MemoryType.CONVERSATION,
            created_at=datetime(2023, 1, 1),
            last_used=datetime(2023, 1, 2)
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        result = await memory_manager_instance.get_backend_stats("test_backend")
        
        assert result is not None
        assert result["name"] == "test_backend"
        assert result["backend_type"] == "conversation"
        assert result["enabled"] is True
        assert result["created_at"] == "2023-01-01T00:00:00"
        assert result["last_used"] == "2023-01-02T00:00:00"

    @pytest.mark.asyncio
    async def test_get_registry_stats(self, memory_manager_instance):
        """Test getting registry statistics"""
        await memory_manager_instance.initialize()
        
        # Register backends
        backends = [
            MemoryBackend(name="conv_backend", backend_type=MemoryType.CONVERSATION),
            MemoryBackend(name="summary_backend", backend_type=MemoryType.SUMMARY),
            MemoryBackend(name="vector_backend", backend_type=MemoryType.VECTOR),
            MemoryBackend(name="conv_backend_2", backend_type=MemoryType.CONVERSATION)
        ]
        
        for backend in backends:
            await memory_manager_instance.register_memory_backend(backend)
        
        result = await memory_manager_instance.get_registry_stats()
        
        assert result["total_backends"] == 4
        assert result["enabled_backends"] == 4
        assert len(result["backend_types"]) == 3
        assert result["backend_types"]["conversation"] == 2
        assert result["backend_types"]["summary"] == 1
        assert result["backend_types"]["vector"] == 1

    def test_get_backend_for_type(self, memory_manager_instance):
        """Test getting backend for specific type"""
        # Register backends
        conv_backend = MemoryBackend(
            name="conv_backend",
            backend_type=MemoryType.CONVERSATION
        )
        vector_backend = MemoryBackend(
            name="vector_backend",
            backend_type=MemoryType.VECTOR
        )
        
        memory_manager_instance._backends = {
            "conv_backend": conv_backend,
            "vector_backend": vector_backend
        }
        
        # Get conversation backend
        result = memory_manager_instance._get_backend_for_type(MemoryType.CONVERSATION)
        assert result == conv_backend
        
        # Get vector backend
        result = memory_manager_instance._get_backend_for_type(MemoryType.VECTOR)
        assert result == vector_backend
        
        # Get non-existent type
        result = memory_manager_instance._get_backend_for_type(MemoryType.SUMMARY)
        assert result is None

    @pytest.mark.asyncio
    async def test_track_memory_operation(
        self, 
        memory_manager_instance, 
        mock_monitoring
    ):
        """Test tracking memory operations"""
        await memory_manager_instance.initialize()
        
        await memory_manager_instance._track_memory_operation(
            operation="create_conversation",
            conversation_id="conv-123",
            success=True,
            duration=0.5,
            metadata={"agent_name": "test_agent"}
        )
        
        mock_monitoring.return_value.track_memory_operation.assert_called_once()
        call_args = mock_monitoring.return_value.track_memory_operation.call_args[1]
        assert call_args["operation"] == "create_conversation"
        assert call_args["conversation_id"] == "conv-123"
        assert call_args["success"] is True
        assert call_args["duration"] == 0.5
        assert call_args["metadata"]["agent_name"] == "test_agent"

    @pytest.mark.asyncio
    async def test_generate_summary(
        self, 
        memory_manager_instance, 
        mock_postgres_client
    ):
        """Test generating conversation summary"""
        await memory_manager_instance.initialize()
        
        # Register backend
        backend = MemoryBackend(
            name="conversation_backend",
            backend_type=MemoryType.CONVERSATION
        )
        await memory_manager_instance.register_memory_backend(backend)
        
        # Mock messages
        mock_messages = [
            {"role": "human", "content": "Hello, can you help me with machine learning?"},
            {"role": "ai", "content": "Of course! I'd be happy to help you with machine learning concepts."},
            {"role": "human", "content": "What is a neural network?"},
            {"role": "ai", "content": "A neural network is a computational model inspired by biological neural networks..."}
        ]
        mock_postgres_client.return_value.get_messages.return_value = mock_messages
        
        # Mock LLM for summarization
        with patch('app.core.langchain.memory_manager.llm_manager') as mock_llm_manager:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Discussion about machine learning and neural networks basics."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_manager.get_llm = AsyncMock(return_value=mock_llm)
            
            result = await memory_manager_instance._generate_summary("conv-123")
            
            assert result == "Discussion about machine learning and neural networks basics."
            mock_postgres_client.return_value.get_messages.assert_called_once_with("conv-123")
            mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding(self, memory_manager_instance):
        """Test generating embeddings"""
        await memory_manager_instance.initialize()
        
        # Mock embedding model
        with patch('app.core.langchain.memory_manager.embeddings') as mock_embeddings:
            mock_embeddings.embed_query.return_value = [0.1] * 1536
            
            result = await memory_manager_instance._generate_embedding("Hello, world!")
            
            assert result == [0.1] * 1536
            mock_embeddings.embed_query.assert_called_once_with("Hello, world!")


class TestGlobalMemoryManager:
    """Test global memory manager instance"""

    def test_global_instance(self):
        """Test that global memory manager instance exists"""
        from app.core.langchain.memory_manager import memory_manager
        assert memory_manager is not None
        assert isinstance(memory_manager, LangChainMemoryManager)