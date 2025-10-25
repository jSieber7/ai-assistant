"""
Unit tests for LangChain Memory Workflow.

This module tests the LangChain-based memory workflow that provides
different strategies for conversation management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple

from app.core.langchain.memory_workflow import (
    MemoryWorkflow,
    MemoryWorkflowType,
    ConversationMemory,
    SummaryMemory,
    VectorMemory,
    ConversationTurn,
    MemoryState,
    memory_workflow,
    create_memory_workflow
)


class TestMemoryWorkflowType:
    """Test MemoryWorkflowType enum"""

    def test_memory_workflow_type_values(self):
        """Test that MemoryWorkflowType has expected values"""
        expected_types = [
            "conversation",
            "summary",
            "vector",
            "hybrid"
        ]
        
        actual_types = [workflow_type.value for workflow_type in MemoryWorkflowType]
        assert actual_types == expected_types


class TestConversationTurn:
    """Test ConversationTurn dataclass"""

    def test_conversation_turn_defaults(self):
        """Test ConversationTurn default values"""
        timestamp = datetime.now()
        turn = ConversationTurn(
            role="user",
            content="Hello, how are you?",
            timestamp=timestamp
        )
        
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.timestamp == timestamp
        assert turn.metadata == {}

    def test_conversation_turn_with_values(self):
        """Test ConversationTurn with provided values"""
        timestamp = datetime.now()
        metadata = {"tokens": 10, "model": "gpt-4"}
        
        turn = ConversationTurn(
            role="assistant",
            content="I'm doing well, thank you!",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert turn.role == "assistant"
        assert turn.content == "I'm doing well, thank you!"
        assert turn.timestamp == timestamp
        assert turn.metadata == metadata

    def test_conversation_turn_to_dict(self):
        """Test ConversationTurn to_dict method"""
        timestamp = datetime.now()
        metadata = {"tokens": 10}
        
        turn = ConversationTurn(
            role="user",
            content="Hello",
            timestamp=timestamp,
            metadata=metadata
        )
        
        result = turn.to_dict()
        
        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["metadata"] == metadata


class TestMemoryState:
    """Test MemoryState dataclass"""

    def test_memory_state_defaults(self):
        """Test MemoryState default values"""
        state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        assert state.conversation_id == "conv-123"
        assert state.workflow_type == MemoryWorkflowType.CONVERSATION
        assert state.turns == []
        assert state.summary == ""
        assert state.metadata == {}

    def test_memory_state_with_values(self):
        """Test MemoryState with provided values"""
        timestamp = datetime.now()
        turns = [
            ConversationTurn("user", "Hello", timestamp),
            ConversationTurn("assistant", "Hi there!", timestamp)
        ]
        metadata = {"user_id": "user-456"}
        
        state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.SUMMARY,
            turns=turns,
            summary="User greeted and assistant responded",
            metadata=metadata
        )
        
        assert state.conversation_id == "conv-123"
        assert state.workflow_type == MemoryWorkflowType.SUMMARY
        assert len(state.turns) == 2
        assert state.turns[0].role == "user"
        assert state.turns[0].content == "Hello"
        assert state.turns[1].role == "assistant"
        assert state.turns[1].content == "Hi there!"
        assert state.summary == "User greeted and assistant responded"
        assert state.metadata == metadata

    def test_memory_state_add_turn(self):
        """Test MemoryState add_turn method"""
        state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        timestamp = datetime.now()
        turn = ConversationTurn("user", "Hello", timestamp)
        
        state.add_turn(turn)
        
        assert len(state.turns) == 1
        assert state.turns[0] == turn

    def test_memory_state_get_recent_turns(self):
        """Test MemoryState get_recent_turns method"""
        timestamp = datetime.now()
        turns = [
            ConversationTurn("user", "Hello", timestamp - timedelta(seconds=3)),
            ConversationTurn("assistant", "Hi", timestamp - timedelta(seconds=2)),
            ConversationTurn("user", "How are you?", timestamp - timedelta(seconds=1)),
            ConversationTurn("assistant", "I'm well", timestamp)
        ]
        
        state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION,
            turns=turns
        )
        
        # Get all turns
        recent_turns = state.get_recent_turns()
        assert len(recent_turns) == 4
        
        # Get last 2 turns
        recent_turns = state.get_recent_turns(2)
        assert len(recent_turns) == 2
        assert recent_turns[0].content == "How are you?"
        assert recent_turns[1].content == "I'm well"

    def test_memory_state_to_dict(self):
        """Test MemoryState to_dict method"""
        timestamp = datetime.now()
        turns = [
            ConversationTurn("user", "Hello", timestamp)
        ]
        metadata = {"user_id": "user-456"}
        
        state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION,
            turns=turns,
            summary="Test summary",
            metadata=metadata
        )
        
        result = state.to_dict()
        
        assert result["conversation_id"] == "conv-123"
        assert result["workflow_type"] == "conversation"
        assert len(result["turns"]) == 1
        assert result["turns"][0]["role"] == "user"
        assert result["turns"][0]["content"] == "Hello"
        assert result["summary"] == "Test summary"
        assert result["metadata"] == metadata


class TestConversationMemory:
    """Test ConversationMemory class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def conversation_memory(self, mock_llm_manager, mock_memory_manager):
        """Create a conversation memory instance"""
        return ConversationMemory(
            llm_manager=mock_llm_manager,
            memory_manager=mock_memory_manager
        )

    @pytest.mark.asyncio
    async def test_initialize(self, conversation_memory, mock_memory_manager):
        """Test conversation memory initialization"""
        conversation_id = "conv-123"
        
        await conversation_memory.initialize(conversation_id)
        
        assert conversation_memory.conversation_id == conversation_id
        assert conversation_memory.state.conversation_id == conversation_id
        assert conversation_memory.state.workflow_type == MemoryWorkflowType.CONVERSATION
        mock_memory_manager.get_memory.assert_called_once_with(
            memory_type="conversation",
            conversation_id=conversation_id
        )

    @pytest.mark.asyncio
    async def test_add_turn(self, conversation_memory):
        """Test adding a turn to conversation memory"""
        await conversation_memory.initialize("conv-123")
        
        await conversation_memory.add_turn(
            role="user",
            content="Hello, how are you?",
            metadata={"tokens": 10}
        )
        
        assert len(conversation_memory.state.turns) == 1
        turn = conversation_memory.state.turns[0]
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.metadata["tokens"] == 10

    @pytest.mark.asyncio
    async def test_get_context(self, conversation_memory):
        """Test getting conversation context"""
        await conversation_memory.initialize("conv-123")
        
        # Add some turns
        await conversation_memory.add_turn("user", "Hello")
        await conversation_memory.add_turn("assistant", "Hi there!")
        await conversation_memory.add_turn("user", "How are you?")
        
        context = await conversation_memory.get_context()
        
        assert len(context) == 3
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Hi there!"
        assert context[2]["role"] == "user"
        assert context[2]["content"] == "How are you?"

    @pytest.mark.asyncio
    async def test_get_context_with_limit(self, conversation_memory):
        """Test getting conversation context with limit"""
        await conversation_memory.initialize("conv-123")
        
        # Add some turns
        await conversation_memory.add_turn("user", "Hello")
        await conversation_memory.add_turn("assistant", "Hi there!")
        await conversation_memory.add_turn("user", "How are you?")
        await conversation_memory.add_turn("assistant", "I'm well")
        
        context = await conversation_memory.get_context(max_turns=2)
        
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "How are you?"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "I'm well"

    @pytest.mark.asyncio
    async def test_get_context_empty(self, conversation_memory):
        """Test getting context from empty conversation"""
        await conversation_memory.initialize("conv-123")
        
        context = await conversation_memory.get_context()
        
        assert context == []

    @pytest.mark.asyncio
    async def test_save_state(self, conversation_memory, mock_memory_manager):
        """Test saving conversation state"""
        await conversation_memory.initialize("conv-123")
        
        await conversation_memory.add_turn("user", "Hello")
        
        await conversation_memory.save_state()
        
        mock_memory_manager.save_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_state(self, conversation_memory, mock_memory_manager):
        """Test loading conversation state"""
        await conversation_memory.initialize("conv-123")
        
        # Mock loaded state
        timestamp = datetime.now()
        loaded_state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION,
            turns=[
                ConversationTurn("user", "Hello", timestamp)
            ],
            summary="Test summary"
        )
        
        mock_memory_manager.load_memory.return_value = loaded_state
        
        await conversation_memory.load_state()
        
        assert len(conversation_memory.state.turns) == 1
        assert conversation_memory.state.turns[0].content == "Hello"
        assert conversation_memory.state.summary == "Test summary"

    @pytest.mark.asyncio
    async def test_clear(self, conversation_memory):
        """Test clearing conversation memory"""
        await conversation_memory.initialize("conv-123")
        
        await conversation_memory.add_turn("user", "Hello")
        await conversation_memory.add_turn("assistant", "Hi there!")
        
        await conversation_memory.clear()
        
        assert len(conversation_memory.state.turns) == 0
        assert conversation_memory.state.summary == ""


class TestSummaryMemory:
    """Test SummaryMemory class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="User greeted and assistant responded")
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def summary_memory(self, mock_llm_manager, mock_memory_manager):
        """Create a summary memory instance"""
        return SummaryMemory(
            llm_manager=mock_llm_manager,
            memory_manager=mock_memory_manager,
            max_turns_before_summary=2
        )

    @pytest.mark.asyncio
    async def test_initialize(self, summary_memory, mock_memory_manager):
        """Test summary memory initialization"""
        conversation_id = "conv-123"
        
        await summary_memory.initialize(conversation_id)
        
        assert summary_memory.conversation_id == conversation_id
        assert summary_memory.state.conversation_id == conversation_id
        assert summary_memory.state.workflow_type == MemoryWorkflowType.SUMMARY
        mock_memory_manager.get_memory.assert_called_once_with(
            memory_type="summary",
            conversation_id=conversation_id
        )

    @pytest.mark.asyncio
    async def test_add_turn_no_summary(self, summary_memory):
        """Test adding turns without triggering summary"""
        await summary_memory.initialize("conv-123")
        
        await summary_memory.add_turn("user", "Hello")
        
        assert len(summary_memory.state.turns) == 1
        assert summary_memory.state.summary == ""

    @pytest.mark.asyncio
    async def test_add_turn_trigger_summary(self, summary_memory):
        """Test adding turns that trigger summary generation"""
        await summary_memory.initialize("conv-123")
        
        await summary_memory.add_turn("user", "Hello")
        await summary_memory.add_turn("assistant", "Hi there!")
        
        # Should have triggered summary
        assert len(summary_memory.state.turns) == 0  # Turns cleared after summary
        assert summary_memory.state.summary == "User greeted and assistant responded"

    @pytest.mark.asyncio
    async def test_generate_summary(self, summary_memory):
        """Test generating summary"""
        await summary_memory.initialize("conv-123")
        
        timestamp = datetime.now()
        summary_memory.state.turns = [
            ConversationTurn("user", "Hello", timestamp),
            ConversationTurn("assistant", "Hi there!", timestamp)
        ]
        
        summary = await summary_memory._generate_summary()
        
        assert summary == "User greeted and assistant responded"
        assert len(summary_memory.state.turns) == 0  # Turns should be cleared

    @pytest.mark.asyncio
    async def test_get_context(self, summary_memory):
        """Test getting context from summary memory"""
        await summary_memory.initialize("conv-123")
        
        # Set up state with summary and recent turns
        summary_memory.state.summary = "User greeted and assistant responded"
        timestamp = datetime.now()
        summary_memory.state.turns = [
            ConversationTurn("user", "How are you?", timestamp),
            ConversationTurn("assistant", "I'm well", timestamp)
        ]
        
        context = await summary_memory.get_context()
        
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "User greeted and assistant responded"
        assert context[1]["role"] == "user"
        assert context[1]["content"] == "How are you?"
        assert context[2]["role"] == "assistant"
        assert context[2]["content"] == "I'm well"

    @pytest.mark.asyncio
    async def test_get_context_empty(self, summary_memory):
        """Test getting context from empty summary memory"""
        await summary_memory.initialize("conv-123")
        
        context = await summary_memory.get_context()
        
        assert context == []

    @pytest.mark.asyncio
    async def test_save_state(self, summary_memory, mock_memory_manager):
        """Test saving summary state"""
        await summary_memory.initialize("conv-123")
        
        await summary_memory.save_state()
        
        mock_memory_manager.save_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_state(self, summary_memory, mock_memory_manager):
        """Test loading summary state"""
        await summary_memory.initialize("conv-123")
        
        # Mock loaded state
        loaded_state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.SUMMARY,
            summary="Test summary"
        )
        
        mock_memory_manager.load_memory.return_value = loaded_state
        
        await summary_memory.load_state()
        
        assert summary_memory.state.summary == "Test summary"

    @pytest.mark.asyncio
    async def test_clear(self, summary_memory):
        """Test clearing summary memory"""
        await summary_memory.initialize("conv-123")
        
        summary_memory.state.summary = "Test summary"
        await summary_memory.add_turn("user", "Hello")
        
        await summary_memory.clear()
        
        assert len(summary_memory.state.turns) == 0
        assert summary_memory.state.summary == ""


class TestVectorMemory:
    """Test VectorMemory class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def vector_memory(self, mock_llm_manager, mock_memory_manager):
        """Create a vector memory instance"""
        return VectorMemory(
            llm_manager=mock_llm_manager,
            memory_manager=mock_memory_manager
        )

    @pytest.mark.asyncio
    async def test_initialize(self, vector_memory, mock_memory_manager):
        """Test vector memory initialization"""
        conversation_id = "conv-123"
        
        await vector_memory.initialize(conversation_id)
        
        assert vector_memory.conversation_id == conversation_id
        assert vector_memory.state.conversation_id == conversation_id
        assert vector_memory.state.workflow_type == MemoryWorkflowType.VECTOR
        mock_memory_manager.get_memory.assert_called_once_with(
            memory_type="vector",
            conversation_id=conversation_id
        )

    @pytest.mark.asyncio
    async def test_add_turn(self, vector_memory):
        """Test adding a turn to vector memory"""
        await vector_memory.initialize("conv-123")
        
        await vector_memory.add_turn(
            role="user",
            content="Hello, how are you?",
            metadata={"tokens": 10}
        )
        
        assert len(vector_memory.state.turns) == 1
        turn = vector_memory.state.turns[0]
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.metadata["tokens"] == 10

    @pytest.mark.asyncio
    async def test_get_context(self, vector_memory):
        """Test getting context from vector memory"""
        await vector_memory.initialize("conv-123")
        
        # Add some turns
        await vector_memory.add_turn("user", "Hello")
        await vector_memory.add_turn("assistant", "Hi there!")
        await vector_memory.add_turn("user", "How are you?")
        
        context = await vector_memory.get_context()
        
        assert len(context) == 3
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Hi there!"
        assert context[2]["role"] == "user"
        assert context[2]["content"] == "How are you?"

    @pytest.mark.asyncio
    async def test_get_context_with_query(self, vector_memory):
        """Test getting context with query-based retrieval"""
        await vector_memory.initialize("conv-123")
        
        # Add some turns
        await vector_memory.add_turn("user", "What is the weather like?")
        await vector_memory.add_turn("assistant", "It's sunny today.")
        await vector_memory.add_turn("user", "How about tomorrow?")
        await vector_memory.add_turn("assistant", "It might rain tomorrow.")
        
        # Mock similarity search
        mock_memory = Mock()
        mock_memory.similarity_search.return_value = [
            Mock(page_content="What is the weather like?"),
            Mock(page_content="It's sunny today.")
        ]
        vector_memory.memory = mock_memory
        
        context = await vector_memory.get_context(query="weather", max_context=2)
        
        assert len(context) == 2
        mock_memory.similarity_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_state(self, vector_memory, mock_memory_manager):
        """Test saving vector state"""
        await vector_memory.initialize("conv-123")
        
        await vector_memory.add_turn("user", "Hello")
        
        await vector_memory.save_state()
        
        mock_memory_manager.save_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_state(self, vector_memory, mock_memory_manager):
        """Test loading vector state"""
        await vector_memory.initialize("conv-123")
        
        # Mock loaded state
        loaded_state = MemoryState(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.VECTOR,
            turns=[
                ConversationTurn("user", "Hello", datetime.now())
            ]
        )
        
        mock_memory_manager.load_memory.return_value = loaded_state
        
        await vector_memory.load_state()
        
        assert len(vector_memory.state.turns) == 1
        assert vector_memory.state.turns[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_clear(self, vector_memory):
        """Test clearing vector memory"""
        await vector_memory.initialize("conv-123")
        
        await vector_memory.add_turn("user", "Hello")
        await vector_memory.add_turn("assistant", "Hi there!")
        
        await vector_memory.clear()
        
        assert len(vector_memory.state.turns) == 0


class TestMemoryWorkflow:
    """Test MemoryWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def memory_workflow(self, mock_llm_manager, mock_memory_manager):
        """Create a memory workflow instance"""
        return MemoryWorkflow(
            llm_manager=mock_llm_manager,
            memory_manager=mock_memory_manager
        )

    @pytest.mark.asyncio
    async def test_create_conversation_memory(self, memory_workflow):
        """Test creating conversation memory"""
        memory = memory_workflow._create_memory(MemoryWorkflowType.CONVERSATION)
        
        assert isinstance(memory, ConversationMemory)
        assert memory.llm_manager == memory_workflow.llm_manager
        assert memory.memory_manager == memory_workflow.memory_manager

    @pytest.mark.asyncio
    async def test_create_summary_memory(self, memory_workflow):
        """Test creating summary memory"""
        memory = memory_workflow._create_memory(MemoryWorkflowType.SUMMARY)
        
        assert isinstance(memory, SummaryMemory)
        assert memory.llm_manager == memory_workflow.llm_manager
        assert memory.memory_manager == memory_workflow.memory_manager

    @pytest.mark.asyncio
    async def test_create_vector_memory(self, memory_workflow):
        """Test creating vector memory"""
        memory = memory_workflow._create_memory(MemoryWorkflowType.VECTOR)
        
        assert isinstance(memory, VectorMemory)
        assert memory.llm_manager == memory_workflow.llm_manager
        assert memory.memory_manager == memory_workflow.memory_manager

    @pytest.mark.asyncio
    async def test_create_hybrid_memory(self, memory_workflow):
        """Test creating hybrid memory"""
        memory = memory_workflow._create_memory(MemoryWorkflowType.HYBRID)
        
        # Hybrid should default to conversation memory for now
        assert isinstance(memory, ConversationMemory)

    @pytest.mark.asyncio
    async def test_initialize(self, memory_workflow):
        """Test memory workflow initialization"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        assert memory_workflow.conversation_id == "conv-123"
        assert memory_workflow.workflow_type == MemoryWorkflowType.CONVERSATION
        assert isinstance(memory_workflow.memory, ConversationMemory)

    @pytest.mark.asyncio
    async def test_add_turn(self, memory_workflow):
        """Test adding a turn through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.add_turn(
            role="user",
            content="Hello, how are you?",
            metadata={"tokens": 10}
        )
        
        # Check turn was added to underlying memory
        assert len(memory_workflow.memory.state.turns) == 1
        turn = memory_workflow.memory.state.turns[0]
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.metadata["tokens"] == 10

    @pytest.mark.asyncio
    async def test_get_context(self, memory_workflow):
        """Test getting context through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.add_turn("user", "Hello")
        await memory_workflow.add_turn("assistant", "Hi there!")
        
        context = await memory_workflow.get_context()
        
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_save_state(self, memory_workflow):
        """Test saving state through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.save_state()
        
        # Should have called save on underlying memory
        assert memory_workflow.memory.save_state.called

    @pytest.mark.asyncio
    async def test_load_state(self, memory_workflow):
        """Test loading state through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.load_state()
        
        # Should have called load on underlying memory
        assert memory_workflow.memory.load_state.called

    @pytest.mark.asyncio
    async def test_clear(self, memory_workflow):
        """Test clearing through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.add_turn("user", "Hello")
        await memory_workflow.add_turn("assistant", "Hi there!")
        
        await memory_workflow.clear()
        
        assert len(memory_workflow.memory.state.turns) == 0

    @pytest.mark.asyncio
    async def test_get_state(self, memory_workflow):
        """Test getting state through workflow"""
        await memory_workflow.initialize(
            conversation_id="conv-123",
            workflow_type=MemoryWorkflowType.CONVERSATION
        )
        
        await memory_workflow.add_turn("user", "Hello")
        
        state = await memory_workflow.get_state()
        
        assert state.conversation_id == "conv-123"
        assert state.workflow_type == MemoryWorkflowType.CONVERSATION
        assert len(state.turns) == 1
        assert state.turns[0].content == "Hello"


class TestHelperFunctions:
    """Test helper functions"""

    @pytest.mark.asyncio
    async def test_create_memory_workflow(self):
        """Test create_memory_workflow helper function"""
        with patch('app.core.langchain.memory_workflow.LLMManager') as mock_llm_manager, \
             patch('app.core.langchain.memory_workflow.MemoryManager') as mock_memory_manager:
            
            workflow = await create_memory_workflow(
                workflow_type=MemoryWorkflowType.CONVERSATION
            )
            
            assert isinstance(workflow, MemoryWorkflow)
            assert workflow.workflow_type == MemoryWorkflowType.CONVERSATION

    def test_global_memory_workflow(self):
        """Test global memory workflow instance"""
        from app.core.langchain.memory_workflow import memory_workflow
        assert memory_workflow is not None
        assert isinstance(memory_workflow, MemoryWorkflow)