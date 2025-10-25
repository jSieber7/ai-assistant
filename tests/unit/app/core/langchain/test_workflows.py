"""
Unit tests for LangChain Workflows.

This module tests the LangGraph-based workflows for different agent types,
including conversational, tool_heavy, multi_agent, researcher, analyst, and synthesizer workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple

from app.core.langchain.workflows import (
    BaseWorkflow,
    WorkflowType,
    WorkflowState,
    ConversationWorkflow,
    ToolHeavyWorkflow,
    MultiAgentWorkflow,
    ResearcherWorkflow,
    AnalystWorkflow,
    SynthesizerWorkflow,
    WorkflowFactory,
    workflow_factory
)


class TestWorkflowType:
    """Test WorkflowType enum"""

    def test_workflow_type_values(self):
        """Test that WorkflowType has expected values"""
        expected_types = [
            "conversational",
            "tool_heavy",
            "multi_agent",
            "researcher",
            "analyst",
            "synthesizer"
        ]
        
        actual_types = [workflow_type.value for workflow_type in WorkflowType]
        assert actual_types == expected_types


class TestWorkflowState:
    """Test WorkflowState dataclass"""

    def test_workflow_state_defaults(self):
        """Test WorkflowState default values"""
        state = WorkflowState(
            input="Hello, how are you?",
            workflow_type=WorkflowType.CONVERSATIONAL
        )
        
        assert state.input == "Hello, how are you?"
        assert state.workflow_type == WorkflowType.CONVERSATIONAL
        assert state.output == ""
        assert state.steps == []
        assert state.metadata == {}

    def test_workflow_state_with_values(self):
        """Test WorkflowState with provided values"""
        steps = ["step1", "step2", "step3"]
        metadata = {"user_id": "user-456", "request_id": "req-123"}
        
        state = WorkflowState(
            input="What is the weather?",
            workflow_type=WorkflowType.TOOL_HEAVY,
            output="The weather is sunny.",
            steps=steps,
            metadata=metadata
        )
        
        assert state.input == "What is the weather?"
        assert state.workflow_type == WorkflowType.TOOL_HEAVY
        assert state.output == "The weather is sunny."
        assert state.steps == steps
        assert state.metadata == metadata

    def test_workflow_state_add_step(self):
        """Test WorkflowState add_step method"""
        state = WorkflowState(
            input="Test input",
            workflow_type=WorkflowType.CONVERSATIONAL
        )
        
        state.add_step("processing")
        state.add_step("generating_response")
        
        assert len(state.steps) == 2
        assert state.steps[0] == "processing"
        assert state.steps[1] == "generating_response"

    def test_workflow_state_to_dict(self):
        """Test WorkflowState to_dict method"""
        steps = ["step1", "step2"]
        metadata = {"test": True}
        
        state = WorkflowState(
            input="Test input",
            workflow_type=WorkflowType.CONVERSATIONAL,
            output="Test output",
            steps=steps,
            metadata=metadata
        )
        
        result = state.to_dict()
        
        assert result["input"] == "Test input"
        assert result["workflow_type"] == "conversational"
        assert result["output"] == "Test output"
        assert result["steps"] == steps
        assert result["metadata"] == metadata


class TestBaseWorkflow:
    """Test BaseWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        registry.get_tool.return_value = Mock()
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def base_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a base workflow instance"""
        return BaseWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_base_workflow_init(self, base_workflow):
        """Test BaseWorkflow initialization"""
        assert base_workflow.llm_manager is not None
        assert base_workflow.tool_registry is not None
        assert base_workflow.memory_manager is not None
        assert base_workflow.workflow_type is None

    @pytest.mark.asyncio
    async def test_base_workflow_initialize(self, base_workflow):
        """Test BaseWorkflow initialize method"""
        conversation_id = "conv-123"
        
        await base_workflow.initialize(conversation_id)
        
        assert base_workflow.conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_base_workflow_execute_not_implemented(self, base_workflow):
        """Test BaseWorkflow execute method raises NotImplementedError"""
        state = WorkflowState(
            input="Test input",
            workflow_type=WorkflowType.CONVERSATIONAL
        )
        
        with pytest.raises(NotImplementedError):
            await base_workflow.execute(state)

    @pytest.mark.asyncio
    async def test_base_workflow_get_context(self, base_workflow, mock_memory_manager):
        """Test BaseWorkflow get_context method"""
        await base_workflow.initialize("conv-123")
        
        mock_memory = Mock()
        mock_memory.get_context.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        mock_memory_manager.get_memory.return_value = mock_memory
        
        context = await base_workflow.get_context()
        
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_base_workflow_save_to_memory(self, base_workflow, mock_memory_manager):
        """Test BaseWorkflow save_to_memory method"""
        await base_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Test input",
            workflow_type=WorkflowType.CONVERSATIONAL,
            output="Test output"
        )
        
        await base_workflow.save_to_memory(state)
        
        mock_memory = mock_memory_manager.get_memory.return_value
        mock_memory.add_turn.assert_called()


class TestConversationWorkflow:
    """Test ConversationWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Hello! How can I help you today?")
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def conversation_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a conversation workflow instance"""
        return ConversationWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_conversation_workflow_init(self, conversation_workflow):
        """Test ConversationWorkflow initialization"""
        assert conversation_workflow.workflow_type == WorkflowType.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_conversation_workflow_execute(self, conversation_workflow):
        """Test ConversationWorkflow execute method"""
        await conversation_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Hello, how are you?",
            workflow_type=WorkflowType.CONVERSATIONAL
        )
        
        result = await conversation_workflow.execute(state)
        
        assert result.output == "Hello! How can I help you today?"
        assert len(result.steps) > 0
        assert "processing" in result.steps

    @pytest.mark.asyncio
    async def test_conversation_workflow_execute_with_context(self, conversation_workflow, mock_memory_manager):
        """Test ConversationWorkflow execute with context"""
        await conversation_workflow.initialize("conv-123")
        
        # Mock context
        mock_memory = mock_memory_manager.get_memory.return_value
        mock_memory.get_context.return_value = [
            {"role": "user", "content": "My name is John"},
            {"role": "assistant", "content": "Nice to meet you, John!"}
        ]
        
        state = WorkflowState(
            input="What's my name?",
            workflow_type=WorkflowType.CONVERSATIONAL
        )
        
        result = await conversation_workflow.execute(state)
        
        assert result.output == "Hello! How can I help you today?"
        mock_memory.get_context.assert_called_once()


class TestToolHeavyWorkflow:
    """Test ToolHeavyWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="I'll help you with that.",
            additional_kwargs={"tool_calls": [
                {"name": "search_tool", "args": {"query": "weather"}}
            ]}
        )
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        mock_tool = Mock()
        mock_tool.execute.return_value = {"result": "It's sunny today"}
        registry.get_tool.return_value = mock_tool
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def tool_heavy_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a tool heavy workflow instance"""
        return ToolHeavyWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_tool_heavy_workflow_init(self, tool_heavy_workflow):
        """Test ToolHeavyWorkflow initialization"""
        assert tool_heavy_workflow.workflow_type == WorkflowType.TOOL_HEAVY

    @pytest.mark.asyncio
    async def test_tool_heavy_workflow_execute_with_tools(self, tool_heavy_workflow):
        """Test ToolHeavyWorkflow execute with tool calls"""
        await tool_heavy_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="What's the weather like today?",
            workflow_type=WorkflowType.TOOL_HEAVY
        )
        
        result = await tool_heavy_workflow.execute(state)
        
        assert result.output == "I'll help you with that."
        assert len(result.steps) > 0
        assert "tool_execution" in result.steps

    @pytest.mark.asyncio
    async def test_tool_heavy_workflow_execute_without_tools(self, tool_heavy_workflow, mock_llm_manager):
        """Test ToolHeavyWorkflow execute without tool calls"""
        await tool_heavy_workflow.initialize("conv-123")
        
        # Mock LLM without tool calls
        mock_llm = mock_llm_manager.get_llm.return_value
        mock_llm.invoke.return_value = Mock(
            content="The weather is sunny today.",
            additional_kwargs={}
        )
        
        state = WorkflowState(
            input="What's the weather like today?",
            workflow_type=WorkflowType.TOOL_HEAVY
        )
        
        result = await tool_heavy_workflow.execute(state)
        
        assert result.output == "The weather is sunny today."
        assert len(result.steps) > 0
        assert "tool_execution" not in result.steps


class TestMultiAgentWorkflow:
    """Test MultiAgentWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def mock_agent_manager(self):
        """Mock agent manager"""
        manager = Mock()
        mock_agent = Mock()
        mock_agent.execute.return_value = {"response": "Agent response"}
        manager.get_agent.return_value = mock_agent
        return manager

    @pytest.fixture
    def multi_agent_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager, mock_agent_manager):
        """Create a multi-agent workflow instance"""
        return MultiAgentWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager,
            agent_manager=mock_agent_manager
        )

    def test_multi_agent_workflow_init(self, multi_agent_workflow):
        """Test MultiAgentWorkflow initialization"""
        assert multi_agent_workflow.workflow_type == WorkflowType.MULTI_AGENT

    @pytest.mark.asyncio
    async def test_multi_agent_workflow_execute(self, multi_agent_workflow, mock_agent_manager):
        """Test MultiAgentWorkflow execute method"""
        await multi_agent_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Help me with this complex task",
            workflow_type=WorkflowType.MULTI_AGENT,
            metadata={"agents": ["researcher", "analyst"]}
        )
        
        result = await multi_agent_workflow.execute(state)
        
        assert result.output == "Agent response"
        assert len(result.steps) > 0
        assert "agent_coordination" in result.steps
        mock_agent_manager.get_agent.assert_called()


class TestResearcherWorkflow:
    """Test ResearcherWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="I'll research this topic for you.",
            additional_kwargs={"tool_calls": [
                {"name": "search_tool", "args": {"query": "research topic"}}
            ]}
        )
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        mock_tool = Mock()
        mock_tool.execute.return_value = {"result": "Research findings"}
        registry.get_tool.return_value = mock_tool
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def researcher_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a researcher workflow instance"""
        return ResearcherWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_researcher_workflow_init(self, researcher_workflow):
        """Test ResearcherWorkflow initialization"""
        assert researcher_workflow.workflow_type == WorkflowType.RESEARCHER

    @pytest.mark.asyncio
    async def test_researcher_workflow_execute(self, researcher_workflow):
        """Test ResearcherWorkflow execute method"""
        await researcher_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Research the latest developments in AI",
            workflow_type=WorkflowType.RESEARCHER
        )
        
        result = await researcher_workflow.execute(state)
        
        assert result.output == "I'll research this topic for you."
        assert len(result.steps) > 0
        assert "research" in result.steps


class TestAnalystWorkflow:
    """Test AnalystWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="Here's my analysis of the data.",
            additional_kwargs={"tool_calls": [
                {"name": "analysis_tool", "args": {"data": "sample data"}}
            ]}
        )
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        mock_tool = Mock()
        mock_tool.execute.return_value = {"result": "Analysis results"}
        registry.get_tool.return_value = mock_tool
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def analyst_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create an analyst workflow instance"""
        return AnalystWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_analyst_workflow_init(self, analyst_workflow):
        """Test AnalystWorkflow initialization"""
        assert analyst_workflow.workflow_type == WorkflowType.ANALYST

    @pytest.mark.asyncio
    async def test_analyst_workflow_execute(self, analyst_workflow):
        """Test AnalystWorkflow execute method"""
        await analyst_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Analyze this data for trends",
            workflow_type=WorkflowType.ANALYST
        )
        
        result = await analyst_workflow.execute(state)
        
        assert result.output == "Here's my analysis of the data."
        assert len(result.steps) > 0
        assert "analysis" in result.steps


class TestSynthesizerWorkflow:
    """Test SynthesizerWorkflow class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="Here's a synthesis of the information."
        )
        manager.get_llm.return_value = mock_llm
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        mock_memory = Mock()
        mock_memory.get_context.return_value = []
        manager.get_memory.return_value = mock_memory
        return manager

    @pytest.fixture
    def synthesizer_workflow(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a synthesizer workflow instance"""
        return SynthesizerWorkflow(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_synthesizer_workflow_init(self, synthesizer_workflow):
        """Test SynthesizerWorkflow initialization"""
        assert synthesizer_workflow.workflow_type == WorkflowType.SYNTHESIZER

    @pytest.mark.asyncio
    async def test_synthesizer_workflow_execute(self, synthesizer_workflow):
        """Test SynthesizerWorkflow execute method"""
        await synthesizer_workflow.initialize("conv-123")
        
        state = WorkflowState(
            input="Synthesize these findings",
            workflow_type=WorkflowType.SYNTHESIZER,
            metadata={
                "sources": [
                    {"content": "Source 1 information"},
                    {"content": "Source 2 information"}
                ]
            }
        )
        
        result = await synthesizer_workflow.execute(state)
        
        assert result.output == "Here's a synthesis of the information."
        assert len(result.steps) > 0
        assert "synthesis" in result.steps


class TestWorkflowFactory:
    """Test WorkflowFactory class"""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock()
        manager.get_llm.return_value = Mock()
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock()
        return registry

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        manager = Mock()
        manager.get_memory.return_value = Mock()
        return manager

    @pytest.fixture
    def workflow_factory(self, mock_llm_manager, mock_tool_registry, mock_memory_manager):
        """Create a workflow factory instance"""
        return WorkflowFactory(
            llm_manager=mock_llm_manager,
            tool_registry=mock_tool_registry,
            memory_manager=mock_memory_manager
        )

    def test_create_conversational_workflow(self, workflow_factory):
        """Test creating conversational workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.CONVERSATIONAL)
        
        assert isinstance(workflow, ConversationWorkflow)
        assert workflow.workflow_type == WorkflowType.CONVERSATIONAL

    def test_create_tool_heavy_workflow(self, workflow_factory):
        """Test creating tool heavy workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.TOOL_HEAVY)
        
        assert isinstance(workflow, ToolHeavyWorkflow)
        assert workflow.workflow_type == WorkflowType.TOOL_HEAVY

    def test_create_multi_agent_workflow(self, workflow_factory):
        """Test creating multi-agent workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.MULTI_AGENT)
        
        assert isinstance(workflow, MultiAgentWorkflow)
        assert workflow.workflow_type == WorkflowType.MULTI_AGENT

    def test_create_researcher_workflow(self, workflow_factory):
        """Test creating researcher workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.RESEARCHER)
        
        assert isinstance(workflow, ResearcherWorkflow)
        assert workflow.workflow_type == WorkflowType.RESEARCHER

    def test_create_analyst_workflow(self, workflow_factory):
        """Test creating analyst workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.ANALYST)
        
        assert isinstance(workflow, AnalystWorkflow)
        assert workflow.workflow_type == WorkflowType.ANALYST

    def test_create_synthesizer_workflow(self, workflow_factory):
        """Test creating synthesizer workflow"""
        workflow = workflow_factory.create_workflow(WorkflowType.SYNTHESIZER)
        
        assert isinstance(workflow, SynthesizerWorkflow)
        assert workflow.workflow_type == WorkflowType.SYNTHESIZER

    def test_create_workflow_invalid_type(self, workflow_factory):
        """Test creating workflow with invalid type"""
        with pytest.raises(ValueError, match="Unknown workflow type"):
            workflow_factory.create_workflow("invalid_type")


class TestHelperFunctions:
    """Test helper functions"""

    @pytest.mark.asyncio
    async def test_workflow_factory_create(self):
        """Test workflow_factory create function"""
        with patch('app.core.langchain.workflows.LLMManager') as mock_llm_manager, \
             patch('app.core.langchain.workflows.ToolRegistry') as mock_tool_registry, \
             patch('app.core.langchain.workflows.MemoryManager') as mock_memory_manager:
            
            factory = await workflow_factory.create()
            
            assert isinstance(factory, WorkflowFactory)

    def test_global_workflow_factory(self):
        """Test global workflow factory instance"""
        from app.core.langchain.workflows import workflow_factory
        assert workflow_factory is not None
        assert isinstance(workflow_factory, WorkflowFactory)