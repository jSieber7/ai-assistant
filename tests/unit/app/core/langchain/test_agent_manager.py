"""
Unit tests for LangChain Agent Manager.

This module tests the LangGraph-based agent manager functionality,
including agent registration, execution, and state management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from app.core.langchain.agent_manager import (
    LangGraphAgentManager,
    AgentType,
    AgentState,
    MultiAgentState,
    AgentConfig,
    agent_manager
)


class TestAgentType:
    """Test AgentType enum"""

    def test_agent_type_values(self):
        """Test that AgentType has expected values"""
        expected_types = [
            "conversational",
            "tool_heavy",
            "multi_agent",
            "researcher",
            "analyst",
            "synthesizer",
            "coordinator"
        ]
        
        actual_types = [agent_type.value for agent_type in AgentType]
        assert actual_types == expected_types


class TestAgentState:
    """Test AgentState dataclass"""

    def test_agent_state_defaults(self):
        """Test AgentState default values"""
        state = AgentState()
        
        assert state.messages == []
        assert state.current_task is None
        assert state.tool_results == []
        assert state.context == {}
        assert state.conversation_id is None
        assert state.agent_name is None
        assert state.iteration_count == 0
        assert state.max_iterations == 5

    def test_agent_state_with_values(self):
        """Test AgentState with provided values"""
        messages = [Mock()]
        tool_results = [{"result": "test"}]
        context = {"key": "value"}
        
        state = AgentState(
            messages=messages,
            current_task="test task",
            tool_results=tool_results,
            context=context,
            conversation_id="conv-123",
            agent_name="test_agent",
            iteration_count=2,
            max_iterations=10
        )
        
        assert state.messages == messages
        assert state.current_task == "test task"
        assert state.tool_results == tool_results
        assert state.context == context
        assert state.conversation_id == "conv-123"
        assert state.agent_name == "test_agent"
        assert state.iteration_count == 2
        assert state.max_iterations == 10


class TestMultiAgentState:
    """Test MultiAgentState dataclass"""

    def test_multi_agent_state_defaults(self):
        """Test MultiAgentState default values"""
        state = MultiAgentState()
        
        assert state.messages == []
        assert state.current_task is None
        assert state.agent_results == {}
        assert state.active_agents == []
        assert state.coordinator_decision is None
        assert state.context == {}
        assert state.conversation_id is None
        assert state.iteration_count == 0
        assert state.max_iterations == 10

    def test_multi_agent_state_with_values(self):
        """Test MultiAgentState with provided values"""
        messages = [Mock()]
        agent_results = {"researcher": {"result": "test"}}
        active_agents = ["researcher", "analyst"]
        
        state = MultiAgentState(
            messages=messages,
            current_task="test task",
            agent_results=agent_results,
            active_agents=active_agents,
            coordinator_decision="continue",
            context={"key": "value"},
            conversation_id="conv-123",
            iteration_count=3,
            max_iterations=15
        )
        
        assert state.messages == messages
        assert state.current_task == "test task"
        assert state.agent_results == agent_results
        assert state.active_agents == active_agents
        assert state.coordinator_decision == "continue"
        assert state.context == {"key": "value"}
        assert state.conversation_id == "conv-123"
        assert state.iteration_count == 3
        assert state.max_iterations == 15


class TestAgentConfig:
    """Test AgentConfig dataclass"""

    def test_agent_config_defaults(self):
        """Test AgentConfig default values"""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        assert config.name == "test_agent"
        assert config.agent_type == AgentType.CONVERSATIONAL
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.tools == []
        assert config.system_prompt is None
        assert config.max_iterations == 5
        assert config.enable_streaming is False
        assert config.memory_backend == "conversation"
        assert config.metadata == {}

    def test_agent_config_with_values(self):
        """Test AgentConfig with provided values"""
        tools = ["search", "calculator"]
        metadata = {"version": "1.0"}
        
        config = AgentConfig(
            name="advanced_agent",
            agent_type=AgentType.TOOL_HEAVY,
            llm_model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2048,
            tools=tools,
            system_prompt="You are an advanced assistant",
            max_iterations=10,
            enable_streaming=True,
            memory_backend="summary",
            metadata=metadata
        )
        
        assert config.name == "advanced_agent"
        assert config.agent_type == AgentType.TOOL_HEAVY
        assert config.llm_model == "gpt-4-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.tools == tools
        assert config.system_prompt == "You are an advanced assistant"
        assert config.max_iterations == 10
        assert config.enable_streaming is True
        assert config.memory_backend == "summary"
        assert config.metadata == metadata


class TestLangGraphAgentManager:
    """Test LangGraphAgentManager class"""

    @pytest.fixture
    def agent_manager_instance(self):
        """Create a fresh agent manager instance for testing"""
        return LangGraphAgentManager()

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        with patch('app.core.langchain.agent_manager.llm_manager') as mock:
            mock.get_llm = AsyncMock(return_value=Mock())
            yield mock

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        with patch('app.core.langchain.agent_manager.tool_registry') as mock:
            mock.get_tool = Mock(return_value=Mock())
            yield mock

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        with patch('app.core.langchain.agent_manager.memory_manager') as mock:
            mock.get_conversation_messages = AsyncMock(return_value=[])
            mock.add_message = AsyncMock(return_value=True)
            yield mock

    @pytest.fixture
    def mock_monitoring(self):
        """Mock monitoring system"""
        with patch('app.core.langchain.agent_manager.LangChainMonitoring') as mock:
            mock_instance = Mock()
            mock_instance.initialize = AsyncMock()
            mock_instance.track_agent_registration = AsyncMock()
            mock_instance.start_agent_execution = AsyncMock(return_value="exec-123")
            mock_instance.complete_agent_execution = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_initialize(self, agent_manager_instance, mock_monitoring):
        """Test agent manager initialization"""
        await agent_manager_instance.initialize()
        
        assert agent_manager_instance._initialized is True
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, agent_manager_instance, mock_monitoring):
        """Test that initialize is idempotent"""
        await agent_manager_instance.initialize()
        await agent_manager_instance.initialize()
        
        assert agent_manager_instance._initialized is True
        # Should only initialize once
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_agent_success(
        self, 
        agent_manager_instance, 
        mock_llm_manager, 
        mock_tool_registry, 
        mock_monitoring
    ):
        """Test successful agent registration"""
        await agent_manager_instance.initialize()
        
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4",
            tools=["search"]
        )
        
        result = await agent_manager_instance.register_agent(config)
        
        assert result is True
        assert "test_agent" in agent_manager_instance._agent_configs
        assert "test_agent" in agent_manager_instance._agents
        assert "test_agent" in agent_manager_instance._agent_states
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_agent_registration.assert_called_once()
        call_args = mock_monitoring.return_value.track_agent_registration.call_args[1]
        assert call_args["agent_name"] == "test_agent"
        assert call_args["agent_type"] == "conversational"
        assert call_args["success"] is True

    @pytest.mark.asyncio
    async def test_register_agent_duplicate(
        self, 
        agent_manager_instance, 
        mock_monitoring
    ):
        """Test registering duplicate agent"""
        await agent_manager_instance.initialize()
        
        config = AgentConfig(
            name="duplicate_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        # Register first time
        result1 = await agent_manager_instance.register_agent(config)
        assert result1 is True
        
        # Register second time
        result2 = await agent_manager_instance.register_agent(config)
        assert result2 is False
        
        # Verify monitoring was called for failure
        assert mock_monitoring.return_value.track_agent_registration.call_count == 2
        failure_call = mock_monitoring.return_value.track_agent_registration.call_args_list[1][1]
        assert failure_call["success"] is False
        assert "already registered" in failure_call["error"]

    @pytest.mark.asyncio
    async def test_register_agent_creation_failure(
        self, 
        agent_manager_instance, 
        mock_llm_manager, 
        mock_monitoring
    ):
        """Test agent registration with creation failure"""
        await agent_manager_instance.initialize()
        
        # Make LLM manager fail
        mock_llm_manager.get_llm.side_effect = Exception("LLM creation failed")
        
        config = AgentConfig(
            name="failing_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        result = await agent_manager_instance.register_agent(config)
        
        assert result is False
        assert "failing_agent" not in agent_manager_instance._agents
        
        # Verify monitoring was called for failure
        mock_monitoring.return_value.track_agent_registration.assert_called_once()
        call_args = mock_monitoring.return_value.track_agent_registration.call_args[1]
        assert call_args["success"] is False
        assert "Failed to create agent instance" in call_args["error"]

    def test_get_agent(self, agent_manager_instance):
        """Test getting an agent"""
        # Create mock agent
        mock_agent = Mock()
        agent_manager_instance._agents["test_agent"] = mock_agent
        
        # Get existing agent
        result = agent_manager_instance.get_agent("test_agent")
        assert result == mock_agent
        
        # Get non-existent agent
        result = agent_manager_instance.get_agent("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_agents(self, agent_manager_instance):
        """Test listing agents"""
        # Create mock configurations
        config1 = AgentConfig(
            name="agent1",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        config2 = AgentConfig(
            name="agent2",
            agent_type=AgentType.TOOL_HEAVY,
            llm_model="gpt-4"
        )
        
        agent_manager_instance._agent_configs = {
            "agent1": config1,
            "agent2": config2
        }
        
        # Add one agent to _agents (active)
        agent_manager_instance._agents = {"agent1": Mock()}
        
        # Add mock states
        agent_manager_instance._agent_states = {
            "agent1": {
                "created_at": "2023-01-01T00:00:00",
                "last_used": "2023-01-02T00:00:00",
                "usage_count": 5,
                "conversations": {}
            },
            "agent2": {
                "created_at": "2023-01-01T00:00:00",
                "last_used": None,
                "usage_count": 0,
                "conversations": {}
            }
        }
        
        # List active agents only
        result = await agent_manager_instance.list_agents(active_only=True)
        assert len(result) == 1
        assert result[0]["name"] == "agent1"
        assert result[0]["enabled"] is True
        
        # List all agents
        result = await agent_manager_instance.list_agents(active_only=False)
        assert len(result) == 2
        names = [agent["name"] for agent in result]
        assert "agent1" in names
        assert "agent2" in names

    @pytest.mark.asyncio
    async def test_invoke_agent_success(
        self, 
        agent_manager_instance, 
        mock_llm_manager, 
        mock_tool_registry, 
        mock_memory_manager, 
        mock_monitoring
    ):
        """Test successful agent invocation"""
        await agent_manager_instance.initialize()
        
        # Create and register an agent
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        await agent_manager_instance.register_agent(config)
        
        # Mock agent execution
        mock_agent = agent_manager_instance._agents["test_agent"]
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="Test response")]
        })
        
        result = await agent_manager_instance.invoke_agent(
            agent_name="test_agent",
            message="Hello, world!",
            conversation_id="conv-123"
        )
        
        assert result["success"] is True
        assert "response" in result
        assert result["agent_name"] == "test_agent"
        assert result["conversation_id"] == "conv-123"
        
        # Verify monitoring was called
        mock_monitoring.return_value.start_agent_execution.assert_called_once()
        mock_monitoring.return_value.complete_agent_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_agent_not_found(
        self, 
        agent_manager_instance, 
        mock_monitoring
    ):
        """Test invoking non-existent agent"""
        await agent_manager_instance.initialize()
        
        result = await agent_manager_instance.invoke_agent(
            agent_name="nonexistent_agent",
            message="Hello, world!"
        )
        
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["agent_name"] == "nonexistent_agent"
        
        # Verify monitoring was called for failure
        mock_monitoring.return_value.track_agent_execution.assert_called_once()
        call_args = mock_monitoring.return_value.track_agent_execution.call_args[1]
        assert call_args["success"] is False
        assert "not found" in call_args["error"]

    @pytest.mark.asyncio
    async def test_create_conversation(
        self, 
        agent_manager_instance, 
        mock_memory_manager
    ):
        """Test creating a conversation"""
        await agent_manager_instance.initialize()
        
        conversation_id = await agent_manager_instance.create_conversation(
            agent_name="test_agent",
            title="Test Conversation",
            metadata={"test": True}
        )
        
        assert conversation_id is not None
        assert len(conversation_id) > 0  # UUID should be generated
        
        # Verify memory manager was called
        mock_memory_manager.create_conversation.assert_called_once()
        call_args = mock_memory_manager.create_conversation.call_args[1]
        assert call_args["agent_name"] == "test_agent"
        assert call_args["title"] == "Test Conversation"
        assert call_args["metadata"]["test"] is True

    @pytest.mark.asyncio
    async def test_get_conversation_history(
        self, 
        agent_manager_instance, 
        mock_memory_manager
    ):
        """Test getting conversation history"""
        await agent_manager_instance.initialize()
        
        # Mock memory manager response
        mock_messages = [
            {"role": "human", "content": "Hello"},
            {"role": "ai", "content": "Hi there!"}
        ]
        mock_memory_manager.get_conversation_messages.return_value = mock_messages
        
        result = await agent_manager_instance.get_conversation_history(
            agent_name="test_agent",
            conversation_id="conv-123"
        )
        
        assert result == mock_messages
        mock_memory_manager.get_conversation_messages.assert_called_once_with("conv-123")

    @pytest.mark.asyncio
    async def test_reset_agent(self, agent_manager_instance):
        """Test resetting an agent"""
        await agent_manager_instance.initialize()
        
        # Set up agent state
        agent_manager_instance._agent_states["test_agent"] = {
            "created_at": "2023-01-01T00:00:00",
            "last_used": "2023-01-02T00:00:00",
            "usage_count": 5,
            "conversations": {"conv-1": {}, "conv-2": {}}
        }
        
        result = await agent_manager_instance.reset_agent("test_agent")
        
        assert result is True
        state = agent_manager_instance._agent_states["test_agent"]
        assert state["conversations"] == {}
        assert state["usage_count"] == 0
        assert state["last_used"] is None

    @pytest.mark.asyncio
    async def test_reset_agent_not_found(self, agent_manager_instance):
        """Test resetting non-existent agent"""
        await agent_manager_instance.initialize()
        
        result = await agent_manager_instance.reset_agent("nonexistent_agent")
        
        assert result is False

    def test_get_agent_stats(self, agent_manager_instance):
        """Test getting agent statistics"""
        # Set up agent state
        agent_manager_instance._agent_states["test_agent"] = {
            "created_at": "2023-01-01T00:00:00",
            "last_used": "2023-01-02T00:00:00",
            "usage_count": 5,
            "conversations": {"conv-1": {}, "conv-2": {}}
        }
        
        result = agent_manager_instance.get_agent_stats("test_agent")
        
        assert result["created_at"] == "2023-01-01T00:00:00"
        assert result["last_used"] == "2023-01-02T00:00:00"
        assert result["usage_count"] == 5
        assert len(result["conversations"]) == 2

    def test_get_agent_stats_not_found(self, agent_manager_instance):
        """Test getting stats for non-existent agent"""
        result = agent_manager_instance.get_agent_stats("nonexistent_agent")
        
        assert result is None

    def test_get_registry_stats(self, agent_manager_instance):
        """Test getting registry statistics"""
        # Set up registry state
        agent_manager_instance._agent_configs = {
            "agent1": Mock(agent_type=AgentType.CONVERSATIONAL),
            "agent2": Mock(agent_type=AgentType.TOOL_HEAVY),
            "agent3": Mock(agent_type=AgentType.RESEARCHER)
        }
        agent_manager_instance._agents = {
            "agent1": Mock(),
            "agent2": Mock()
        }
        agent_manager_instance._workflows = {
            "conversational": Mock(),
            "tool_heavy": Mock(),
            "multi_agent": Mock()
        }
        
        result = agent_manager_instance.get_registry_stats()
        
        assert result["total_agents"] == 3
        assert result["active_agents"] == 2
        assert result["total_workflows"] == 3
        assert len(result["agent_types"]) == 3
        assert result["agent_types"]["conversational"] == 1
        assert result["agent_types"]["tool_heavy"] == 1
        assert result["agent_types"]["researcher"] == 1
        assert len(result["available_workflows"]) == 3

    @pytest.mark.asyncio
    async def test_create_agent_instance(
        self, 
        agent_manager_instance, 
        mock_llm_manager, 
        mock_tool_registry
    ):
        """Test creating an agent instance"""
        await agent_manager_instance.initialize()
        
        # Set up workflows
        agent_manager_instance._workflows = {
            "conversational": Mock()
        }
        
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4",
            tools=["search"]
        )
        
        result = await agent_manager_instance._create_agent_instance(config)
        
        assert result is not None
        mock_llm_manager.get_llm.assert_called_once_with(
            "gpt-4",
            temperature=0.7,
            max_tokens=None,
            streaming=False
        )
        mock_tool_registry.get_tool.assert_called_once_with("search")

    @pytest.mark.asyncio
    async def test_create_agent_instance_workflow_not_found(
        self, 
        agent_manager_instance, 
        mock_llm_manager
    ):
        """Test creating agent instance with missing workflow"""
        await agent_manager_instance.initialize()
        
        # Don't set up workflows
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        result = await agent_manager_instance._create_agent_instance(config)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_invoke_agent_sync(
        self, 
        agent_manager_instance, 
        mock_memory_manager
    ):
        """Test synchronous agent invocation"""
        await agent_manager_instance.initialize()
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="Test response")]
        })
        
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        state = AgentState(
            messages=[Mock(content="Hello")],
            current_task="Hello",
            conversation_id="conv-123",
            agent_name="test_agent"
        )
        
        result = await agent_manager_instance._invoke_agent_sync(
            mock_agent, state, config
        )
        
        assert result["success"] is True
        assert "response" in result
        assert result["agent_name"] == "test_agent"
        assert result["conversation_id"] == "conv-123"
        
        # Verify memory manager was called
        mock_memory_manager.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_agent_sync_failure(
        self, 
        agent_manager_instance
    ):
        """Test synchronous agent invocation with failure"""
        await agent_manager_instance.initialize()
        
        # Create mock agent that fails
        mock_agent = Mock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("Agent execution failed"))
        
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        state = AgentState(
            messages=[Mock(content="Hello")],
            current_task="Hello",
            conversation_id="conv-123",
            agent_name="test_agent"
        )
        
        result = await agent_manager_instance._invoke_agent_sync(
            mock_agent, state, config
        )
        
        assert result["success"] is False
        assert "error" in result
        assert result["agent_name"] == "test_agent"
        assert result["conversation_id"] == "conv-123"

    @pytest.mark.asyncio
    async def test_invoke_agent_stream(
        self, 
        agent_manager_instance, 
        mock_memory_manager
    ):
        """Test streaming agent invocation"""
        await agent_manager_instance.initialize()
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="Test response")]
        })
        
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CONVERSATIONAL,
            llm_model="gpt-4"
        )
        
        state = AgentState(
            messages=[Mock(content="Hello")],
            current_task="Hello",
            conversation_id="conv-123",
            agent_name="test_agent"
        )
        
        # Stream mode should fall back to sync for now
        result = await agent_manager_instance._invoke_agent_stream(
            mock_agent, state, config
        )
        
        assert result["success"] is True
        assert "response" in result


class TestGlobalAgentManager:
    """Test global agent manager instance"""

    def test_global_instance(self):
        """Test that global agent manager instance exists"""
        from app.core.langchain.agent_manager import agent_manager
        assert agent_manager is not None
        assert isinstance(agent_manager, LangGraphAgentManager)