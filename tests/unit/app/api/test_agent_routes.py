"""
Unit tests for agent API routes.

Tests for the agent system endpoints including agent-based chat completions,
agent management, and agent registry operations.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app


class TestAgentChatCompletions:
    """Test cases for the /api/v1/agents/chat/completions endpoint"""

    @pytest.mark.asyncio
    async def test_agent_chat_completions_legacy(self, mock_agent_chat_request):
        """Test agent chat completions with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent result
                    mock_result = MagicMock()
                    mock_result.response = "Test agent response"
                    mock_result.agent_name = "test-agent"
                    mock_result.tool_results = []
                    mock_result.conversation_id = str(uuid.uuid4())
                    mock_result.execution_time = 1.5
                    mock_result.metadata = {}
                    mock_registry.process_message.return_value = mock_result
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/chat/completions", json=mock_agent_chat_request.dict())
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["object"] == "agent.chat.completion"
                    assert data["agent_name"] == "test-agent"
                    assert data["choices"][0]["message"]["content"] == "Test agent response"

    @pytest.mark.asyncio
    async def test_agent_chat_completions_langchain(self, mock_agent_chat_request):
        """Test agent chat completions with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.invoke_agent = AsyncMock(return_value=MagicMock(
                    response="Test LangChain response",
                    agent_name="langchain-agent",
                    tool_results=[],
                    conversation_id=str(uuid.uuid4()),
                    execution_time=1.2,
                    metadata={}
                ))
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/api/v1/agents/chat/completions", json=mock_agent_chat_request.dict())
                
                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "agent.chat.completion"
                assert data["agent_name"] == "langchain-agent"

    @pytest.mark.asyncio
    async def test_agent_chat_completions_disabled(self, mock_agent_chat_request):
        """Test agent chat completions when agent system is disabled"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = False
            
            client = TestClient(app)
            response = client.post("/api/v1/agents/chat/completions", json=mock_agent_chat_request.dict())
            
            assert response.status_code == 503
            assert "Agent system is disabled" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_agent_chat_completions_no_user_message(self):
        """Test agent chat completions with no user message"""
        # Create request with only system messages
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}
            ],
            "agent_name": "test-agent"
        }
        
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            client = TestClient(app)
            response = client.post("/api/v1/agents/chat/completions", json=request_data)
            
            assert response.status_code == 400
            assert "No user messages found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_agent_chat_completions_with_tool_results(self, mock_agent_chat_request):
        """Test agent chat completions with tool results"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent result with tool results
                    mock_tool_result = MagicMock()
                    mock_tool_result.tool_name = "calculator"
                    mock_tool_result.success = True
                    mock_tool_result.execution_time = 0.5
                    mock_tool_result.data = {"result": 42}
                    mock_tool_result.error = None
                    mock_tool_result.metadata = {}
                    
                    mock_result = MagicMock()
                    mock_result.response = "Calculation complete"
                    mock_result.agent_name = "test-agent"
                    mock_result.tool_results = [mock_tool_result]
                    mock_result.conversation_id = str(uuid.uuid4())
                    mock_registry.process_message.return_value = mock_result
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/chat/completions", json=mock_agent_chat_request.dict())
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "tool_results" in data
                    assert len(data["tool_results"]) == 1
                    assert data["tool_results"][0]["tool_name"] == "calculator"
                    assert data["tool_results"][0]["success"] is True


class TestAgentListing:
    """Test cases for agent listing endpoints"""

    @pytest.mark.asyncio
    async def test_list_agents_legacy(self):
        """Test listing agents with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent
                    mock_agent = MagicMock()
                    mock_agent.name = "test-agent"
                    mock_agent.description = "Test agent"
                    mock_agent.version = "1.0.0"
                    mock_agent.state.value = "active"
                    mock_agent.categories = ["general"]
                    mock_agent.get_usage_stats.return_value = {
                        "usage_count": 10,
                        "last_used": 1234567890
                    }
                    mock_registry.list_agents.return_value = [mock_agent]
                    
                    client = TestClient(app)
                    response = client.get("/api/v1/agents/")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "agents" in data
                    assert len(data["agents"]) == 1
                    assert data["agents"][0]["name"] == "test-agent"
                    assert data["total_count"] == 1

    @pytest.mark.asyncio
    async def test_list_agents_langchain(self):
        """Test listing agents with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.list_agents.return_value = [
                    {
                        "name": "langchain-agent",
                        "description": "LangChain agent",
                        "type": "conversational",
                        "enabled": True,
                        "usage_count": 5,
                        "last_used": 1234567890
                    }
                ]
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.get("/api/v1/agents/")
                
                assert response.status_code == 200
                data = response.json()
                assert "agents" in data
                assert len(data["agents"]) == 1
                assert data["agents"][0]["name"] == "langchain-agent"

    @pytest.mark.asyncio
    async def test_list_agents_disabled(self):
        """Test listing agents when agent system is disabled"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = False
            
            client = TestClient(app)
            response = client.get("/api/v1/agents/")
            
            assert response.status_code == 200
            data = response.json()
            assert data["agents"] == []
            assert "Agent system is disabled" in data["message"]

    @pytest.mark.asyncio
    async def test_get_agent_info_legacy(self):
        """Test getting specific agent info with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent
                    mock_agent = MagicMock()
                    mock_agent.name = "test-agent"
                    mock_agent.description = "Test agent"
                    mock_agent.version = "1.0.0"
                    mock_agent.state.value = "active"
                    mock_agent.categories = ["general"]
                    mock_agent.get_usage_stats.return_value = {
                        "usage_count": 10,
                        "last_used": 1234567890,
                        "current_conversation_id": str(uuid.uuid4()),
                        "conversation_history_length": 5
                    }
                    mock_registry.get_agent.return_value = mock_agent
                    mock_registry.get_registry_stats.return_value = {
                        "default_agent": "test-agent",
                        "active_agents": ["test-agent"]
                    }
                    
                    client = TestClient(app)
                    response = client.get("/api/v1/agents/test-agent")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["agent"]["name"] == "test-agent"
                    assert data["registry_info"]["is_default"] is True
                    assert data["registry_info"]["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_agent_info_not_found(self):
        """Test getting info for non-existent agent"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.get_agent.return_value = None
                    
                    client = TestClient(app)
                    response = client.get("/api/v1/agents/non-existent-agent")
                    
                    assert response.status_code == 404
                    assert "not found" in response.json()["detail"]


class TestAgentActivation:
    """Test cases for agent activation/deactivation endpoints"""

    @pytest.mark.asyncio
    async def test_activate_agent_legacy(self):
        """Test activating an agent with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.activate_agent.return_value = True
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/test-agent/activate")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "activated successfully" in data["message"]
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_activate_agent_langchain(self):
        """Test activating an agent with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.activate_agent = AsyncMock(return_value=True)
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/api/v1/agents/test-agent/activate")
                
                assert response.status_code == 200
                data = response.json()
                assert "activated successfully" in data["message"]
                assert data["registry"] == "langgraph"

    @pytest.mark.asyncio
    async def test_deactivate_agent_legacy(self):
        """Test deactivating an agent with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.deactivate_agent.return_value = True
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/test-agent/deactivate")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "deactivated successfully" in data["message"]
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_set_default_agent_legacy(self):
        """Test setting an agent as default with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.set_default_agent.return_value = True
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/test-agent/set-default")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "set as default successfully" in data["message"]
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_set_default_agent_langchain(self):
        """Test setting an agent as default with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/api/v1/agents/test-agent/set-default")
                
                assert response.status_code == 501
                assert "not supported in LangGraph mode" in response.json()["detail"]


class TestAgentConversation:
    """Test cases for agent conversation endpoints"""

    @pytest.mark.asyncio
    async def test_get_conversation_history_legacy(self):
        """Test getting conversation history with legacy registry"""
        conversation_id = str(uuid.uuid4())
        
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent
                    mock_agent = MagicMock()
                    mock_agent.get_conversation_history.return_value = [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ]
                    mock_registry.get_agent.return_value = mock_agent
                    
                    client = TestClient(app)
                    response = client.get(f"/api/v1/agents/test-agent/conversation/{conversation_id}")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["agent_name"] == "test-agent"
                    assert data["conversation_id"] == conversation_id
                    assert len(data["history"]) == 2
                    assert data["message_count"] == 2
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_get_conversation_history_langchain(self):
        """Test getting conversation history with LangChain integration"""
        conversation_id = str(uuid.uuid4())
        
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.get_conversation_history.return_value = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.get(f"/api/v1/agents/test-agent/conversation/{conversation_id}")
                
                assert response.status_code == 200
                data = response.json()
                assert data["agent_name"] == "test-agent"
                assert data["conversation_id"] == conversation_id
                assert len(data["history"]) == 2
                assert data["registry"] == "langgraph"


class TestAgentReset:
    """Test cases for agent reset endpoints"""

    @pytest.mark.asyncio
    async def test_reset_agent_legacy(self):
        """Test resetting an agent with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    # Mock agent
                    mock_agent = MagicMock()
                    mock_agent.reset = MagicMock()
                    mock_registry.get_agent.return_value = mock_agent
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/test-agent/reset")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "reset successfully" in data["message"]
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_reset_all_agents_legacy(self):
        """Test resetting all agents with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.reset_all_agents = MagicMock()
                    
                    client = TestClient(app)
                    response = client.post("/api/v1/agents/registry/reset")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "All agents reset successfully" in data["message"]
                    assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_reset_all_agents_langchain(self):
        """Test resetting all agents with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.reset_all_agents = AsyncMock()
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/api/v1/agents/registry/reset")
                
                assert response.status_code == 200
                data = response.json()
                assert "All agents reset successfully" in data["message"]
                assert data["registry"] == "langgraph"


class TestAgentRegistry:
    """Test cases for agent registry endpoints"""

    @pytest.mark.asyncio
    async def test_get_registry_info_legacy(self):
        """Test getting registry info with legacy registry"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.agent_routes.agent_registry') as mock_registry:
                    mock_registry.get_registry_stats.return_value = {
                        "total_agents": 5,
                        "active_agents": 3,
                        "default_agent": "test-agent",
                        "categories": ["general", "search"],
                        "agents_by_category": {"general": 3, "search": 2}
                    }
                    
                    client = TestClient(app)
                    response = client.get("/api/v1/agents/registry/info")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["total_agents"] == 5
                    assert data["active_agents"] == 3
                    assert data["default_agent"] == "test-agent"
                    assert "general" in data["categories"]
                    assert data["agents_by_category"]["general"] == 3

    @pytest.mark.asyncio
    async def test_get_registry_info_langchain(self):
        """Test getting registry info with LangChain integration"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_agents = True
                mock_integration.get_agent_registry_stats.return_value = {
                    "total_agents": 5,
                    "active_agents": 3,
                    "agent_types": {"conversational": 3, "search": 2}
                }
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.get("/api/v1/agents/registry/info")
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_agents"] == 5
                assert data["active_agents"] == 3
                assert data["default_agent"] is None  # LangGraph doesn't have default concept
                assert "conversational" in data["categories"]


class TestAgentTools:
    """Test cases for agent tools endpoints"""

    @pytest.mark.asyncio
    async def test_get_available_tools(self):
        """Test getting available tools"""
        with patch('app.api.agent_routes.tool_registry') as mock_registry:
            # Mock tool
            mock_tool = MagicMock()
            mock_tool.name = "calculator"
            mock_tool.description = "Performs calculations"
            mock_tool.version = "1.0.0"
            mock_tool.author = "AI Assistant Team"
            mock_tool.enabled = True
            mock_tool.keywords = ["math", "calculation"]
            mock_tool.categories = ["utility"]
            mock_tool.get_usage_stats.return_value = {
                "usage_count": 10,
                "last_used": 1234567890
            }
            mock_tool.timeout = 30
            
            mock_registry.list_tools.return_value = [mock_tool]
            mock_registry.get_registry_stats.return_value = {
                "total_tools": 1,
                "enabled_tools": 1,
                "categories": ["utility"]
            }
            
            client = TestClient(app)
            response = client.get("/api/v1/agents/tools/available")
            
            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert len(data["tools"]) == 1
            assert data["tools"][0]["name"] == "calculator"
            assert data["total_tools"] == 1
            assert data["enabled_tools"] == 1


class TestAgentHealth:
    """Test cases for agent health endpoint"""

    def test_agent_system_health_enabled(self):
        """Test agent system health when enabled"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = True
            
            with patch('app.api.agent_routes.agent_registry') as mock_registry:
                mock_registry.get_registry_stats.return_value = {
                    "total_agents": 5,
                    "active_agents": 3
                }
                
                with patch('app.api.agent_routes.tool_registry') as mock_tool_registry:
                    mock_tool_registry.get_registry_stats.return_value = {
                        "total_tools": 10,
                        "enabled_tools": 8
                    }
                    
                    client = TestClient(app)
                    response = client.get("/api/v1/agents/health")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["agent_system_enabled"] is True
                    assert data["agent_registry_healthy"] is True
                    assert data["tool_registry_healthy"] is True
                    assert data["active_agents"] == 3
                    assert data["enabled_tools"] == 8
                    assert data["status"] == "healthy"

    def test_agent_system_health_disabled(self):
        """Test agent system health when disabled"""
        with patch('app.api.agent_routes.settings') as mock_settings:
            mock_settings.agent_system_enabled = False
            
            with patch('app.api.agent_routes.tool_registry') as mock_tool_registry:
                mock_tool_registry.get_registry_stats.return_value = {
                    "total_tools": 10,
                    "enabled_tools": 8
                }
                
                client = TestClient(app)
                response = client.get("/api/v1/agents/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data["agent_system_enabled"] is False
                assert data["agent_registry_healthy"] is True  # True when disabled
                assert data["tool_registry_healthy"] is True
                assert data["active_agents"] == 0
                assert data["status"] == "disabled"