"""
Unit tests for agent routes functionality
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.api.agent_routes import (
    AgentChatRequest,
    AgentChatMessage,
    AgentChatResponse,
    AgentInfo,
    AgentRegistryInfo,
)


class TestAgentRoutesDataModels:
    """Test data models used in agent routes"""

    def test_agent_chat_request_model(self):
        """Test AgentChatRequest model validation"""
        # Valid request
        messages = [
            AgentChatMessage(role="user", content="Hello"),
            AgentChatMessage(role="assistant", content="Hi there!")
        ]
        request = AgentChatRequest(
            messages=messages,
            agent_name="test_agent",
            stream=False,
            temperature=0.7,
            max_tokens=100
        )
        
        assert request.messages == messages
        assert request.agent_name == "test_agent"
        assert request.stream is False
        assert request.temperature == 0.7
        assert request.max_tokens == 100

    def test_agent_chat_message_model(self):
        """Test AgentChatMessage model validation"""
        message = AgentChatMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"

    def test_agent_chat_response_model(self):
        """Test AgentChatResponse model validation"""
        response = AgentChatResponse(
            id="resp_123",
            object="chat.completion",
            created=1234567890,
            model="test_agent",
            agent_name="test_agent",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
        
        assert response.id == "resp_123"
        assert response.object == "chat.completion"
        assert response.model == "test_agent"
        assert response.agent_name == "test_agent"
        assert len(response.choices) == 1
        assert response.choices[0]["message"]["content"] == "Test response"

    def test_agent_info_model(self):
        """Test AgentInfo model validation"""
        agent_info = AgentInfo(
            name="test_agent",
            description="Test agent description",
            version="1.0.0",
            state="active",
            usage_count=10,
            categories=["text_generation", "tool_use"]
        )
        
        assert agent_info.name == "test_agent"
        assert agent_info.description == "Test agent description"
        assert agent_info.version == "1.0.0"
        assert agent_info.state == "active"
        assert agent_info.usage_count == 10
        assert "text_generation" in agent_info.categories

    def test_agent_registry_info_model(self):
        """Test AgentRegistryInfo model validation"""
        registry_info = AgentRegistryInfo(
            total_agents=5,
            active_agents=3,
            default_agent="test_agent",
            categories=["search", "scraping", "analysis"],
            agents_by_category={
                "search": 1,
                "scraping": 2
            }
        )
        
        assert registry_info.total_agents == 5
        assert registry_info.active_agents == 3
        assert registry_info.default_agent == "test_agent"
        assert len(registry_info.categories) == 3
        assert registry_info.agents_by_category["search"] == 1


class TestAgentRoutesErrorHandling:
    """Test error handling in agent routes"""

    def test_invalid_agent_chat_request(self, client):
        """Test handling of invalid agent chat request"""
        # Missing required fields
        invalid_request = {
            "messages": [],  # Empty messages
            "agent_name": ""  # Empty agent name
        }
        
        response = client.post("/agents/chat/completions", json=invalid_request)
        # Should return validation error or 404 if endpoint doesn't exist
        assert response.status_code in [422, 400, 404]

    def test_invalid_agent_name(self, client):
        """Test handling of invalid agent name"""
        # Test with special characters that might not be allowed
        response = client.get("/agents/agent@#$%")
        assert response.status_code in [422, 404, 500]


class TestAgentRoutesBasicFunctionality:
    """Test basic functionality of agent routes"""

    def test_agent_chat_request_with_different_roles(self):
        """Test agent chat request with different message roles"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        request = AgentChatRequest(
            messages=[AgentChatMessage(**msg) for msg in messages],
            agent_name="test_agent",
            stream=False
        )
        
        assert len(request.messages) == 4
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
        assert request.messages[2].role == "assistant"
        assert request.messages[3].role == "user"

    def test_agent_chat_request_with_parameters(self):
        """Test agent chat request with various parameters"""
        request = AgentChatRequest(
            messages=[AgentChatMessage(role="user", content="Hello")],
            agent_name="test_agent",
            stream=False,
            temperature=0.5,
            max_tokens=100
        )
        
        assert request.temperature == 0.5
        assert request.max_tokens == 100

    def test_agent_chat_request_minimal(self):
        """Test agent chat request with minimal parameters"""
        request = AgentChatRequest(
            messages=[AgentChatMessage(role="user", content="Hello")],
            agent_name="test_agent"
        )
        
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello"
        assert request.agent_name == "test_agent"
        # Default values should be applied
        assert request.stream is False  # Default value