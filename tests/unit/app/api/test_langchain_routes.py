"""
Unit tests for LangChain API routes.

This module tests the API routes that have been updated to use LangChain components,
ensuring proper integration and backward compatibility.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.core.langchain.integration import LangChainIntegration
from app.core.langchain.llm_manager import LangChainLLMManager
from app.core.langchain.tool_registry import LangChainToolRegistry
from app.core.langchain.agent_manager import LangGraphAgentManager
from app.core.langchain.memory_manager import LangChainMemoryManager
from app.core.langchain.monitoring import LangChainMonitoring


class TestLangChainRoutes:
    """Test cases for LangChain API routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_integration(self):
        """Mock LangChain integration"""
        integration = Mock(spec=LangChainIntegration)
        integration.initialize = AsyncMock()
        integration.get_integration_mode = Mock(return_value="langchain")
        integration.is_component_enabled = Mock(return_value=True)
        integration.health_check = AsyncMock(return_value={
            "overall_status": "healthy",
            "components": {
                "llm_manager": {"status": "healthy"},
                "tool_registry": {"status": "healthy"},
                "agent_manager": {"status": "healthy"},
                "memory_manager": {"status": "healthy"},
                "monitoring": {"status": "healthy"}
            }
        })
        
        # Mock component managers
        integration._llm_manager = Mock(spec=LangChainLLMManager)
        integration._tool_registry = Mock(spec=LangChainToolRegistry)
        integration._agent_manager = Mock(spec=LangGraphAgentManager)
        integration._memory_manager = Mock(spec=LangChainMemoryManager)
        integration._monitoring = Mock(spec=LangChainMonitoring)
        
        return integration
    
    def test_langchain_health_endpoint(self, client, mock_integration):
        """Test LangChain health check endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "overall_status" in data
            assert "components" in data
            assert data["overall_status"] == "healthy"
    
    def test_langchain_status_endpoint(self, client, mock_integration):
        """Test LangChain status endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/status")
            
            assert response.status_code == 200
            data = response.json()
            assert "integration_mode" in data
            assert "components" in data
            assert data["integration_mode"] == "langchain"
    
    def test_langchain_llm_providers_endpoint(self, client, mock_integration):
        """Test LangChain LLM providers endpoint"""
        mock_integration._llm_manager.list_providers.return_value = [
            {"name": "openai", "models": ["gpt-3.5-turbo", "gpt-4"]},
            {"name": "ollama", "models": ["llama2", "mistral"]}
        ]
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/llm/providers")
            
            assert response.status_code == 200
            data = response.json()
            assert "providers" in data
            assert len(data["providers"]) == 2
    
    def test_langchain_llm_request_endpoint(self, client, mock_integration):
        """Test LangChain LLM request endpoint"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = "This is a test response"
        mock_integration._llm_manager.get_llm.return_value = mock_llm
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/llm/request",
                json={
                    "model": "gpt-3.5-turbo",
                    "prompt": "Hello, world!"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "model" in data
            assert data["response"] == "This is a test response"
            assert data["model"] == "gpt-3.5-turbo"
    
    def test_langchain_tools_list_endpoint(self, client, mock_integration):
        """Test LangChain tools list endpoint"""
        mock_integration._tool_registry.list_tools.return_value = [
            {"name": "search_tool", "description": "Search the web"},
            {"name": "scraper_tool", "description": "Scrape web pages"}
        ]
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/tools")
            
            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert len(data["tools"]) == 2
    
    def test_langchain_tool_execute_endpoint(self, client, mock_integration):
        """Test LangChain tool execute endpoint"""
        mock_integration._tool_registry.execute_tool.return_value = {
            "success": True,
            "result": "Tool executed successfully"
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/tools/execute",
                json={
                    "tool_name": "search_tool",
                    "parameters": {"query": "test query"}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert "success" in data
            assert data["success"] is True
    
    def test_langchain_agents_list_endpoint(self, client, mock_integration):
        """Test LangChain agents list endpoint"""
        mock_integration._agent_manager.list_agents.return_value = [
            {"name": "research_agent", "description": "Research agent"},
            {"name": "writer_agent", "description": "Writing agent"}
        ]
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/agents")
            
            assert response.status_code == 200
            data = response.json()
            assert "agents" in data
            assert len(data["agents"]) == 2
    
    def test_langchain_agent_invoke_endpoint(self, client, mock_integration):
        """Test LangChain agent invoke endpoint"""
        mock_integration._agent_manager.invoke_agent.return_value = {
            "result": "Agent executed successfully",
            "messages": [{"role": "assistant", "content": "Response"}]
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/agents/invoke",
                json={
                    "agent_name": "research_agent",
                    "input_data": {"messages": [{"role": "user", "content": "Hello"}]},
                    "config": {"thread_id": "test_thread"}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert "messages" in data
            assert len(data["messages"]) == 1
    
    def test_langchain_memory_conversations_endpoint(self, client, mock_integration):
        """Test LangChain memory conversations endpoint"""
        mock_integration._memory_manager.list_conversations.return_value = [
            {"conversation_id": "conv1", "agent_name": "agent1", "message_count": 5},
            {"conversation_id": "conv2", "agent_name": "agent2", "message_count": 3}
        ]
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/memory/conversations")
            
            assert response.status_code == 200
            data = response.json()
            assert "conversations" in data
            assert len(data["conversations"]) == 2
    
    def test_langchain_memory_messages_endpoint(self, client, mock_integration):
        """Test LangChain memory messages endpoint"""
        mock_integration._memory_manager.get_conversation_messages.return_value = [
            {"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2023-01-01T00:00:01"}
        ]
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/memory/conversations/conv1/messages")
            
            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            assert len(data["messages"]) == 2
    
    def test_langchain_memory_create_endpoint(self, client, mock_integration):
        """Test LangChain memory create conversation endpoint"""
        mock_integration._memory_manager.create_conversation.return_value = {
            "conversation_id": "new_conv",
            "agent_name": "test_agent",
            "created_at": "2023-01-01T00:00:00"
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/memory/conversations",
                json={
                    "conversation_id": "new_conv",
                    "agent_name": "test_agent"
                }
            )
            
            assert response.status_code == 201
            data = response.json()
            assert "conversation_id" in data
            assert data["conversation_id"] == "new_conv"
    
    def test_langchain_memory_add_message_endpoint(self, client, mock_integration):
        """Test LangChain memory add message endpoint"""
        mock_integration._memory_manager.add_message.return_value = {
            "message_id": "msg1",
            "conversation_id": "conv1",
            "role": "user",
            "content": "Hello",
            "timestamp": "2023-01-01T00:00:00"
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/memory/conversations/conv1/messages",
                json={
                    "role": "user",
                    "content": "Hello"
                }
            )
            
            assert response.status_code == 201
            data = response.json()
            assert "message_id" in data
            assert data["conversation_id"] == "conv1"
            assert data["role"] == "user"
    
    def test_langchain_monitoring_metrics_endpoint(self, client, mock_integration):
        """Test LangChain monitoring metrics endpoint"""
        mock_integration._monitoring.get_metrics.return_value = {
            "llm_requests": {"count": 100, "avg_duration": 1.5},
            "tool_executions": {"count": 50, "success_rate": 0.95},
            "agent_invocations": {"count": 25, "avg_duration": 2.0}
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/monitoring/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert "llm_requests" in data["metrics"]
            assert "tool_executions" in data["metrics"]
            assert "agent_invocations" in data["metrics"]
    
    def test_langchain_monitoring_component_metrics_endpoint(self, client, mock_integration):
        """Test LangChain monitoring component metrics endpoint"""
        mock_integration._monitoring.get_component_metrics.return_value = {
            "component_type": "llm",
            "component_name": "openai",
            "metrics": {
                "request_count": 50,
                "success_rate": 0.98,
                "avg_duration": 1.2
            }
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/monitoring/metrics/llm/openai")
            
            assert response.status_code == 200
            data = response.json()
            assert "component_type" in data
            assert "component_name" in data
            assert "metrics" in data
            assert data["component_type"] == "llm"
            assert data["component_name"] == "openai"
    
    def test_langchain_monitoring_performance_endpoint(self, client, mock_integration):
        """Test LangChain monitoring performance endpoint"""
        mock_integration._monitoring.get_performance_metrics.return_value = {
            "system_performance": {
                "avg_response_time": 1.5,
                "throughput": 10.0,
                "error_rate": 0.02
            },
            "component_performance": {
                "llm_manager": {"avg_response_time": 1.2},
                "tool_registry": {"avg_response_time": 0.5},
                "agent_manager": {"avg_response_time": 2.0}
            }
        }
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/monitoring/performance")
            
            assert response.status_code == 200
            data = response.json()
            assert "system_performance" in data
            assert "component_performance" in data
            assert "avg_response_time" in data["system_performance"]
    
    def test_langchain_specialized_agents_endpoint(self, client, mock_integration):
        """Test LangChain specialized agents endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/langchain/agents/specialized")
            
            assert response.status_code == 200
            data = response.json()
            assert "agents" in data
            assert len(data["agents"]) >= 9  # Should have at least 9 specialized agents
    
    def test_langchain_summarize_agent_endpoint(self, client, mock_integration):
        """Test LangChain summarize agent endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            with patch('app.core.langchain.specialized_agents.SummarizeAgent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.summarize.return_value = {
                    "summary": "This is a summary",
                    "word_count": 4,
                    "compression_ratio": 0.5
                }
                mock_agent_class.return_value = mock_agent
                
                response = client.post(
                    "/api/langchain/agents/specialized/summarize",
                    json={
                        "text": "This is a longer text that needs to be summarized.",
                        "target_length": 10
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "summary" in data
                assert "word_count" in data
                assert data["summary"] == "This is a summary"
    
    def test_langchain_search_query_agent_endpoint(self, client, mock_integration):
        """Test LangChain search query agent endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            with patch('app.core.langchain.specialized_agents.SearchQueryAgent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.generate_query.return_value = {
                    "query": "best search query",
                    "keywords": ["search", "query"],
                    "intent": "informational"
                }
                mock_agent_class.return_value = mock_agent
                
                response = client.post(
                    "/api/langchain/agents/specialized/search-query",
                    json={
                        "topic": "machine learning applications"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "query" in data
                assert "keywords" in data
                assert data["query"] == "best search query"
    
    def test_langchain_creative_story_agent_endpoint(self, client, mock_integration):
        """Test LangChain creative story agent endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            with patch('app.core.langchain.specialized_agents.CreativeStoryAgent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.generate.return_value = {
                    "story": "Once upon a time...",
                    "genre": "fantasy",
                    "word_count": 100
                }
                mock_agent_class.return_value = mock_agent
                
                response = client.post(
                    "/api/langchain/agents/specialized/creative-story",
                    json={
                        "prompt": "A magical forest",
                        "genre": "fantasy"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "story" in data
                assert "genre" in data
                assert data["story"] == "Once upon a time..."
    
    def test_langchain_fact_checker_agent_endpoint(self, client, mock_integration):
        """Test LangChain fact checker agent endpoint"""
        with patch('app.api.routes.langchain_integration', mock_integration):
            with patch('app.core.langchain.specialized_agents.FactCheckerAgent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.check_fact.return_value = {
                    "verdict": "true",
                    "confidence": 0.95,
                    "sources": ["https://example.com/source1"],
                    "evidence": ["Evidence supporting the claim"]
                }
                mock_agent_class.return_value = mock_agent
                
                response = client.post(
                    "/api/langchain/agents/specialized/fact-checker",
                    json={
                        "claim": "The Earth is round"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "verdict" in data
                assert "confidence" in data
                assert data["verdict"] == "true"
    
    def test_langchain_error_handling(self, client, mock_integration):
        """Test LangChain error handling"""
        mock_integration._llm_manager.get_llm.side_effect = Exception("LLM error")
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.post(
                "/api/langchain/llm/request",
                json={
                    "model": "gpt-3.5-turbo",
                    "prompt": "Hello, world!"
                }
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "LLM error" in data["error"]
    
    def test_langchain_backward_compatibility(self, client, mock_integration):
        """Test backward compatibility with legacy endpoints"""
        # Test that legacy endpoints still work when integration mode is 'legacy'
        mock_integration.get_integration_mode.return_value = "legacy"
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            response = client.get("/api/health")
            
            # Should still respond with health status
            assert response.status_code == 200
    
    def test_langchain_feature_flags(self, client, mock_integration):
        """Test LangChain feature flags"""
        # Test with different component enabled states
        mock_integration.is_component_enabled.side_effect = lambda component: {
            "llm_manager": True,
            "tool_registry": False,
            "agent_manager": True,
            "memory_manager": False,
            "monitoring": True
        }.get(component, False)
        
        with patch('app.api.routes.langchain_integration', mock_integration):
            # LLM endpoints should work
            response = client.get("/api/langchain/llm/providers")
            assert response.status_code == 200
            
            # Tool endpoints should return 503 (disabled)
            response = client.get("/api/langchain/tools")
            assert response.status_code == 503
            
            # Agent endpoints should work
            response = client.get("/api/langchain/agents")
            assert response.status_code == 200
            
            # Memory endpoints should return 503 (disabled)
            response = client.get("/api/langchain/memory/conversations")
            assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__])