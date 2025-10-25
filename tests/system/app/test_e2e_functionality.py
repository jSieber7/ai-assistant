"""
System tests for end-to-end functionality.

This module tests the entire application stack with real user scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import httpx
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.core.config import Settings
from app.core.caching.layers.memory import MemoryCache
from app.core.tools.execution.dynamic_executor import DynamicExecutor
from app.core.tools.execution.registry import ToolRegistry
from app.core.tools.web.searxng_tool import SearXNGTool
from app.core.tools.web.firecrawl_tool import FirecrawlTool


class TestE2EFunctionality:
    """Test end-to-end functionality"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def test_settings(self):
        """Create test settings"""
        return Settings(
            environment="test",
            debug=True,
            api_host="localhost",
            api_port=8000,
            redis_url="redis://localhost:6379/0",
            cache_enabled=True,
            cache_ttl=300,
            llm_provider="openai",
            openai_api_key="test-key",
            ollama_base_url="http://localhost:11434"
        )

    @pytest.mark.asyncio
    async def test_complete_chat_workflow(self, client, test_settings):
        """Test complete chat workflow from request to response"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock LLM response
            mock_response = {
                "id": "chatcmpl-e2e-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I'm an AI assistant. How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 15,
                    "total_tokens": 30
                }
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send chat request
                chat_request = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, introduce yourself."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
                
                response = client.post("/chat/completions", json=chat_request)
                
                # Verify response
                assert response.status_code == 200
                data = response.json()
                assert "choices" in data
                assert len(data["choices"]) > 0
                assert "message" in data["choices"][0]
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert "Hello!" in data["choices"][0]["message"]["content"]
                
                # Verify usage tracking
                assert "usage" in data
                assert data["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_agent_with_tools_workflow(self, client, test_settings):
        """Test agent workflow with tool usage"""
        with patch('app.api.routes.agents.get_settings', return_value=test_settings):
            # Mock agent response with tool call
            mock_agent_response = {
                "id": "agent-e2e-test",
                "object": "agent.completion",
                "created": int(time.time()),
                "agent_type": "tool_heavy",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I'll search for that information for you."
                        },
                        "finish_reason": "tool_calls"
                    }
                ],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "searxng_search",
                            "arguments": json.dumps({
                                "query": "latest AI developments",
                                "num_results": 5
                            })
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 20,
                    "total_tokens": 45
                }
            }
            
            # Mock tool execution result
            mock_tool_result = {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "Latest AI Developments in 2023",
                            "url": "https://example.com/ai-news",
                            "content": "Recent advances in AI include..."
                        }
                    ],
                    "query": "latest AI developments",
                    "num_results": 1
                },
                "message": "Tool executed successfully",
                "execution_time": 0.5
            }
            
            with patch('app.api.routes.agents.create_agent_completion', return_value=mock_agent_response):
                with patch('app.api.routes.agents.executor.execute_tool', return_value=mock_tool_result):
                    # Send agent request
                    agent_request = {
                        "agent_type": "tool_heavy",
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": "Search for latest AI developments"}
                        ],
                        "tools_enabled": True,
                        "temperature": 0.7
                    }
                    
                    response = client.post("/agents/chat", json=agent_request)
                    
                    # Verify response
                    assert response.status_code == 200
                    data = response.json()
                    assert "choices" in data
                    assert "tool_calls" in data
                    assert len(data["tool_calls"]) > 0
                    assert data["tool_calls"][0]["function"]["name"] == "searxng_search"

    @pytest.mark.asyncio
    async def test_tool_execution_workflow(self, client, test_settings):
        """Test direct tool execution workflow"""
        with patch('app.api.routes.tools.get_settings', return_value=test_settings):
            # Mock tool execution
            mock_result = {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "Python Programming Guide",
                            "url": "https://example.com/python-guide",
                            "content": "A comprehensive guide to Python programming..."
                        }
                    ],
                    "query": "Python programming tutorial",
                    "num_results": 1
                },
                "message": "Tool executed successfully",
                "execution_time": 0.8
            }
            
            with patch('app.api.routes.tools.executor.execute_tool', return_value=mock_result):
                # Send tool execution request
                tool_request = {
                    "tool_name": "searxng_search",
                    "parameters": {
                        "query": "Python programming tutorial",
                        "num_results": 5
                    }
                }
                
                response = client.post("/tools/execute", json=tool_request)
                
                # Verify response
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "data" in data
                assert "results" in data["data"]
                assert len(data["data"]["results"]) > 0
                assert "Python Programming Guide" in data["data"]["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_conversation_memory_workflow(self, client, test_settings):
        """Test conversation memory across multiple requests"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock responses for conversation
            responses = [
                {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello! My name is AI Assistant."
                            }
                        }
                    ]
                },
                {
                    "id": "chatcmpl-2",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "I told you my name is AI Assistant. Do you remember?"
                            }
                        }
                    ]
                }
            ]
            
            with patch('app.api.routes.chat.create_chat_completion', side_effect=responses):
                # First message
                request1 = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "What's your name?"}
                    ],
                    "conversation_id": str(uuid.uuid4())
                }
                
                response1 = client.post("/chat/completions", json=request1)
                assert response1.status_code == 200
                assert "AI Assistant" in response1.json()["choices"][0]["message"]["content"]
                
                # Second message (should have memory)
                request2 = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "What did you tell me your name was?"}
                    ],
                    "conversation_id": request1["conversation_id"]
                }
                
                response2 = client.post("/chat/completions", json=request2)
                assert response2.status_code == 200
                assert "remember" in response2.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_caching_workflow(self, client, test_settings):
        """Test caching workflow across requests"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock cache
            mock_cache = MemoryCache()
            mock_cache.configure(ttl=300, max_size=1000)
            
            # Mock LLM response
            mock_response = {
                "id": "chatcmpl-cache-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a cached response."
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.cache', mock_cache):
                with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                    # First request
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": "Cache test message"}
                        ]
                    }
                    
                    response1 = client.post("/chat/completions", json=request)
                    assert response1.status_code == 200
                    
                    # Second request (should use cache)
                    response2 = client.post("/chat/completions", json=request)
                    assert response2.status_code == 200
                    
                    # Both responses should be identical
                    assert response1.json() == response2.json()

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, client, test_settings):
        """Test error handling workflow"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Test with invalid model
            request = {
                "model": "invalid-model-name",
                "messages": [
                    {"role": "user", "content": "Test message"}
                ]
            }
            
            response = client.post("/chat/completions", json=request)
            assert response.status_code == 422  # Validation error
            
            # Test with missing required field
            request = {
                "messages": [
                    {"role": "user", "content": "Test message"}
                ]
                # Missing model field
            }
            
            response = client.post("/chat/completions", json=request)
            assert response.status_code == 422  # Validation error
            
            # Test with malformed messages
            request = {
                "model": "gpt-3.5-turbo",
                "messages": "invalid-messages-format"
            }
            
            response = client.post("/chat/completions", json=request)
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_provider_switching_workflow(self, client, test_settings):
        """Test switching between different LLM providers"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock responses for different providers
            openai_response = {
                "id": "chatcmpl-openai",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Response from OpenAI"
                        }
                    }
                ]
            }
            
            ollama_response = {
                "id": "chatcmpl-ollama",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Response from Ollama"
                        }
                    }
                ]
            }
            
            # Test OpenAI provider
            with patch('app.api.routes.chat.create_chat_completion', return_value=openai_response):
                request = {
                    "model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "messages": [
                        {"role": "user", "content": "Test OpenAI"}
                    ]
                }
                
                response = client.post("/chat/completions", json=request)
                assert response.status_code == 200
                assert "OpenAI" in response.json()["choices"][0]["message"]["content"]
            
            # Test Ollama provider
            with patch('app.api.routes.chat.create_chat_completion', return_value=ollama_response):
                request = {
                    "model": "llama2",
                    "provider": "ollama",
                    "messages": [
                        {"role": "user", "content": "Test Ollama"}
                    ]
                }
                
                response = client.post("/chat/completions", json=request)
                assert response.status_code == 200
                assert "Ollama" in response.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_response_workflow(self, client, test_settings):
        """Test streaming response workflow"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock streaming chunks
            mock_chunks = [
                'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                'data: {"choices": [{"delta": {"content": " there"}}]}\n\n',
                'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
                'data: [DONE]\n\n'
            ]
            
            with patch('app.api.routes.chat.create_chat_completion_stream', return_value=mock_chunks):
                request = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Stream this response"}
                    ],
                    "stream": True
                }
                
                response = client.post("/chat/completions", json=request)
                
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                
                # Verify streaming content
                content = response.content.decode()
                assert "Hello" in content
                assert "there" in content
                assert "!" in content

    @pytest.mark.asyncio
    async def test_concurrent_requests_workflow(self, client, test_settings):
        """Test handling of concurrent requests"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-concurrent",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Concurrent response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send multiple concurrent requests
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Concurrent test {uuid.uuid4()}"}
                        ]
                    }
                    return client.post("/chat/completions", json=request)
                
                # Send 5 concurrent requests
                tasks = [send_request() for _ in range(5)]
                responses = await asyncio.gather(*tasks)
                
                # Verify all requests succeeded
                assert all(response.status_code == 200 for response in responses)
                assert all("Concurrent response" in response.json()["choices"][0]["message"]["content"] 
                          for response in responses)

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, client, test_settings):
        """Test multi-agent collaboration workflow"""
        with patch('app.api.routes.agents.get_settings', return_value=test_settings):
            # Mock agent responses
            researcher_response = {
                "id": "agent-researcher",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll research the topic for you."
                        }
                    }
                ],
                "tool_calls": [
                    {
                        "function": {
                            "name": "searxng_search",
                            "arguments": json.dumps({"query": "climate change research"})
                        }
                    }
                ]
            }
            
            analyst_response = {
                "id": "agent-analyst",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Based on the research, here's my analysis..."
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.agents.create_agent_completion', side_effect=[researcher_response, analyst_response]):
                # Start with researcher agent
                request1 = {
                    "agent_type": "researcher",
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Research climate change"}
                    ],
                    "tools_enabled": True
                }
                
                response1 = client.post("/agents/chat", json=request1)
                assert response1.status_code == 200
                assert "tool_calls" in response1.json()
                
                # Continue with analyst agent
                request2 = {
                    "agent_type": "analyst",
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Analyze the research results"}
                    ],
                    "tools_enabled": False
                }
                
                response2 = client.post("/agents/chat", json=request2)
                assert response2.status_code == 200
                assert "analysis" in response2.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_system_health_monitoring_workflow(self, client, test_settings):
        """Test system health monitoring workflow"""
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "components" in health_data
        
        # Check individual components
        components = health_data["components"]
        assert "database" in components
        assert "cache" in components
        assert "llm" in components
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert "requests" in metrics_data
        assert "performance" in metrics_data
        assert "errors" in metrics_data

    @pytest.mark.asyncio
    async def test_configuration_management_workflow(self, client, test_settings):
        """Test configuration management workflow"""
        with patch('app.api.routes.config.get_settings', return_value=test_settings):
            # Get current configuration
            response = client.get("/config")
            assert response.status_code == 200
            
            config_data = response.json()
            assert "environment" in config_data
            assert "api_version" in config_data
            assert "features" in config_data
            assert "providers" in config_data
            
            # Update configuration
            update_request = {
                "cache_ttl": 600,
                "max_tokens": 2000,
                "temperature": 0.8
            }
            
            response = client.put("/config", json=update_request)
            assert response.status_code == 200
            
            # Verify update
            response = client.get("/config")
            updated_config = response.json()
            assert updated_config["cache_ttl"] == 600
            assert updated_config["max_tokens"] == 2000
            assert updated_config["temperature"] == 0.8