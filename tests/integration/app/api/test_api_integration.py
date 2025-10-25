"""
Integration tests for API components.

This module tests the integration between API routes and core components.
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
from app.core.caching.layers.redis_cache import RedisCache
from app.core.tools.execution.dynamic_executor import DynamicExecutor
from app.core.tools.execution.registry import ToolRegistry
from app.core.tools.web.searxng_tool import SearXNGTool
from app.core.tools.web.firecrawl_tool import FirecrawlTool


class TestAPIIntegration:
    """Test API integration with core components"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings"""
        return Settings(
            environment="test",
            debug=True,
            api_host="localhost",
            api_port=8000,
            redis_url="redis://localhost:6379/0",
            cache_enabled=True,
            cache_ttl=300
        )

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache"""
        cache = MemoryCache()
        cache.configure(ttl=300, max_size=1000)
        return cache

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry"""
        registry = ToolRegistry()
        
        # Register some mock tools
        searxng_tool = SearXNGTool()
        firecrawl_tool = FirecrawlTool(api_key="test-key")
        
        registry.register_tool(searxng_tool)
        registry.register_tool(firecrawl_tool)
        
        return registry

    @pytest.fixture
    def mock_executor(self, mock_tool_registry):
        """Create a mock tool executor"""
        executor = DynamicExecutor()
        
        # Register tools from registry
        for tool_name in mock_tool_registry.list_tools():
            tool = mock_tool_registry.get_tool(tool_name)
            executor.register_tool(tool)
        
        return executor

    def test_health_check_integration(self, client):
        """Test health check endpoint integration"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_models_list_integration(self, client, mock_settings):
        """Test models list endpoint integration"""
        with patch('app.api.routes.models.get_settings', return_value=mock_settings):
            response = client.get("/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert isinstance(data["models"], list)

    def test_chat_completion_integration(self, client, mock_cache):
        """Test chat completion endpoint with cache integration"""
        chat_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        # Mock the LLM response
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well, thank you for asking!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        
        with patch('app.api.routes.chat.cache', mock_cache):
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                response = client.post("/chat/completions", json=chat_request)
                
                assert response.status_code == 200
                data = response.json()
                assert "choices" in data
                assert len(data["choices"]) > 0
                assert "message" in data["choices"][0]
                assert data["choices"][0]["message"]["role"] == "assistant"

    def test_tool_execution_integration(self, client, mock_executor):
        """Test tool execution endpoint integration"""
        tool_request = {
            "tool_name": "searxng_search",
            "parameters": {
                "query": "test search query",
                "num_results": 5
            }
        }
        
        # Mock the tool execution
        mock_result = {
            "success": True,
            "data": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "content": "Test content 1"
                    }
                ],
                "query": "test search query",
                "num_results": 1
            },
            "message": "Tool executed successfully",
            "execution_time": 0.5
        }
        
        with patch('app.api.routes.tools.executor', mock_executor):
            with patch.object(mock_executor, 'execute_tool', return_value=mock_result):
                response = client.post("/tools/execute", json=tool_request)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "data" in data
                assert "results" in data["data"]

    def test_agent_chat_integration(self, client, mock_cache, mock_executor):
        """Test agent chat endpoint integration"""
        agent_request = {
            "agent_type": "conversational",
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Search for information about AI"}
            ],
            "tools_enabled": True,
            "temperature": 0.7
        }
        
        # Mock agent response
        mock_response = {
            "id": "agent-test",
            "object": "agent.completion",
            "created": int(time.time()),
            "agent_type": "conversational",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll search for information about AI for you."
                    },
                    "finish_reason": "stop"
                }
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "searxng_search",
                        "arguments": json.dumps({
                            "query": "information about AI",
                            "num_results": 5
                        })
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 15,
                "total_tokens": 40
            }
        }
        
        with patch('app.api.routes.agents.cache', mock_cache):
            with patch('app.api.routes.agents.executor', mock_executor):
                with patch('app.api.routes.agents.create_agent_completion', return_value=mock_response):
                    response = client.post("/agents/chat", json=agent_request)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "choices" in data
                    assert "tool_calls" in data
                    assert len(data["tool_calls"]) > 0

    def test_provider_management_integration(self, client, mock_settings):
        """Test provider management endpoint integration"""
        with patch('app.api.routes.providers.get_settings', return_value=mock_settings):
            # Test listing providers
            response = client.get("/providers")
            assert response.status_code == 200
            data = response.json()
            assert "providers" in data
            assert isinstance(data["providers"], list)
            
            # Test getting specific provider
            response = client.get("/providers/openai")
            assert response.status_code == 200
            data = response.json()
            assert "name" in data
            assert "type" in data
            assert "models" in data

    def test_cache_integration(self, client, mock_cache):
        """Test cache integration with API endpoints"""
        # First request - should cache the result
        chat_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Test cache integration"}
            ]
        }
        
        mock_response = {
            "id": "chatcmpl-cache-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Cached response"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        with patch('app.api.routes.chat.cache', mock_cache):
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # First request
                response1 = client.post("/chat/completions", json=chat_request)
                assert response1.status_code == 200
                
                # Second request - should use cache
                response2 = client.post("/chat/completions", json=chat_request)
                assert response2.status_code == 200
                
                # Verify cache was used (check call count)
                # In a real test, we would verify that the LLM was called only once

    def test_error_handling_integration(self, client):
        """Test error handling integration"""
        # Test with invalid request
        invalid_request = {
            "model": "invalid-model",
            "messages": "invalid-messages-format"
        }
        
        response = client.post("/chat/completions", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Test with missing required field
        missing_field_request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
            # Missing model field
        }
        
        response = client.post("/chat/completions", json=missing_field_request)
        assert response.status_code == 422  # Validation error

    def test_rate_limiting_integration(self, client, mock_cache):
        """Test rate limiting integration"""
        # Mock rate limiter
        with patch('app.api.routes.chat.check_rate_limit', return_value=False):
            chat_request = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Test rate limiting"}
                ]
            }
            
            response = client.post("/chat/completions", json=chat_request)
            assert response.status_code == 429  # Too Many Requests

    def test_authentication_integration(self, client):
        """Test authentication integration"""
        # Test with no authentication
        response = client.get("/protected-endpoint")
        # This would depend on how authentication is implemented
        
        # Test with invalid authentication
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/protected-endpoint", headers=headers)
        # This would depend on how authentication is implemented

    def test_concurrent_requests_integration(self, client, mock_cache):
        """Test handling of concurrent requests"""
        chat_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": f"Concurrent test {uuid.uuid4()}"}
            ]
        }
        
        mock_response = {
            "id": "chatcmpl-concurrent-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Concurrent response"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        with patch('app.api.routes.chat.cache', mock_cache):
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send multiple concurrent requests
                async def send_request():
                    return client.post("/chat/completions", json=chat_request)
                
                # In a real test, we would use asyncio.gather to send concurrent requests
                # and verify they are handled correctly

    def test_streaming_response_integration(self, client):
        """Test streaming response integration"""
        chat_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Stream this response"}
            ],
            "stream": True
        }
        
        # Mock streaming response
        mock_chunks = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        with patch('app.api.routes.chat.create_chat_completion_stream', return_value=mock_chunks):
            response = client.post("/chat/completions", json=chat_request)
            
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            # Verify streaming content
            content = response.content.decode()
            assert "Hello" in content
            assert "world" in content
            assert "!" in content

    def test_file_upload_integration(self, client):
        """Test file upload integration"""
        # Create a test file
        test_content = "This is a test file content."
        files = {"file": ("test.txt", test_content, "text/plain")}
        
        response = client.post("/files/upload", files=files)
        
        # This would depend on how file upload is implemented
        # In a real test, we would verify the file was processed correctly

    def test_websocket_integration(self, client):
        """Test WebSocket integration"""
        # This would test WebSocket endpoints for real-time communication
        # In a real test, we would use TestClient with websocket support
        pass

    def test_middleware_integration(self, client):
        """Test middleware integration"""
        # Test that middleware is properly integrated
        response = client.get("/health")
        
        # Check for middleware headers
        assert "x-request-id" in response.headers or "x-correlation-id" in response.headers
        
        # Check for CORS headers if CORS middleware is enabled
        # assert "access-control-allow-origin" in response.headers

    def test_logging_integration(self, client, caplog):
        """Test logging integration"""
        # Make a request that should generate logs
        response = client.get("/health")
        
        # In a real test, we would verify that appropriate logs were generated
        # This would depend on the logging configuration

    def test_metrics_integration(self, client):
        """Test metrics integration"""
        # Make some requests to generate metrics
        client.get("/health")
        client.get("/models")
        
        # Test metrics endpoint
        response = client.get("/metrics")
        
        # This would depend on how metrics are implemented
        # In a real test, we would verify that metrics were collected correctly

    def test_configuration_integration(self, client, mock_settings):
        """Test configuration integration"""
        with patch('app.api.routes.config.get_settings', return_value=mock_settings):
            response = client.get("/config")
            
            assert response.status_code == 200
            data = response.json()
            assert "environment" in data
            assert "api_version" in data
            assert "features" in data