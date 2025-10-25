"""
Unit tests for main API routes.

Tests for the core API endpoints including model listing, chat completions,
health checks, and provider management.
"""

import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.routes import (
    router,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    AddProviderRequest,
    AddProviderResponse,
)
from app.main import app


class TestModelsEndpoint:
    """Test cases for the /v1/models endpoint"""

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_model_info_list):
        """Test successful model listing"""
        with patch('app.api.routes.get_available_models', return_value=mock_model_info_list):
            with patch('app.api.routes.provider_registry') as mock_registry:
                # Mock provider registry
                mock_provider = MagicMock()
                mock_provider.provider_type.value = "openai"
                mock_provider.name = "OpenAI"
                mock_registry.list_providers.return_value = [mock_provider]
                
                client = TestClient(app)
                response = client.get("/v1/models")
                
                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "list"
                assert len(data["data"]) > 0
                
                # Check first model structure
                model = data["data"][0]
                assert "id" in model
                assert "object" in model
                assert model["object"] == "model"
                assert "owned_by" in model
                assert "permission" in model
                assert "root" in model

    @pytest.mark.asyncio
    async def test_list_models_fallback(self):
        """Test model listing fallback on error"""
        with patch('app.api.routes.get_available_models', side_effect=Exception("Test error")):
            client = TestClient(app)
            response = client.get("/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "langchain-agent-hub"


class TestChatCompletionsEndpoint:
    """Test cases for the /v1/chat/completions endpoint"""

    @pytest.mark.asyncio
    async def test_chat_completions_with_deep_agents(self, mock_chat_request):
        """Test chat completions with deep agents enabled"""
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = True
            mock_settings.agent_system_enabled = False
            
            with patch('app.api.routes.deep_agent_manager') as mock_deep_agent:
                mock_deep_agent.invoke.return_value = "Test response"
                
                with patch('app.api.routes.get_postgresql_client') as mock_db_client:
                    # Mock database operations
                    mock_conn = AsyncMock()
                    mock_conn.fetchrow.return_value = {"id": uuid.uuid4()}
                    mock_db_client.return_value.pool.acquire.return_value.__aenter__.return_value = mock_conn
                    
                    client = TestClient(app)
                    response = client.post("/v1/chat/completions", json=mock_chat_request.dict())
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["object"] == "chat.completion"
                    assert "choices" in data
                    assert len(data["choices"]) > 0
                    assert data["choices"][0]["message"]["role"] == "assistant"
                    assert "conversation_id" in data

    @pytest.mark.asyncio
    async def test_chat_completions_with_agent_system(self, mock_chat_request):
        """Test chat completions with agent system enabled"""
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = False
            mock_settings.agent_system_enabled = True
            mock_settings.default_agent = "test-agent"
            
            with patch('app.api.routes.agent_registry') as mock_registry:
                # Mock agent result
                mock_result = MagicMock()
                mock_result.response = "Test agent response"
                mock_result.agent_name = "test-agent"
                mock_result.tool_results = []
                mock_result.conversation_id = str(uuid.uuid4())
                mock_registry.process_message.return_value = mock_result
                
                with patch('app.api.routes.get_postgresql_client') as mock_db_client:
                    # Mock database operations
                    mock_conn = AsyncMock()
                    mock_conn.fetchrow.return_value = {"id": uuid.uuid4()}
                    mock_db_client.return_value.pool.acquire.return_value.__aenter__.return_value = mock_conn
                    
                    client = TestClient(app)
                    response = client.post("/v1/chat/completions", json=mock_chat_request.dict())
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["object"] == "chat.completion"
                    assert data["agent_name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_chat_completions_direct_llm(self, mock_chat_request):
        """Test chat completions with direct LLM (no agents)"""
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = False
            mock_settings.agent_system_enabled = False
            
            with patch('app.api.routes.get_llm') as mock_get_llm:
                # Mock LLM
                mock_llm = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = "Test LLM response"
                mock_llm.ainvoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                
                with patch('app.api.routes.get_postgresql_client') as mock_db_client:
                    # Mock database operations
                    mock_conn = AsyncMock()
                    mock_conn.fetchrow.return_value = {"id": uuid.uuid4()}
                    mock_db_client.return_value.pool.acquire.return_value.__aenter__.return_value = mock_conn
                    
                    client = TestClient(app)
                    response = client.post("/v1/chat/completions", json=mock_chat_request.dict())
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["object"] == "chat.completion"
                    assert data["choices"][0]["message"]["content"] == "Test LLM response"

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, mock_chat_request):
        """Test streaming chat completions"""
        mock_chat_request.stream = True
        
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = False
            mock_settings.agent_system_enabled = False
            
            with patch('app.api.routes.get_llm') as mock_get_llm:
                # Mock streaming LLM
                mock_llm = AsyncMock()
                mock_llm.streaming = True
                mock_chunk = MagicMock()
                mock_chunk.content = "Test chunk"
                mock_llm.astream.return_value = [mock_chunk]
                mock_get_llm.return_value = mock_llm
                
                client = TestClient(app)
                response = client.post("/v1/chat/completions", json=mock_chat_request.dict())
                
                assert response.status_code == 200
                # Streaming responses should have content-type text/plain
                assert "text/plain" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_chat_completions_no_user_message(self):
        """Test chat completions with no user message"""
        # Create request with only system messages
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}
            ],
            "model": "test-model"
        }
        
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = True
            mock_settings.agent_system_enabled = False
            
            client = TestClient(app)
            response = client.post("/v1/chat/completions", json=request_data)
            
            assert response.status_code == 400
            assert "No user messages found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_completions_validation_error(self):
        """Test chat completions with invalid request"""
        # Create invalid request
        request_data = {
            "messages": "invalid",  # Should be a list
            "model": "test-model"
        }
        
        client = TestClient(app)
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_completions_connection_error(self, mock_chat_request):
        """Test chat completions with connection error"""
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.deep_agents_enabled = False
            mock_settings.agent_system_enabled = False
            
            with patch('app.api.routes.get_llm', side_effect=ConnectionError("Connection failed")):
                client = TestClient(app)
                response = client.post("/v1/chat/completions", json=mock_chat_request.dict())
                
                assert response.status_code == 503
                assert "temporarily unavailable" in response.json()["detail"]


class TestHealthEndpoint:
    """Test cases for the /health endpoint"""

    def test_health_check(self):
        """Test basic health check"""
        with patch('app.api.routes.settings') as mock_settings:
            mock_settings.environment = "test"
            mock_settings.openai_settings.api_key = "test-key"
            mock_settings.openrouter_api_key = "test-key"
            mock_settings.custom_reranker_enabled = True
            mock_settings.ollama_reranker_enabled = True
            mock_settings.jina_reranker_api_key = "test-key"
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "langchain-agent-hub"
            assert data["environment"] == "test"
            assert "api_keys_configured" in data


class TestProvidersEndpoint:
    """Test cases for provider-related endpoints"""

    @pytest.mark.asyncio
    async def test_list_providers(self):
        """Test listing available providers"""
        with patch('app.api.routes.provider_registry') as mock_registry:
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.name = "Test Provider"
            mock_provider.provider_type.value = "openai"
            mock_provider.is_configured = True
            mock_provider.is_healthy.return_value = True
            mock_registry.list_providers.return_value = [mock_provider]
            mock_registry._default_provider = None
            
            client = TestClient(app)
            response = client.get("/v1/providers")
            
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["name"] == "Test Provider"
            assert data["data"][0]["type"] == "openai"
            assert data["data"][0]["configured"] is True
            assert data["data"][0]["healthy"] is True

    @pytest.mark.asyncio
    async def test_list_provider_models_success(self):
        """Test listing models for a specific provider"""
        with patch('app.api.routes.provider_registry') as mock_registry:
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.is_configured = True
            
            # Mock model
            mock_model = MagicMock()
            mock_model.name = "test-model"
            mock_model.description = "Test model description"
            mock_model.context_length = 4096
            mock_model.supports_streaming = True
            mock_model.supports_tools = True
            mock_provider.list_models.return_value = [mock_model]
            
            mock_registry.get_provider.return_value = mock_provider
            
            with patch('app.api.routes.ProviderType') as mock_provider_type:
                mock_provider_type.return_value = "openai"
                
                client = TestClient(app)
                response = client.get("/v1/providers/openai/models")
                
                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["id"] == "test-model"
                assert data["data"][0]["description"] == "Test model description"

    @pytest.mark.asyncio
    async def test_list_provider_models_invalid_type(self):
        """Test listing models for invalid provider type"""
        client = TestClient(app)
        response = client.get("/v1/providers/invalid-type/models")
        
        assert response.status_code == 400
        assert "Invalid provider type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_provider_models_not_found(self):
        """Test listing models for non-existent provider"""
        with patch('app.api.routes.provider_registry') as mock_registry:
            with patch('app.api.routes.ProviderType') as mock_provider_type:
                mock_provider_type.return_value = "openai"
                mock_registry.get_provider.return_value = None
                
                client = TestClient(app)
                response = client.get("/v1/providers/openai/models")
                
                assert response.status_code == 404
                assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_health_check_providers(self):
        """Test health check for all providers"""
        with patch('app.api.routes.provider_registry') as mock_registry:
            # Mock provider
            mock_provider = MagicMock()
            mock_provider.name = "Test Provider"
            mock_provider.provider_type.value = "openai"
            mock_provider.is_configured = True
            mock_provider.is_healthy.return_value = True
            mock_registry.list_providers.return_value = [mock_provider]
            
            client = TestClient(app)
            response = client.post("/v1/providers/health-check")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert len(data["providers"]) == 1
            assert data["providers"][0]["name"] == "Test Provider"


class TestAddProviderEndpoint:
    """Test cases for the /v1/providers (POST) endpoint"""

    @pytest.mark.asyncio
    async def test_add_provider_success(self):
        """Test successfully adding a new provider"""
        request_data = {
            "name": "Test Provider",
            "type": "openai_compatible",
            "api_key": "test-api-key",
            "api_base": "https://api.test.com",
            "is_default": True,
            "model_list": ["test-model"]
        }
        
        with patch('app.api.routes.secure_settings') as mock_secure_settings:
            with patch('app.api.routes.initialize_llm_providers') as mock_init:
                with patch('app.api.routes.settings') as mock_settings:
                    mock_settings.preferred_provider = None
                    
                    client = TestClient(app)
                    response = client.post("/v1/providers", json=request_data)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert "added successfully" in data["message"]
                    assert data["provider"]["name"] == "Test Provider"
                    assert data["provider"]["type"] == "openai_compatible"

    @pytest.mark.asyncio
    async def test_add_provider_validation_error(self):
        """Test adding provider with validation error"""
        # Missing API key for non-local provider
        request_data = {
            "name": "Test Provider",
            "type": "openai_compatible",
            # api_key is missing
        }
        
        client = TestClient(app)
        response = client.post("/v1/providers", json=request_data)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_add_provider_local_no_api_key(self):
        """Test adding local provider without API key"""
        request_data = {
            "name": "Local Ollama",
            "type": "ollama",
            # api_key is not required for local providers
            "api_base": "http://localhost:11434"
        }
        
        with patch('app.api.routes.secure_settings') as mock_secure_settings:
            with patch('app.api.routes.initialize_llm_providers') as mock_init:
                client = TestClient(app)
                response = client.post("/v1/providers", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    @pytest.mark.asyncio
    async def test_add_provider_server_error(self):
        """Test adding provider with server error"""
        request_data = {
            "name": "Test Provider",
            "type": "openai_compatible",
            "api_key": "test-api-key"
        }
        
        with patch('app.api.routes.secure_settings', side_effect=Exception("Server error")):
            client = TestClient(app)
            response = client.post("/v1/providers", json=request_data)
            
            assert response.status_code == 500
            assert "internal error" in response.json()["detail"]


class TestStreamingResponse:
    """Test cases for streaming response functionality"""

    @pytest.mark.asyncio
    async def test_stream_response_native_streaming(self):
        """Test streaming response with native streaming support"""
        from app.api.routes import _stream_response
        from fastapi.responses import StreamingResponse
        
        # Mock messages and LLM
        mock_messages = [MagicMock()]
        mock_llm = MagicMock()
        mock_llm.streaming = True
        mock_llm.astream.return_value = [MagicMock(content="Test chunk")]
        
        response = await _stream_response(mock_messages, mock_llm, "test-model")
        
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/plain"

    @pytest.mark.asyncio
    async def test_stream_response_fallback_streaming(self):
        """Test streaming response with fallback streaming"""
        from app.api.routes import _stream_response
        from fastapi.responses import StreamingResponse
        
        # Mock messages and LLM
        mock_messages = [MagicMock()]
        mock_llm = MagicMock()
        mock_llm.streaming = False
        mock_response = MagicMock()
        mock_response.content = "Test response content"
        mock_llm.ainvoke.return_value = mock_response
        
        response = await _stream_response(mock_messages, mock_llm, "test-model")
        
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/plain"