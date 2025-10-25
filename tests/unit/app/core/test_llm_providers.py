"""
Unit tests for LLM Provider System.

This module tests the LLM provider system including OpenAI-compatible,
OpenRouter, and Ollama providers, as well as the provider registry.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import httpx

from app.core.llm_providers import (
    ProviderType,
    ModelInfo,
    LLMProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    OllamaProvider,
    LLMProviderRegistry
)


class TestProviderType:
    """Test ProviderType enum"""

    def test_provider_type_values(self):
        """Test that ProviderType has expected values"""
        expected_types = [
            "openai",
            "openrouter",
            "ollama",
            "anthropic",
            "google",
            "custom"
        ]
        
        actual_types = [provider_type.value for provider_type in ProviderType]
        assert actual_types == expected_types


class TestModelInfo:
    """Test ModelInfo dataclass"""

    def test_model_info_defaults(self):
        """Test ModelInfo default values"""
        model_info = ModelInfo(
            name="gpt-4",
            provider=ProviderType.OPENAI
        )
        
        assert model_info.name == "gpt-4"
        assert model_info.provider == ProviderType.OPENAI
        assert model_info.display_name == "gpt-4"
        assert model_info.description == ""
        assert model_info.max_tokens is None
        assert model_info.context_window is None
        assert model_info.input_cost_per_1k is None
        assert model_info.output_cost_per_1k is None
        assert model_info.supports_functions is False
        assert model_info.supports_vision is False
        assert model_info.supports_streaming is True
        assert model_info.metadata == {}

    def test_model_info_with_values(self):
        """Test ModelInfo with provided values"""
        metadata = {"version": "1.0", "family": "gpt-4"}
        
        model_info = ModelInfo(
            name="gpt-4-turbo",
            provider=ProviderType.OPENAI,
            display_name="GPT-4 Turbo",
            description="Advanced GPT-4 model with improved capabilities",
            max_tokens=4096,
            context_window=128000,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            metadata=metadata
        )
        
        assert model_info.name == "gpt-4-turbo"
        assert model_info.provider == ProviderType.OPENAI
        assert model_info.display_name == "GPT-4 Turbo"
        assert model_info.description == "Advanced GPT-4 model with improved capabilities"
        assert model_info.max_tokens == 4096
        assert model_info.context_window == 128000
        assert model_info.input_cost_per_1k == 0.01
        assert model_info.output_cost_per_1k == 0.03
        assert model_info.supports_functions is True
        assert model_info.supports_vision is True
        assert model_info.supports_streaming is True
        assert model_info.metadata == metadata

    def test_model_info_to_dict(self):
        """Test ModelInfo to_dict method"""
        metadata = {"test": True}
        
        model_info = ModelInfo(
            name="gpt-4",
            provider=ProviderType.OPENAI,
            display_name="GPT-4",
            description="Test model",
            max_tokens=4096,
            metadata=metadata
        )
        
        result = model_info.to_dict()
        
        assert result["name"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["display_name"] == "GPT-4"
        assert result["description"] == "Test model"
        assert result["max_tokens"] == 4096
        assert result["metadata"] == metadata


class TestOpenAICompatibleProvider:
    """Test OpenAICompatibleProvider class"""

    @pytest.fixture
    def provider(self):
        """Create an OpenAI-compatible provider instance"""
        return OpenAICompatibleProvider(
            provider_type=ProviderType.OPENAI,
            api_key="test-api-key",
            base_url="https://api.openai.com/v1"
        )

    def test_provider_init(self, provider):
        """Test OpenAICompatibleProvider initialization"""
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_detect_provider_name_openai(self):
        """Test provider name detection for OpenAI"""
        provider = OpenAICompatibleProvider(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        
        assert provider._detect_provider_name("https://api.openai.com/v1") == "openai"

    def test_detect_provider_name_anthropic(self):
        """Test provider name detection for Anthropic"""
        provider = OpenAICompatibleProvider(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            base_url="https://api.anthropic.com/v1"
        )
        
        assert provider._detect_provider_name("https://api.anthropic.com/v1") == "anthropic"

    def test_detect_provider_name_google(self):
        """Test provider name detection for Google"""
        provider = OpenAICompatibleProvider(
            provider_type=ProviderType.GOOGLE,
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1"
        )
        
        assert provider._detect_provider_name("https://generativelanguage.googleapis.com/v1") == "google"

    def test_detect_provider_name_openrouter(self):
        """Test provider name detection for OpenRouter"""
        provider = OpenAICompatibleProvider(
            provider_type=ProviderType.OPENROUTER,
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1"
        )
        
        assert provider._detect_provider_name("https://openrouter.ai/api/v1") == "openrouter"

    def test_detect_provider_name_custom(self):
        """Test provider name detection for custom provider"""
        provider = OpenAICompatibleProvider(
            provider_type=ProviderType.CUSTOM,
            api_key="test-key",
            base_url="https://custom-api.example.com/v1"
        )
        
        assert provider._detect_provider_name("https://custom-api.example.com/v1") == "custom"

    @pytest.mark.asyncio
    async def test_create_llm(self, provider):
        """Test creating an LLM instance"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm
            
            result = await provider.create_llm("gpt-4", temperature=0.7, max_tokens=1000)
            
            assert result == mock_llm
            mock_chat_openai.assert_called_once_with(
                model="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                max_tokens=1000
            )

    @pytest.mark.asyncio
    async def test_create_llm_with_custom_kwargs(self, provider):
        """Test creating an LLM instance with custom kwargs"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm
            
            result = await provider.create_llm(
                "gpt-4",
                temperature=0.5,
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.1
            )
            
            assert result == mock_llm
            mock_chat_openai.assert_called_once_with(
                model="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.5,
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.1
            )

    @pytest.mark.asyncio
    async def test_list_models_openai(self, provider):
        """Test listing OpenAI models"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            models = await provider.list_models()
            
            assert len(models) == 2
            assert models[0].name == "gpt-4"
            assert models[0].provider == ProviderType.OPENAI
            assert models[1].name == "gpt-3.5-turbo"
            assert models[1].provider == ProviderType.OPENAI

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing models with error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("API error")
            
            models = await provider.list_models()
            
            assert models == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await provider.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            result = await provider.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    async def test_check_openrouter_model_access_success(self, provider):
        """Test successful OpenRouter model access check"""
        provider.provider_type = ProviderType.OPENROUTER
        provider.api_key = "test-key"
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await provider._check_openrouter_model_access(
                "openai/gpt-4",
                {"Authorization": "Bearer test-key"},
                mock_client.return_value.__aenter__.return_value
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_check_openrouter_model_access_failure(self, provider):
        """Test failed OpenRouter model access check"""
        provider.provider_type = ProviderType.OPENROUTER
        provider.api_key = "test-key"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("API error")
            
            result = await provider._check_openrouter_model_access(
                "openai/gpt-4",
                {"Authorization": "Bearer test-key"},
                mock_client.return_value.__aenter__.return_value
            )
            
            assert result is False


class TestOpenRouterProvider:
    """Test OpenRouterProvider class"""

    @pytest.fixture
    def provider(self):
        """Create an OpenRouter provider instance"""
        return OpenRouterProvider(
            api_key="test-api-key"
        )

    def test_provider_init(self, provider):
        """Test OpenRouterProvider initialization"""
        assert provider.provider_type == ProviderType.OPENROUTER
        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_provider_init_custom_url(self):
        """Test OpenRouterProvider initialization with custom URL"""
        provider = OpenRouterProvider(
            api_key="test-api-key",
            base_url="https://custom.openrouter.ai/api/v1"
        )
        
        assert provider.provider_type == ProviderType.OPENROUTER
        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://custom.openrouter.ai/api/v1"


class TestOllamaProvider:
    """Test OllamaProvider class"""

    @pytest.fixture
    def provider(self):
        """Create an Ollama provider instance"""
        return OllamaProvider(
            base_url="http://localhost:11434"
        )

    def test_provider_init(self, provider):
        """Test OllamaProvider initialization"""
        assert provider.provider_type == ProviderType.OLLAMA
        assert provider.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_create_llm(self, provider):
        """Test creating an Ollama LLM instance"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_chat_ollama.return_value = mock_llm
            
            result = await provider.create_llm("llama2", temperature=0.7)
            
            assert result == mock_llm
            mock_chat_ollama.assert_called_once_with(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.7
            )

    @pytest.mark.asyncio
    async def test_create_llm_with_custom_kwargs(self, provider):
        """Test creating an Ollama LLM instance with custom kwargs"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_chat_ollama.return_value = mock_llm
            
            result = await provider.create_llm(
                "llama2",
                temperature=0.5,
                top_p=0.9,
                num_ctx=4096
            )
            
            assert result == mock_llm
            mock_chat_ollama.assert_called_once_with(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.5,
                top_p=0.9,
                num_ctx=4096
            )

    @pytest.mark.asyncio
    async def test_create_llm_error(self, provider):
        """Test creating an Ollama LLM instance with error"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_chat_ollama.side_effect = Exception("Import error")
            
            with pytest.raises(Exception, match="Import error"):
                await provider.create_llm("llama2")

    @pytest.mark.asyncio
    async def test_list_models_success(self, provider):
        """Test successful Ollama models listing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest", "size": 1234567890},
                {"name": "mistral:latest", "size": 987654321}
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            models = await provider.list_models()
            
            assert len(models) == 2
            assert models[0].name == "llama2:latest"
            assert models[0].provider == ProviderType.OLLAMA
            assert models[1].name == "mistral:latest"
            assert models[1].provider == ProviderType.OLLAMA

    @pytest.mark.asyncio
    async def test_list_models_remote_url(self, provider):
        """Test listing models from remote Ollama URL"""
        provider.base_url = "https://remote-ollama.example.com"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            await provider.list_models()
            
            # Should call with headers for remote URL
            call_args = mock_client.return_value.__aenter__.return_value.get.call_args
            assert "headers" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing Ollama models with error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            models = await provider.list_models()
            
            assert models == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful Ollama health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await provider.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_remote_url(self, provider):
        """Test Ollama health check with remote URL"""
        provider.base_url = "https://remote-ollama.example.com"
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            await provider.health_check()
            
            # Should call with headers for remote URL
            call_args = mock_client.return_value.__aenter__.return_value.get.call_args
            assert "headers" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed Ollama health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            result = await provider.health_check()
            
            assert result is False


class TestLLMProviderRegistry:
    """Test LLMProviderRegistry class"""

    @pytest.fixture
    def registry(self):
        """Create a provider registry instance"""
        return LLMProviderRegistry()

    def test_registry_init(self, registry):
        """Test LLMProviderRegistry initialization"""
        assert len(registry.providers) == 0
        assert registry.default_provider is None

    def test_register_provider(self, registry):
        """Test registering a provider"""
        provider = Mock(spec=LLMProvider)
        provider.provider_type = ProviderType.OPENAI
        
        registry.register_provider(provider)
        
        assert len(registry.providers) == 1
        assert registry.providers[ProviderType.OPENAI] == provider

    def test_register_provider_sets_default(self, registry):
        """Test that registering first provider sets it as default"""
        provider = Mock(spec=LLMProvider)
        provider.provider_type = ProviderType.OPENAI
        
        registry.register_provider(provider)
        
        assert registry.default_provider == provider

    def test_get_provider(self, registry):
        """Test getting a registered provider"""
        provider = Mock(spec=LLMProvider)
        provider.provider_type = ProviderType.OPENAI
        
        registry.register_provider(provider)
        
        result = registry.get_provider(ProviderType.OPENAI)
        
        assert result == provider

    def test_get_provider_not_found(self, registry):
        """Test getting a provider that doesn't exist"""
        result = registry.get_provider(ProviderType.OPENAI)
        
        assert result is None

    def test_get_default_provider(self, registry):
        """Test getting the default provider"""
        provider = Mock(spec=LLMProvider)
        provider.provider_type = ProviderType.OPENAI
        
        registry.register_provider(provider)
        
        result = registry.get_default_provider()
        
        assert result == provider

    def test_get_default_provider_none(self, registry):
        """Test getting default provider when none is set"""
        result = registry.get_default_provider()
        
        assert result is None

    def test_set_default_provider(self, registry):
        """Test setting the default provider"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        registry.set_default_provider(ProviderType.OLLAMA)
        
        assert registry.default_provider == provider2

    def test_set_default_provider_not_found(self, registry):
        """Test setting default provider that doesn't exist"""
        with pytest.raises(ValueError, match="Provider not found"):
            registry.set_default_provider(ProviderType.OPENAI)

    def test_list_configured_providers(self, registry):
        """Test listing configured providers"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        result = registry.list_configured_providers()
        
        assert len(result) == 2
        assert provider1 in result
        assert provider2 in result

    @pytest.mark.asyncio
    async def test_health_check_all(self, registry):
        """Test health check for all providers"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        provider1.health_check = AsyncMock(return_value=True)
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        provider2.health_check = AsyncMock(return_value=False)
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        results = await registry.health_check_all()
        
        assert results[ProviderType.OPENAI] is True
        assert results[ProviderType.OLLAMA] is False

    @pytest.mark.asyncio
    async def test_resolve_model_with_provider_prefix(self, registry):
        """Test resolving model with provider prefix"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        result = await registry.resolve_model("openai:gpt-4")
        
        assert result == (provider1, "gpt-4")

    @pytest.mark.asyncio
    async def test_resolve_model_without_provider_prefix(self, registry):
        """Test resolving model without provider prefix"""
        provider = Mock(spec=LLMProvider)
        provider.provider_type = ProviderType.OPENAI
        
        registry.register_provider(provider)
        registry.set_default_provider(ProviderType.OPENAI)
        
        result = await registry.resolve_model("gpt-4")
        
        assert result == (provider, "gpt-4")

    @pytest.mark.asyncio
    async def test_resolve_model_no_default_provider(self, registry):
        """Test resolving model without default provider"""
        with pytest.raises(ValueError, match="No default provider configured"):
            await registry.resolve_model("gpt-4")

    @pytest.mark.asyncio
    async def test_resolve_model_provider_not_found(self, registry):
        """Test resolving model with unknown provider"""
        with pytest.raises(ValueError, match="Provider 'unknown' not found"):
            await registry.resolve_model("unknown:gpt-4")

    @pytest.mark.asyncio
    async def test_list_all_models(self, registry):
        """Test listing all models from all providers"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        provider1.list_models = AsyncMock(return_value=[
            ModelInfo("gpt-4", ProviderType.OPENAI),
            ModelInfo("gpt-3.5-turbo", ProviderType.OPENAI)
        ])
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        provider2.list_models = AsyncMock(return_value=[
            ModelInfo("llama2", ProviderType.OLLAMA)
        ])
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        models = await registry.list_all_models()
        
        assert len(models) == 3
        assert any(m.name == "gpt-4" for m in models)
        assert any(m.name == "gpt-3.5-turbo" for m in models)
        assert any(m.name == "llama2" for m in models)

    @pytest.mark.asyncio
    async def test_list_all_models_with_error(self, registry):
        """Test listing all models with provider error"""
        provider1 = Mock(spec=LLMProvider)
        provider1.provider_type = ProviderType.OPENAI
        provider1.list_models = AsyncMock(side_effect=Exception("API error"))
        
        provider2 = Mock(spec=LLMProvider)
        provider2.provider_type = ProviderType.OLLAMA
        provider2.list_models = AsyncMock(return_value=[
            ModelInfo("llama2", ProviderType.OLLAMA)
        ])
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        models = await registry.list_all_models()
        
        # Should still return models from working provider
        assert len(models) == 1
        assert models[0].name == "llama2"