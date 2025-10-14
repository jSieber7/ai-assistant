"""
Tests for Ollama integration functionality
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.core.llm_providers import (
    OllamaProvider,
    OpenRouterProvider,
    OpenAICompatibleProvider,
    LLMProviderRegistry,
    ProviderType,
    ModelInfo,
)
from app.core.config import OllamaSettings


@pytest.mark.unit
class TestOllamaProvider:
    """Test Ollama provider functionality"""

    def test_ollama_provider_init(self):
        """Test Ollama provider initialization"""
        provider = OllamaProvider("http://localhost:11434")

        assert provider.provider_type == ProviderType.OLLAMA
        assert provider.name == "Ollama"
        assert provider.base_url == "http://localhost:11434"
        assert provider.is_configured

    @pytest.mark.asyncio
    async def test_ollama_health_check_success(self):
        """Test successful Ollama health check"""
        provider = OllamaProvider("http://localhost:11434")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await provider.health_check()

            assert result
            assert provider.is_healthy()

    @pytest.mark.asyncio
    async def test_ollama_health_check_failure(self):
        """Test Ollama health check failure"""
        provider = OllamaProvider("http://localhost:11434")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("Connection failed")
            )

            result = await provider.health_check()

            assert not result
            assert not provider.is_healthy()

    @pytest.mark.asyncio
    async def test_ollama_list_models(self):
        """Test listing Ollama models"""
        provider = OllamaProvider("http://localhost:11434")

        mock_response_data = {
            "models": [
                {"name": "llama2:latest", "details": {"context_length": 4096}},
                {"name": "codellama:latest", "details": {"context_length": 16384}},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            models = await provider.list_models()

            assert len(models) == 2
            assert models[0].name == "llama2:latest"
            assert models[0].provider == ProviderType.OLLAMA
            assert models[0].context_length == 4096
            assert models[1].name == "codellama:latest"
            assert models[1].context_length == 16384

    @pytest.mark.asyncio
    async def test_ollama_create_llm(self):
        """Test creating Ollama LLM instance"""
        provider = OllamaProvider("http://localhost:11434")

        # Mock model info
        provider.get_model_info = AsyncMock(
            return_value=ModelInfo(
                name="llama2", provider=ProviderType.OLLAMA, display_name="Llama 2"
            )
        )

        # Mock the ChatOllama import from langchain_community
        with patch("langchain_community.chat_models.ChatOllama") as mock_chat_ollama:
            mock_llm = Mock()
            mock_chat_ollama.return_value = mock_llm

            result = await provider.create_llm("llama2", temperature=0.7)

            assert result == mock_llm
            mock_chat_ollama.assert_called_once_with(
                base_url="http://localhost:11434",
                model="llama2",
                temperature=0.7,
                num_predict=None,
                streaming=False,
            )


@pytest.mark.unit
class TestLLMProviderRegistry:
    """Test LLM provider registry functionality"""

    def test_registry_init(self):
        """Test registry initialization"""
        registry = LLMProviderRegistry()

        assert len(registry._providers) == 0
        assert registry._default_provider is None

    def test_register_provider(self):
        """Test provider registration"""
        registry = LLMProviderRegistry()
        provider = OllamaProvider("http://localhost:11434")

        registry.register_provider(provider)

        assert len(registry._providers) == 1
        assert ProviderType.OLLAMA in registry._providers
        assert registry._default_provider == ProviderType.OLLAMA

    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        registry = LLMProviderRegistry()

        ollama_provider = OllamaProvider("http://localhost:11434")
        openrouter_provider = OpenRouterProvider("test_key")

        registry.register_provider(ollama_provider)
        registry.register_provider(openrouter_provider)

        assert len(registry._providers) == 2
        assert (
            registry._default_provider == ProviderType.OLLAMA
        )  # First registered becomes default

    def test_set_default_provider(self):
        """Test setting default provider"""
        registry = LLMProviderRegistry()

        ollama_provider = OllamaProvider("http://localhost:11434")
        openrouter_provider = OpenRouterProvider("test_key")

        registry.register_provider(ollama_provider)
        registry.register_provider(openrouter_provider)

        registry.set_default_provider(ProviderType.OPENROUTER)

        assert registry._default_provider == ProviderType.OPENROUTER

    def test_get_provider(self):
        """Test getting provider by type"""
        registry = LLMProviderRegistry()
        provider = OllamaProvider("http://localhost:11434")

        registry.register_provider(provider)

        retrieved = registry.get_provider(ProviderType.OLLAMA)

        assert retrieved == provider
        assert retrieved.name == "Ollama"

    @pytest.mark.asyncio
    async def test_resolve_model_with_provider_prefix(self):
        """Test model resolution with provider prefix"""
        registry = LLMProviderRegistry()
        provider = OllamaProvider("http://localhost:11434")

        registry.register_provider(provider)

        resolved_provider, model_name = await registry.resolve_model("ollama:llama2")

        assert resolved_provider == provider
        assert model_name == "llama2"

    @pytest.mark.asyncio
    async def test_resolve_model_without_prefix(self):
        """Test model resolution without provider prefix"""
        registry = LLMProviderRegistry()
        provider = OllamaProvider("http://localhost:11434")

        # Mock model info
        provider.get_model_info = AsyncMock(
            return_value=ModelInfo(
                name="llama2", provider=ProviderType.OLLAMA, display_name="Llama 2"
            )
        )

        registry.register_provider(provider)
        registry.set_default_provider(ProviderType.OLLAMA)

        resolved_provider, model_name = await registry.resolve_model("llama2")

        assert resolved_provider == provider
        assert model_name == "llama2"

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health check for all providers"""
        registry = LLMProviderRegistry()

        ollama_provider = OllamaProvider("http://localhost:11434")
        openrouter_provider = OpenRouterProvider("test_key")

        # Mock health checks
        ollama_provider.health_check = AsyncMock(return_value=True)
        openrouter_provider.health_check = AsyncMock(return_value=True)

        registry.register_provider(ollama_provider)
        registry.register_provider(openrouter_provider)

        await registry.health_check_all()

        assert ollama_provider.is_healthy()
        assert openrouter_provider.is_healthy()


@pytest.mark.unit
class TestOllamaSettings:
    """Test Ollama settings configuration"""

    def test_ollama_settings_defaults(self):
        """Test Ollama settings default values"""
        settings = OllamaSettings()

        assert settings.enabled
        assert settings.base_url == "http://localhost:11434"
        assert settings.default_model == "llama2"
        assert settings.timeout == 30
        assert settings.temperature == 0.7
        assert settings.streaming
        assert settings.health_check_interval == 60
        assert settings.auto_health_check

    def test_ollama_settings_custom_values(self):
        """Test Ollama settings with custom values"""
        settings = OllamaSettings(
            enabled=False,
            base_url="http://custom:11434",
            default_model="codellama",
            timeout=60,
            temperature=0.5,
            streaming=False,
        )

        assert not settings.enabled
        assert settings.base_url == "http://custom:11434"
        assert settings.default_model == "codellama"
        assert settings.timeout == 60
        assert settings.temperature == 0.5
        assert not settings.streaming


@pytest.mark.unit
class TestModelInfo:
    """Test ModelInfo data class"""

    def test_model_info_creation(self):
        """Test ModelInfo creation"""
        model = ModelInfo(
            name="llama2",
            provider=ProviderType.OLLAMA,
            display_name="Llama 2",
            description="Open source LLM",
            context_length=4096,
            supports_streaming=True,
            supports_tools=False,
        )

        assert model.name == "llama2"
        assert model.provider == ProviderType.OLLAMA
        assert model.display_name == "Llama 2"
        assert model.description == "Open source LLM"
        assert model.context_length == 4096
        assert model.supports_streaming
        assert not model.supports_tools

    def test_model_info_optional_fields(self):
        """Test ModelInfo with optional fields"""
        model = ModelInfo(
            name="llama2", provider=ProviderType.OLLAMA, display_name="Llama 2"
        )

        assert model.name == "llama2"
        assert model.provider == ProviderType.OLLAMA
        assert model.display_name == "Llama 2"
        assert model.description is None
        assert model.context_length is None
        assert model.supports_streaming  # Default
        assert model.supports_tools  # Default


@pytest.mark.unit
class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider functionality"""

    def test_openai_compatible_provider_init_openrouter(self):
        """Test OpenAI-compatible provider initialization with OpenRouter URL"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://openrouter.ai/api/v1"
        )

        assert provider.provider_type == ProviderType.OPENAI_COMPATIBLE
        assert provider.name == "OpenRouter"
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.api_key == "test_key"
        assert provider.is_configured

    def test_openai_compatible_provider_init_openai(self):
        """Test OpenAI-compatible provider initialization with OpenAI URL"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        assert provider.provider_type == ProviderType.OPENAI_COMPATIBLE
        assert provider.name == "OpenAI"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_openai_compatible_provider_init_custom_name(self):
        """Test OpenAI-compatible provider with custom name"""
        provider = OpenAICompatibleProvider(
            api_key="test_key",
            base_url="https://custom.example.com/v1",
            provider_name="Custom Provider",
        )

        assert provider.name == "Custom Provider"

    def test_openai_compatible_provider_init_custom_headers(self):
        """Test OpenAI-compatible provider with custom headers"""
        custom_headers = {"X-Custom-Header": "value"}
        provider = OpenAICompatibleProvider(
            api_key="test_key",
            base_url="https://api.openai.com/v1",
            custom_headers=custom_headers,
        )

        assert provider.custom_headers == custom_headers

    @pytest.mark.asyncio
    async def test_openai_compatible_health_check_success(self):
        """Test successful OpenAI-compatible health check"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await provider.health_check()

            assert result
            assert provider.is_healthy()

    @pytest.mark.asyncio
    async def test_openai_compatible_health_check_failure(self):
        """Test OpenAI-compatible health check failure"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("Connection failed")
            )

            result = await provider.health_check()

            assert not result
            assert not provider.is_healthy()

    @pytest.mark.asyncio
    async def test_openai_compatible_list_models_openai_format(self):
        """Test listing models in OpenAI format"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        mock_response_data = {
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            models = await provider.list_models()

            assert len(models) == 2
            assert models[0].name == "gpt-4"
            assert models[0].provider == ProviderType.OPENAI_COMPATIBLE
            assert models[1].name == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_openai_compatible_list_models_fallback(self):
        """Test listing models with fallback when API fails"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("API unavailable")
            )

            models = await provider.list_models()

            # Should return fallback models
            assert len(models) > 0
            assert all(
                model.provider == ProviderType.OPENAI_COMPATIBLE for model in models
            )

    @pytest.mark.asyncio
    async def test_openai_compatible_create_llm(self):
        """Test creating OpenAI-compatible LLM instance"""
        provider = OpenAICompatibleProvider(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

        # Mock the ChatOpenAI import from langchain_openai
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm

            result = await provider.create_llm("gpt-4", temperature=0.7)

            assert result == mock_llm
            mock_chat_openai.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                api_key="test_key",
                model="gpt-4",
                temperature=0.7,
                max_tokens=None,
                streaming=False,
                default_headers=None,
            )

    @pytest.mark.asyncio
    async def test_openai_compatible_create_llm_with_custom_headers(self):
        """Test creating LLM instance with custom headers"""
        custom_headers = {"X-Custom-Header": "value"}
        provider = OpenAICompatibleProvider(
            api_key="test_key",
            base_url="https://api.openai.com/v1",
            custom_headers=custom_headers,
        )

        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm

            result = await provider.create_llm("gpt-4")

            assert result == mock_llm
            # The implementation includes both Authorization and custom headers
            expected_headers = {"Authorization": "Bearer test_key", **custom_headers}
            mock_chat_openai.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                api_key="test_key",
                model="gpt-4",
                temperature=0.7,
                max_tokens=None,
                streaming=False,
                default_headers=expected_headers,
            )


@pytest.mark.unit
class TestOpenRouterBackwardCompatibility:
    """Test OpenRouter backward compatibility"""

    def test_openrouter_provider_inheritance(self):
        """Test that OpenRouterProvider inherits from OpenAICompatibleProvider"""
        provider = OpenRouterProvider(
            api_key="test_key", base_url="https://openrouter.ai/api/v1"
        )

        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_type == ProviderType.OPENROUTER
        assert provider.name == "OpenRouter"

    @pytest.mark.asyncio
    async def test_openrouter_provider_create_llm(self):
        """Test OpenRouter provider creates LLM correctly"""
        provider = OpenRouterProvider(
            api_key="test_key", base_url="https://openrouter.ai/api/v1"
        )

        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm

            result = await provider.create_llm("anthropic/claude-3.5-sonnet")

            assert result == mock_llm
            mock_chat_openai.assert_called_once_with(
                base_url="https://openrouter.ai/api/v1",
                api_key="test_key",
                model="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=None,
                streaming=False,
                default_headers=None,
            )
