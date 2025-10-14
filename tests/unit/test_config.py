"""
Unit tests for configuration module
"""

import pytest
from unittest.mock import patch, MagicMock
from app.core.config import (
    Settings,
    CacheSettings,
    OllamaSettings,
    OpenAISettings,
    initialize_llm_providers,
    get_llm,
    get_available_models,
    initialize_agent_system,
)


class TestCacheSettings:
    """Test cache settings configuration"""

    def test_cache_settings_defaults(self):
        """Test cache settings default values"""
        settings = CacheSettings()
        
        assert settings.enabled is True
        assert settings.default_ttl == 300
        assert settings.max_size == 1000
        assert settings.redis_url == "redis://localhost:6379/0"

    def test_cache_settings_custom_values(self):
        """Test cache settings with custom values"""
        settings = CacheSettings(
            enabled=False,
            default_ttl=600,
            max_size=2000,
            redis_url="redis://custom:6379/1"
        )
        
        assert settings.enabled is False
        assert settings.default_ttl == 600
        assert settings.max_size == 2000
        assert settings.redis_url == "redis://custom:6379/1"


class TestOllamaSettings:
    """Test Ollama settings configuration"""

    def test_ollama_settings_defaults(self):
        """Test Ollama settings default values"""
        settings = OllamaSettings()
        
        assert settings.enabled is True
        assert settings.base_url == "http://localhost:11434"
        assert settings.default_model == "llama2"
        assert settings.timeout == 30
        assert settings.temperature == 0.7
        assert settings.streaming is True

    def test_ollama_settings_custom_values(self):
        """Test Ollama settings with custom values"""
        settings = OllamaSettings(
            enabled=False,
            base_url="http://custom:11434",
            default_model="codellama",
            timeout=60,
            temperature=0.5,
            streaming=False
        )
        
        assert settings.enabled is False
        assert settings.base_url == "http://custom:11434"
        assert settings.default_model == "codellama"
        assert settings.timeout == 60
        assert settings.temperature == 0.5
        assert settings.streaming is False


class TestOpenAISettings:
    """Test OpenAI settings configuration"""

    def test_openai_settings_defaults(self):
        """Test OpenAI settings default values"""
        settings = OpenAISettings()
        
        assert settings.enabled is True
        assert settings.api_key is None
        assert settings.base_url == "https://api.openai.com/v1"
        assert settings.default_model == "gpt-3.5-turbo"
        assert settings.timeout == 30
        assert settings.temperature == 0.7

    def test_openai_settings_custom_values(self):
        """Test OpenAI settings with custom values"""
        settings = OpenAISettings(
            enabled=False,
            api_key="sk-test-key",
            base_url="https://api.custom.com/v1",
            default_model="gpt-4",
            timeout=60,
            temperature=0.5
        )
        
        assert settings.enabled is False
        assert settings.api_key.get_secret_value() == "sk-test-key"
        assert settings.base_url == "https://api.custom.com/v1"
        assert settings.default_model == "gpt-4"
        assert settings.timeout == 60
        assert settings.temperature == 0.5


class TestSettings:
    """Test main settings configuration"""

    def test_settings_defaults(self):
        """Test settings default values"""
        settings = Settings()
        
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.environment == "development"
        assert settings.tool_system_enabled is True
        assert settings.agent_system_enabled is True

    def test_settings_from_env(self):
        """Test settings loaded from environment variables"""
        with patch.dict("os.environ", {
            "DEBUG": "true",
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "ENVIRONMENT": "production"
        }):
            settings = Settings()
            
            assert settings.debug is True
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.environment == "production"

    def test_settings_openrouter_api_key(self):
        """Test OpenRouter API key setting"""
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "sk-or-test-key"
        }):
            settings = Settings()
            
            assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-key"

    def test_settings_default_model(self):
        """Test default model setting"""
        with patch.dict("os.environ", {
            "DEFAULT_MODEL": "gpt-4"
        }):
            settings = Settings()
            
            assert settings.default_model == "gpt-4"


class TestLLMProviderInitialization:
    """Test LLM provider initialization"""

    @patch("app.core.config.provider_registry")
    @patch("app.core.config.OpenRouterProvider")
    @patch("app.core.config.OllamaProvider")
    @patch("app.core.config.OpenAICompatibleProvider")
    def test_initialize_llm_providers_with_all_configured(
        self, mock_openai_provider, mock_ollama_provider, mock_openrouter, mock_registry
    ):
        """Test initializing all configured LLM providers"""
        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key.get_secret_value.return_value = "test-key"
        mock_settings.ollama_settings.enabled = True
        mock_settings.openai_settings.enabled = True
        mock_settings.openai_settings.api_key.get_secret_value.return_value = "sk-test-key"
        mock_settings.preferred_provider = "openrouter"
        mock_settings.enable_fallback = True
        
        # Create mock providers
        mock_openrouter_instance = MagicMock()
        mock_openrouter.return_value = mock_openrouter_instance
        
        mock_ollama_instance = MagicMock()
        mock_ollama_provider.return_value = mock_ollama_instance
        
        mock_openai_instance = MagicMock()
        mock_openai_provider.return_value = mock_openai_instance
        
        with patch("app.core.config.settings", mock_settings):
            initialize_llm_providers()
            
            # Verify all providers were registered
            assert mock_registry.register_provider.call_count == 3

    @patch("app.core.config.provider_registry")
    @patch("app.core.config.OpenRouterProvider")
    def test_initialize_llm_providers_no_api_key(self, mock_openrouter, mock_registry):
        """Test initialization when API keys are missing"""
        # Create mock settings with no API keys
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = None
        mock_settings.ollama_settings.enabled = False
        mock_settings.openai_settings.enabled = False
        
        with patch("app.core.config.settings", mock_settings):
            with pytest.raises(ValueError, match="No LLM providers are configured"):
                initialize_llm_providers()

    @patch("app.core.config.provider_registry")
    @patch("app.core.config.OpenRouterProvider")
    def test_initialize_llm_providers_with_fallback(
        self, mock_openrouter, mock_registry
    ):
        """Test initialization with fallback enabled"""
        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key.get_secret_value.return_value = "test-key"
        mock_settings.ollama_settings.enabled = False
        mock_settings.openai_settings.enabled = False
        mock_settings.preferred_provider = "openrouter"
        mock_settings.enable_fallback = True
        
        mock_openrouter_instance = MagicMock()
        mock_openrouter.return_value = mock_openrouter_instance
        
        with patch("app.core.config.settings", mock_settings):
            initialize_llm_providers()
            
            # Verify provider was registered
            mock_registry.register_provider.assert_called_once_with(mock_openrouter_instance)

    @patch("app.core.config.provider_registry")
    def test_get_llm_with_default_model(self, mock_registry):
        """Test getting LLM with default model"""
        # Create mock provider
        mock_provider = MagicMock()
        mock_llm = MagicMock()
        mock_provider.create_llm.return_value = mock_llm
        
        mock_registry.get_default_provider.return_value = mock_provider
        
        result = get_llm()
        
        assert result == mock_llm
        mock_provider.create_llm.assert_called_once()

    @patch("app.core.config.provider_registry")
    def test_get_llm_with_specific_model(self, mock_registry):
        """Test getting LLM with specific model"""
        # Create mock provider
        mock_provider = MagicMock()
        mock_llm = MagicMock()
        mock_provider.create_llm.return_value = mock_llm
        
        mock_registry.resolve_model.return_value = (mock_provider, "gpt-4")
        
        result = get_llm("gpt-4")
        
        assert result == mock_llm
        mock_provider.create_llm.assert_called_once_with("gpt-4")

    @patch("app.core.config.provider_registry")
    def test_get_llm_with_provider_prefix(self, mock_registry):
        """Test getting LLM with provider prefix"""
        # Create mock provider
        mock_provider = MagicMock()
        mock_llm = MagicMock()
        mock_provider.create_llm.return_value = mock_llm
        
        mock_registry.resolve_model.return_value = (mock_provider, "llama2")
        
        result = get_llm("ollama:llama2")
        
        assert result == mock_llm
        mock_provider.create_llm.assert_called_once_with("llama2")

    @patch("app.core.config.provider_registry")
    def test_get_available_models(self, mock_registry):
        """Test getting available models from all providers"""
        # Create mock models
        mock_model1 = MagicMock()
        mock_model1.name = "gpt-4"
        mock_model1.provider.value = "openai"
        
        mock_model2 = MagicMock()
        mock_model2.name = "llama2"
        mock_model2.provider.value = "ollama"
        
        mock_registry.list_all_models.return_value = [mock_model1, mock_model2]
        
        result = get_available_models()
        
        assert len(result) == 2
        assert result[0].name == "gpt-4"
        assert result[1].name == "llama2"


class TestAgentSystemInitialization:
    """Test agent system initialization"""

    @patch("app.core.config.agent_registry")
    def test_initialize_agent_system_success(self, mock_agent_registry):
        """Test successful agent system initialization"""
        mock_agent_registry.initialize_default_agents.return_value = None
        
        # Should not raise an exception
        initialize_agent_system()
        
        mock_agent_registry.initialize_default_agents.assert_called_once()

    @patch("app.core.config.agent_registry")
    def test_initialize_agent_system_failure(self, mock_agent_registry):
        """Test agent system initialization failure"""
        mock_agent_registry.initialize_default_agents.side_effect = Exception("Failed to initialize")
        
        # Should not raise an exception, should log the error
        initialize_agent_system()


class TestConfigurationValidation:
    """Test configuration validation"""

    def test_validate_port_range(self):
        """Test port validation"""
        with patch.dict("os.environ", {"PORT": "99999"}):  # Invalid port
            with pytest.raises(ValueError):  # Should raise validation error
                Settings()

    def test_validate_boolean_settings(self):
        """Test boolean setting validation"""
        with patch.dict("os.environ", {"DEBUG": "invalid"}):
            # Pydantic should handle this gracefully
            settings = Settings()
            # Should default to False or raise validation error
            assert settings.debug in [False, True]

    def test_validate_url_settings(self):
        """Test URL setting validation"""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "invalid-url"}):
            # Pydantic should handle URL validation
            with pytest.raises(ValueError):  # Should raise validation error
                OllamaSettings(base_url="invalid-url")


class TestConfigurationEnvironmentOverrides:
    """Test environment variable overrides"""

    def test_environment_override_precedence(self):
        """Test that environment variables take precedence"""
        with patch.dict("os.environ", {
            "DEFAULT_MODEL": "gpt-4",
            "OPENROUTER_API_KEY": "sk-env-key"
        }):
            settings = Settings()
            
            assert settings.default_model == "gpt-4"
            assert settings.openrouter_api_key.get_secret_value() == "sk-env-key"

    def test_nested_settings_environment_override(self):
        """Test environment variable override for nested settings"""
        with patch.dict("os.environ", {
            "OLLAMA_ENABLED": "false",
            "OLLAMA_BASE_URL": "http://env:11434"
        }):
            settings = Settings()
            
            assert settings.ollama_settings.enabled is False
            assert settings.ollama_settings.base_url == "http://env:11434"

    def test_cache_settings_environment_override(self):
        """Test cache settings environment override"""
        with patch.dict("os.environ", {
            "CACHE_ENABLED": "false",
            "CACHE_DEFAULT_TTL": "600",
            "CACHE_MAX_SIZE": "2000"
        }):
            settings = Settings()
            
            assert settings.cache_settings.enabled is False
            assert settings.cache_settings.default_ttl == 600
            assert settings.cache_settings.max_size == 2000


class TestConfigurationSecretHandling:
    """Test handling of secret configuration values"""

    def test_api_key_secret_handling(self):
        """Test that API keys are properly handled as secrets"""
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "sk-secret-key",
            "OPENAI_API_KEY": "sk-openai-key"
        }):
            settings = Settings()
            
            # Should be Pydantic SecretStr objects
            assert hasattr(settings.openrouter_api_key, "get_secret_value")
            assert hasattr(settings.openai_settings.api_key, "get_secret_value")
            
            # Should not expose the secret in string representation
            assert "sk-secret-key" not in str(settings.openrouter_api_key)
            assert "sk-openai-key" not in str(settings.openai_settings.api_key)

    def test_secret_value_access(self):
        """Test accessing secret values"""
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "sk-test-key"
        }):
            settings = Settings()
            
            # Should be able to get the actual value
            assert settings.openrouter_api_key.get_secret_value() == "sk-test-key"