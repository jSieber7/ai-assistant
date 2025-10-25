"""
Unit tests for LangChain LLM Manager.

This module tests the LangChain-based LLM manager functionality,
including provider registration, LLM creation, and management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from app.core.langchain.llm_manager import (
    LangChainLLMManager,
    LLMProvider,
    LLMConfig,
    LLMStats,
    llm_manager
)


class TestLLMProvider:
    """Test LLMProvider enum"""

    def test_llm_provider_values(self):
        """Test that LLMProvider has expected values"""
        expected_providers = [
            "openai",
            "anthropic",
            "google",
            "ollama",
            "huggingface"
        ]
        
        actual_providers = [provider.value for provider in LLMProvider]
        assert actual_providers == expected_providers


class TestLLMConfig:
    """Test LLMConfig dataclass"""

    def test_llm_config_defaults(self):
        """Test LLMConfig default values"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.top_p is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None
        assert config.stop_sequences is None
        assert config.streaming is False
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.metadata == {}

    def test_llm_config_with_values(self):
        """Test LLMConfig with provided values"""
        metadata = {"version": "1.0", "custom": True}
        
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["\n\n", "###"],
            streaming=True,
            timeout=120,
            max_retries=5,
            metadata=metadata
        )
        
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model_name == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.1
        assert config.stop_sequences == ["\n\n", "###"]
        assert config.streaming is True
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.metadata == metadata


class TestLLMStats:
    """Test LLMStats dataclass"""

    def test_llm_stats_defaults(self):
        """Test LLMStats default values"""
        stats = LLMStats(
            model_name="gpt-4",
            provider="openai"
        )
        
        assert stats.model_name == "gpt-4"
        assert stats.provider == "openai"
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0
        assert stats.total_response_time == 0.0
        assert stats.average_response_time == 0.0
        assert stats.last_used is None
        assert stats.created_at is not None

    def test_llm_stats_with_values(self):
        """Test LLMStats with provided values"""
        created_at = datetime.now()
        last_used = datetime.now()
        
        stats = LLMStats(
            model_name="gpt-4",
            provider="openai",
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
            total_tokens=5000,
            total_response_time=45.5,
            average_response_time=4.55,
            last_used=last_used,
            created_at=created_at
        )
        
        assert stats.model_name == "gpt-4"
        assert stats.provider == "openai"
        assert stats.total_requests == 10
        assert stats.successful_requests == 9
        assert stats.failed_requests == 1
        assert stats.total_tokens == 5000
        assert stats.total_response_time == 45.5
        assert stats.average_response_time == 4.55
        assert stats.last_used == last_used
        assert stats.created_at == created_at


class TestLangChainLLMManager:
    """Test LangChainLLMManager class"""

    @pytest.fixture
    def llm_manager_instance(self):
        """Create a fresh LLM manager instance for testing"""
        return LangChainLLMManager()

    @pytest.fixture
    def mock_monitoring(self):
        """Mock monitoring system"""
        with patch('app.core.langchain.llm_manager.LangChainMonitoring') as mock:
            mock_instance = Mock()
            mock_instance.initialize = AsyncMock()
            mock_instance.track_llm_request = AsyncMock()
            mock_instance.track_metric = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI LangChain integration"""
        with patch('app.core.langchain.llm_manager.ChatOpenAI') as mock:
            mock.return_value = Mock()
            yield mock

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic LangChain integration"""
        with patch('app.core.langchain.llm_manager.ChatAnthropic') as mock:
            mock.return_value = Mock()
            yield mock

    @pytest.fixture
    def mock_google(self):
        """Mock Google LangChain integration"""
        with patch('app.core.langchain.llm_manager.ChatGoogleGenerativeAI') as mock:
            mock.return_value = Mock()
            yield mock

    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama LangChain integration"""
        with patch('app.core.langchain.llm_manager.Ollama') as mock:
            mock.return_value = Mock()
            yield mock

    @pytest.mark.asyncio
    async def test_initialize(self, llm_manager_instance, mock_monitoring):
        """Test LLM manager initialization"""
        await llm_manager_instance.initialize()
        
        assert llm_manager_instance._initialized is True
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, llm_manager_instance, mock_monitoring):
        """Test that initialize is idempotent"""
        await llm_manager_instance.initialize()
        await llm_manager_instance.initialize()
        
        assert llm_manager_instance._initialized is True
        # Should only initialize once
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_llm_config(self, llm_manager_instance):
        """Test registering an LLM configuration"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5
        )
        
        result = await llm_manager_instance.register_llm_config(config)
        
        assert result is True
        key = "openai:gpt-4"
        assert key in llm_manager_instance._llm_configs
        assert key in llm_manager_instance._llm_stats

    @pytest.mark.asyncio
    async def test_register_llm_config_duplicate(self, llm_manager_instance):
        """Test registering duplicate LLM configuration"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        
        # Register first time
        result1 = await llm_manager_instance.register_llm_config(config)
        assert result1 is True
        
        # Register second time
        result2 = await llm_manager_instance.register_llm_config(config)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_get_llm_openai(
        self, 
        llm_manager_instance, 
        mock_openai, 
        mock_monitoring
    ):
        """Test getting OpenAI LLM"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get LLM
        result = await llm_manager_instance.get_llm("gpt-4")
        
        assert result is not None
        mock_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.5,
            max_tokens=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            stop=None,
            streaming=False,
            timeout=60,
            max_retries=3
        )

    @pytest.mark.asyncio
    async def test_get_llm_anthropic(
        self, 
        llm_manager_instance, 
        mock_anthropic, 
        mock_monitoring
    ):
        """Test getting Anthropic LLM"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus",
            temperature=0.3
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get LLM
        result = await llm_manager_instance.get_llm("claude-3-opus")
        
        assert result is not None
        mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_google(
        self, 
        llm_manager_instance, 
        mock_google, 
        mock_monitoring
    ):
        """Test getting Google LLM"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-pro"
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get LLM
        result = await llm_manager_instance.get_llm("gemini-pro")
        
        assert result is not None
        mock_google.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_ollama(
        self, 
        llm_manager_instance, 
        mock_ollama, 
        mock_monitoring
    ):
        """Test getting Ollama LLM"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama2"
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get LLM
        result = await llm_manager_instance.get_llm("llama2")
        
        assert result is not None
        mock_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_with_overrides(
        self, 
        llm_manager_instance, 
        mock_openai, 
        mock_monitoring
    ):
        """Test getting LLM with parameter overrides"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get LLM with overrides
        result = await llm_manager_instance.get_llm(
            "gpt-4",
            temperature=0.1,
            max_tokens=1000,
            streaming=True
        )
        
        assert result is not None
        mock_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.1,  # Override
            max_tokens=1000,   # Override
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            stop=None,
            streaming=True,     # Override
            timeout=60,
            max_retries=3
        )

    @pytest.mark.asyncio
    async def test_get_llm_not_registered(self, llm_manager_instance):
        """Test getting LLM for unregistered model"""
        await llm_manager_instance.initialize()
        
        result = await llm_manager_instance.get_llm("unregistered-model")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_llm_with_provider(
        self, 
        llm_manager_instance, 
        mock_openai, 
        mock_anthropic
    ):
        """Test getting LLM with explicit provider"""
        await llm_manager_instance.initialize()
        
        # Register configs
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        anthropic_config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="gpt-4"  # Same model name, different provider
        )
        
        await llm_manager_instance.register_llm_config(openai_config)
        await llm_manager_instance.register_llm_config(anthropic_config)
        
        # Get OpenAI version
        result = await llm_manager_instance.get_llm("gpt-4", provider="openai")
        assert result is not None
        mock_openai.assert_called_once()
        
        # Get Anthropic version
        result = await llm_manager_instance.get_llm("gpt-4", provider="anthropic")
        assert result is not None
        mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_available_models(self, llm_manager_instance):
        """Test listing available models"""
        await llm_manager_instance.initialize()
        
        # Register multiple configs
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3-opus"),
            LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama2")
        ]
        
        for config in configs:
            await llm_manager_instance.register_llm_config(config)
        
        result = await llm_manager_instance.list_available_models()
        
        assert len(result) == 4
        model_names = [model["model_name"] for model in result]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names
        assert "claude-3-opus" in model_names
        assert "llama2" in model_names
        
        # Check provider info
        gpt4_model = next(m for m in result if m["model_name"] == "gpt-4")
        assert gpt4_model["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_list_available_models_by_provider(self, llm_manager_instance):
        """Test listing available models filtered by provider"""
        await llm_manager_instance.initialize()
        
        # Register multiple configs
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3-opus"),
            LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama2")
        ]
        
        for config in configs:
            await llm_manager_instance.register_llm_config(config)
        
        # Filter by OpenAI
        result = await llm_manager_instance.list_available_models(provider="openai")
        
        assert len(result) == 2
        model_names = [model["model_name"] for model in result]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names
        assert "claude-3-opus" not in model_names

    @pytest.mark.asyncio
    async def test_get_model_stats(self, llm_manager_instance):
        """Test getting model statistics"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Get stats
        result = await llm_manager_instance.get_model_stats("gpt-4")
        
        assert result is not None
        assert result["model_name"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["total_requests"] == 0
        assert result["successful_requests"] == 0
        assert result["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_get_model_stats_not_found(self, llm_manager_instance):
        """Test getting stats for non-existent model"""
        await llm_manager_instance.initialize()
        
        result = await llm_manager_instance.get_model_stats("nonexistent-model")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_track_request_success(
        self, 
        llm_manager_instance, 
        mock_monitoring
    ):
        """Test tracking a successful request"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Track request
        await llm_manager_instance._track_request(
            model_name="gpt-4",
            success=True,
            response_time=2.5,
            tokens_used=100
        )
        
        # Check stats
        stats = llm_manager_instance._llm_stats["openai:gpt-4"]
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_tokens == 100
        assert stats.total_response_time == 2.5
        assert stats.average_response_time == 2.5
        assert stats.last_used is not None
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_llm_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_request_failure(
        self, 
        llm_manager_instance, 
        mock_monitoring
    ):
        """Test tracking a failed request"""
        await llm_manager_instance.initialize()
        
        # Register config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        await llm_manager_instance.register_llm_config(config)
        
        # Track failed request
        await llm_manager_instance._track_request(
            model_name="gpt-4",
            success=False,
            response_time=1.0,
            tokens_used=0,
            error="API error"
        )
        
        # Check stats
        stats = llm_manager_instance._llm_stats["openai:gpt-4"]
        assert stats.total_requests == 1
        assert stats.successful_requests == 0
        assert stats.failed_requests == 1
        assert stats.total_tokens == 0
        assert stats.total_response_time == 1.0
        assert stats.average_response_time == 1.0

    @pytest.mark.asyncio
    async def test_get_provider_stats(self, llm_manager_instance):
        """Test getting provider statistics"""
        await llm_manager_instance.initialize()
        
        # Register multiple configs for OpenAI
        openai_configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")
        ]
        
        # Register one config for Anthropic
        anthropic_config = LLMConfig(
            provider=LLMProvider.ANTHROPIC, 
            model_name="claude-3-opus"
        )
        
        for config in openai_configs + [anthropic_config]:
            await llm_manager_instance.register_llm_config(config)
        
        # Get OpenAI stats
        result = await llm_manager_instance.get_provider_stats("openai")
        
        assert result is not None
        assert result["provider"] == "openai"
        assert result["total_models"] == 2
        assert len(result["models"]) == 2
        model_names = [model["model_name"] for model in result["models"]]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names

    @pytest.mark.asyncio
    async def test_get_registry_stats(self, llm_manager_instance):
        """Test getting registry statistics"""
        await llm_manager_instance.initialize()
        
        # Register configs for multiple providers
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3-opus"),
            LLMConfig(provider=LLMProvider.GOOGLE, model_name="gemini-pro"),
            LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama2")
        ]
        
        for config in configs:
            await llm_manager_instance.register_llm_config(config)
        
        # Get registry stats
        result = await llm_manager_instance.get_registry_stats()
        
        assert result["total_models"] == 5
        assert result["total_providers"] == 5
        assert len(result["providers"]) == 5
        assert result["providers"]["openai"]["model_count"] == 2
        assert result["providers"]["anthropic"]["model_count"] == 1
        assert result["providers"]["google"]["model_count"] == 1
        assert result["providers"]["ollama"]["model_count"] == 1

    @pytest.mark.asyncio
    async def test_reset_stats(self, llm_manager_instance):
        """Test resetting model statistics"""
        await llm_manager_instance.initialize()
        
        # Register config and track some requests
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4"
        )
        await llm_manager_instance.register_llm_config(config)
        
        await llm_manager_instance._track_request(
            model_name="gpt-4",
            success=True,
            response_time=2.5,
            tokens_used=100
        )
        
        # Reset stats
        result = await llm_manager_instance.reset_stats("gpt-4")
        
        assert result is True
        stats = llm_manager_instance._llm_stats["openai:gpt-4"]
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0
        assert stats.total_response_time == 0.0
        assert stats.average_response_time == 0.0

    @pytest.mark.asyncio
    async def test_reset_stats_not_found(self, llm_manager_instance):
        """Test resetting stats for non-existent model"""
        await llm_manager_instance.initialize()
        
        result = await llm_manager_instance.reset_stats("nonexistent-model")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_reset_all_stats(self, llm_manager_instance):
        """Test resetting all statistics"""
        await llm_manager_instance.initialize()
        
        # Register multiple configs and track requests
        configs = [
            LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4"),
            LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-3-opus")
        ]
        
        for config in configs:
            await llm_manager_instance.register_llm_config(config)
            await llm_manager_instance._track_request(
                model_name=config.model_name,
                success=True,
                response_time=2.0,
                tokens_used=50
            )
        
        # Reset all stats
        await llm_manager_instance.reset_all_stats()
        
        # Check all stats are reset
        for key, stats in llm_manager_instance._llm_stats.items():
            assert stats.total_requests == 0
            assert stats.successful_requests == 0
            assert stats.failed_requests == 0
            assert stats.total_tokens == 0
            assert stats.total_response_time == 0.0
            assert stats.average_response_time == 0.0

    def test_get_config_key(self, llm_manager_instance):
        """Test getting configuration key"""
        # Test with provider
        key = llm_manager_instance._get_config_key("gpt-4", "openai")
        assert key == "openai:gpt-4"
        
        # Test without provider (should default to OpenAI)
        key = llm_manager_instance._get_config_key("gpt-4")
        assert key == "openai:gpt-4"

    def test_parse_model_key(self, llm_manager_instance):
        """Test parsing model key"""
        # Test valid key
        provider, model = llm_manager_instance._parse_model_key("openai:gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"
        
        # Test key without provider
        provider, model = llm_manager_instance._parse_model_key("gpt-4")
        assert provider is None
        assert model == "gpt-4"

    @pytest.mark.asyncio
    async def test_create_openai_llm(self, llm_manager_instance):
        """Test creating OpenAI LLM"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1000
        )
        
        with patch('app.core.langchain.llm_manager.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            result = await llm_manager_instance._create_openai_llm(config)
            
            assert result == mock_llm
            mock_openai.assert_called_once_with(
                model="gpt-4",
                temperature=0.5,
                max_tokens=1000,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop=None,
                streaming=False,
                timeout=60,
                max_retries=3
            )

    @pytest.mark.asyncio
    async def test_create_anthropic_llm(self, llm_manager_instance):
        """Test creating Anthropic LLM"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus",
            temperature=0.3
        )
        
        with patch('app.core.langchain.llm_manager.ChatAnthropic') as mock_anthropic:
            mock_llm = Mock()
            mock_anthropic.return_value = mock_llm
            
            result = await llm_manager_instance._create_anthropic_llm(config)
            
            assert result == mock_llm
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_google_llm(self, llm_manager_instance):
        """Test creating Google LLM"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-pro"
        )
        
        with patch('app.core.langchain.llm_manager.ChatGoogleGenerativeAI') as mock_google:
            mock_llm = Mock()
            mock_google.return_value = mock_llm
            
            result = await llm_manager_instance._create_google_llm(config)
            
            assert result == mock_llm
            mock_google.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_ollama_llm(self, llm_manager_instance):
        """Test creating Ollama LLM"""
        await llm_manager_instance.initialize()
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama2"
        )
        
        with patch('app.core.langchain.llm_manager.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            result = await llm_manager_instance._create_ollama_llm(config)
            
            assert result == mock_llm
            mock_ollama.assert_called_once()


class TestGlobalLLMManager:
    """Test global LLM manager instance"""

    def test_global_instance(self):
        """Test that global LLM manager instance exists"""
        from app.core.langchain.llm_manager import llm_manager
        assert llm_manager is not None
        assert isinstance(llm_manager, LangChainLLMManager)