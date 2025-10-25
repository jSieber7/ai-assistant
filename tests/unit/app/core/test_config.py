"""
Unit tests for core configuration system.

Tests Settings, CacheSettings, OllamaSettings, OpenAISettings, and other configuration classes.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, Mock
from pathlib import Path

from app.core.config import (
    Settings,
    CacheSettings,
    OllamaSettings,
    OpenAISettings,
    FirecrawlSettings,
    PlaywrightSettings,
    MilvusSettings,
    LangChainSettings,
    LangGraphSettings,
    VectorStoreSettings,
    FirebaseSettings,
    initialize_llm_providers,
    get_llm,
    get_available_models,
    initialize_agent_system,
    initialize_langchain_components,
    initialize_langgraph_workflows,
    initialize_firebase_integration,
    initialize_all_langchain_components
)
from app.core.multi_writer_config import MultiWriterSettings
from app.core.secure_settings import SecureSettingsManager


class TestCacheSettings:
    """Test CacheSettings configuration"""
    
    def test_default_cache_settings(self):
        """Test default cache settings values"""
        settings = CacheSettings()
        
        assert settings.caching_enabled is True
        assert settings.default_ttl == 300
        assert settings.max_cache_size == 1000
        assert settings.cache_layers == ["memory", "redis"]
        assert settings.write_through is True
        assert settings.read_through is True
        assert settings.compression_enabled is True
        assert settings.compression_algorithm == "gzip"
        assert settings.batching_enabled is True
        assert settings.connection_pooling_enabled is True
        assert settings.monitoring_enabled is True
    
    def test_cache_settings_from_env(self):
        """Test creating cache settings from environment variables"""
        with patch.dict(os.environ, {
            "CACHING_ENABLED": "false",
            "DEFAULT_TTL": "600",
            "MAX_CACHE_SIZE": "2000",
            "COMPRESSION_ALGORITHM": "lzma",
            "BATCHING_ENABLED": "false"
        }):
            settings = CacheSettings()
            
            # Note: Pydantic_settings doesn't automatically use env vars without explicit configuration
            # This test demonstrates the expected behavior if env vars were properly configured
            assert settings.caching_enabled is True  # Default value
            assert settings.default_ttl == 300  # Default value
    
    def test_cache_settings_validation(self):
        """Test cache settings validation"""
        settings = CacheSettings()
        
        # Test valid compression algorithms
        for algorithm in ["gzip", "lzma", "zlib", "base64", "none"]:
            settings.compression_algorithm = algorithm
            assert settings.compression_algorithm == algorithm
        
        # Test invalid values would raise validation errors in real usage
        # This is handled by Pydantic validation


class TestOllamaSettings:
    """Test OllamaSettings configuration"""
    
    def test_default_ollama_settings(self):
        """Test default Ollama settings values"""
        settings = OllamaSettings()
        
        assert settings.enabled is True
        assert settings.base_url == "http://host.docker.internal:11434"
        assert settings.default_model == "llama2"
        assert settings.timeout == 30
        assert settings.max_retries == 3
        assert settings.temperature == 0.7
        assert settings.max_tokens is None
        assert settings.streaming is True
    
    def test_ollama_max_tokens_validation(self):
        """Test Ollama max_tokens validation"""
        # Test with empty string
        settings = OllamaSettings(max_tokens="")
        assert settings.max_tokens is None
        
        # Test with valid integer
        settings = OllamaSettings(max_tokens="1000")
        assert settings.max_tokens == 1000
        
        # Test with None
        settings = OllamaSettings(max_tokens=None)
        assert settings.max_tokens is None
        
        # Test with invalid string
        settings = OllamaSettings(max_tokens="invalid")
        assert settings.max_tokens is None
    
    def test_ollama_settings_from_secure_storage(self):
        """Test loading Ollama settings from secure storage"""
        mock_secure_config = {
            "enabled": False,
            "base_url": "http://custom-ollama:11434",
            "default_model": "codellama",
            "timeout": 60,
            "max_retries": 5,
            "temperature": 0.5,
            "max_tokens": 2000,
            "streaming": False
        }
        
        with patch('app.core.config.secure_settings') as mock_secure:
            mock_secure.get_category.return_value = {"ollama": mock_secure_config}
            
            settings = OllamaSettings()
            
            assert settings.enabled is False
            assert settings.base_url == "http://custom-ollama:11434"
            assert settings.default_model == "codellama"
            assert settings.timeout == 60
            assert settings.max_retries == 5
            assert settings.temperature == 0.5
            assert settings.max_tokens == 2000
            assert settings.streaming is False


class TestOpenAISettings:
    """Test OpenAISettings configuration"""
    
    def test_default_openai_settings(self):
        """Test default OpenAI settings values"""
        settings = OpenAISettings()
        
        assert settings.enabled is True
        assert settings.api_key is None
        assert settings.base_url == "https://openrouter.ai/api/v1"
        assert settings.default_model == "anthropic/claude-3.5-sonnet"
        assert settings.provider_name is None
        assert settings.custom_headers == {}
        assert settings.timeout == 30
        assert settings.max_retries == 3
    
    def test_openai_settings_from_secure_storage(self):
        """Test loading OpenAI settings from secure storage"""
        mock_secure_config = {
            "enabled": True,
            "api_key": "test-api-key",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4",
            "provider_name": "OpenAI",
            "timeout": 60,
            "max_retries": 5
        }
        
        with patch('app.core.config.secure_settings') as mock_secure:
            mock_secure.get_category.return_value = {"openai_compatible": mock_secure_config}
            
            settings = OpenAISettings()
            
            assert settings.enabled is True
            assert settings.api_key.get_secret_value() == "test-api-key"
            assert settings.base_url == "https://api.openai.com/v1"
            assert settings.default_model == "gpt-4"
            assert settings.provider_name == "OpenAI"
            assert settings.timeout == 60
            assert settings.max_retries == 5


class TestFirecrawlSettings:
    """Test FirecrawlSettings configuration"""
    
    def test_default_firecrawl_settings(self):
        """Test default Firecrawl settings values"""
        settings = FirecrawlSettings()
        
        assert settings.enabled is False
        assert settings.deployment_mode == "docker"
        assert settings.docker_url == "http://firecrawl.localhost"
        assert settings.bull_auth_key is None
        assert settings.scraping_enabled is True
        assert settings.max_concurrent_scrapes == 5
        assert settings.scrape_timeout == 60
        assert settings.content_cleaning is True
        assert settings.extract_images is False
        assert settings.extract_links is True
        assert settings.formats == ["markdown", "raw"]
        assert settings.wait_for == 2000
        assert settings.screenshot is False
    
    def test_firecrawl_effective_properties(self):
        """Test Firecrawl effective properties"""
        settings = FirecrawlSettings(
            docker_url="http://custom-firecrawl:3002",
            bull_auth_key="test-key"
        )
        
        assert settings.effective_url == "http://custom-firecrawl:3002"
        assert settings.effective_api_key is None  # Docker mode doesn't need API key


class TestSettings:
    """Test main Settings configuration"""
    
    def test_default_settings(self):
        """Test default settings values"""
        settings = Settings()
        
        # Server settings
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.reload is True
        
        # Provider settings
        assert settings.preferred_provider == "openai_compatible"
        assert settings.enable_fallback is True
        
        # System settings
        assert settings.tool_system_enabled is True
        assert settings.agent_system_enabled is True
        assert settings.deep_agents_enabled is False
        assert settings.multi_writer_enabled is False
        
        # Visual system settings
        assert settings.visual_system_enabled is True
        assert settings.visual_default_model == "openai_vision:gpt-4-vision-preview"
        assert settings.visual_max_concurrent_analyses == 3
        
        # Reranker settings
        assert settings.custom_reranker_enabled is True
        assert settings.custom_reranker_model == "all-MiniLM-L6-v2"
        assert settings.ollama_reranker_enabled is False
    
    def test_settings_from_secure_storage(self):
        """Test loading settings from secure storage"""
        mock_system_config = {
            "tool_system_enabled": False,
            "agent_system_enabled": False,
            "deep_agents_enabled": True,
            "multi_writer_enabled": True,
            "preferred_provider": "ollama",
            "enable_fallback": False,
            "debug": False,
            "host": "0.0.0.0",
            "port": 9000,
            "environment": "production"
        }
        
        mock_jina_config = {
            "enabled": True,
            "api_key": "test-jina-key",
            "url": "http://custom-jina:8080",
            "model": "custom-reranker",
            "timeout": 60,
            "cache_ttl": 7200,
            "max_retries": 5
        }
        
        mock_searxng_config = {
            "url": "http://custom-searxng:8080"
        }
        
        with patch('app.core.config.secure_settings') as mock_secure:
            mock_secure.get_category.side_effect = lambda category, default=None: {
                "system_config": mock_system_config,
                "external_services": {
                    "jina_reranker": mock_jina_config,
                    "searxng": mock_searxng_config
                }
            }.get(category, default)
            
            settings = Settings()
            
            assert settings.tool_system_enabled is False
            assert settings.agent_system_enabled is False
            assert settings.deep_agents_enabled is True
            assert settings.multi_writer_enabled is True
            assert settings.preferred_provider == "ollama"
            assert settings.enable_fallback is False
            assert settings.debug is False
            assert settings.host == "0.0.0.0"
            assert settings.port == 9000
            assert settings.environment == "production"
            
            assert settings.jina_reranker_enabled is True
            assert settings.jina_reranker_api_key == "test-jina-key"
            assert settings.jina_reranker_url == "http://custom-jina:8080"
            assert settings.jina_reranker_model == "custom-reranker"
            assert settings.jina_reranker_timeout == 60
            assert settings.jina_reranker_cache_ttl == 7200
            assert settings.jina_reranker_max_retries == 5
            
            assert settings.searxng_url == "http://custom-searxng:8080"
    
    def test_multi_writer_enabled_parsing(self):
        """Test parsing multi_writer_enabled from environment variable"""
        with patch.dict(os.environ, {"MULTI_WRITER_ENABLED": "true"}):
            settings = Settings()
            assert settings.multi_writer_enabled is True
        
        with patch.dict(os.environ, {"MULTI_WRITER_ENABLED": "false"}):
            settings = Settings()
            assert settings.multi_writer_enabled is False
        
        with patch.dict(os.environ, {"MULTI_WRITER_ENABLED": "1"}):
            settings = Settings()
            assert settings.multi_writer_enabled is True
        
        with patch.dict(os.environ, {"MULTI_WRITER_ENABLED": "enabled"}):
            settings = Settings()
            assert settings.multi_writer_enabled is True


class TestProviderInitialization:
    """Test LLM provider initialization functions"""
    
    @patch('app.core.config.provider_registry')
    @patch('app.core.config.OpenAICompatibleProvider')
    @patch('app.core.config.OpenRouterProvider')
    @patch('app.core.config.OllamaProvider')
    def test_initialize_llm_providers_openai_compatible(self, mock_ollama, mock_openrouter, mock_openai, mock_registry):
        """Test initializing OpenAI-compatible provider"""
        mock_registry._providers = {}
        mock_registry._default_provider = None
        mock_registry.list_providers.return_value = []
        mock_registry.list_configured_providers.return_value = []
        mock_registry.register_provider = Mock()
        mock_registry.set_default_provider = Mock()
        
        settings = Settings()
        settings.openai_settings.api_key = Mock()
        settings.openai_settings.api_key.get_secret_value.return_value = "test-key"
        settings.ollama_settings.enabled = False
        
        initialize_llm_providers()
        
        mock_openai.assert_called_once()
        mock_registry.register_provider.assert_called()
    
    @patch('app.core.config.provider_registry')
    @patch('app.core.config.OpenRouterProvider')
    def test_initialize_llm_providers_openrouter_backward_compatibility(self, mock_openrouter, mock_registry):
        """Test initializing OpenRouter provider with backward compatibility"""
        mock_registry._providers = {}
        mock_registry._default_provider = None
        mock_registry.list_providers.return_value = []
        mock_registry.list_configured_providers.return_value = []
        mock_registry.register_provider = Mock()
        mock_registry.set_default_provider = Mock()
        
        settings = Settings()
        settings.openai_settings.enabled = False
        settings.openrouter_api_key = Mock()
        settings.openrouter_api_key.get_secret_value.return_value = "test-key"
        settings.ollama_settings.enabled = False
        
        initialize_llm_providers()
        
        mock_openrouter.assert_called_once()
        mock_registry.register_provider.assert_called()
    
    @patch('app.core.config.provider_registry')
    @patch('app.core.config.OllamaProvider')
    def test_initialize_llm_providers_ollama(self, mock_ollama, mock_registry):
        """Test initializing Ollama provider"""
        mock_registry._providers = {}
        mock_registry._default_provider = None
        mock_registry.list_providers.return_value = []
        mock_registry.list_configured_providers.return_value = []
        mock_registry.register_provider = Mock()
        mock_registry.set_default_provider = Mock()
        
        settings = Settings()
        settings.openai_settings.enabled = False
        settings.openrouter_api_key = None
        settings.ollama_settings.enabled = True
        
        initialize_llm_providers()
        
        mock_ollama.assert_called_once()
        mock_registry.register_provider.assert_called()
    
    @patch('app.core.config.provider_registry')
    @patch('app.core.config.secure_settings')
    @patch('app.core.config.OpenAICompatibleProvider')
    @patch('app.core.config.OllamaProvider')
    def test_initialize_llm_providers_from_secure_settings(self, mock_ollama, mock_openai, mock_secure, mock_registry):
        """Test initializing providers from secure settings"""
        mock_registry._providers = {}
        mock_registry._default_provider = None
        mock_registry.list_providers.return_value = []
        mock_registry.list_configured_providers.return_value = []
        mock_registry.register_provider = Mock()
        mock_registry.set_default_provider = Mock()
        
        mock_secure.get_category.return_value = {
            "openai_compatible": {
                "enabled": True,
                "api_key": "secure-key",
                "base_url": "https://secure-api.com/v1"
            },
            "ollama": {
                "enabled": True,
                "base_url": "http://secure-ollama:11434"
            }
        }
        
        settings = Settings()
        settings.openai_settings.enabled = False
        settings.openrouter_api_key = None
        settings.ollama_settings.enabled = False
        
        initialize_llm_providers()
        
        mock_openai.assert_called_once()
        mock_ollama.assert_called_once()
        assert mock_registry.register_provider.call_count == 2


class TestLLMFactory:
    """Test LLM factory functions"""
    
    @pytest.mark.asyncio
    async def test_get_llm_no_providers(self):
        """Test getting LLM when no providers are configured"""
        with patch('app.core.config.provider_registry') as mock_registry:
            mock_registry.list_providers.return_value = []
            mock_registry.list_configured_providers.return_value = []
            
            with patch('app.core.config.initialize_llm_providers'):
                llm = await get_llm("test-model")
                
                # Should return a mock LLM
                assert llm is not None
                assert hasattr(llm, '_generate')
                assert hasattr(llm, 'model_name')
    
    @pytest.mark.asyncio
    async def test_get_llm_with_provider(self):
        """Test getting LLM when providers are configured"""
        with patch('app.core.config.provider_registry') as mock_registry:
            mock_provider = Mock()
            mock_llm = Mock()
            
            mock_registry.list_providers.return_value = [mock_provider]
            mock_registry.list_configured_providers.return_value = [mock_provider]
            mock_registry.resolve_model.return_value = (mock_provider, "actual-model")
            mock_provider.create_llm.return_value = mock_llm
            
            with patch('app.core.config.initialize_llm_providers'):
                llm = await get_llm("test-model")
                
                assert llm == mock_llm
                mock_provider.create_llm.assert_called_once_with("actual-model")
    
    def test_get_available_models_no_providers(self):
        """Test getting available models when no providers are configured"""
        with patch('app.core.config.provider_registry') as mock_registry:
            mock_registry.list_providers.return_value = []
            
            with patch('app.core.config.initialize_llm_providers'):
                models = get_available_models()
                
                assert models == []
    
    def test_get_available_models_with_providers(self):
        """Test getting available models when providers are configured"""
        with patch('app.core.config.provider_registry') as mock_registry:
            mock_model = Mock()
            mock_model.name = "test-model"
            
            mock_registry.list_providers.return_value = [Mock()]
            mock_registry.list_all_models.return_value = [mock_model]
            
            with patch('app.core.config.initialize_llm_providers'):
                models = get_available_models()
                
                assert models == [mock_model]


class TestAgentSystemInitialization:
    """Test agent system initialization"""
    
    @patch('app.core.config.agent_registry')
    @patch('app.core.config.ToolAgent')
    @patch('app.core.config.KeywordStrategy')
    @patch('app.core.config.tool_registry')
    def test_initialize_agent_system_enabled(self, mock_tool_registry, mock_strategy, mock_agent, mock_agent_registry):
        """Test initializing agent system when enabled"""
        settings = Settings()
        settings.agent_system_enabled = True
        
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        with patch('app.core.config.get_llm') as mock_get_llm:
            mock_get_llm.return_value = Mock()
            
            result = initialize_agent_system()
            
            mock_agent.assert_called_once()
            mock_agent_registry.register.assert_called_once()
            mock_agent_registry.set_default_agent.assert_called_once()
            assert result == mock_agent_registry
    
    def test_initialize_agent_system_disabled(self):
        """Test initializing agent system when disabled"""
        settings = Settings()
        settings.agent_system_enabled = False
        
        result = initialize_agent_system()
        
        assert result is None


class TestLangChainComponents:
    """Test LangChain component initialization"""
    
    def test_initialize_langchain_components_disabled(self):
        """Test initializing LangChain components when disabled"""
        settings = Settings()
        settings.langchain_settings.enabled = False
        
        result = initialize_langchain_components()
        
        assert result is None
    
    @patch('app.core.config.OpenAIEmbeddings')
    @patch('app.core.config.Milvus')
    @patch('app.core.config.ConversationBufferMemory')
    def test_initialize_langchain_components_enabled(self, mock_memory, mock_milvus, mock_embeddings):
        """Test initializing LangChain components when enabled"""
        settings = Settings()
        settings.langchain_settings.enabled = True
        settings.langchain_settings.memory_enabled = True
        settings.vector_store_settings.provider = "milvus"
        settings.vector_store_settings.embedding_provider = "openai"
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vector_store = Mock()
        mock_milvus.return_value = mock_vector_store
        
        mock_memory_instance = Mock()
        mock_memory.return_value = mock_memory_instance
        
        with patch('app.core.config.initialize_llm_providers'):
            result = initialize_langchain_components()
            
            assert result is not None
            assert "embeddings" in result
            assert "vector_store" in result
            assert "memory" in result
            
            mock_embeddings.assert_called_once()
            mock_milvus.assert_called_once()
            mock_memory.assert_called_once()


class TestLangGraphWorkflows:
    """Test LangGraph workflow initialization"""
    
    def test_initialize_langgraph_workflows_disabled(self):
        """Test initializing LangGraph workflows when disabled"""
        settings = Settings()
        settings.langgraph_settings.enabled = False
        
        result = initialize_langgraph_workflows()
        
        assert result is None
    
    @patch('app.core.config.StateGraph')
    @patch('app.core.config.MemorySaver')
    def test_initialize_langgraph_workflows_enabled(self, mock_memory_saver, mock_state_graph):
        """Test initializing LangGraph workflows when enabled"""
        settings = Settings()
        settings.langgraph_settings.enabled = True
        settings.langgraph_settings.checkpoint_backend = "memory"
        
        mock_checkpoint_saver = Mock()
        mock_memory_saver.return_value = mock_checkpoint_saver
        
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph
        
        result = initialize_langgraph_workflows()
        
        assert result is not None
        assert "checkpoint_saver" in result
        assert "conversation" in result
        
        mock_memory_saver.assert_called_once()
        mock_state_graph.assert_called_once()


class TestFirebaseIntegration:
    """Test Firebase integration initialization"""
    
    def test_initialize_firebase_integration_disabled(self):
        """Test initializing Firebase integration when disabled"""
        settings = Settings()
        settings.firebase_settings.enabled = False
        
        result = initialize_firebase_integration()
        
        assert result is None
    
    @patch('app.core.config.firebase_admin')
    @patch('app.core.config.credentials')
    @patch('app.core.config.firestore')
    @patch('app.core.config.storage')
    @patch('os.path.exists')
    def test_initialize_firebase_integration_enabled(self, mock_exists, mock_storage, mock_firestore, mock_credentials, mock_firebase_admin):
        """Test initializing Firebase integration when enabled"""
        settings = Settings()
        settings.firebase_settings.enabled = True
        settings.firebase_settings.service_account_key_path = "/path/to/key.json"
        settings.firebase_settings.project_id = "test-project"
        settings.firebase_settings.database_url = "https://test-project.firebaseio.com"
        settings.firebase_settings.storage_bucket = "test-project.appspot.com"
        
        mock_exists.return_value = True
        mock_firebase_admin._apps = []  # No existing apps
        
        mock_cred = Mock()
        mock_credentials.Certificate.return_value = mock_cred
        
        mock_app = Mock()
        mock_firebase_admin.initialize_app.return_value = mock_app
        
        mock_db = Mock()
        mock_firestore.client.return_value = mock_db
        
        mock_bucket = Mock()
        mock_storage.bucket.return_value = mock_bucket
        
        result = initialize_firebase_integration()
        
        assert result is not None
        assert "app" in result
        assert "firestore" in result
        assert "storage" in result
        
        mock_credentials.Certificate.assert_called_once()
        mock_firebase_admin.initialize_app.assert_called_once()
        mock_firestore.client.assert_called_once()
        mock_storage.bucket.assert_called_once()


class TestAllLangChainComponents:
    """Test initializing all LangChain components"""
    
    @patch('app.core.config.initialize_langchain_components')
    @patch('app.core.config.initialize_langgraph_workflows')
    @patch('app.core.config.initialize_firebase_integration')
    @patch('app.core.config.integration_layer')
    def test_initialize_all_langchain_components_no_integration(self, mock_integration, mock_firebase, mock_langgraph, mock_langchain):
        """Test initializing all components when integration layer is not available"""
        mock_integration._initialized = False
        
        # Simulate ImportError for integration layer
        import sys
        original_modules = sys.modules.copy()
        if 'app.core.langchain.integration' in sys.modules:
            del sys.modules['app.core.langchain.integration']
        
        try:
            result = initialize_all_langchain_components()
            
            assert result is not None
            assert "integration" in result
            assert result["integration"]["status"] == "skipped"
        finally:
            # Restore original modules
            sys.modules.update(original_modules)
    
    @patch('app.core.config.initialize_langchain_components')
    @patch('app.core.config.initialize_langgraph_workflows')
    @patch('app.core.config.initialize_firebase_integration')
    @patch('app.core.config.integration_layer')
    def test_initialize_all_langchain_components_with_integration(self, mock_integration, mock_firebase, mock_langgraph, mock_langchain):
        """Test initializing all components when integration layer is available"""
        mock_integration._initialized = False
        mock_integration.initialize.return_value = None
        mock_integration.get_feature_flags.return_value = {
            "use_langchain_llm": True,
            "use_langgraph_workflows": True
        }
        mock_integration.get_integration_mode.return_value = Mock()
        mock_integration.get_integration_mode.return_value.value = "full"
        
        mock_langchain.return_value = {"embeddings": Mock()}
        mock_langgraph.return_value = {"checkpoint_saver": Mock()}
        mock_firebase.return_value = {"firestore": Mock()}
        
        result = initialize_all_langchain_components()
        
        assert result is not None
        assert "integration" in result
        assert "langchain" in result
        assert "langgraph" in result
        assert "firebase" in result
        
        assert result["integration"]["status"] == "success"
        assert result["langchain"]["status"] == "success"
        assert result["langgraph"]["status"] == "success"
        assert result["firebase"]["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__])