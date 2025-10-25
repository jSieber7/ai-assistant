"""
Test configuration fixtures for AI Assistant application.

Contains configuration objects for different test scenarios.
"""

import os
from typing import Dict, Any, List
from unittest.mock import Mock

from app.core.config import (
    Settings,
    LLMProviderSettings,
    OpenRouterSettings,
    OpenAISettings,
    OllamaSettings,
    VisualLLMSettings,
    MultiWriterSettings,
    DatabaseSettings,
    RedisSettings,
    MonitoringSettings
)


class TestSettingsFactory:
    """Factory for creating test configuration objects"""
    
    @staticmethod
    def create_minimal_settings() -> Settings:
        """Create minimal settings for basic testing"""
        return Settings(
            environment="test",
            debug=True,
            log_level="DEBUG",
            database=DatabaseSettings(
                url="sqlite:///:memory:",
                echo=False
            ),
            redis=RedisSettings(
                url="redis://localhost:6379/1",
                decode_responses=True
            ),
            monitoring=MonitoringSettings(
                enabled=False,
                metrics_port=9091
            )
        )
    
    @staticmethod
    def create_full_settings() -> Settings:
        """Create full settings for comprehensive testing"""
        return Settings(
            environment="test",
            debug=True,
            log_level="DEBUG",
            api_host="127.0.0.1",
            api_port=8000,
            api_prefix="/api/v1",
            cors_origins=["http://localhost:3000"],
            database=DatabaseSettings(
                url="postgresql://test:test@localhost:5432/test_db",
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            ),
            redis=RedisSettings(
                url="redis://localhost:6379/1",
                decode_responses=True,
                max_connections=10,
                retry_on_timeout=True,
                health_check_interval=30
            ),
            llm_providers=TestSettingsFactory._create_llm_providers(),
            visual_llm=TestSettingsFactory._create_visual_llm_settings(),
            multi_writer=TestSettingsFactory._create_multi_writer_settings(),
            monitoring=TestSettingsFactory._create_monitoring_settings()
        )
    
    @staticmethod
    def _create_llm_providers() -> LLMProviderSettings:
        """Create LLM provider settings"""
        return LLMProviderSettings(
            default_provider="openrouter",
            openrouter=OpenRouterSettings(
                api_key="test_openrouter_key",
                base_url="https://openrouter.ai/api/v1",
                timeout=30,
                max_retries=3,
                models=["deepseek/deepseek-v3.1-terminus", "anthropic/claude-3-opus"]
            ),
            openai=OpenAISettings(
                api_key="test_openai_key",
                base_url="https://api.openai.com/v1",
                timeout=30,
                max_retries=3,
                models=["gpt-4", "gpt-3.5-turbo"]
            ),
            ollama=OllamaSettings(
                base_url="http://localhost:11434",
                timeout=60,
                max_retries=2,
                models=["llama2", "codellama"]
            )
        )
    
    @staticmethod
    def _create_visual_llm_settings() -> VisualLLMSettings:
        """Create visual LLM settings"""
        return VisualLLMSettings(
            default_provider="openai",
            openai=OpenAISettings(
                api_key="test_openai_vision_key",
                base_url="https://api.openai.com/v1",
                timeout=60,
                max_retries=3,
                models=["gpt-4-vision-preview"]
            ),
            max_image_size=20 * 1024 * 1024,  # 20MB
            supported_formats=["jpg", "jpeg", "png", "webp", "gif"]
        )
    
    @staticmethod
    def _create_multi_writer_settings() -> MultiWriterSettings:
        """Create multi-writer settings"""
        return MultiWriterSettings(
            default_model="deepseek/deepseek-v3.1-terminus",
            max_workflow_duration=1800,  # 30 minutes
            max_parallel_tasks=3,
            checkers=["grammar_checker", "fact_checker", "style_checker"],
            output_formats=["markdown", "html", "pdf"]
        )
    
    @staticmethod
    def _create_monitoring_settings() -> MonitoringSettings:
        """Create monitoring settings"""
        return MonitoringSettings(
            enabled=True,
            metrics_port=9091,
            health_check_interval=30,
            log_requests=True,
            log_errors=True,
            log_performance=True
        )


class TestEnvironmentConfigs:
    """Test environment configurations"""
    
    @staticmethod
    def development() -> Dict[str, Any]:
        """Development environment configuration"""
        return {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "database": {
                "url": "sqlite:///./dev.db",
                "echo": True
            },
            "redis": {
                "url": "redis://localhost:6379/0"
            },
            "llm_providers": {
                "default_provider": "openrouter",
                "openrouter": {
                    "api_key": "dev_openrouter_key"
                }
            }
        }
    
    @staticmethod
    def testing() -> Dict[str, Any]:
        """Testing environment configuration"""
        return {
            "environment": "testing",
            "debug": True,
            "log_level": "INFO",
            "database": {
                "url": "sqlite:///:memory:",
                "echo": False
            },
            "redis": {
                "url": "redis://localhost:6379/1"
            },
            "llm_providers": {
                "default_provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434"
                }
            },
            "monitoring": {
                "enabled": False
            }
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Production environment configuration"""
        return {
            "environment": "production",
            "debug": False,
            "log_level": "WARNING",
            "database": {
                "url": "postgresql://user:pass@db:5432/prod_db",
                "echo": False,
                "pool_size": 20,
                "max_overflow": 30
            },
            "redis": {
                "url": "redis://redis:6379/0",
                "max_connections": 50
            },
            "llm_providers": {
                "default_provider": "openrouter",
                "openrouter": {
                    "api_key": "prod_openrouter_key",
                    "timeout": 60,
                    "max_retries": 5
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9091
            }
        }


class TestFeatureFlags:
    """Test feature flag configurations"""
    
    @staticmethod
    def all_enabled() -> Dict[str, bool]:
        """All features enabled"""
        return {
            "enable_visual_llm": True,
            "enable_multi_writer": True,
            "enable_agent_designer": True,
            "enable_advanced_caching": True,
            "enable_monitoring": True,
            "enable_rate_limiting": True,
            "enable_authentication": True
        }
    
    @staticmethod
    def minimal() -> Dict[str, bool]:
        """Minimal features enabled"""
        return {
            "enable_visual_llm": False,
            "enable_multi_writer": False,
            "enable_agent_designer": False,
            "enable_advanced_caching": False,
            "enable_monitoring": False,
            "enable_rate_limiting": False,
            "enable_authentication": False
        }
    
    @staticmethod
    def core_only() -> Dict[str, bool]:
        """Core features only"""
        return {
            "enable_visual_llm": True,
            "enable_multi_writer": False,
            "enable_agent_designer": True,
            "enable_advanced_caching": True,
            "enable_monitoring": True,
            "enable_rate_limiting": False,
            "enable_authentication": False
        }


class TestProviderConfigs:
    """Test provider configurations"""
    
    @staticmethod
    def openrouter_only() -> Dict[str, Any]:
        """OpenRouter only configuration"""
        return {
            "default_provider": "openrouter",
            "openrouter": {
                "api_key": "test_openrouter_key",
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 30,
                "max_retries": 3,
                "models": ["deepseek/deepseek-v3.1-terminus"]
            }
        }
    
    @staticmethod
    def openai_only() -> Dict[str, Any]:
        """OpenAI only configuration"""
        return {
            "default_provider": "openai",
            "openai": {
                "api_key": "test_openai_key",
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "max_retries": 3,
                "models": ["gpt-4", "gpt-3.5-turbo"]
            }
        }
    
    @staticmethod
    def ollama_only() -> Dict[str, Any]:
        """Ollama only configuration"""
        return {
            "default_provider": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 60,
                "max_retries": 2,
                "models": ["llama2", "codellama"]
            }
        }
    
    @staticmethod
    def all_providers() -> Dict[str, Any]:
        """All providers configuration"""
        return {
            "default_provider": "openrouter",
            "openrouter": {
                "api_key": "test_openrouter_key",
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 30,
                "max_retries": 3,
                "models": ["deepseek/deepseek-v3.1-terminus"]
            },
            "openai": {
                "api_key": "test_openai_key",
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "max_retries": 3,
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 60,
                "max_retries": 2,
                "models": ["llama2", "codellama"]
            }
        }


class TestScenarioConfigs:
    """Test scenario configurations"""
    
    @staticmethod
    def load_testing() -> Dict[str, Any]:
        """Configuration for load testing"""
        return {
            "environment": "testing",
            "debug": False,
            "log_level": "WARNING",
            "database": {
                "url": "sqlite:///:memory:",
                "pool_size": 50,
                "max_overflow": 100
            },
            "redis": {
                "url": "redis://localhost:6379/2",
                "max_connections": 100
            },
            "monitoring": {
                "enabled": True,
                "log_requests": True,
                "log_performance": True
            }
        }
    
    @staticmethod
    def integration_testing() -> Dict[str, Any]:
        """Configuration for integration testing"""
        return {
            "environment": "testing",
            "debug": True,
            "log_level": "INFO",
            "database": {
                "url": "postgresql://test:test@localhost:5432/test_integration",
                "echo": False
            },
            "redis": {
                "url": "redis://localhost:6379/3"
            },
            "llm_providers": TestProviderConfigs.all_providers(),
            "monitoring": {
                "enabled": True
            }
        }
    
    @staticmethod
    def performance_testing() -> Dict[str, Any]:
        """Configuration for performance testing"""
        return {
            "environment": "testing",
            "debug": False,
            "log_level": "ERROR",
            "database": {
                "url": "sqlite:///:memory:",
                "pool_size": 20,
                "max_overflow": 40
            },
            "redis": {
                "url": "redis://localhost:6379/4",
                "max_connections": 50
            },
            "monitoring": {
                "enabled": True,
                "log_performance": True,
                "metrics_port": 9092
            }
        }


class TestMockConfigs:
    """Mock configurations for testing"""
    
    @staticmethod
    def mock_openai_client() -> Mock:
        """Create a mock OpenAI client"""
        client = Mock()
        client.chat.completions.create.return_value = {
            "id": "test-chat-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        return client
    
    @staticmethod
    def mock_database_client() -> Mock:
        """Create a mock database client"""
        client = Mock()
        client.execute.return_value = [{"id": 1, "name": "test"}]
        client.fetch_one.return_value = {"id": 1, "name": "test"}
        client.fetch_all.return_value = [{"id": 1, "name": "test"}]
        return client
    
    @staticmethod
    def mock_redis_client() -> Mock:
        """Create a mock Redis client"""
        client = Mock()
        client.get.return_value = "test_value"
        client.set.return_value = True
        client.delete.return_value = 1
        client.exists.return_value = True
        return client
    
    @staticmethod
    def mock_http_client() -> Mock:
        """Create a mock HTTP client"""
        client = Mock()
        client.get.return_value = Mock(
            status_code=200,
            json=lambda: {"status": "ok"},
            raise_for_status=lambda: None
        )
        client.post.return_value = Mock(
            status_code=201,
            json=lambda: {"id": 1},
            raise_for_status=lambda: None
        )
        return client


# Utility functions for creating test configurations
def create_test_config(
    environment: str = "testing",
    features: Dict[str, bool] = None,
    providers: Dict[str, Any] = None
) -> Settings:
    """Create a test configuration with custom parameters"""
    if environment == "development":
        config_dict = TestEnvironmentConfigs.development()
    elif environment == "production":
        config_dict = TestEnvironmentConfigs.production()
    else:
        config_dict = TestEnvironmentConfigs.testing()
    
    if features:
        config_dict.update(features)
    
    if providers:
        config_dict["llm_providers"] = providers
    
    # Convert dict to Settings object
    return Settings(**config_dict)


def create_minimal_test_config() -> Settings:
    """Create a minimal test configuration"""
    return TestSettingsFactory.create_minimal_settings()


def create_full_test_config() -> Settings:
    """Create a full test configuration"""
    return TestSettingsFactory.create_full_settings()