from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, ClassVar
from pydantic import SecretStr
import logging
import os

# Import MultiWriterSettings before using it
from .multi_writer_config import MultiWriterSettings

logger = logging.getLogger(__name__)


class CacheSettings(BaseSettings):
    """Caching system configuration."""

    # General caching settings
    caching_enabled: bool = True
    default_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000

    # Multi-layer cache settings
    cache_layers: List[str] = ["memory", "redis"]  # Layer priority order
    write_through: bool = True
    read_through: bool = True

    # Memory cache settings
    memory_cache_max_size: int = 500
    memory_cache_cleanup_interval: int = 60

    # Redis cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_max_connections: int = 10
    redis_connection_timeout: int = 5
    redis_retry_attempts: int = 3

    # Compression settings
    compression_enabled: bool = True
    compression_algorithm: str = "gzip"  # gzip, lzma, zlib, base64, none
    compression_level: int = 6
    compression_threshold: int = 100  # Minimum size to compress (bytes)

    # Batching settings
    batching_enabled: bool = True
    max_batch_size: int = 10
    max_batch_wait_time: float = 0.1  # 100ms
    max_batch_queue_size: int = 1000

    # Connection pooling settings
    connection_pooling_enabled: bool = True
    max_http_connections: int = 10
    max_database_connections: int = 5
    connection_timeout: float = 5.0
    acquire_timeout: float = 10.0

    # Performance monitoring
    monitoring_enabled: bool = True
    stats_collection_interval: int = 60  # seconds


class OllamaSettings(BaseSettings):
    """Ollama local model configuration."""

    # Ollama server settings
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    default_model: str = "llama2"

    # Connection settings
    timeout: int = 30
    max_retries: int = 3

    # Model settings
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = True

    # Health check settings
    health_check_interval: int = 60  # seconds
    auto_health_check: bool = True


class OpenAISettings(BaseSettings):
    """Generic OpenAI-compatible API configuration"""

    enabled: bool = True
    api_key: Optional[SecretStr] = None
    base_url: str = "https://openrouter.ai/api/v1"  # Default for backward compatibility
    default_model: str = "anthropic/claude-3.5-sonnet"
    provider_name: Optional[str] = None  # Auto-detected from base_url if not provided
    custom_headers: Dict[str, str] = {}
    timeout: int = 30
    max_retries: int = 3

    class Config:
        env_prefix = "OPENAI_COMPATIBLE_"


class Settings(BaseSettings):
    # OpenAI-compatible provider settings (new generic approach)
    openai_settings: OpenAISettings = OpenAISettings()

    # Backward compatibility settings
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"

    # Provider selection
    preferred_provider: str = (
        "openai_compatible"  # "openai_compatible", "ollama", or "auto"
    )
    enable_fallback: bool = True  # Fall back to other providers if preferred fails

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    environment: str = "development"
    debug: bool = True
    reload: bool = True

    # Tool system settings
    tool_system_enabled: bool = True
    max_concurrent_tools: int = 5

    # Agent system settings
    agent_system_enabled: bool = False
    default_agent: Optional[str] = None
    max_agent_iterations: int = 3
    agent_timeout: int = 60

    # Caching system settings
    cache_settings: CacheSettings = CacheSettings()

    # Ollama settings
    ollama_settings: OllamaSettings = OllamaSettings()

    # Multi-writer system settings
    multi_writer_settings: ClassVar[MultiWriterSettings] = MultiWriterSettings()

    # Models unused in the current stage of development
    router_model: str = "deepseek/deepseek-chat"
    logic_model: str = "anthropic/claude-3.5-sonnet"
    human_interface_model: str = "anthropic/claude-3.5-sonnet"

    # Unused in the current stage of development
    postgres_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    searxng_url: Optional[str] = None
    secret_key: Optional[str] = None

    class Config:
        env_file = ".env"
        ignored_types = (
            MultiWriterSettings,
        )  # Ignore the class itself, not an instance

        @classmethod
        def customize_sources(cls, init_settings, env_settings, file_secret_settings):
            # Prioritize .env.test when in testing environment
            if os.getenv("ENVIRONMENT") == "testing":
                test_env_file = ".env.test"
                if os.path.exists(test_env_file):
                    return (
                        init_settings,
                        env_settings,
                        file_secret_settings(test_env_file),
                    )
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


settings = Settings()


def initialize_llm_providers():
    """Initialize all configured LLM providers with backward compatibility"""
    from .llm_providers import (
        OpenAICompatibleProvider,
        OpenRouterProvider,
        OllamaProvider,
        provider_registry,
    )

    # Initialize generic OpenAI-compatible provider
    openai_provider = None

    # Check for new generic settings first
    if settings.openai_settings.enabled and settings.openai_settings.api_key:
        openai_provider = OpenAICompatibleProvider(
            api_key=settings.openai_settings.api_key.get_secret_value(),
            base_url=settings.openai_settings.base_url,
            provider_name=settings.openai_settings.provider_name,
            custom_headers=settings.openai_settings.custom_headers,
        )
        provider_registry.register_provider(openai_provider)
        logger.info(f"OpenAI-compatible provider initialized: {openai_provider.name}")

    # Backward compatibility: check for OpenRouter settings
    elif settings.openrouter_api_key:
        # Create OpenRouter provider using the new generic class
        openrouter_provider = OpenRouterProvider(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
        )
        provider_registry.register_provider(openrouter_provider)
        logger.info("OpenRouter provider initialized (backward compatibility mode)")
        openai_provider = openrouter_provider
    else:
        logger.warning("No OpenAI-compatible provider configured - missing API key")

    # Initialize Ollama provider if enabled
    if settings.ollama_settings.enabled:
        ollama_provider = OllamaProvider(base_url=settings.ollama_settings.base_url)
        provider_registry.register_provider(ollama_provider)
        logger.info(
            f"Ollama provider initialized at {settings.ollama_settings.base_url}"
        )
    else:
        logger.info("Ollama provider disabled in settings")

    # Set default provider based on preferences
    configured_providers = provider_registry.list_configured_providers()
    if not configured_providers:
        logger.warning("No LLM providers are configured - some features may not work")
        # Don't raise an error, just return the registry without a default provider
        return provider_registry

    # Handle provider preference with backward compatibility
    preferred = settings.preferred_provider.lower()

    # Map old provider names to new ones for backward compatibility
    provider_mapping = {
        "openrouter": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "ollama": "ollama",
    }

    mapped_preferred = provider_mapping.get(preferred, preferred)

    for provider in configured_providers:
        if provider.provider_type.value == mapped_preferred:
            provider_registry.set_default_provider(provider.provider_type)
            logger.info(f"Set {provider.name} as default provider")
            break
    else:
        # Use first available provider as default
        default_provider = configured_providers[0]
        provider_registry.set_default_provider(default_provider.provider_type)
        logger.info(
            f"Set {default_provider.name} as default provider (no preferred provider found)"
        )

    return provider_registry


async def get_llm(model_name: Optional[str] = None, **kwargs):
    """
    Factory function to create LLM instances with multi-provider support

    Args:
        model_name: Model name (supports format "provider:model" or just "model")
        **kwargs: Additional LLM parameters (temperature, max_tokens, etc.)

    Returns:
        LangChain LLM instance
    """
    from .llm_providers import provider_registry

    if not provider_registry.list_providers():
        # Initialize providers if not already done
        initialize_llm_providers()

    # Use default model if none specified
    if not model_name:
        model_name = settings.default_model

    try:
        # Resolve model to provider and actual model name
        provider, actual_model = await provider_registry.resolve_model(model_name)

        # Create LLM instance
        llm = await provider.create_llm(actual_model, **kwargs)

        logger.info(f"Created {provider.name} LLM with model '{actual_model}'")
        return llm

    except ValueError as e:
        # Handle "Model not found" errors specifically
        if "not found in any configured provider" in str(e):
            # Get list of available models for a more helpful error message
            try:
                available_models = get_available_models()
                if available_models:
                    model_list = ", ".join([m.name for m in available_models[:5]])
                    if len(available_models) > 5:
                        model_list += f" and {len(available_models) - 5} more"
                    error_msg = f"Model '{model_name}' not found. Available models: {model_list}"
                else:
                    error_msg = f"Model '{model_name}' not found. No models are currently available."

                if settings.enable_fallback:
                    error_msg += " Trying fallback providers..."

                logger.warning(error_msg)
            except Exception:
                logger.warning(
                    f"Model '{model_name}' not found in any configured provider"
                )

        # Try fallback providers if enabled
        if settings.enable_fallback:
            logger.warning(f"Attempting fallback for model '{model_name}'")

            configured_providers = provider_registry.list_configured_providers()
            for fallback_provider in configured_providers:
                # Skip the provider that already failed if we can determine it
                if "not found" in str(e) and ":" in model_name:
                    failed_provider_name = model_name.split(":", 1)[0]
                    try:
                        if (
                            fallback_provider.provider_type.value
                            == failed_provider_name
                        ):
                            continue
                    except ValueError:
                        pass

                try:
                    # Try provider's default model
                    models = await fallback_provider.list_models()
                    if models:
                        fallback_model = models[0].name
                        llm = await fallback_provider.create_llm(
                            fallback_model, **kwargs
                        )
                        logger.info(
                            f"Created fallback {fallback_provider.name} LLM with model '{fallback_model}'"
                        )
                        return llm
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback to {fallback_provider.name} failed: {str(fallback_error)}"
                    )
                    continue

        # If we get here, all fallback attempts failed
        if "not found in any configured provider" in str(e):
            # Re-raise with a more helpful error message
            try:
                available_models = get_available_models()
                if available_models:
                    model_list = ", ".join([m.name for m in available_models[:5]])
                    if len(available_models) > 5:
                        model_list += f" and {len(available_models) - 5} more"
                    raise ValueError(
                        f"Model '{model_name}' not found. Available models: {model_list}"
                    )
                else:
                    raise ValueError(
                        f"Model '{model_name}' not found. No models are currently available."
                    )
            except Exception:
                raise ValueError(
                    f"Model '{model_name}' not found in any configured provider"
                )
        else:
            raise ValueError(f"Failed to create LLM for model '{model_name}': {str(e)}")

    except Exception as e:
        # Handle other types of errors
        logger.error(
            f"Unexpected error creating LLM for model '{model_name}': {str(e)}"
        )
        raise ValueError(f"Failed to create LLM for model '{model_name}': {str(e)}")


def get_available_models():
    """Get all available models from all configured providers"""
    from .llm_providers import provider_registry

    if not provider_registry.list_providers():
        initialize_llm_providers()

    import asyncio

    # Create a simple synchronous wrapper
    def _sync_get_models():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create a coroutine and run it
                async def _get_models():
                    return await provider_registry.list_all_models()

                return loop.run_until_complete(_get_models())
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            return []

    # Use threading to avoid event loop issues
    import concurrent.futures

    # Use a thread pool to run the async function
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_sync_get_models)
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to get models from thread: {str(e)}")
            return []


def initialize_agent_system():
    """Initialize the agent system with default agents"""
    from .agents.registry import agent_registry
    from .agents.tool_agent import ToolAgent
    from .agents.strategies import KeywordStrategy
    from .tools.registry import tool_registry

    if not settings.agent_system_enabled:
        return

    # Initialize LLM providers first
    initialize_llm_providers()

    # Create default agent
    import asyncio

    # Get LLM asynchronously
    try:
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, need to run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_llm())
                llm = future.result(timeout=10)  # Add 10 second timeout
        except RuntimeError:
            # No running loop, we can use asyncio.run
            llm = asyncio.run(get_llm())
    except concurrent.futures.TimeoutError:
        logger.error("Agent system initialization timed out while getting LLM")
        raise ValueError(
            "Agent system initialization timed out - LLM provider not responding"
        )
    except Exception as e:
        logger.error(f"Failed to get LLM for agent system: {str(e)}")
        raise

    tool_agent = ToolAgent(
        tool_registry=tool_registry,
        llm=llm,
        selection_strategy=KeywordStrategy(),
        max_iterations=settings.max_agent_iterations,
    )

    # Register the agent
    agent_registry.register(tool_agent, category="default")
    agent_registry.set_default_agent(tool_agent.name)

    return agent_registry
