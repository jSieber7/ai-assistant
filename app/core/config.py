from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import SecretStr
import logging

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


class Settings(BaseSettings):
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"

    # Provider selection
    preferred_provider: str = "openrouter"  # "openrouter", "ollama", or "auto"
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


settings = Settings()


def initialize_llm_providers():
    """Initialize all configured LLM providers"""
    from .llm_providers import OpenRouterProvider, OllamaProvider, provider_registry

    # Initialize OpenRouter provider if API key is available
    if settings.openrouter_api_key:
        openrouter_provider = OpenRouterProvider(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
        )
        provider_registry.register_provider(openrouter_provider)
        logger.info("OpenRouter provider initialized")
    else:
        logger.warning("OpenRouter provider not initialized - missing API key")

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
        raise ValueError("No LLM providers are configured")

    # Set preferred provider if available
    preferred = settings.preferred_provider.lower()
    for provider in configured_providers:
        if provider.provider_type.value == preferred:
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

    except Exception as e:
        if settings.enable_fallback:
            # Try fallback providers
            logger.warning(f"Failed to create LLM for model '{model_name}': {str(e)}")

            configured_providers = provider_registry.list_configured_providers()
            for fallback_provider in configured_providers:
                if (
                    fallback_provider.provider_type.value
                    == settings.preferred_provider.lower()
                ):
                    continue  # Skip the one that already failed

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

        raise ValueError(f"Failed to create LLM for model '{model_name}': {str(e)}")


def get_available_models():
    """Get all available models from all configured providers"""
    from .llm_providers import provider_registry

    if not provider_registry.list_providers():
        initialize_llm_providers()

    import asyncio

    # Run async function in sync context
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(provider_registry.list_all_models())


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
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    llm = loop.run_until_complete(get_llm())

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
