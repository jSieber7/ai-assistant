from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, ClassVar
from pydantic import SecretStr, validator
import logging
import os

# Import MultiWriterSettings before using it
from .multi_writer_config import MultiWriterSettings
from .secure_settings import secure_settings

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

    @validator("max_tokens", pre=True, always=True)
    @classmethod
    def parse_max_tokens(cls, v):
        """Parse max_tokens from environment variable, handling empty strings."""
        if v == "" or v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    # Health check settings
    health_check_interval: int = 60  # seconds
    auto_health_check: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        # Load from secure settings if available
        try:
            secure_ollama_config = secure_settings.get_category("llm_providers").get("ollama", {})
            if secure_ollama_config:
                self.enabled = secure_ollama_config.get("enabled", self.enabled)
                self.base_url = secure_ollama_config.get("base_url", self.base_url)
                self.default_model = secure_ollama_config.get("default_model", self.default_model)
                self.timeout = secure_ollama_config.get("timeout", self.timeout)
                self.max_retries = secure_ollama_config.get("max_retries", self.max_retries)
                self.temperature = secure_ollama_config.get("temperature", self.temperature)
                self.max_tokens = secure_ollama_config.get("max_tokens", self.max_tokens)
                self.streaming = secure_ollama_config.get("streaming", self.streaming)
        except Exception as e:
            logger.warning(f"Failed to load Ollama settings from secure storage: {e}")

    class Config:
        env_prefix = "OLLAMA_SETTINGS_"


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

    def __init__(self, **data):
        super().__init__(**data)
        # Load from secure settings if available
        try:
            secure_openai_config = secure_settings.get_category("llm_providers").get("openai_compatible", {})
            if secure_openai_config:
                self.enabled = secure_openai_config.get("enabled", self.enabled)
                if secure_openai_config.get("api_key"):
                    self.api_key = SecretStr(secure_openai_config["api_key"])
                self.base_url = secure_openai_config.get("base_url", self.base_url)
                self.default_model = secure_openai_config.get("default_model", self.default_model)
                self.provider_name = secure_openai_config.get("provider_name", self.provider_name)
                self.timeout = secure_openai_config.get("timeout", self.timeout)
                self.max_retries = secure_openai_config.get("max_retries", self.max_retries)
        except Exception as e:
            logger.warning(f"Failed to load OpenAI settings from secure storage: {e}")

    class Config:
        env_prefix = "OPENAI_COMPATIBLE_"


class FirecrawlSettings(BaseSettings):
    """Firecrawl configuration for web scraping and data storage (Docker-only)"""

    enabled: bool = False
    deployment_mode: str = "docker"  # Docker-only mode

    # Docker Configuration
    docker_url: str = "http://firecrawl-api:3002"
    bull_auth_key: Optional[str] = None

    # Web scraping specific settings
    scraping_enabled: bool = True
    max_concurrent_scrapes: int = 5
    scrape_timeout: int = 60

    # Data processing settings
    content_cleaning: bool = True
    extract_images: bool = False
    extract_links: bool = True
    formats: List[str] = ["markdown", "raw"]
    wait_for: int = 2000
    screenshot: bool = False
    include_tags: List[str] = ["article", "main", "content"]
    exclude_tags: List[str] = ["nav", "footer", "aside", "script", "style"]

    # Health check timeout
    health_check_timeout: int = 10  # Timeout for Docker health check

    def __init__(self, **data):
        super().__init__(**data)
        # Load from secure settings if available
        try:
            secure_firecrawl_config = secure_settings.get_category("external_services").get("firecrawl", {})
            if secure_firecrawl_config:
                self.enabled = secure_firecrawl_config.get("enabled", self.enabled)
                self.deployment_mode = secure_firecrawl_config.get("deployment_mode", self.deployment_mode)
                self.docker_url = secure_firecrawl_config.get("docker_url", self.docker_url)
                self.bull_auth_key = secure_firecrawl_config.get("bull_auth_key", self.bull_auth_key)
                self.scraping_enabled = secure_firecrawl_config.get("scraping_enabled", self.scraping_enabled)
                self.max_concurrent_scrapes = secure_firecrawl_config.get("max_concurrent_scrapes", self.max_concurrent_scrapes)
                self.scrape_timeout = secure_firecrawl_config.get("scrape_timeout", self.scrape_timeout)
        except Exception as e:
            logger.warning(f"Failed to load Firecrawl settings from secure storage: {e}")

    @property
    def effective_url(self) -> str:
        """Get the Docker URL"""
        return self.docker_url

    @property
    def effective_api_key(self) -> Optional[str]:
        """Docker mode doesn't need API key"""
        return None

    class Config:
        env_prefix = "FIRECRAWL_"


class PlaywrightSettings(BaseSettings):
    """Playwright configuration for browser automation"""

    enabled: bool = False
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    timeout: int = 30
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    class Config:
        env_prefix = "PLAYWRIGHT_"


class MilvusSettings(BaseSettings):
    """Milvus vector database configuration for Deep Search agent"""

    # Connection settings
    enabled: bool = True
    host: str = "milvus"
    port: int = 19530
    user: Optional[str] = None
    password: Optional[str] = None
    database: str = "default"

    # Collection settings
    collection_prefix: str = "deep_search_"
    embedding_dimension: int = 1536  # Default for OpenAI embeddings
    index_type: str = "HNSW"  # HNSW, IVF_FLAT, IVF_PQ, etc.
    metric_type: str = "COSINE"  # COSINE, L2, IP

    # Performance settings
    connection_pool_size: int = 10
    connection_timeout: int = 10
    max_retries: int = 3

    # Collection management
    auto_cleanup: bool = True  # Auto-cleanup temporary collections
    cleanup_delay: int = 3600  # Delay before cleanup (seconds)

    class Config:
        env_prefix = "MILVUS_"


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

    # Reverse proxy settings
    base_url: str = "http://localhost"  # Base URL when behind reverse proxy
    behind_proxy: bool = True  # Whether the app is behind a reverse proxy

    # Tool system settings
    tool_system_enabled: bool = True
    max_concurrent_tools: int = 5

    # Agent system settings
    agent_system_enabled: bool = True
    default_agent: Optional[str] = None
    max_agent_iterations: int = 3
    agent_timeout: int = 60

    # Caching system settings
    cache_settings: CacheSettings = CacheSettings()

    # Ollama settings
    ollama_settings: OllamaSettings = OllamaSettings()

    # Multi-writer system settings
    multi_writer_settings: ClassVar[MultiWriterSettings] = MultiWriterSettings()

    # Firecrawl settings
    firecrawl_settings: FirecrawlSettings = FirecrawlSettings()

    # Playwright settings
    playwright_settings: PlaywrightSettings = PlaywrightSettings()

    # Milvus settings
    milvus_settings: MilvusSettings = MilvusSettings()

    # Jina Reranker settings
    jina_reranker_enabled: bool = False
    jina_reranker_url: str = "http://jina-reranker:8080"
    jina_reranker_model: str = "jina-reranker-v2-base-multilingual"
    jina_reranker_timeout: int = 30
    jina_reranker_cache_ttl: int = 3600
    jina_reranker_max_retries: int = 3
    jina_reranker_api_key: Optional[str] = None

    # Models unused in the current stage of development
    router_model: str = "deepseek/deepseek-chat"
    logic_model: str = "anthropic/claude-3.5-sonnet"
    human_interface_model: str = "anthropic/claude-3.5-sonnet"

    # Unused in the current stage of development
    postgres_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    secret_key: Optional[str] = None
    # SearXNG URL is hardcoded since it's an internal service
    searxng_url: str = "http://searxng:8080"

    def __init__(self, **data):
        super().__init__(**data)
        # Load system config from secure settings
        try:
            secure_system_config = secure_settings.get_category("system_config", {})
            if secure_system_config:
                self.tool_system_enabled = secure_system_config.get("tool_system_enabled", self.tool_system_enabled)
                self.agent_system_enabled = secure_system_config.get("agent_system_enabled", self.agent_system_enabled)
                self.preferred_provider = secure_system_config.get("preferred_provider", self.preferred_provider)
                self.enable_fallback = secure_system_config.get("enable_fallback", self.enable_fallback)
                self.debug = secure_system_config.get("debug", self.debug)
                self.host = secure_system_config.get("host", self.host)
                self.port = secure_system_config.get("port", self.port)
                self.environment = secure_system_config.get("environment", self.environment)
                self.secret_key = secure_system_config.get("secret_key", self.secret_key)

            # Load Jina Reranker settings from secure settings
            secure_jina_config = secure_settings.get_category("external_services", {}).get("jina_reranker", {})
            if secure_jina_config:
                self.jina_reranker_enabled = secure_jina_config.get("enabled", self.jina_reranker_enabled)
                self.jina_reranker_api_key = secure_jina_config.get("api_key", self.jina_reranker_api_key)
                self.jina_reranker_url = secure_jina_config.get("url", self.jina_reranker_url)
                self.jina_reranker_model = secure_jina_config.get("model", self.jina_reranker_model)
                self.jina_reranker_timeout = secure_jina_config.get("timeout", self.jina_reranker_timeout)
                self.jina_reranker_cache_ttl = secure_jina_config.get("cache_ttl", self.jina_reranker_cache_ttl)
                self.jina_reranker_max_retries = secure_jina_config.get("max_retries", self.jina_reranker_max_retries)

            # Load SearXNG settings from secure settings
            secure_searxng_config = secure_settings.get_category("external_services", {}).get("searxng", {})
            if secure_searxng_config:
                self.searxng_url = secure_searxng_config.get("url", self.searxng_url)

        except Exception as e:
            logger.warning(f"Failed to load system settings from secure storage: {e}")

    class Config:
        env_file = ".env"
        ignored_types = (
            MultiWriterSettings,
        )  # Ignore the class itself, not an instance
        extra = "ignore"  # Allow extra fields to avoid validation errors

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


def initialize_firecrawl_system():
    """Initialize the Firecrawl Docker scraping system with tools and agents"""
    from .tools.registry import tool_registry
    from .tools.firecrawl_tool import FirecrawlTool
    from .agents.registry import agent_registry
    from .agents.firecrawl_agent import FirecrawlAgent

    if not settings.firecrawl_settings.enabled:
        logger.info("Firecrawl system disabled in settings")
        return

    # Initialize LLM providers first
    initialize_llm_providers()

    # Create Firecrawl scraper tool (Docker-only)
    firecrawl_tool = FirecrawlTool()
    tool_registry.register(firecrawl_tool, category="firecrawl")
    logger.info("Firecrawl Docker scraper tool registered")

    # Create Firecrawl scraper agent
    try:
        import asyncio

        # Get LLM asynchronously
        try:
            asyncio.get_running_loop()
            # We're in an async context, need to run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_llm())
                llm = future.result(timeout=10)
        except RuntimeError:
            # No running loop, we can use asyncio.run
            llm = asyncio.run(get_llm())

        firecrawl_agent = FirecrawlAgent(llm=llm)
        agent_registry.register(firecrawl_agent, category="firecrawl")
        logger.info("Firecrawl Docker scraper agent registered")

    except Exception as e:
        logger.error(f"Failed to create Firecrawl scraper agent: {str(e)}")

    return tool_registry, agent_registry


def initialize_playwright_system():
    """Initialize the Playwright browser automation system with tools"""
    from .tools.registry import tool_registry
    from .tools.playwright_tool import PlaywrightTool

    if not settings.playwright_settings.enabled:
        logger.info("Playwright system disabled in settings")
        return

    # Create Playwright tool
    playwright_tool = PlaywrightTool(
        headless=settings.playwright_settings.headless,
        browser_type=settings.playwright_settings.browser_type,
    )
    tool_registry.register(playwright_tool, category="playwright")
    logger.info("Playwright automation tool registered")

    return tool_registry
