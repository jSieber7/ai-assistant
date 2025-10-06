from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import SecretStr


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


class Settings(BaseSettings):
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"

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
    agent_system_enabled: bool = True
    default_agent: str = "tool_agent"
    max_agent_iterations: int = 3
    agent_timeout: int = 60

    # Caching system settings
    cache_settings: CacheSettings = CacheSettings()

    # Models unused in the current stage of development
    router_model: str = "deepseek/deepseek-chat"
    logic_model: str = "anthropic/claude-3.5-sonnet"
    human_interface_model: str = "anthropic/claude-3.5-sonnet"

    # Unused in the current stage of development
    postgres_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    searxng_url: Optional[str] = None
    secret_key: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()


def get_llm(model_name: Optional[str] = None):
    """Factory function to create LLM instances"""
    from langchain_openai import ChatOpenAI

    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment")

    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=model_name or settings.default_model,
        temperature=0.7,
    )


def initialize_agent_system():
    """Initialize the agent system with default agents"""
    from .agents.registry import agent_registry
    from .agents.tool_agent import ToolAgent
    from .agents.strategies import KeywordStrategy
    from .tools.registry import tool_registry

    if not settings.agent_system_enabled:
        return

    # Create default agent
    llm = get_llm()
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
