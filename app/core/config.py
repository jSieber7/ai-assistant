from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import SecretStr

class Settings(BaseSettings):
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"
    debug: bool = True
    reload: bool = True
    
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
        temperature=0.7
    )