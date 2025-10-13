"""
LLM Provider System for AI Assistant

This module provides a flexible provider system that supports multiple LLM providers
including OpenRouter and Ollama, with automatic fallback and health checking capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types"""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    provider: ProviderType
    display_name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = True


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self._is_healthy = True
        self._last_health_check = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass
    
    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured"""
        pass
    
    @abstractmethod
    async def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create LLM instance for the given model"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models from this provider"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible"""
        pass
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        models = await self.list_models()
        for model in models:
            if model.name == model_name:
                return model
        return None
    
    def is_healthy(self) -> bool:
        """Check if provider is currently marked as healthy"""
        return self._is_healthy
    
    def set_health_status(self, is_healthy: bool):
        """Set the health status of the provider"""
        self._is_healthy = is_healthy


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        super().__init__(ProviderType.OPENROUTER)
        self.api_key = api_key
        self.base_url = base_url
        self._common_models = [
            ModelInfo(
                name="anthropic/claude-3.5-sonnet",
                provider=ProviderType.OPENROUTER,
                display_name="Claude 3.5 Sonnet",
                description="Anthropic's most capable model",
                context_length=200000,
                supports_streaming=True,
                supports_tools=True
            ),
            ModelInfo(
                name="deepseek/deepseek-chat",
                provider=ProviderType.OPENROUTER,
                display_name="DeepSeek Chat",
                description="DeepSeek's conversational model",
                context_length=32768,
                supports_streaming=True,
                supports_tools=True
            ),
            ModelInfo(
                name="openai/gpt-4o",
                provider=ProviderType.OPENROUTER,
                display_name="GPT-4o",
                description="OpenAI's latest multimodal model",
                context_length=128000,
                supports_streaming=True,
                supports_tools=True
            ),
        ]
    
    @property
    def name(self) -> str:
        return "OpenRouter"
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    async def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create LangChain ChatOpenAI instance for OpenRouter"""
        from langchain_openai import ChatOpenAI
        
        if not self.is_configured:
            raise ValueError("OpenRouter provider is not configured with API key")
        
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens'),
            streaming=kwargs.get('streaming', False),
        )
    
    async def list_models(self) -> List[ModelInfo]:
        """List available OpenRouter models"""
        return self._common_models
    
    async def health_check(self) -> bool:
        """Check OpenRouter API health"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                is_healthy = response.status_code == 200
                self.set_health_status(is_healthy)
                return is_healthy
        except Exception as e:
            logger.error(f"OpenRouter health check failed: {str(e)}")
            self.set_health_status(False)
            return False


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(ProviderType.OLLAMA)
        self.base_url = base_url.rstrip('/')
    
    @property
    def name(self) -> str:
        return "Ollama"
    
    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)
    
    async def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create LangChain ChatOllama instance"""
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            # Fallback to older import path
            try:
                from langchain.chat_models import ChatOllama
            except ImportError:
                raise ImportError(
                    "LangChain Ollama integration not found. "
                    "Install with: pip install langchain-community"
                )
        
        if not self.is_configured:
            raise ValueError("Ollama provider is not configured with base URL")
        
        # Validate model exists
        model_info = await self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in Ollama")
        
        return ChatOllama(
            base_url=self.base_url,
            model=model_name,
            temperature=kwargs.get('temperature', 0.7),
            num_predict=kwargs.get('max_tokens'),
            streaming=kwargs.get('streaming', False),
        )
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Ollama models"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned {response.status_code}")
                
                data = response.json()
                models = []
                
                for model in data.get('models', []):
                    model_info = ModelInfo(
                        name=model['name'],
                        provider=ProviderType.OLLAMA,
                        display_name=model['name'].split(':')[0],  # Remove tag
                        description=f"Ollama model: {model['name']}",
                        context_length=model.get('details', {}).get('context_length'),
                        supports_streaming=True,
                        supports_tools=False  # Ollama doesn't support tool calling yet
                    )
                    models.append(model_info)
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            self.set_health_status(False)
            return []
    
    async def health_check(self) -> bool:
        """Check Ollama server health"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                is_healthy = response.status_code == 200
                self.set_health_status(is_healthy)
                return is_healthy
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            self.set_health_status(False)
            return False


class LLMProviderRegistry:
    """Registry for managing LLM providers"""
    
    def __init__(self):
        self._providers: Dict[ProviderType, LLMProvider] = {}
        self._default_provider: Optional[ProviderType] = None
    
    def register_provider(self, provider: LLMProvider):
        """Register a new provider"""
        self._providers[provider.provider_type] = provider
        logger.info(f"Registered {provider.name} provider")
        
        # Set as default if no default exists
        if self._default_provider is None and provider.is_configured:
            self._default_provider = provider.provider_type
    
    def get_provider(self, provider_type: ProviderType) -> Optional[LLMProvider]:
        """Get a specific provider"""
        return self._providers.get(provider_type)
    
    def get_default_provider(self) -> Optional[LLMProvider]:
        """Get the default provider"""
        if self._default_provider:
            return self._providers.get(self._default_provider)
        return None
    
    def set_default_provider(self, provider_type: ProviderType):
        """Set the default provider"""
        if provider_type in self._providers:
            self._default_provider = provider_type
            logger.info(f"Set {provider_type.value} as default provider")
        else:
            raise ValueError(f"Provider {provider_type.value} not registered")
    
    def list_providers(self) -> List[LLMProvider]:
        """List all registered providers"""
        return list(self._providers.values())
    
    def list_configured_providers(self) -> List[LLMProvider]:
        """List only configured and healthy providers"""
        return [
            provider for provider in self._providers.values()
            if provider.is_configured and provider.is_healthy()
        ]
    
    async def health_check_all(self):
        """Health check all providers"""
        tasks = []
        for provider in self._providers.values():
            if provider.is_configured:
                tasks.append(provider.health_check())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def resolve_model(self, model_name: str) -> tuple[LLMProvider, str]:
        """
        Resolve model name to provider and actual model name.
        
        Supports formats:
        - "provider:model" (e.g., "ollama:llama2")
        - "model" (uses default provider or tries all providers)
        """
        if ':' in model_name:
            # Explicit provider specified
            provider_name, actual_model = model_name.split(':', 1)
            try:
                provider_type = ProviderType(provider_name)
                provider = self.get_provider(provider_type)
                if not provider:
                    raise ValueError(f"Provider {provider_name} not found")
                if not provider.is_configured:
                    raise ValueError(f"Provider {provider_name} not configured")
                return provider, actual_model
            except ValueError:
                raise ValueError(f"Unknown provider: {provider_name}")
        
        # Try default provider first
        default_provider = self.get_default_provider()
        if default_provider and default_provider.is_configured:
            model_info = await default_provider.get_model_info(model_name)
            if model_info:
                return default_provider, model_name
        
        # Try all configured providers
        for provider in self.list_configured_providers():
            model_info = await provider.get_model_info(model_name)
            if model_info:
                return provider, model_name
        
        raise ValueError(f"Model '{model_name}' not found in any configured provider")
    
    async def list_all_models(self) -> List[ModelInfo]:
        """List all available models from all providers"""
        all_models = []
        for provider in self.list_configured_providers():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception as e:
                logger.error(f"Failed to list models from {provider.name}: {str(e)}")
        return all_models


# Global provider registry
provider_registry = LLMProviderRegistry()