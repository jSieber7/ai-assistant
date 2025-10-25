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
import httpx

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types"""

    OPENAI_COMPATIBLE = "openai_compatible"
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


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible API provider"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        provider_name: str = None,
        custom_headers: Dict[str, str] = None,
    ):
        super().__init__(ProviderType.OPENAI_COMPATIBLE)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._provider_name = provider_name or self._detect_provider_name(base_url)
        self.custom_headers = custom_headers or {}

        # Common fallback models for endpoints without model listing
        self._fallback_models = [
            ModelInfo(
                name="gpt-4",
                provider=ProviderType.OPENAI_COMPATIBLE,
                display_name="GPT-4",
                description="OpenAI's GPT-4 model",
                context_length=8192,
                supports_streaming=True,
                supports_tools=True,
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI_COMPATIBLE,
                display_name="GPT-3.5 Turbo",
                description="OpenAI's GPT-3.5 Turbo model",
                context_length=4096,
                supports_streaming=True,
                supports_tools=True,
            ),
            ModelInfo(
                name="claude-3-sonnet",
                provider=ProviderType.OPENAI_COMPATIBLE,
                display_name="Claude 3 Sonnet",
                description="Anthropic's Claude 3 Sonnet model",
                context_length=200000,
                supports_streaming=True,
                supports_tools=True,
            ),
        ]

    def _detect_provider_name(self, base_url: str) -> str:
        """Detect provider name from base URL"""
        if "openrouter.ai" in base_url:
            return "OpenRouter"
        elif "api.openai.com" in base_url:
            return "OpenAI"
        elif "together.ai" in base_url:
            return "Together AI"
        elif "azure.com" in base_url:
            return "Azure OpenAI"
        else:
            return "OpenAI Compatible"

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create LangChain ChatOpenAI instance for OpenAI-compatible API"""
        from langchain_openai import ChatOpenAI

        if not self.is_configured:
            raise ValueError(f"{self.name} provider is not configured with API key")

        # Prepare headers
        headers = {"Authorization": f"Bearer {self.api_key}"}
        headers.update(self.custom_headers)

        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            streaming=kwargs.get("streaming", False),
            default_headers=headers if self.custom_headers else None,
        )

    async def list_models(self) -> List[ModelInfo]:
        """List available models from the OpenAI-compatible API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            headers.update(self.custom_headers)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    models = []

                    # Handle different response formats
                    if "data" in data:
                        # OpenAI format
                        for model in data["data"]:
                            model_name = model.get("id", model.get("model", ""))
                            
                            # For OpenRouter, check if user has access to the model
                            if "openrouter.ai" in self.base_url.lower():
                                if await self._check_openrouter_model_access(model_name, headers, client):
                                    display_name = model_name.split("/")[-1]
                                    description = model.get("object", "model")
                                else:
                                    # Skip models the user doesn't have access to
                                    logger.debug(f"Skipping inaccessible OpenRouter model: {model_name}")
                                    continue
                            else:
                                display_name = model_name.split("/")[-1]
                                description = model.get("object", "model")
                            
                            model_info = ModelInfo(
                                name=model_name,
                                provider=ProviderType.OPENAI_COMPATIBLE,
                                display_name=display_name,
                                description=description,
                                context_length=None,  # Not typically provided in list endpoint
                                supports_streaming=True,
                                supports_tools=True,
                            )
                            models.append(model_info)
                    elif isinstance(data, list):
                        # Direct list format
                        for model in data:
                            if isinstance(model, dict):
                                model_name = model.get("id", model.get("name", ""))
                                
                                # For OpenRouter, check if user has access to the model
                                if "openrouter.ai" in self.base_url.lower():
                                    if await self._check_openrouter_model_access(model_name, headers, client):
                                        display_name = model_name.split("/")[-1]
                                        description = model.get("description", f"Model from {self.name}")
                                    else:
                                        # Skip models the user doesn't have access to
                                        logger.debug(f"Skipping inaccessible OpenRouter model: {model_name}")
                                        continue
                                else:
                                    display_name = model_name.split("/")[-1]
                                    description = model.get(
                                        "description", f"Model from {self.name}"
                                    )
                                
                                model_info = ModelInfo(
                                    name=model_name,
                                    provider=ProviderType.OPENAI_COMPATIBLE,
                                    display_name=display_name,
                                    description=description,
                                    context_length=model.get("context_length"),
                                    supports_streaming=model.get(
                                        "supports_streaming", True
                                    ),
                                    supports_tools=model.get("supports_tools", True),
                                )
                                models.append(model_info)

                    if models:
                        return models

                # If API call fails or returns unexpected format, use fallback models
                logger.warning(
                    f"Could not fetch models from {self.name}, using fallback models"
                )
                return self._fallback_models

        except Exception as e:
            logger.error(f"Failed to list models from {self.name}: {str(e)}")
            return self._fallback_models

    async def _check_openrouter_model_access(self, model_name: str, headers: dict, client: httpx.AsyncClient) -> bool:
        """
        Check if the user has access to a specific OpenRouter model.
        
        This method makes a lightweight request to verify model accessibility.
        """
        try:
            # Make a minimal request to check model access
            # We use the chat completions endpoint with a minimal message
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False
            }
            
            test_response = await client.post(
                f"{self.base_url}/chat/completions",
                json=test_payload,
                headers=headers,
                timeout=5.0
            )
            
            # If we get a 200 response, the model is accessible
            if test_response.status_code == 200:
                return True
            # If we get a 403 or 401, the model is not accessible
            elif test_response.status_code in [401, 403]:
                logger.debug(f"Model {model_name} not accessible: HTTP {test_response.status_code}")
                return False
            # For other status codes, we'll assume the model might be accessible
            # (could be temporary issues, rate limits, etc.)
            else:
                logger.debug(f"Model {model_name} access check returned HTTP {test_response.status_code}, assuming accessible")
                return True
                
        except httpx.TimeoutException:
            # Timeout doesn't necessarily mean lack of access
            logger.debug(f"Model {model_name} access check timed out, assuming accessible")
            return True
        except Exception as e:
            # Other errors don't necessarily mean lack of access
            logger.debug(f"Model {model_name} access check failed: {str(e)}, assuming accessible")
            return True

    async def health_check(self) -> bool:
        """Check OpenAI-compatible API health"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            headers.update(self.custom_headers)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                is_healthy = response.status_code == 200
                self.set_health_status(is_healthy)
                return is_healthy
        except Exception as e:
            logger.error(f"{self.name} health check failed: {str(e)}")
            self.set_health_status(False)
            return False




class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(ProviderType.OLLAMA)
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "Ollama"

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)

    async def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create LangChain ChatOllama instance"""
        # LangChain 1.0 compatibility - use langchain_ollama package
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # This is the correct fallback for LangChain 1.0
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError(
                    "LangChain Ollama integration not found. "
                    "Install with: pip install langchain-ollama"
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
            temperature=kwargs.get("temperature", 0.7),
            num_predict=kwargs.get("max_tokens"),
            streaming=kwargs.get("streaming", False),
        )

    async def list_models(self) -> List[ModelInfo]:
        """List available Ollama models"""
        try:
            # Check if Ollama is running in Docker
            if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                logger.warning(f"Ollama base URL {self.base_url} may not be accessible from within Docker")
                logger.info("Consider using 'http://host.docker.internal:11434' for Docker environments")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned {response.status_code}: {response.text}")

                data = response.json()
                models = []

                for model in data.get("models", []):
                    model_info = ModelInfo(
                        name=model["name"],
                        provider=ProviderType.OLLAMA,
                        display_name=model["name"].split(":")[0],  # Remove tag
                        description=f"Ollama model: {model['name']}",
                        context_length=model.get("details", {}).get("context_length"),
                        supports_streaming=True,
                        supports_tools=False,  # Ollama doesn't support tool calling yet
                    )
                    models.append(model_info)

                self.set_health_status(True)
                return models

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            logger.info("Make sure Ollama is running and accessible from the application")
            self.set_health_status(False)
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            self.set_health_status(False)
            return []

    async def health_check(self) -> bool:
        """Check Ollama server health"""
        try:
            # Check if Ollama is running in Docker
            if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                logger.warning(f"Ollama base URL {self.base_url} may not be accessible from within Docker")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                is_healthy = response.status_code == 200
                self.set_health_status(is_healthy)
                if not is_healthy:
                    logger.warning(f"Ollama health check failed with status {response.status_code}")
                return is_healthy
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            self.set_health_status(False)
            return False
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
            provider
            for provider in self._providers.values()
            if provider.is_configured  # Only check if configured, not if healthy
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
        if ":" in model_name:
            # Explicit provider specified
            provider_name, actual_model = model_name.split(":", 1)
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



async def get_llm(model_name: Optional[str] = None, **kwargs):
    """
    Factory function to create LLM instances with multi-provider support

    Args:
        model_name: Model name (supports format "provider:model" or just "model")
        **kwargs: Additional LLM parameters (temperature, max_tokens, etc.)
    
    Returns:
        LLM instance configured with specified provider and model
    """
    # Get default provider
    default_provider = provider_registry.get_default_provider()
    if not default_provider:
        raise ValueError("No LLM providers configured")
    
    # Resolve model to provider
    provider, actual_model = await provider_registry.resolve_model(model_name or "gpt-4")
    
    # Create and return LLM instance
    return await provider.create_llm(actual_model, **kwargs)