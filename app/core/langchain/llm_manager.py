"""
LangChain-based LLM Manager for AI Assistant.

This module provides comprehensive LLM management using LangChain's native
model integrations, supporting multiple providers and model types.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.secure_settings import secure_settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers in LangChain"""
    
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"


@dataclass
class ModelInfo:
    """Information about an available model"""
    
    name: str
    provider: LLMProvider
    display_name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_function_calling: bool = True
    max_tokens: Optional[int] = None
    input_cost_per_1k: Optional[float] = None
    output_cost_per_1k: Optional[float] = None


class LangChainLLMManager:
    """
    Comprehensive LLM manager using LangChain integrations.
    
    This class replaces the custom provider system with LangChain's
    native model integrations, providing better performance,
    reliability, and ecosystem integration.
    """
    
    def __init__(self):
        self._providers: Dict[LLMProvider, Dict[str, Any]] = {}
        self._model_cache: Dict[str, BaseChatModel] = {}
        self._default_provider: Optional[LLMProvider] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LLM manager with configured providers"""
        if self._initialized:
            return
            
        logger.info("Initializing LangChain LLM Manager...")
        
        # Initialize OpenAI/OpenAI-compatible provider
        await self._initialize_openai_provider()
        
        # Initialize Ollama provider
        await self._initialize_ollama_provider()
        
        # Initialize Anthropic provider
        await self._initialize_anthropic_provider()
        
        # Initialize Google provider
        await self._initialize_google_provider()
        
        # Set default provider
        self._set_default_provider()
        
        self._initialized = True
        logger.info("LangChain LLM Manager initialized successfully")
        
    async def _initialize_openai_provider(self):
        """Initialize OpenAI/OpenAI-compatible provider"""
        try:
            # Check for OpenAI API key
            openai_api_key = secure_settings.get_setting("llm_providers", "openai_compatible", "api_key")
            if not openai_api_key:
                openai_api_key = secure_settings.get_setting("llm_providers", "openai", "api_key")
                
            if openai_api_key:
                base_url = secure_settings.get_setting("llm_providers", "openai_compatible", "base_url")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
                    
                self._providers[LLMProvider.OPENAI] = {
                    "api_key": openai_api_key,
                    "base_url": base_url,
                    "models": [
                        ModelInfo(
                            name="gpt-4",
                            provider=LLMProvider.OPENAI,
                            display_name="GPT-4",
                            description="OpenAI's GPT-4 model",
                            context_length=8192,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=4096,
                            input_cost_per_1k=0.03,
                            output_cost_per_1k=0.06
                        ),
                        ModelInfo(
                            name="gpt-4-turbo",
                            provider=LLMProvider.OPENAI,
                            display_name="GPT-4 Turbo",
                            description="OpenAI's GPT-4 Turbo model",
                            context_length=128000,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=4096,
                            input_cost_per_1k=0.01,
                            output_cost_per_1k=0.03
                        ),
                        ModelInfo(
                            name="gpt-3.5-turbo",
                            provider=LLMProvider.OPENAI,
                            display_name="GPT-3.5 Turbo",
                            description="OpenAI's GPT-3.5 Turbo model",
                            context_length=16385,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=4096,
                            input_cost_per_1k=0.0015,
                            output_cost_per_1k=0.002
                        ),
                    ]
                }
                logger.info("OpenAI provider initialized")
            else:
                logger.info("OpenAI API key not found, skipping OpenAI provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            
    async def _initialize_ollama_provider(self):
        """Initialize Ollama provider"""
        try:
            base_url = secure_settings.get_setting("llm_providers", "ollama", "base_url")
            if not base_url:
                base_url = "http://localhost:11434"
                
            # Test Ollama connectivity
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = []
                        for model in data.get("models", []):
                            model_name = model["name"]
                            models.append(
                                ModelInfo(
                                    name=model_name,
                                    provider=LLMProvider.OLLAMA,
                                    display_name=model_name.split(":")[0],
                                    description=f"Ollama model: {model_name}",
                                    context_length=model.get("details", {}).get("context_length"),
                                    supports_streaming=True,
                                    supports_tools=False,  # Ollama doesn't support tool calling yet
                                    supports_function_calling=False,
                                )
                            )
                            
                        self._providers[LLMProvider.OLLAMA] = {
                            "base_url": base_url,
                            "models": models
                        }
                        logger.info(f"Ollama provider initialized with {len(models)} models")
                    else:
                        logger.warning(f"Ollama not accessible at {base_url}")
                except Exception as e:
                    logger.warning(f"Failed to connect to Ollama at {base_url}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {str(e)}")
            
    async def _initialize_anthropic_provider(self):
        """Initialize Anthropic provider"""
        try:
            api_key = secure_settings.get_setting("llm_providers", "anthropic", "api_key")
            if api_key:
                self._providers[LLMProvider.ANTHROPIC] = {
                    "api_key": api_key,
                    "models": [
                        ModelInfo(
                            name="claude-3-sonnet-20240229",
                            provider=LLMProvider.ANTHROPIC,
                            display_name="Claude 3 Sonnet",
                            description="Anthropic's Claude 3 Sonnet model",
                            context_length=200000,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=4096,
                            input_cost_per_1k=0.003,
                            output_cost_per_1k=0.015
                        ),
                        ModelInfo(
                            name="claude-3-haiku-20240307",
                            provider=LLMProvider.ANTHROPIC,
                            display_name="Claude 3 Haiku",
                            description="Anthropic's Claude 3 Haiku model",
                            context_length=200000,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=4096,
                            input_cost_per_1k=0.00025,
                            output_cost_per_1k=0.00125
                        ),
                    ]
                }
                logger.info("Anthropic provider initialized")
            else:
                logger.info("Anthropic API key not found, skipping Anthropic provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
            
    async def _initialize_google_provider(self):
        """Initialize Google provider"""
        try:
            api_key = secure_settings.get_setting("llm_providers", "google", "api_key")
            if api_key:
                self._providers[LLMProvider.GOOGLE] = {
                    "api_key": api_key,
                    "models": [
                        ModelInfo(
                            name="gemini-pro",
                            provider=LLMProvider.GOOGLE,
                            display_name="Gemini Pro",
                            description="Google's Gemini Pro model",
                            context_length=32768,
                            supports_streaming=True,
                            supports_tools=True,
                            supports_function_calling=True,
                            max_tokens=2048,
                        ),
                    ]
                }
                logger.info("Google provider initialized")
            else:
                logger.info("Google API key not found, skipping Google provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {str(e)}")
            
    def _set_default_provider(self):
        """Set the default provider based on configuration and availability"""
        preferred_provider = secure_settings.get_setting("system_config", "preferred_provider")
        
        if preferred_provider:
            try:
                provider_enum = LLMProvider(preferred_provider)
                if provider_enum in self._providers:
                    self._default_provider = provider_enum
                    logger.info(f"Set default provider to {preferred_provider}")
                    return
            except ValueError:
                logger.warning(f"Invalid preferred provider: {preferred_provider}")
                
        # Fallback to first available provider
        if self._providers:
            self._default_provider = next(iter(self._providers.keys()))
            logger.info(f"Set default provider to {self._default_provider.value}")
        else:
            logger.warning("No providers available")
            
    async def get_llm(
        self, 
        model_name: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        **kwargs
    ) -> BaseChatModel:
        """
        Get a LangChain LLM instance for the specified model.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "ollama:llama2")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            streaming: Whether to enable streaming
            **kwargs: Additional model parameters
            
        Returns:
            LangChain BaseChatModel instance
            
        Raises:
            ValueError: If model is not found or provider is not configured
        """
        if not self._initialized:
            await self.initialize()
            
        # Check cache first
        cache_key = f"{model_name}:{temperature}:{max_tokens}:{streaming}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        # Parse model name to extract provider if specified
        provider, actual_model_name = self._parse_model_name(model_name)
        
        # Get provider configuration
        if provider not in self._providers:
            raise ValueError(f"Provider {provider.value} is not configured")
            
        provider_config = self._providers[provider]
        
        # Create LLM instance based on provider
        llm = await self._create_llm_instance(
            provider, actual_model_name, provider_config,
            temperature, max_tokens, streaming, **kwargs
        )
        
        # Cache the instance
        self._model_cache[cache_key] = llm
        
        return llm
        
    def _parse_model_name(self, model_name: str) -> tuple[LLMProvider, str]:
        """
        Parse model name to extract provider and actual model name.
        
        Args:
            model_name: Model name (e.g., "gpt-4", "ollama:llama2")
            
        Returns:
            Tuple of (provider, actual_model_name)
        """
        if ":" in model_name:
            provider_name, actual_model = model_name.split(":", 1)
            try:
                provider = LLMProvider(provider_name)
                return provider, actual_model
            except ValueError:
                logger.warning(f"Unknown provider: {provider_name}, using default")
                
        # Use default provider
        if self._default_provider:
            return self._default_provider, model_name
            
        # Fallback to trying all providers
        for provider, config in self._providers.items():
            for model in config.get("models", []):
                if model.name == model_name:
                    return provider, model_name
                    
        raise ValueError(f"Model '{model_name}' not found in any configured provider")
        
    async def _create_llm_instance(
        self,
        provider: LLMProvider,
        model_name: str,
        provider_config: Dict[str, Any],
        temperature: float,
        max_tokens: Optional[int],
        streaming: bool,
        **kwargs
    ) -> BaseChatModel:
        """Create LLM instance based on provider type"""
        
        if provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=model_name,
                api_key=provider_config["api_key"],
                base_url=provider_config["base_url"],
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                **kwargs
            )
            
        elif provider == LLMProvider.OLLAMA:
            return ChatOllama(
                model=model_name,
                base_url=provider_config["base_url"],
                temperature=temperature,
                num_predict=max_tokens,
                streaming=streaming,
                **kwargs
            )
            
        elif provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=model_name,
                api_key=provider_config["api_key"],
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                **kwargs
            )
            
        elif provider == LLMProvider.GOOGLE:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=provider_config["api_key"],
                temperature=temperature,
                max_output_tokens=max_tokens,
                streaming=streaming,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    async def list_models(self, provider: Optional[LLMProvider] = None) -> List[ModelInfo]:
        """
        List available models from all or specific provider.
        
        Args:
            provider: Specific provider to list models from (optional)
            
        Returns:
            List of ModelInfo objects
        """
        if not self._initialized:
            await self.initialize()
            
        models = []
        
        if provider:
            if provider in self._providers:
                models.extend(self._providers[provider].get("models", []))
        else:
            for provider_config in self._providers.values():
                models.extend(provider_config.get("models", []))
                
        return models
        
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        models = await self.list_models()
        for model in models:
            if model.name == model_name:
                return model
        return None
        
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(self._providers.keys())
        
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available and configured"""
        return provider in self._providers
        
    def clear_cache(self):
        """Clear the LLM instance cache"""
        self._model_cache.clear()
        logger.info("LLM cache cleared")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_models": len(self._model_cache),
            "cache_keys": list(self._model_cache.keys())
        }


# Global LLM manager instance
llm_manager = LangChainLLMManager()