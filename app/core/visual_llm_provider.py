"""
Visual LMM Provider System for AI Assistant

This module provides a specialized provider system for Visual Large Multimodal Models (LMMs)
that can process both text and images, enabling visual understanding and analysis capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, BinaryIO
from enum import Enum
import logging
import asyncio
import base64
import io
from dataclasses import dataclass
import mimetypes

logger = logging.getLogger(__name__)


class VisualProviderType(Enum):
    """Supported Visual LMM provider types"""
    
    OPENAI_VISION = "openai_vision"
    ANTHROPIC_VISION = "anthropic_vision"
    GOOGLE_VISION = "google_vision"
    OLLAMA_VISION = "ollama_vision"
    LOCAL_VISION = "local_vision"


@dataclass
class ImageContent:
    """Represents image content for visual analysis"""
    
    data: Union[str, bytes, BinaryIO]  # Base64 string, bytes, or file-like object
    media_type: str  # MIME type (e.g., "image/jpeg", "image/png")
    name: Optional[str] = None
    description: Optional[str] = None
    
    def to_base64(self) -> str:
        """Convert image data to base64 string"""
        if isinstance(self.data, str):
            # Assume already base64 encoded
            return self.data
        elif isinstance(self.data, bytes):
            return base64.b64encode(self.data).decode('utf-8')
        elif hasattr(self.data, 'read'):
            # File-like object
            return base64.b64encode(self.data.read()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image data type: {type(self.data)}")
    
    def validate(self) -> bool:
        """Validate image content"""
        # Check media type
        if not self.media_type.startswith('image/'):
            return False
        
        # Check data exists
        if not self.data:
            return False
            
        return True


@dataclass
class VisualModelInfo:
    """Information about an available visual model"""
    
    name: str
    provider: VisualProviderType
    display_name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    supports_streaming: bool = True
    max_image_size: Optional[int] = None  # Max image size in bytes
    supported_formats: List[str] = None  # Supported image formats
    vision_capabilities: List[str] = None  # Vision capabilities
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["image/jpeg", "image/png", "image/webp"]
        if self.vision_capabilities is None:
            self.vision_capabilities = ["analysis", "description", "ocr"]


class VisualLMMProvider(ABC):
    """Abstract base class for Visual LMM providers"""
    
    def __init__(self, provider_type: VisualProviderType):
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
    async def create_visual_llm(self, model_name: str, **kwargs) -> Any:
        """Create visual LLM instance for the given model"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[VisualModelInfo]:
        """List available visual models from this provider"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible"""
        pass
    
    @abstractmethod
    async def analyze_image(
        self,
        model_name: str,
        image: ImageContent,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze an image with the given prompt
        
        Args:
            model_name: Name of the visual model to use
            image: ImageContent object containing the image
            prompt: Text prompt for the analysis
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing the analysis result
        """
        pass
    
    @abstractmethod
    async def analyze_images(
        self,
        model_name: str,
        images: List[ImageContent],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze multiple images with the given prompt
        
        Args:
            model_name: Name of the visual model to use
            images: List of ImageContent objects
            prompt: Text prompt for the analysis
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing the analysis result
        """
        pass
    
    async def get_model_info(self, model_name: str) -> Optional[VisualModelInfo]:
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
    
    def prepare_image_for_api(self, image: ImageContent) -> Dict[str, Any]:
        """
        Prepare image content for API submission
        
        Args:
            image: ImageContent object
            
        Returns:
            Dictionary formatted for the specific provider's API
        """
        if not image.validate():
            raise ValueError("Invalid image content")
        
        return {
            "type": "image",
            "media_type": image.media_type,
            "data": image.to_base64(),
            "name": image.name,
            "description": image.description,
        }


class OpenAIVisionProvider(VisualLMMProvider):
    """OpenAI Vision API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(VisualProviderType.OPENAI_VISION)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        
        # Common OpenAI vision models
        self._fallback_models = [
            VisualModelInfo(
                name="gpt-4-vision-preview",
                provider=VisualProviderType.OPENAI_VISION,
                display_name="GPT-4 Vision Preview",
                description="OpenAI's GPT-4 model with vision capabilities",
                context_length=128000,
                supports_streaming=True,
                max_image_size=20 * 1024 * 1024,  # 20MB
                supported_formats=["image/jpeg", "image/png", "image/webp", "image/gif"],
                vision_capabilities=["analysis", "description", "ocr", "detection"],
            ),
            VisualModelInfo(
                name="gpt-4o",
                provider=VisualProviderType.OPENAI_VISION,
                display_name="GPT-4O",
                description="OpenAI's GPT-4O model with multimodal capabilities",
                context_length=128000,
                supports_streaming=True,
                max_image_size=20 * 1024 * 1024,  # 20MB
                supported_formats=["image/jpeg", "image/png", "image/webp", "image/gif"],
                vision_capabilities=["analysis", "description", "ocr", "detection"],
            ),
        ]
    
    @property
    def name(self) -> str:
        return "OpenAI Vision"
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    async def create_visual_llm(self, model_name: str, **kwargs) -> Any:
        """Create LangChain ChatOpenAI instance with vision capabilities"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("LangChain OpenAI integration not found. Install with: pip install langchain-openai")
        
        if not self.is_configured:
            raise ValueError(f"{self.name} provider is not configured with API key")
        
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            streaming=kwargs.get("streaming", False),
        )
    
    async def list_models(self) -> List[VisualModelInfo]:
        """List available OpenAI vision models"""
        try:
            import httpx
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    # Filter for vision-capable models
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        if any(vision_model in model_id for vision_model in ["gpt-4-vision", "gpt-4o"]):
                            model_info = VisualModelInfo(
                                name=model_id,
                                provider=VisualProviderType.OPENAI_VISION,
                                display_name=model_id,
                                description=model.get("object", "vision model"),
                                context_length=128000,  # Default for GPT-4
                                supports_streaming=True,
                                max_image_size=20 * 1024 * 1024,
                                supported_formats=["image/jpeg", "image/png", "image/webp", "image/gif"],
                                vision_capabilities=["analysis", "description", "ocr", "detection"],
                            )
                            models.append(model_info)
                    
                    if models:
                        return models
                
                # If API call fails, use fallback models
                logger.warning(f"Could not fetch models from {self.name}, using fallback models")
                return self._fallback_models
                
        except Exception as e:
            logger.error(f"Failed to list models from {self.name}: {str(e)}")
            return self._fallback_models
    
    async def health_check(self) -> bool:
        """Check OpenAI Vision API health"""
        try:
            import httpx
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
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
    
    async def analyze_image(
        self,
        model_name: str,
        image: ImageContent,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze an image using OpenAI Vision API"""
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Prepare the message content
            content = [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{image.to_base64()}",
                        "detail": kwargs.get("detail", "auto"),
                    },
                },
            ]
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": model_name,
                        "provider": self.name,
                    }
                else:
                    error_msg = f"OpenAI Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "model": model_name,
                        "provider": self.name,
                    }
                    
        except Exception as e:
            error_msg = f"OpenAI Vision analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": model_name,
                "provider": self.name,
            }
    
    async def analyze_images(
        self,
        model_name: str,
        images: List[ImageContent],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze multiple images using OpenAI Vision API"""
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Prepare the message content with multiple images
            content = [
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
            
            # Add all images
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{image.to_base64()}",
                        "detail": kwargs.get("detail", "auto"),
                    },
                })
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": model_name,
                        "provider": self.name,
                        "image_count": len(images),
                    }
                else:
                    error_msg = f"OpenAI Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "model": model_name,
                        "provider": self.name,
                        "image_count": len(images),
                    }
                    
        except Exception as e:
            error_msg = f"OpenAI Vision analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": model_name,
                "provider": self.name,
                "image_count": len(images),
            }


class OllamaVisionProvider(VisualLMMProvider):
    """Ollama Vision provider for local visual models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(VisualProviderType.OLLAMA_VISION)
        self.base_url = base_url.rstrip("/")
        
        # Common Ollama vision models
        self._fallback_models = [
            VisualModelInfo(
                name="llava",
                provider=VisualProviderType.OLLAMA_VISION,
                display_name="LLaVA",
                description="LLaVA visual language model",
                context_length=4096,
                supports_streaming=True,
                max_image_size=10 * 1024 * 1024,  # 10MB
                supported_formats=["image/jpeg", "image/png", "image/webp"],
                vision_capabilities=["analysis", "description", "ocr"],
            ),
            VisualModelInfo(
                name="bakllava",
                provider=VisualProviderType.OLLAMA_VISION,
                display_name="BakLLaVA",
                description="BakLLaVA visual language model",
                context_length=4096,
                supports_streaming=True,
                max_image_size=10 * 1024 * 1024,  # 10MB
                supported_formats=["image/jpeg", "image/png", "image/webp"],
                vision_capabilities=["analysis", "description", "ocr"],
            ),
        ]
    
    @property
    def name(self) -> str:
        return "Ollama Vision"
    
    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)
    
    async def create_visual_llm(self, model_name: str, **kwargs) -> Any:
        """Create Ollama instance with vision capabilities"""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError("LangChain Ollama integration not found. Install with: pip install langchain-ollama")
        
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
    
    async def list_models(self) -> List[VisualModelInfo]:
        """List available Ollama vision models"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned {response.status_code}")
                
                data = response.json()
                models = []
                
                # Filter for vision-capable models
                for model in data.get("models", []):
                    model_name = model["name"]
                    # Check if it's a known vision model
                    if any(vision_model in model_name.lower() for vision_model in ["llava", "bakllava", "moondream", "cogvlm"]):
                        model_info = VisualModelInfo(
                            name=model_name,
                            provider=VisualProviderType.OLLAMA_VISION,
                            display_name=model_name.split(":")[0],  # Remove tag
                            description=f"Ollama vision model: {model_name}",
                            context_length=model.get("details", {}).get("context_length", 4096),
                            supports_streaming=True,
                            max_image_size=10 * 1024 * 1024,  # 10MB
                            supported_formats=["image/jpeg", "image/png", "image/webp"],
                            vision_capabilities=["analysis", "description", "ocr"],
                        )
                        models.append(model_info)
                
                if models:
                    return models
                
                # If no vision models found, return fallback
                logger.warning("No vision models found in Ollama, using fallback models")
                return self._fallback_models
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            self.set_health_status(False)
            return self._fallback_models
    
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
    
    async def analyze_image(
        self,
        model_name: str,
        image: ImageContent,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze an image using Ollama Vision API"""
        try:
            import httpx
            
            # Prepare the payload for Ollama
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image.to_base64()],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data.get("response", ""),
                        "model": model_name,
                        "provider": self.name,
                        "done": data.get("done", False),
                        "context": data.get("context", []),
                    }
                else:
                    error_msg = f"Ollama Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "model": model_name,
                        "provider": self.name,
                    }
                    
        except Exception as e:
            error_msg = f"Ollama Vision analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": model_name,
                "provider": self.name,
            }
    
    async def analyze_images(
        self,
        model_name: str,
        images: List[ImageContent],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze multiple images using Ollama Vision API"""
        try:
            import httpx
            
            # Prepare the payload for Ollama with multiple images
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img.to_base64() for img in images],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data.get("response", ""),
                        "model": model_name,
                        "provider": self.name,
                        "image_count": len(images),
                        "done": data.get("done", False),
                        "context": data.get("context", []),
                    }
                else:
                    error_msg = f"Ollama Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "model": model_name,
                        "provider": self.name,
                        "image_count": len(images),
                    }
                    
        except Exception as e:
            error_msg = f"Ollama Vision analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": model_name,
                "provider": self.name,
                "image_count": len(images),
            }


class VisualLLMProviderRegistry:
    """Registry for managing Visual LMM providers"""
    
    def __init__(self):
        self._providers: Dict[VisualProviderType, VisualLMMProvider] = {}
        self._default_provider: Optional[VisualProviderType] = None
    
    def register_provider(self, provider: VisualLMMProvider):
        """Register a new visual provider"""
        self._providers[provider.provider_type] = provider
        logger.info(f"Registered {provider.name} visual provider")
        
        # Set as default if no default exists
        if self._default_provider is None and provider.is_configured:
            self._default_provider = provider.provider_type
    
    def get_provider(self, provider_type: VisualProviderType) -> Optional[VisualLMMProvider]:
        """Get a specific visual provider"""
        return self._providers.get(provider_type)
    
    def get_default_provider(self) -> Optional[VisualLMMProvider]:
        """Get the default visual provider"""
        if self._default_provider:
            return self._providers.get(self._default_provider)
        return None
    
    def set_default_provider(self, provider_type: VisualProviderType):
        """Set the default visual provider"""
        if provider_type in self._providers:
            self._default_provider = provider_type
            logger.info(f"Set {provider_type.value} as default visual provider")
        else:
            raise ValueError(f"Visual provider {provider_type.value} not registered")
    
    def list_providers(self) -> List[VisualLMMProvider]:
        """List all registered visual providers"""
        return list(self._providers.values())
    
    def list_configured_providers(self) -> List[VisualLMMProvider]:
        """List only configured visual providers"""
        return [
            provider
            for provider in self._providers.values()
            if provider.is_configured
        ]
    
    async def health_check_all(self):
        """Health check all visual providers"""
        tasks = []
        for provider in self._providers.values():
            if provider.is_configured:
                tasks.append(provider.health_check())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def resolve_model(self, model_name: str) -> tuple[VisualLMMProvider, str]:
        """
        Resolve model name to visual provider and actual model name.
        
        Supports formats:
        - "provider:model" (e.g., "openai_vision:gpt-4-vision-preview")
        - "model" (uses default provider or tries all providers)
        """
        if ":" in model_name:
            # Explicit provider specified
            provider_name, actual_model = model_name.split(":", 1)
            try:
                provider_type = VisualProviderType(provider_name)
                provider = self.get_provider(provider_type)
                if not provider:
                    raise ValueError(f"Visual provider {provider_name} not found")
                if not provider.is_configured:
                    raise ValueError(f"Visual provider {provider_name} not configured")
                return provider, actual_model
            except ValueError:
                raise ValueError(f"Unknown visual provider: {provider_name}")
        
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
        
        raise ValueError(f"Visual model '{model_name}' not found in any configured provider")
    
    async def list_all_models(self) -> List[VisualModelInfo]:
        """List all available visual models from all providers"""
        all_models = []
        for provider in self.list_configured_providers():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception as e:
                logger.error(f"Failed to list models from {provider.name}: {str(e)}")
        return all_models


# Global visual provider registry
visual_provider_registry = VisualLLMProviderRegistry()