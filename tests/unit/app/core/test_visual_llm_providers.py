"""
Unit tests for Visual LLM Provider System.

This module tests the visual LLM provider system including OpenAI Vision
and Ollama Vision providers, as well as the visual provider registry.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import httpx
import base64

from app.core.visual_llm_provider import (
    VisualProviderType,
    ImageContent,
    VisualModelInfo,
    VisualLMMProvider,
    OpenAIVisionProvider,
    OllamaVisionProvider,
    VisualLLMProviderRegistry
)


class TestVisualProviderType:
    """Test VisualProviderType enum"""

    def test_visual_provider_type_values(self):
        """Test that VisualProviderType has expected values"""
        expected_types = [
            "openai",
            "ollama",
            "anthropic",
            "google",
            "custom"
        ]
        
        actual_types = [provider_type.value for provider_type in VisualProviderType]
        assert actual_types == expected_types


class TestImageContent:
    """Test ImageContent dataclass"""

    def test_image_content_defaults(self):
        """Test ImageContent default values"""
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="image/jpeg"
        )
        
        assert image_content.data == b"fake-image-data"
        assert image_content.media_type == "image/jpeg"
        assert image_content.width is None
        assert image_content.height is None
        assert image_content.filename is None
        assert image_content.metadata == {}

    def test_image_content_with_values(self):
        """Test ImageContent with provided values"""
        metadata = {"source": "upload", "user_id": "user-123"}
        
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="image/png",
            width=1024,
            height=768,
            filename="test.png",
            metadata=metadata
        )
        
        assert image_content.data == b"fake-image-data"
        assert image_content.media_type == "image/png"
        assert image_content.width == 1024
        assert image_content.height == 768
        assert image_content.filename == "test.png"
        assert image_content.metadata == metadata

    def test_image_content_to_base64(self):
        """Test ImageContent to_base64 method"""
        image_data = b"fake-image-data"
        expected_base64 = base64.b64encode(image_data).decode('utf-8')
        
        image_content = ImageContent(
            data=image_data,
            media_type="image/jpeg"
        )
        
        result = image_content.to_base64()
        
        assert result == expected_base64

    def test_image_content_validate_valid(self):
        """Test ImageContent validate method with valid content"""
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="image/jpeg"
        )
        
        result = image_content.validate()
        
        assert result is True

    def test_image_content_validate_empty_data(self):
        """Test ImageContent validate method with empty data"""
        image_content = ImageContent(
            data=b"",
            media_type="image/jpeg"
        )
        
        result = image_content.validate()
        
        assert result is False

    def test_image_content_validate_invalid_media_type(self):
        """Test ImageContent validate method with invalid media type"""
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="text/plain"
        )
        
        result = image_content.validate()
        
        assert result is False

    def test_image_content_validate_none_data(self):
        """Test ImageContent validate method with None data"""
        image_content = ImageContent(
            data=None,
            media_type="image/jpeg"
        )
        
        result = image_content.validate()
        
        assert result is False


class TestVisualModelInfo:
    """Test VisualModelInfo dataclass"""

    def test_visual_model_info_defaults(self):
        """Test VisualModelInfo default values"""
        model_info = VisualModelInfo(
            name="gpt-4-vision-preview",
            provider=VisualProviderType.OPENAI
        )
        
        assert model_info.name == "gpt-4-vision-preview"
        assert model_info.provider == VisualProviderType.OPENAI
        assert model_info.display_name == "gpt-4-vision-preview"
        assert model_info.description == ""
        assert model_info.max_tokens is None
        assert model_info.context_window is None
        assert model_info.input_cost_per_1k is None
        assert model_info.output_cost_per_1k is None
        assert model_info.supports_functions is False
        assert model_info.supports_streaming is True
        assert model_info.max_image_size is None
        assert model_info.supported_formats == []
        assert model_info.metadata == {}

    def test_visual_model_info_with_values(self):
        """Test VisualModelInfo with provided values"""
        metadata = {"version": "1.0", "family": "gpt-4-vision"}
        supported_formats = ["image/jpeg", "image/png", "image/webp"]
        
        model_info = VisualModelInfo(
            name="gpt-4-vision-preview",
            provider=VisualProviderType.OPENAI,
            display_name="GPT-4 Vision Preview",
            description="Advanced GPT-4 model with vision capabilities",
            max_tokens=4096,
            context_window=128000,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
            supports_functions=True,
            supports_streaming=True,
            max_image_size=20 * 1024 * 1024,  # 20MB
            supported_formats=supported_formats,
            metadata=metadata
        )
        
        assert model_info.name == "gpt-4-vision-preview"
        assert model_info.provider == VisualProviderType.OPENAI
        assert model_info.display_name == "GPT-4 Vision Preview"
        assert model_info.description == "Advanced GPT-4 model with vision capabilities"
        assert model_info.max_tokens == 4096
        assert model_info.context_window == 128000
        assert model_info.input_cost_per_1k == 0.01
        assert model_info.output_cost_per_1k == 0.03
        assert model_info.supports_functions is True
        assert model_info.supports_streaming is True
        assert model_info.max_image_size == 20 * 1024 * 1024
        assert model_info.supported_formats == supported_formats
        assert model_info.metadata == metadata

    def test_visual_model_info_post_init(self):
        """Test VisualModelInfo __post_init__ method"""
        model_info = VisualModelInfo(
            name="test-model",
            provider=VisualProviderType.OPENAI,
            max_image_size=1024
        )
        
        # Should convert bytes to MB
        assert model_info.max_image_size == 1024 / (1024 * 1024)

    def test_visual_model_info_to_dict(self):
        """Test VisualModelInfo to_dict method"""
        metadata = {"test": True}
        supported_formats = ["image/jpeg"]
        
        model_info = VisualModelInfo(
            name="gpt-4-vision-preview",
            provider=VisualProviderType.OPENAI,
            display_name="GPT-4 Vision",
            description="Test model",
            max_tokens=4096,
            supported_formats=supported_formats,
            metadata=metadata
        )
        
        result = model_info.to_dict()
        
        assert result["name"] == "gpt-4-vision-preview"
        assert result["provider"] == "openai"
        assert result["display_name"] == "GPT-4 Vision"
        assert result["description"] == "Test model"
        assert result["max_tokens"] == 4096
        assert result["supported_formats"] == supported_formats
        assert result["metadata"] == metadata


class TestOpenAIVisionProvider:
    """Test OpenAIVisionProvider class"""

    @pytest.fixture
    def provider(self):
        """Create an OpenAI Vision provider instance"""
        return OpenAIVisionProvider(
            api_key="test-api-key",
            base_url="https://api.openai.com/v1"
        )

    def test_provider_init(self, provider):
        """Test OpenAIVisionProvider initialization"""
        assert provider.provider_type == VisualProviderType.OPENAI
        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_provider_init_custom_url(self):
        """Test OpenAIVisionProvider initialization with custom URL"""
        provider = OpenAIVisionProvider(
            api_key="test-api-key",
            base_url="https://custom.openai.com/v1"
        )
        
        assert provider.provider_type == VisualProviderType.OPENAI
        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://custom.openai.com/v1"

    @pytest.mark.asyncio
    async def test_create_visual_llm(self, provider):
        """Test creating a visual LLM instance"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm
            
            result = await provider.create_visual_llm("gpt-4-vision-preview", temperature=0.7, max_tokens=1000)
            
            assert result == mock_llm
            mock_chat_openai.assert_called_once_with(
                model="gpt-4-vision-preview",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                max_tokens=1000
            )

    @pytest.mark.asyncio
    async def test_create_visual_llm_error(self, provider):
        """Test creating a visual LLM instance with error"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.side_effect = Exception("Import error")
            
            with pytest.raises(Exception, match="Import error"):
                await provider.create_visual_llm("gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        """Test listing OpenAI vision models"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4-vision-preview",
                    "object": "model",
                    "capabilities": ["vision", "text"]
                },
                {
                    "id": "gpt-4",
                    "object": "model",
                    "capabilities": ["text"]
                }
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            models = await provider.list_models()
            
            assert len(models) == 1  # Only vision models
            assert models[0].name == "gpt-4-vision-preview"
            assert models[0].provider == VisualProviderType.OPENAI

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing models with error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("API error")
            
            models = await provider.list_models()
            
            assert models == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await provider.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            result = await provider.health_check()
            
            assert result is False

    def test_prepare_image_for_api(self, provider):
        """Test preparing image for API"""
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="image/jpeg"
        )
        
        result = provider.prepare_image_for_api(image_content)
        
        expected_base64 = base64.b64encode(b"fake-image-data").decode('utf-8')
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{expected_base64}"

    @pytest.mark.asyncio
    async def test_analyze_image(self, provider):
        """Test analyzing a single image"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "This is an image of a cat"
            mock_llm.invoke.return_value = mock_response
            mock_chat_openai.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            result = await provider.analyze_image(
                image=image_content,
                prompt="What do you see in this image?",
                model="gpt-4-vision-preview"
            )
            
            assert result == "This is an image of a cat"

    @pytest.mark.asyncio
    async def test_analyze_image_with_options(self, provider):
        """Test analyzing an image with options"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Detailed analysis of the image"
            mock_llm.invoke.return_value = mock_response
            mock_chat_openai.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            result = await provider.analyze_image(
                image=image_content,
                prompt="Analyze this image in detail",
                model="gpt-4-vision-preview",
                max_tokens=2000,
                temperature=0.3,
                detail="high"
            )
            
            assert result == "Detailed analysis of the image"

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, provider):
        """Test analyzing an image with error"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("API error")
            mock_chat_openai.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            with pytest.raises(Exception, match="API error"):
                await provider.analyze_image(
                    image=image_content,
                    prompt="What do you see?",
                    model="gpt-4-vision-preview"
                )

    @pytest.mark.asyncio
    async def test_analyze_images(self, provider):
        """Test analyzing multiple images"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Analysis of multiple images"
            mock_llm.invoke.return_value = mock_response
            mock_chat_openai.return_value = mock_llm
            
            image_contents = [
                ImageContent(data=b"fake-image-data-1", media_type="image/jpeg"),
                ImageContent(data=b"fake-image-data-2", media_type="image/png")
            ]
            
            result = await provider.analyze_images(
                images=image_contents,
                prompt="Compare these images",
                model="gpt-4-vision-preview"
            )
            
            assert result == "Analysis of multiple images"

    @pytest.mark.asyncio
    async def test_analyze_images_error(self, provider):
        """Test analyzing multiple images with error"""
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("API error")
            mock_chat_openai.return_value = mock_llm
            
            image_contents = [
                ImageContent(data=b"fake-image-data", media_type="image/jpeg")
            ]
            
            with pytest.raises(Exception, match="API error"):
                await provider.analyze_images(
                    images=image_contents,
                    prompt="Analyze these images",
                    model="gpt-4-vision-preview"
                )


class TestOllamaVisionProvider:
    """Test OllamaVisionProvider class"""

    @pytest.fixture
    def provider(self):
        """Create an Ollama Vision provider instance"""
        return OllamaVisionProvider(
            base_url="http://localhost:11434"
        )

    def test_provider_init(self, provider):
        """Test OllamaVisionProvider initialization"""
        assert provider.provider_type == VisualProviderType.OLLAMA
        assert provider.base_url == "http://localhost:11434"

    def test_provider_init_custom_url(self):
        """Test OllamaVisionProvider initialization with custom URL"""
        provider = OllamaVisionProvider(
            base_url="https://remote-ollama.example.com"
        )
        
        assert provider.provider_type == VisualProviderType.OLLAMA
        assert provider.base_url == "https://remote-ollama.example.com"

    @pytest.mark.asyncio
    async def test_create_visual_llm(self, provider):
        """Test creating an Ollama vision LLM instance"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_chat_ollama.return_value = mock_llm
            
            result = await provider.create_visual_llm("llava", temperature=0.7)
            
            assert result == mock_llm
            mock_chat_ollama.assert_called_once_with(
                model="llava",
                base_url="http://localhost:11434",
                temperature=0.7
            )

    @pytest.mark.asyncio
    async def test_create_visual_llm_error(self, provider):
        """Test creating an Ollama vision LLM instance with error"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_chat_ollama.side_effect = Exception("Import error")
            
            with pytest.raises(Exception, match="Import error"):
                await provider.create_visual_llm("llava")

    @pytest.mark.asyncio
    async def test_list_models_success(self, provider):
        """Test successful Ollama vision models listing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llava:latest",
                    "size": 1234567890,
                    "details": {"formats": ["png", "jpg"]}
                },
                {
                    "name": "mistral:latest",
                    "size": 987654321,
                    "details": {"formats": []}
                }
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            models = await provider.list_models()
            
            assert len(models) == 1  # Only vision models
            assert models[0].name == "llava:latest"
            assert models[0].provider == VisualProviderType.OLLAMA

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing Ollama models with error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            models = await provider.list_models()
            
            assert models == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful Ollama health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await provider.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed Ollama health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            result = await provider.health_check()
            
            assert result is False

    def test_prepare_image_for_api(self, provider):
        """Test preparing image for API"""
        image_content = ImageContent(
            data=b"fake-image-data",
            media_type="image/jpeg"
        )
        
        result = provider.prepare_image_for_api(image_content)
        
        assert isinstance(result, bytes)
        assert result == b"fake-image-data"

    @pytest.mark.asyncio
    async def test_analyze_image(self, provider):
        """Test analyzing a single image"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "This is an image of a dog"
            mock_llm.invoke.return_value = mock_response
            mock_chat_ollama.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            result = await provider.analyze_image(
                image=image_content,
                prompt="What do you see in this image?",
                model="llava"
            )
            
            assert result == "This is an image of a dog"

    @pytest.mark.asyncio
    async def test_analyze_image_with_options(self, provider):
        """Test analyzing an image with options"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Detailed analysis of the image"
            mock_llm.invoke.return_value = mock_response
            mock_chat_ollama.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            result = await provider.analyze_image(
                image=image_content,
                prompt="Analyze this image in detail",
                model="llava",
                temperature=0.3,
                top_p=0.9
            )
            
            assert result == "Detailed analysis of the image"

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, provider):
        """Test analyzing an image with error"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("API error")
            mock_chat_ollama.return_value = mock_llm
            
            image_content = ImageContent(
                data=b"fake-image-data",
                media_type="image/jpeg"
            )
            
            with pytest.raises(Exception, match="API error"):
                await provider.analyze_image(
                    image=image_content,
                    prompt="What do you see?",
                    model="llava"
                )

    @pytest.mark.asyncio
    async def test_analyze_images(self, provider):
        """Test analyzing multiple images"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Analysis of multiple images"
            mock_llm.invoke.return_value = mock_response
            mock_chat_ollama.return_value = mock_llm
            
            image_contents = [
                ImageContent(data=b"fake-image-data-1", media_type="image/jpeg"),
                ImageContent(data=b"fake-image-data-2", media_type="image/png")
            ]
            
            result = await provider.analyze_images(
                images=image_contents,
                prompt="Compare these images",
                model="llava"
            )
            
            assert result == "Analysis of multiple images"

    @pytest.mark.asyncio
    async def test_analyze_images_error(self, provider):
        """Test analyzing multiple images with error"""
        with patch('langchain_community.chat_models.ChatOllama') as mock_chat_ollama:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("API error")
            mock_chat_ollama.return_value = mock_llm
            
            image_contents = [
                ImageContent(data=b"fake-image-data", media_type="image/jpeg")
            ]
            
            with pytest.raises(Exception, match="API error"):
                await provider.analyze_images(
                    images=image_contents,
                    prompt="Analyze these images",
                    model="llava"
                )


class TestVisualLLMProviderRegistry:
    """Test VisualLLMProviderRegistry class"""

    @pytest.fixture
    def registry(self):
        """Create a visual provider registry instance"""
        return VisualLLMProviderRegistry()

    def test_registry_init(self, registry):
        """Test VisualLLMProviderRegistry initialization"""
        assert len(registry.providers) == 0
        assert registry.default_provider is None

    def test_register_provider(self, registry):
        """Test registering a provider"""
        provider = Mock(spec=VisualLMMProvider)
        provider.provider_type = VisualProviderType.OPENAI
        
        registry.register_provider(provider)
        
        assert len(registry.providers) == 1
        assert registry.providers[VisualProviderType.OPENAI] == provider

    def test_register_provider_sets_default(self, registry):
        """Test that registering first provider sets it as default"""
        provider = Mock(spec=VisualLMMProvider)
        provider.provider_type = VisualProviderType.OPENAI
        
        registry.register_provider(provider)
        
        assert registry.default_provider == provider

    def test_get_provider(self, registry):
        """Test getting a registered provider"""
        provider = Mock(spec=VisualLMMProvider)
        provider.provider_type = VisualProviderType.OPENAI
        
        registry.register_provider(provider)
        
        result = registry.get_provider(VisualProviderType.OPENAI)
        
        assert result == provider

    def test_get_provider_not_found(self, registry):
        """Test getting a provider that doesn't exist"""
        result = registry.get_provider(VisualProviderType.OPENAI)
        
        assert result is None

    def test_get_default_provider(self, registry):
        """Test getting the default provider"""
        provider = Mock(spec=VisualLMMProvider)
        provider.provider_type = VisualProviderType.OPENAI
        
        registry.register_provider(provider)
        
        result = registry.get_default_provider()
        
        assert result == provider

    def test_get_default_provider_none(self, registry):
        """Test getting default provider when none is set"""
        result = registry.get_default_provider()
        
        assert result is None

    def test_set_default_provider(self, registry):
        """Test setting the default provider"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        registry.set_default_provider(VisualProviderType.OLLAMA)
        
        assert registry.default_provider == provider2

    def test_set_default_provider_not_found(self, registry):
        """Test setting default provider that doesn't exist"""
        with pytest.raises(ValueError, match="Provider not found"):
            registry.set_default_provider(VisualProviderType.OPENAI)

    def test_list_configured_providers(self, registry):
        """Test listing configured providers"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        result = registry.list_configured_providers()
        
        assert len(result) == 2
        assert provider1 in result
        assert provider2 in result

    @pytest.mark.asyncio
    async def test_health_check_all(self, registry):
        """Test health check for all providers"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        provider1.health_check = AsyncMock(return_value=True)
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        provider2.health_check = AsyncMock(return_value=False)
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        results = await registry.health_check_all()
        
        assert results[VisualProviderType.OPENAI] is True
        assert results[VisualProviderType.OLLAMA] is False

    @pytest.mark.asyncio
    async def test_resolve_model_with_provider_prefix(self, registry):
        """Test resolving model with provider prefix"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        result = await registry.resolve_model("openai:gpt-4-vision-preview")
        
        assert result == (provider1, "gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_resolve_model_without_provider_prefix(self, registry):
        """Test resolving model without provider prefix"""
        provider = Mock(spec=VisualLMMProvider)
        provider.provider_type = VisualProviderType.OPENAI
        
        registry.register_provider(provider)
        registry.set_default_provider(VisualProviderType.OPENAI)
        
        result = await registry.resolve_model("gpt-4-vision-preview")
        
        assert result == (provider, "gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_resolve_model_no_default_provider(self, registry):
        """Test resolving model without default provider"""
        with pytest.raises(ValueError, match="No default provider configured"):
            await registry.resolve_model("gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_resolve_model_provider_not_found(self, registry):
        """Test resolving model with unknown provider"""
        with pytest.raises(ValueError, match="Provider 'unknown' not found"):
            await registry.resolve_model("unknown:gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_list_all_models(self, registry):
        """Test listing all models from all providers"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        provider1.list_models = AsyncMock(return_value=[
            VisualModelInfo("gpt-4-vision-preview", VisualProviderType.OPENAI)
        ])
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        provider2.list_models = AsyncMock(return_value=[
            VisualModelInfo("llava:latest", VisualProviderType.OLLAMA)
        ])
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        models = await registry.list_all_models()
        
        assert len(models) == 2
        assert any(m.name == "gpt-4-vision-preview" for m in models)
        assert any(m.name == "llava:latest" for m in models)

    @pytest.mark.asyncio
    async def test_list_all_models_with_error(self, registry):
        """Test listing all models with provider error"""
        provider1 = Mock(spec=VisualLMMProvider)
        provider1.provider_type = VisualProviderType.OPENAI
        provider1.list_models = AsyncMock(side_effect=Exception("API error"))
        
        provider2 = Mock(spec=VisualLMMProvider)
        provider2.provider_type = VisualProviderType.OLLAMA
        provider2.list_models = AsyncMock(return_value=[
            VisualModelInfo("llava:latest", VisualProviderType.OLLAMA)
        ])
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        models = await registry.list_all_models()
        
        # Should still return models from working provider
        assert len(models) == 1
        assert models[0].name == "llava:latest"