"""
Visual Analysis Tool for Visual LMM System

This module provides comprehensive visual analysis capabilities using Visual LMM providers,
including image description, OCR, object detection, and comparative analysis.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import json

from app.core.tools.base.base import BaseTool, ToolExecutionError
from app.core.tools.visual.image_processor import ImageProcessorTool
from app.core.visual_llm_provider import (
    VisualLLMProviderRegistry, 
    ImageContent, 
    visual_provider_registry
)

logger = logging.getLogger(__name__)


class VisualAnalyzerTool(BaseTool):
    """Comprehensive visual analysis tool using Visual LMM providers"""
    
    def __init__(self, default_model: str = "openai_vision:gpt-4-vision-preview"):
        super().__init__()
        self.default_model = default_model
        self.image_processor = ImageProcessorTool()
        self.provider_registry = visual_provider_registry
        
        # Predefined analysis prompts
        self.prompts = {
            "describe": "Provide a detailed description of this image, including the main subjects, setting, colors, composition, and any notable details.",
            "ocr": "Extract all text visible in this image. Preserve the layout and formatting as much as possible. If no text is visible, respond with 'No text found'.",
            "objects": "Identify and list all objects, people, and elements visible in this image. Be specific and include their positions and relationships.",
            "analyze": "Analyze this image comprehensively. Describe what you see, identify key elements, assess the composition, and provide insights about the content, context, and potential meaning.",
            "compare": "Compare these images and identify similarities, differences, and relationships between them. Focus on content, composition, style, and any notable changes.",
            "technical": "Analyze the technical aspects of this image including lighting, composition, color scheme, focus, depth of field, and any photographic techniques used.",
            "accessibility": "Describe this image for accessibility purposes. Include all important visual information that would be needed for someone who cannot see the image.",
        }
    
    @property
    def name(self) -> str:
        return "visual_analyzer"
    
    @property
    def description(self) -> str:
        return "Comprehensive visual analysis tool using Visual LMM providers for image description, OCR, object detection, and more"
    
    @property
    def keywords(self) -> List[str]:
        return [
            "visual",
            "analyze",
            "describe",
            "ocr",
            "image",
            "vision",
            "detect",
            "compare",
            "extract",
        ]
    
    @property
    def categories(self) -> List[str]:
        return ["visual", "analysis", "ai"]
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "images": {
                "type": Union[str, List[str]],
                "description": "Image source(s): URL, file path, base64 data, or list of sources",
                "required": True,
            },
            "analysis_type": {
                "type": str,
                "description": f"Type of analysis: {', '.join(self.prompts.keys())} or custom prompt",
                "required": False,
                "default": "describe",
            },
            "custom_prompt": {
                "type": str,
                "description": "Custom analysis prompt (overrides analysis_type if provided)",
                "required": False,
            },
            "model": {
                "type": str,
                "description": "Visual model to use (e.g., 'openai_vision:gpt-4-vision-preview', 'ollama_vision:llava')",
                "required": False,
                "default": self.default_model,
            },
            "process_images": {
                "type": bool,
                "description": "Whether to preprocess images before analysis",
                "required": False,
                "default": True,
            },
            "image_options": {
                "type": Dict[str, Any],
                "description": "Image processing options (resize, format, quality, etc.)",
                "required": False,
                "default": {},
            },
            "model_options": {
                "type": Dict[str, Any],
                "description": "Model-specific options (temperature, max_tokens, etc.)",
                "required": False,
                "default": {},
            },
        }
    
    async def _prepare_images(
        self, 
        images: Union[str, List[str]], 
        process_images: bool = True,
        image_options: Dict[str, Any] = None
    ) -> List[ImageContent]:
        """Prepare images for analysis"""
        image_options = image_options or {}
        
        # Ensure we have a list
        if isinstance(images, str):
            images = [images]
        
        # Process images if requested
        if process_images:
            processed_results = await self.image_processor.batch_process(
                sources=images,
                **image_options
            )
            
            image_contents = []
            for i, result in enumerate(processed_results):
                if result.get("success", False):
                    image_data = result["image_content"]
                    image_content = ImageContent(
                        data=image_data["data"],
                        media_type=image_data["media_type"],
                        name=image_data["name"],
                        description=image_data["description"],
                    )
                    image_contents.append(image_content)
                else:
                    logger.warning(f"Failed to process image {images[i]}: {result.get('error')}")
                    # Try to create ImageContent from original source
                    try:
                        image_content = ImageContent(
                            data=images[i],
                            media_type="image/jpeg",  # Default
                            name=f"image_{i}",
                            description=f"Image from {images[i]}",
                        )
                        image_contents.append(image_content)
                    except Exception as e:
                        logger.error(f"Failed to create ImageContent for {images[i]}: {str(e)}")
            
            return image_contents
        else:
            # Create ImageContent objects directly from sources
            image_contents = []
            for i, image_source in enumerate(images):
                try:
                    image_content = ImageContent(
                        data=image_source,
                        media_type="image/jpeg",  # Default
                        name=f"image_{i}",
                        description=f"Image from {image_source}",
                    )
                    image_contents.append(image_content)
                except Exception as e:
                    logger.error(f"Failed to create ImageContent for {image_source}: {str(e)}")
            
            return image_contents
    
    def _get_analysis_prompt(self, analysis_type: str, custom_prompt: str = None) -> str:
        """Get the analysis prompt based on type"""
        if custom_prompt:
            return custom_prompt
        elif analysis_type in self.prompts:
            return self.prompts[analysis_type]
        else:
            # Use as custom prompt if not in predefined types
            return analysis_type
    
    async def _analyze_with_provider(
        self,
        provider,
        model_name: str,
        images: List[ImageContent],
        prompt: str,
        model_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze images using a specific provider"""
        model_options = model_options or {}
        
        try:
            if len(images) == 1:
                result = await provider.analyze_image(
                    model_name=model_name,
                    image=images[0],
                    prompt=prompt,
                    **model_options
                )
            else:
                result = await provider.analyze_images(
                    model_name=model_name,
                    images=images,
                    prompt=prompt,
                    **model_options
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis with {provider.name} failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "provider": provider.name,
                "model": model_name,
            }
    
    async def execute(
        self,
        images: Union[str, List[str]],
        analysis_type: str = "describe",
        custom_prompt: str = None,
        model: str = None,
        process_images: bool = True,
        image_options: Dict[str, Any] = None,
        model_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute visual analysis"""
        
        model = model or self.default_model
        model_options = model_options or {}
        
        try:
            # Resolve provider and model
            provider, actual_model = await self.provider_registry.resolve_model(model)
            
            if not provider:
                raise ToolExecutionError(f"Visual provider not found for model: {model}")
            
            # Prepare images
            image_contents = await self._prepare_images(
                images=images,
                process_images=process_images,
                image_options=image_options
            )
            
            if not image_contents:
                raise ToolExecutionError("No valid images to analyze")
            
            # Get analysis prompt
            prompt = self._get_analysis_prompt(analysis_type, custom_prompt)
            
            # Perform analysis
            analysis_result = await self._analyze_with_provider(
                provider=provider,
                model_name=actual_model,
                images=image_contents,
                prompt=prompt,
                model_options=model_options
            )
            
            # Prepare result
            result = {
                "success": analysis_result.get("success", False),
                "analysis": analysis_result.get("content", ""),
                "model": actual_model,
                "provider": provider.name,
                "analysis_type": analysis_type,
                "image_count": len(image_contents),
                "prompt_used": prompt,
                "processing_info": {
                    "images_processed": process_images,
                    "image_options": image_options,
                },
            }
            
            # Add additional info from analysis result
            if "usage" in analysis_result:
                result["usage"] = analysis_result["usage"]
            
            if "error" in analysis_result:
                result["error"] = analysis_result["error"]
            
            return result
            
        except Exception as e:
            error_msg = f"Visual analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "analysis_type": analysis_type,
                "image_count": len(images) if isinstance(images, list) else 1,
            }
    
    async def batch_analyze(
        self,
        image_batches: List[Union[str, List[str]]],
        analysis_type: str = "describe",
        custom_prompt: str = None,
        model: str = None,
        process_images: bool = True,
        image_options: Dict[str, Any] = None,
        model_options: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple image batches concurrently"""
        
        semaphore = asyncio.Semaphore(3)  # Limit concurrent analyses
        
        async def analyze_with_semaphore(images: Union[str, List[str]]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.execute(
                        images=images,
                        analysis_type=analysis_type,
                        custom_prompt=custom_prompt,
                        model=model,
                        process_images=process_images,
                        image_options=image_options,
                        model_options=model_options,
                    )
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "images": images,
                    }
        
        tasks = [analyze_with_semaphore(batch) for batch in image_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "batch_index": i,
                })
            else:
                result["batch_index"] = i
                processed_results.append(result)
        
        return processed_results
    
    async def compare_images(
        self,
        images: List[str],
        comparison_focus: str = "general",
        model: str = None,
        process_images: bool = True,
        model_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Compare multiple images with specific focus"""
        
        if len(images) < 2:
            raise ToolExecutionError("At least 2 images are required for comparison")
        
        # Create comparison prompt based on focus
        comparison_prompts = {
            "general": "Compare these images and identify similarities, differences, and relationships. Focus on content, composition, style, and notable changes.",
            "differences": "Focus on identifying all differences between these images. Be thorough and specific about what has changed.",
            "similarities": "Focus on identifying all similarities between these images. Describe what remains consistent across the images.",
            "technical": "Compare the technical aspects of these images including lighting, composition, color, focus, and photographic techniques.",
            "content": "Compare the content of these images. What subjects, objects, or scenes are present in each? How do they relate?",
        }
        
        prompt = comparison_prompts.get(comparison_focus, comparison_prompts["general"])
        
        return await self.execute(
            images=images,
            custom_prompt=prompt,
            model=model,
            process_images=process_images,
            model_options=model_options,
        )
    
    async def extract_text(
        self,
        images: Union[str, List[str]],
        preserve_layout: bool = True,
        model: str = None,
        process_images: bool = True,
        model_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Extract text from images using OCR"""
        
        ocr_prompt = (
            "Extract all text visible in this image. "
            f"{'Preserve the layout and formatting as much as possible.' if preserve_layout else 'List all text content.'} "
            "If no text is visible, respond with 'No text found'."
        )
        
        return await self.execute(
            images=images,
            custom_prompt=ocr_prompt,
            model=model,
            process_images=process_images,
            model_options=model_options,
        )
    
    async def detect_objects(
        self,
        images: Union[str, List[str]],
        detail_level: str = "detailed",
        model: str = None,
        process_images: bool = True,
        model_options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Detect and identify objects in images"""
        
        object_prompts = {
            "basic": "List all main objects, people, and elements visible in this image.",
            "detailed": "Identify and list all objects, people, and elements visible in this image. Be specific and include their positions, relationships, and any notable details.",
            "technical": "Perform a technical object detection analysis. Identify all objects with their approximate positions, sizes, and any technical characteristics that can be determined.",
        }
        
        prompt = object_prompts.get(detail_level, object_prompts["detailed"])
        
        return await self.execute(
            images=images,
            custom_prompt=prompt,
            model=model,
            process_images=process_images,
            model_options=model_options,
        )
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available visual models"""
        try:
            models = await self.provider_registry.list_all_models()
            
            # Group by provider
            models_by_provider = {}
            for model in models:
                provider_name = model.provider.value
                if provider_name not in models_by_provider:
                    models_by_provider[provider_name] = []
                
                models_by_provider[provider_name].append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "context_length": model.context_length,
                    "supports_streaming": model.supports_streaming,
                    "max_image_size": model.max_image_size,
                    "supported_formats": model.supported_formats,
                    "vision_capabilities": model.vision_capabilities,
                })
            
            return {
                "success": True,
                "models_by_provider": models_by_provider,
                "total_models": len(models),
                "providers": list(models_by_provider.keys()),
            }
            
        except Exception as e:
            error_msg = f"Failed to get available models: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }
    
    async def cleanup(self):
        """Clean up resources"""
        await self.image_processor.cleanup()
        logger.info("Visual analyzer resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running, can't cleanup
            pass