"""
Image Processing Tool for Visual LMM System

This module provides comprehensive image processing capabilities for the Visual LMM system,
including image loading, conversion, preprocessing, and validation.
"""

import asyncio
import logging
import base64
import io
import re
from typing import Dict, Any, List, Optional, Union, BinaryIO
from urllib.parse import urlparse
from pathlib import Path
import mimetypes

from app.core.tools.base.base import BaseTool, ToolExecutionError
from app.core.visual_llm_provider import ImageContent

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOps, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Some image processing features will be limited.")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. URL-based image loading will not work.")


class ImageProcessorTool(BaseTool):
    """Comprehensive image processing tool for Visual LMM system"""
    
    def __init__(self, max_size: int = 20 * 1024 * 1024, default_format: str = "JPEG"):
        super().__init__()
        self.max_size = max_size  # Maximum image size in bytes
        self.default_format = default_format
        self.supported_formats = ["JPEG", "PNG", "WebP", "GIF", "BMP"]
        self._client = None
    
    @property
    def name(self) -> str:
        return "image_processor"
    
    @property
    def description(self) -> str:
        return "Comprehensive image processing tool for loading, converting, and optimizing images for visual analysis"
    
    @property
    def keywords(self) -> List[str]:
        return [
            "image",
            "process",
            "convert",
            "resize",
            "optimize",
            "load",
            "download",
            "base64",
            "format",
        ]
    
    @property
    def categories(self) -> List[str]:
        return ["visual", "processing", "utility"]
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "source": {
                "type": str,
                "description": "Image source: URL, file path, or base64 data",
                "required": True,
            },
            "source_type": {
                "type": str,
                "description": "Type of source: 'url', 'file', 'base64', or 'auto' (auto-detect)",
                "required": False,
                "default": "auto",
            },
            "resize": {
                "type": Dict[str, int],
                "description": "Resize dimensions: {'width': 800, 'height': 600} or {'max_dimension': 1024}",
                "required": False,
            },
            "format": {
                "type": str,
                "description": f"Output format: {', '.join(self.supported_formats)}",
                "required": False,
                "default": self.default_format,
            },
            "quality": {
                "type": int,
                "description": "Output quality (1-100, for JPEG/WebP)",
                "required": False,
                "default": 85,
            },
            "max_size": {
                "type": int,
                "description": "Maximum output size in bytes",
                "required": False,
                "default": self.max_size,
            },
            "auto_orient": {
                "type": bool,
                "description": "Automatically orient image based on EXIF data",
                "required": False,
                "default": True,
            },
            "strip_metadata": {
                "type": bool,
                "description": "Remove EXIF and other metadata",
                "required": False,
                "default": False,
            },
        }
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for URL downloads"""
        if not HTTPX_AVAILABLE:
            raise ToolExecutionError("httpx not available for URL downloads")
        
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        
        return self._client
    
    def _detect_source_type(self, source: str) -> str:
        """Auto-detect the type of image source"""
        # Check if it's a URL
        if source.startswith(("http://", "https://")):
            return "url"
        
        # Check if it's a file path
        if "/" in source or "\\" in source or source.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
            return "file"
        
        # Check if it looks like base64
        if source.startswith(("data:image/", "base64,")) or re.match(r'^[A-Za-z0-9+/]+={0,2}$', source):
            return "base64"
        
        # Default to treating as file path
        return "file"
    
    def _extract_base64_from_data_url(self, data_url: str) -> tuple[str, str]:
        """Extract base64 data and media type from data URL"""
        # Pattern: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
        match = re.match(r'data:([^;]+);base64,(.+)', data_url)
        if match:
            media_type = match.group(1)
            base64_data = match.group(2)
            return base64_data, media_type
        
        # If no match, assume it's just base64 data
        return data_url, "image/jpeg"
    
    async def _load_from_url(self, url: str) -> tuple[bytes, str]:
        """Load image data from URL"""
        client = self._get_client()
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            # Get content type from headers
            content_type = response.headers.get("content-type", "image/jpeg")
            if not content_type.startswith("image/"):
                # Try to guess from URL
                ext = Path(urlparse(url).path).suffix.lower()
                if ext in [".jpg", ".jpeg"]:
                    content_type = "image/jpeg"
                elif ext == ".png":
                    content_type = "image/png"
                elif ext == ".gif":
                    content_type = "image/gif"
                elif ext == ".webp":
                    content_type = "image/webp"
                else:
                    content_type = "image/jpeg"
            
            return response.content, content_type
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to load image from URL {url}: {str(e)}")
    
    def _load_from_file(self, file_path: str) -> tuple[bytes, str]:
        """Load image data from file path"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ToolExecutionError(f"Image file not found: {file_path}")
            
            # Read file content
            with open(path, "rb") as f:
                image_data = f.read()
            
            # Detect content type
            content_type, _ = mimetypes.guess_type(str(path))
            if not content_type or not content_type.startswith("image/"):
                # Try to detect from file extension
                ext = path.suffix.lower()
                if ext in [".jpg", ".jpeg"]:
                    content_type = "image/jpeg"
                elif ext == ".png":
                    content_type = "image/png"
                elif ext == ".gif":
                    content_type = "image/gif"
                elif ext == ".webp":
                    content_type = "image/webp"
                elif ext == ".bmp":
                    content_type = "image/bmp"
                else:
                    content_type = "image/jpeg"
            
            return image_data, content_type
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to load image from file {file_path}: {str(e)}")
    
    def _load_from_base64(self, base64_data: str) -> tuple[bytes, str]:
        """Load image data from base64 string"""
        try:
            # Extract base64 and media type if it's a data URL
            if base64_data.startswith("data:"):
                base64_data, media_type = self._extract_base64_from_data_url(base64_data)
            else:
                # Default to JPEG if we can't determine the type
                media_type = "image/jpeg"
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            return image_bytes, media_type
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to decode base64 image data: {str(e)}")
    
    def _process_image_with_pil(
        self,
        image_data: bytes,
        media_type: str,
        resize: Optional[Dict[str, int]] = None,
        format: str = "JPEG",
        quality: int = 85,
        auto_orient: bool = True,
        strip_metadata: bool = False,
    ) -> tuple[bytes, str]:
        """Process image using PIL/Pillow"""
        if not PIL_AVAILABLE:
            raise ToolExecutionError("PIL/Pillow not available for image processing")
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if format.upper() == "JPEG" and image.mode in ["RGBA", "P"]:
                image = image.convert("RGB")
            elif format.upper() == "PNG" and image.mode not in ["RGBA", "RGB", "L"]:
                image = image.convert("RGBA")
            
            # Auto-orient based on EXIF
            if auto_orient:
                try:
                    image = ImageOps.exif_transpose(image)
                except Exception:
                    # EXIF processing failed, continue without orientation
                    pass
            
            # Resize if requested
            if resize:
                if "width" in resize and "height" in resize:
                    # Specific dimensions
                    new_size = (resize["width"], resize["height"])
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                elif "max_dimension" in resize:
                    # Maintain aspect ratio, limit max dimension
                    max_dim = resize["max_dimension"]
                    width, height = image.size
                    if max(width, height) > max_dim:
                        if width > height:
                            new_width = max_dim
                            new_height = int(height * max_dim / width)
                        else:
                            new_height = max_dim
                            new_width = int(width * max_dim / height)
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Strip metadata if requested
            if strip_metadata:
                # Remove EXIF data by creating a new image
                if image.mode in ["RGBA", "RGB", "L"]:
                    image = Image.new(image.mode, image.size, (0, 0, 0) if image.mode == "RGB" else 0)
                    image.paste(Image.open(io.BytesIO(image_data)))
            
            # Save to output format
            output_buffer = io.BytesIO()
            
            save_kwargs = {}
            if format.upper() in ["JPEG", "WEBP"]:
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif format.upper() == "PNG":
                save_kwargs["optimize"] = True
            
            image.save(output_buffer, format=format, **save_kwargs)
            processed_data = output_buffer.getvalue()
            
            # Update media type
            output_media_type = f"image/{format.lower()}"
            
            return processed_data, output_media_type
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to process image with PIL: {str(e)}")
    
    async def execute(
        self,
        source: str,
        source_type: str = "auto",
        resize: Optional[Dict[str, int]] = None,
        format: str = "JPEG",
        quality: int = 85,
        max_size: Optional[int] = None,
        auto_orient: bool = True,
        strip_metadata: bool = False,
    ) -> Dict[str, Any]:
        """Execute image processing"""
        
        # Auto-detect source type if needed
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        # Load image data
        if source_type == "url":
            image_data, media_type = await self._load_from_url(source)
            source_info = {"type": "url", "url": source}
        elif source_type == "file":
            image_data, media_type = self._load_from_file(source)
            source_info = {"type": "file", "path": source}
        elif source_type == "base64":
            image_data, media_type = self._load_from_base64(source)
            source_info = {"type": "base64", "original_media_type": media_type}
        else:
            raise ToolExecutionError(f"Unsupported source type: {source_type}")
        
        # Validate image size
        max_size = max_size or self.max_size
        if len(image_data) > max_size:
            raise ToolExecutionError(f"Image size ({len(image_data)} bytes) exceeds maximum ({max_size} bytes)")
        
        # Process image if PIL is available
        if PIL_AVAILABLE:
            try:
                processed_data, output_media_type = self._process_image_with_pil(
                    image_data=image_data,
                    media_type=media_type,
                    resize=resize,
                    format=format,
                    quality=quality,
                    auto_orient=auto_orient,
                    strip_metadata=strip_metadata,
                )
                
                # Use processed data
                final_data = processed_data
                final_media_type = output_media_type
                processing_info = {
                    "processed": True,
                    "original_format": media_type,
                    "output_format": output_media_type,
                    "original_size": len(image_data),
                    "output_size": len(processed_data),
                    "compression_ratio": len(processed_data) / len(image_data) if image_data else 1.0,
                    "resize_applied": resize is not None,
                    "auto_oriented": auto_orient,
                    "metadata_stripped": strip_metadata,
                }
                
            except Exception as e:
                logger.warning(f"PIL processing failed, using original image: {str(e)}")
                # Fall back to original data
                final_data = image_data
                final_media_type = media_type
                processing_info = {
                    "processed": False,
                    "error": str(e),
                    "original_format": media_type,
                    "output_format": media_type,
                    "original_size": len(image_data),
                    "output_size": len(image_data),
                }
        else:
            # No PIL available, use original data
            final_data = image_data
            final_media_type = media_type
            processing_info = {
                "processed": False,
                "error": "PIL not available",
                "original_format": media_type,
                "output_format": media_type,
                "original_size": len(image_data),
                "output_size": len(image_data),
            }
        
        # Convert to base64 for output
        base64_data = base64.b64encode(final_data).decode('utf-8')
        
        # Create ImageContent object
        image_content = ImageContent(
            data=base64_data,
            media_type=final_media_type,
            name=f"processed_image.{format.lower()}",
            description=f"Processed image from {source_type}",
        )
        
        return {
            "success": True,
            "image_content": {
                "data": base64_data,
                "media_type": final_media_type,
                "name": image_content.name,
                "description": image_content.description,
            },
            "source_info": source_info,
            "processing_info": processing_info,
            "size": len(final_data),
            "format": format,
        }
    
    async def batch_process(
        self,
        sources: List[str],
        source_type: str = "auto",
        resize: Optional[Dict[str, int]] = None,
        format: str = "JPEG",
        quality: int = 85,
        max_size: Optional[int] = None,
        auto_orient: bool = True,
        strip_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process multiple images concurrently"""
        
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
        
        async def process_with_semaphore(source: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.execute(
                        source=source,
                        source_type=source_type,
                        resize=resize,
                        format=format,
                        quality=quality,
                        max_size=max_size,
                        auto_orient=auto_orient,
                        strip_metadata=strip_metadata,
                    )
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "source": source,
                    }
        
        tasks = [process_with_semaphore(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "source": sources[i],
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def extract_images_from_html(self, html_content: str, base_url: str = "") -> List[str]:
        """Extract image URLs from HTML content"""
        import re
        
        # Find all img tags
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(img_pattern, html_content, re.IGNORECASE)
        
        image_urls = []
        for match in matches:
            # Handle relative URLs
            if match.startswith("/") and base_url:
                from urllib.parse import urljoin
                full_url = urljoin(base_url, match)
                image_urls.append(full_url)
            elif match.startswith(("http://", "https://")):
                image_urls.append(match)
            # Skip data URLs and relative paths without base URL
        
        return image_urls
    
    async def cleanup(self):
        """Clean up resources"""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
        
        self._client = None
        logger.info("Image processor HTTP client cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running, can't cleanup
            pass