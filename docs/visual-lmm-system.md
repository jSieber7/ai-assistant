# Visual LMM System Documentation

## Overview

The Visual LMM (Large Multimodal Model) System extends the AI Assistant with advanced visual understanding capabilities, enabling it to analyze images, extract text from visuals, control browsers with visual understanding, and perform comprehensive visual analysis tasks.

## Key Features

### 🖼️ Image Analysis
- **Image Description**: Generate detailed descriptions of images
- **Object Detection**: Identify and categorize objects in images
- **OCR (Optical Character Recognition)**: Extract text from images
- **Image Comparison**: Compare multiple images for similarities and differences
- **Technical Analysis**: Analyze composition, lighting, and technical aspects

### 🌐 Web Visual Capabilities
- **Screenshot Analysis**: Take and analyze screenshots of web pages
- **Visual Web Scraping**: Extract and analyze images from web content
- **Browser Control with Vision**: Control browsers based on visual descriptions
- **Layout Analysis**: Understand web page layouts and structures
- **Accessibility Analysis**: Evaluate web pages for accessibility issues

### 🤖 Intelligent Agent
- **Visual Request Understanding**: Automatically determine the best visual approach
- **Multi-step Workflows**: Execute complex visual interaction sequences
- **Context-Aware Processing**: Understand context and provide relevant insights
- **Error Handling**: Graceful fallbacks when visual understanding fails

## Architecture

### Visual LMM Providers
The system supports multiple visual model providers:

- **OpenAI Vision**: GPT-4V, GPT-4O with vision capabilities
- **Ollama Vision**: Local vision models like LLaVA, BakLLaVA
- **Extensible Design**: Easy to add new visual model providers

### Visual Tools
1. **ImageProcessorTool**: Image loading, conversion, and optimization
2. **VisualAnalyzerTool**: Core visual analysis capabilities
3. **VisualBrowserTool**: Browser control with visual understanding

### Visual Agent
The `VisualAgent` orchestrates all visual capabilities and provides a unified interface for visual tasks.

## Setup and Configuration

### Environment Variables

```bash
# Enable Visual LMM System
VISUAL_SYSTEM_ENABLED=true

# OpenAI Vision Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Ollama Configuration (for local vision models)
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434

# Visual System Settings
VISUAL_DEFAULT_MODEL=openai_vision:gpt-4-vision-preview
VISUAL_MAX_CONCURRENT_ANALYSES=3
VISUAL_SCREENSHOT_QUALITY=85
VISUAL_BROWSER_CONTROL_ENABLED=true
```

### Provider Configuration

#### OpenAI Vision Provider
```python
from app.core.visual_llm_provider import OpenAIVisionProvider, visual_provider_registry

provider = OpenAIVisionProvider(
    api_key="your_api_key",
    base_url="https://api.openai.com/v1"
)
visual_provider_registry.register_provider(provider)
```

#### Ollama Vision Provider
```python
from app.core.visual_llm_provider import OllamaVisionProvider, visual_provider_registry

provider = OllamaVisionProvider(
    base_url="http://localhost:11434"
)
visual_provider_registry.register_provider(provider)
```

## Usage Examples

### Basic Image Analysis

```python
from app.core.tools.visual.visual_analyzer import VisualAnalyzerTool

# Initialize the analyzer
analyzer = VisualAnalyzerTool()

# Analyze an image from URL
result = await analyzer.execute(
    images="https://example.com/image.jpg",
    analysis_type="describe"
)

print(result["analysis"])
```

### OCR Text Extraction

```python
# Extract text from an image
result = await analyzer.execute(
    images="https://example.com/document.jpg",
    analysis_type="ocr"
)

extracted_text = result["analysis"]
```

### Web Page Visual Analysis

```python
from app.core.agents.specialized.visual_agent import VisualAgent

# Initialize the visual agent
agent = VisualAgent(llm=your_llm, tool_registry=tool_registry)

# Analyze a web page visually
result = await agent.analyze_webpage_visual(
    url="https://example.com",
    analysis_types=["describe", "analyze"],
    include_screenshot=True,
    include_extracted_images=True
)
```

### Browser Control with Visual Understanding

```python
from app.core.tools.visual.visual_browser import VisualBrowserTool

# Initialize visual browser
browser = VisualBrowserTool()

# Navigate and interact with visual descriptions
result = await browser.execute(
    url="https://example.com",
    action="click",
    visual_description="red submit button in the form"
)

# Multi-step interaction workflow
workflow_result = await browser.multi_step_interaction(
    url="https://example.com",
    steps=[
        {
            "action": "screenshot",
            "description": "Take initial screenshot"
        },
        {
            "action": "click",
            "visual_description": "search box in the header",
            "description": "Click search box"
        },
        {
            "action": "type",
            "visual_description": "search box",
            "text": "visual AI",
            "description": "Type search query"
        },
        {
            "action": "click",
            "visual_description": "search button",
            "description": "Submit search"
        }
    ]
)
```

### Image Comparison

```python
# Compare multiple images
result = await analyzer.execute(
    images=["image1.jpg", "image2.jpg"],
    analysis_type="compare"
)

comparison_analysis = result["analysis"]
```

### Multi-Image Batch Processing

```python
# Process multiple images concurrently
images = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg"
]

results = await analyzer.batch_analyze(
    image_batches=images,
    analysis_type="describe"
)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['analysis']}")
```

## API Integration

### Using the Visual Agent via API

```python
import requests

# Send a visual analysis request
response = requests.post("http://localhost:8000/agent/process", json={
    "message": "Analyze this image: https://example.com/image.jpg",
    "agent_name": "visual_agent"
})

result = response.json()
print(result["response"])
```

### Direct Tool Usage via API

```python
# Use visual analyzer tool directly
response = requests.post("http://localhost:8000/tools/visual_analyzer/execute", json={
    "images": ["https://example.com/image.jpg"],
    "analysis_type": "describe"
})

result = response.json()
print(result["analysis"])
```

## Configuration Options

### Visual Provider Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `visual_system_enabled` | `true` | Enable/disable the visual system |
| `visual_default_model` | `openai_vision:gpt-4-vision-preview` | Default visual model to use |
| `visual_max_concurrent_analyses` | `3` | Maximum concurrent visual analyses |
| `visual_screenshot_quality` | `85` | Screenshot quality (1-100) |
| `visual_browser_control_enabled` | `true` | Enable visual browser control |

### Image Processing Options

```python
# Advanced image processing
result = await image_processor.execute(
    source="https://example.com/large_image.jpg",
    resize={"max_dimension": 1024},
    format="JPEG",
    quality=85,
    auto_orient=True,
    strip_metadata=True
)
```

### Visual Analysis Options

```python
# Custom analysis with specific options
result = await analyzer.execute(
    images="image.jpg",
    analysis_type="custom",
    custom_prompt="Analyze this image for accessibility issues",
    model="openai_vision:gpt-4o",
    model_options={
        "temperature": 0.1,
        "max_tokens": 500
    }
)
```

## Troubleshooting

### Common Issues

#### Visual Provider Not Configured
```
Error: No visual providers configured
```
**Solution**: Ensure you have configured at least one visual provider with valid API keys.

#### Image Processing Failed
```
Error: Failed to process image
```
**Solution**: Check image format and size. Supported formats: JPEG, PNG, WebP, GIF, BMP. Maximum size: 20MB.

#### Browser Control Issues
```
Error: Failed to initialize browser
```
**Solution**: Ensure Playwright is installed and browser dependencies are available.

### Health Check

```python
from app.core.config.visual_agent_init import health_check_visual_system

# Check visual system health
health = await health_check_visual_system()
print(health)
```

### Debug Mode

Enable debug logging for visual components:

```python
import logging
logging.getLogger("app.core.visual_llm_provider").setLevel(logging.DEBUG)
logging.getLogger("app.core.tools.visual").setLevel(logging.DEBUG)
logging.getLogger("app.core.agents.specialized.visual_agent").setLevel(logging.DEBUG)
```

## Performance Considerations

### Optimization Tips

1. **Image Size**: Resize large images before processing to improve performance
2. **Batch Processing**: Use batch analysis for multiple images
3. **Model Selection**: Choose appropriate models for your use case
4. **Caching**: Enable caching for repeated analyses
5. **Concurrent Limits**: Adjust `visual_max_concurrent_analyses` based on your hardware

### Resource Management

```python
# Clean up resources when done
await analyzer.cleanup()
await browser.cleanup()
await agent.cleanup()
```

## Security Considerations

### API Keys
- Store API keys securely using environment variables
- Use secure settings management for production deployments
- Rotate API keys regularly

### Image Content
- Be aware of privacy implications when processing images
- Implement content filtering if needed
- Consider data retention policies

### Browser Control
- Use headless mode in production environments
- Implement proper access controls for browser automation
- Monitor browser automation activities

## Extending the System

### Adding New Visual Providers

```python
from app.core.visual_llm_provider import VisualLMMProvider, VisualProviderType

class CustomVisionProvider(VisualLMMProvider):
    def __init__(self, api_key: str):
        super().__init__(VisualProviderType.CUSTOM_VISION)
        self.api_key = api_key
    
    # Implement required methods...
    
# Register the provider
provider = CustomVisionProvider(api_key="your_key")
visual_provider_registry.register_provider(provider)
```

### Creating Custom Visual Tools

```python
from app.core.tools.base.base import BaseTool

class CustomVisualTool(BaseTool):
    @property
    def name(self) -> str:
        return "custom_visual_tool"
    
    async def execute(self, **kwargs):
        # Your custom visual processing logic
        pass

# Register the tool
tool_registry.register(CustomVisualTool(), category="visual")
```

## Examples Repository

For more complete examples and use cases, see the `examples/visual-lmm/` directory in the repository.

## Support and Contributing

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Contribute to improving documentation
- **Code**: Submit pull requests for enhancements
- **Community**: Join discussions in the project forums

---

*This documentation covers the Visual LMM System v1.0. For the latest updates and detailed API references, please check the source code documentation.*