# Firecrawl Web Scraping Tool

The Firecrawl Web Scraping Tool provides advanced web content extraction capabilities using the Firecrawl API with comprehensive data processing and multiple output formats.

## Features

- Scrape web content using Firecrawl API with advanced rendering
- Multiple output formats (markdown, raw HTML, etc.)
- Handle JavaScript-heavy websites with dynamic content
- Extract links, images, and metadata
- Customizable tag inclusion/exclusion
- Screenshot capabilities
- Batch processing support
- Error handling and retry mechanisms

## Configuration

### Firecrawl Setup

1. **Enable Firecrawl in your project**:
   ```bash
   # Add to your .env file
   FIRECRAWL_ENABLED=true
   ```

2. **Configure Firecrawl API**:
   ```bash
   # Firecrawl API Configuration
   FIRECRAWL_API_KEY=your-firecrawl-api-key
   FIRECRAWL_BASE_URL=https://api.firecrawl.dev
   FIRECRAWL_SCRAPING_ENABLED=true
   FIRECRAWL_MAX_CONCURRENT_SCRAPES=5
   FIRECRAWL_SCRAPE_TIMEOUT=60
   ```

3. **Configure extraction options**:
   ```bash
   # Data Processing Settings
   FIRECRAWL_CONTENT_CLEANING=true
   FIRECRAWL_EXTRACT_IMAGES=false
   FIRECRAWL_EXTRACT_LINKS=true
   FIRECRAWL_FORMATS=["markdown", "raw"]
   FIRECRAWL_WAIT_FOR=2000
   FIRECRAWL_SCREENSHOT=false
   FIRECRAWL_INCLUDE_TAGS=["article", "main", "content"]
   FIRECRAWL_EXCLUDE_TAGS=["nav", "footer", "aside", "script", "style"]
   ```

## Usage

### Basic Tool Usage

The Firecrawl Tool can be used directly:

```python
from app.core.tools import FirecrawlTool

# Create tool instance
scraper = FirecrawlTool(api_key="your-api-key")

# Scrape a single URL
result = await scraper.execute(
    url="https://example.com",
    formats=["markdown", "raw"],
    extract_links=True,
    extract_images=True,
    wait_for=3000
)

print(f"Title: {result['title']}")
print(f"Content: {result['content'][:200]}...")
print(f"Links found: {result['link_count']}")
```

### Batch Scraping

Scrape multiple URLs concurrently:

```python
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

results = await scraper.batch_scrape(
    urls=urls,
    formats=["markdown"],
    timeout=30
)

for result in results:
    if result.get("success", True):
        print(f"Scraped {result['url']}: {result['title']}")
    else:
        print(f"Failed to scrape {result['url']}: {result.get('error')}")
```

### Agent Usage

The Firecrawl Agent provides intelligent scraping with automatic analysis:

```python
from app.core.agents import FirecrawlAgent
from app.core.config import get_llm

# Get LLM instance
llm = await get_llm()
agent = FirecrawlAgent(llm=llm)

# Natural language scraping request
result = await agent.execute(
    "Please scrape the product information from https://shop.example.com"
)

if result["success"]:
    print(f"Scraped {result['total_urls']} URLs")
    print(f"Summary: {result['summary']['overall_assessment']}")
else:
    print(f"Scraping failed: {result['error']}")
```

## Configuration Options

### FirecrawlTool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | URL to scrape |
| `formats` | list | `["markdown", "raw"]` | Output formats |
| `wait_for` | int | `2000` | Wait time in milliseconds |
| `screenshot` | bool | `false` | Take screenshot |
| `include_tags` | list | `["article", "main", "content"]` | Tags to include |
| `exclude_tags` | list | `["nav", "footer", "aside", "script", "style"]` | Tags to exclude |
| `extract_images` | bool | `false` | Extract images |
| `extract_links` | bool | `true` | Extract links |
| `timeout` | int | `30` | Request timeout |

### FirecrawlAgent Features

The Firecrawl Agent provides additional intelligence:

- **Automatic URL extraction** from natural language requests
- **Content analysis** to determine optimal scraping approach
- **Format selection** based on content type
- **Error handling** with retry logic
- **Comprehensive summaries** with quality assessment

## Response Format

### Tool Response

```python
{
    "url": "https://example.com",
    "title": "Example Page",
    "content": "Page content in markdown format...",
    "description": "Page description from metadata",
    "links": [
        {"url": "https://example.com/page2", "text": "Next Page"}
    ],
    "images": [
        {"src": "https://example.com/image.jpg", "alt": "Example image"}
    ],
    "metadata": {
        "author": "Author Name",
        "publish_date": "2023-01-01"
    },
    "content_length": 1500,
    "link_count": 5,
    "image_count": 3,
    "formats": ["markdown", "raw"]
}
```

### Agent Response

```python
{
    "success": true,
    "results": [
        {
            "url": "https://example.com",
            "data": {...},  # Tool response data
            "formats": ["markdown"],
            "analysis": {...}  # Task analysis
        }
    ],
    "summary": {
        "overall_assessment": "Scraping completed successfully",
        "content_quality_score": 85,
        "data_completeness": "Good",
        "issues_found": [],
        "recommendations": ["Consider extracting more metadata"],
        "key_insights": ["High-quality content extracted"]
    },
    "total_urls": 1,
    "total_content": 1500,
    "analysis": {...}
}
```

## Advanced Features

### Custom Tag Selection

Control which HTML elements are extracted:

```python
# Focus on product information
result = await scraper.execute(
    url="https://shop.example.com",
    include_tags=["product", "price", "description", "image"],
    exclude_tags=["advertisement", "sidebar"]
)
```

### Screenshot Capture

Take screenshots of pages:

```python
result = await scraper.execute(
    url="https://example.com",
    screenshot=True,
    wait_for=5000  # Wait longer for page to fully load
)
```

### Error Handling

The tool includes comprehensive error handling:

- **Network Errors**: Automatic retry with exponential backoff
- **API Errors**: Clear error messages with status codes
- **Timeout Errors**: Configurable timeout handling
- **Content Validation**: Data quality checks

## Best Practices

### Performance Optimization

1. **Use appropriate formats**: Choose formats based on your needs
2. **Set reasonable timeouts**: Balance between reliability and speed
3. **Limit concurrent requests**: Don't overwhelm target websites
4. **Cache results**: Store frequently accessed data

### Content Quality

1. **Customize tag selection**: Focus on relevant content
2. **Use wait times**: Allow dynamic content to load
3. **Validate extracted data**: Check for required information
4. **Handle errors gracefully**: Provide fallback options

### Security Considerations

1. **Rate limiting**: Respect website rate limits
2. **User agents**: Use appropriate user agent strings
3. **Terms of service**: Follow website terms of service
4. **Data privacy**: Handle sensitive data appropriately

## Troubleshooting

### Common Issues

**API Key Errors:**
- Verify your Firecrawl API key is valid
- Check if your account has sufficient credits
- Ensure the API key is properly configured

**Timeout Errors:**
- Increase the timeout value for slow websites
- Use longer wait times for JavaScript-heavy pages
- Check network connectivity

**Content Extraction Issues:**
- Adjust include/exclude tags for better results
- Try different output formats
- Use screenshots to verify page rendering

**Rate Limiting:**
- Reduce concurrent requests
- Add delays between requests
- Implement exponential backoff

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Firecrawl API error: 401` | Invalid API key | Check API key configuration |
| `Firecrawl API error: 429` | Rate limited | Reduce request frequency |
| `Firecrawl request timed out` | Slow website | Increase timeout value |
| `Firecrawl tool error` | Configuration issue | Check settings and parameters |

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from app.core.tools import FirecrawlTool

app = FastAPI()
scraper = FirecrawlTool()

@app.post("/scrape")
async def scrape_url(url: str):
    try:
        result = await scraper.execute(url=url)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Background Processing

```python
import asyncio
from app.core.agents import FirecrawlAgent

async def process_scraping_queue():
    agent = FirecrawlAgent(llm=await get_llm())
    
    while True:
        # Get URLs from queue
        urls = await get_urls_from_queue()
        
        if urls:
            # Process in batches
            results = await agent.batch_scrape(urls)
            
            # Store results
            await store_scraping_results(results)
        
        await asyncio.sleep(60)  # Wait before next batch
```

## Additional Resources

- [Firecrawl API Documentation](https://docs.firecrawl.dev)
- [Configure API Rate Limits](../deployment/monitoring.md) for production use
- [Set Up Monitoring](../deployment/monitoring.md) for scraping performance
- [Security Best Practices](../development/security-and-api-key-handling.md)