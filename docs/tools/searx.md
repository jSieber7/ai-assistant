# SearX Integration

SearX is a privacy-respecting, hackable metasearch engine that allows the AI assistant to perform real-time web searches. This integration provides up-to-date information beyond the LLM's training data cutoff.

## Overview

The SearX integration enables the AI assistant to:
- Search the web for current information
- Access real-time data and news
- Bypass model knowledge limitations
- Provide citations and sources for information

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# SearX instance URL (self-hosted or public instance)
SEARXNG_URL=http://localhost:8080

# Optional: API key if using a secured instance
SEARX_API_KEY=your_api_key_here

# Search settings
SEARX_TIMEOUT=30
SEARX_MAX_RESULTS=10
```

### Supported SearX Instances

You can use:
- **Self-hosted**: Run your own SearX instance for maximum privacy
- **Public instances**: Use reliable public SearX instances
- **Cloud deployment**: Deploy to cloud providers for scalability

## Implementation Status

**Current Status**: *Planned Feature*

This integration is part of the roadmap and will be implemented in future releases.

## Planned Features

### Basic Search Functionality
```python
# Planned implementation
async def search_web(query: str, categories: List[str] = None) -> List[SearchResult]:
    """
    Perform web search using SearX
    
    Args:
        query: Search query
        categories: Optional search categories (news, images, etc.)
    
    Returns:
        List of search results with titles, URLs, and snippets
    """
    pass
```

### Advanced Search Options
- **Category filtering**: News, images, videos, maps
- **Language targeting**: Search in specific languages
- **Time filtering**: Recent results only
- **Region targeting**: Country-specific results

### Result Processing
- **Relevance scoring**: Rank results by relevance
- **Content extraction**: Extract key information from pages
- **Citation generation**: Create proper citations for responses
- **Duplicate detection**: Filter duplicate results

## Usage Examples

### Basic Search Integration
```python
# Planned usage in tool calling
@tool
async def web_search(query: str) -> str:
    """
    Search the web for current information
    
    Use this tool when you need up-to-date information that 
    may not be in the model's training data.
    """
    results = await searx_client.search(query)
    return format_search_results(results)
```

### Integration with AI Agent
The SearX tool will be automatically available to the AI assistant when needed. The agent will determine when real-time information is required and use the search tool accordingly.

## Setup Instructions

### Option 1: Self-Hosted SearX (Recommended)

1. **Install SearX**:
```bash
# Using Docker (easiest method)
docker run -d -p 8080:8080 --name searx searx/searx
```

2. **Verify installation**:
```bash
curl http://localhost:8080
```

3. **Configure the AI assistant**:
```bash
echo "SEARXNG_URL=http://localhost:8080" >> .env
```

### Option 2: Public Instance

1. **Find a reliable public instance** from [searx.space](https://searx.space)
2. **Update configuration**:
```bash
echo "SEARXNG_URL=https://public.searx.instance" >> .env
```

## Security Considerations

### Privacy Protection
- SearX doesn't track users or store search history
- Requests are anonymized through the SearX instance
- No personal data is shared with search engines

### Rate Limiting
- Implement appropriate rate limiting to avoid abuse
- Respect search engine terms of service
- Use caching to reduce duplicate requests

### Content Safety
- Implement result filtering for safe content
- Validate URLs before accessing content
- Use HTTPS for secure connections

## Error Handling

### Common Issues
- **Instance unavailable**: Fallback to cached results or disable feature
- **Rate limiting**: Implement exponential backoff
- **Network errors**: Retry with circuit breaker pattern

### Planned Error Handling
```python
class SearXClient:
    async def search_with_fallback(self, query: str) -> SearchResults:
        try:
            return await self.search(query)
        except SearXError as e:
            logger.warning(f"SearX search failed: {e}")
            return await self.get_cached_results(query)
```

## Performance Optimization

### Caching Strategy
- **Query result caching**: Cache search results for common queries
- **Content caching**: Cache extracted content from pages
- **TTL settings**: Set appropriate cache expiration times

### Async Operations
- Use async/await for non-blocking operations
- Implement connection pooling for HTTP requests
- Use background tasks for content processing

## Testing

### Unit Tests
```python
@pytest.mark.asyncio
async def test_searx_search():
    """Test basic search functionality"""
    client = SearXClient("http://localhost:8080")
    results = await client.search("test query")
    assert len(results) > 0
```

### Integration Tests
```python
@pytest.mark.integration
async def test_searx_integration():
    """Test full integration with AI agent"""
    # Test that agent can use search tool effectively
    pass
```

## Roadmap

### Phase 1: Basic Integration
- [ ] Implement basic search functionality
- [ ] Add error handling and retry logic
- [ ] Create unit tests

### Phase 2: Advanced Features
- [ ] Add result filtering and ranking
- [ ] Implement content extraction
- [ ] Add citation generation

### Phase 3: Production Ready
- [ ] Performance optimization
- [ ] Comprehensive error handling
- [ ] Security hardening

## Related Documentation

- [SearX Official Documentation](https://docs.searxng.org/)
- [AI Assistant Architecture](architecture/overview.md)
- [Tool System Design](architecture/tools.md)

## Support

For issues with SearX integration:
1. Check the SearX instance is running and accessible
2. Verify network connectivity and firewall settings
3. Review logs for detailed error information
4. Consult the [troubleshooting guide](development/troubleshooting.md)