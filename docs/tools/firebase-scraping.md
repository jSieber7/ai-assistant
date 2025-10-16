# Firebase Web Scraping Tool

The Firebase Web Scraping Tool provides advanced web content extraction capabilities using Firebase infrastructure with Selenium rendering and Firestore storage.

## Overview

This tool enables the AI Assistant to:
- Scrape web content using both HTTP requests and Selenium browser automation
- Store scraped data in Firebase Firestore for persistent storage
- Handle JavaScript-heavy websites with dynamic content
- Extract structured data including text, links, images, and metadata

## Configuration

### Firebase Setup

1. **Enable Firebase in your project**:
   ```bash
   FIREBASE_ENABLED=true
   ```

2. **Configure Firebase Service Account**:
   ```bash
   # Firebase Service Account Credentials
   FIREBASE_PROJECT_ID=your-firebase-project-id
   FIREBASE_PRIVATE_KEY_ID=your-private-key-id
   FIREBASE_PRIVATE_KEY=your-private-key-here
   FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
   FIREBASE_CLIENT_ID=your-client-id
   FIREBASE_DATABASE_URL=https://your-project.firebaseio.com
   FIREBASE_STORAGE_BUCKET=your-project.appspot.com
   ```

3. **Web Scraping Settings**:
   ```bash
   FIREBASE_SCRAPING_ENABLED=true
   FIREBASE_MAX_CONCURRENT_SCRAPES=5
   FIREBASE_SCRAPE_TIMEOUT=60
   FIREBASE_SCRAPING_COLLECTION=scraped_data
   ```

4. **Selenium Settings**:
   ```bash
   FIREBASE_USE_SELENIUM=true
   FIREBASE_SELENIUM_DRIVER_TYPE=chrome
   FIREBASE_HEADLESS_BROWSER=true
   FIREBASE_BROWSER_TIMEOUT=30
   ```

## Usage

### Tool Usage

The Firebase Scraper Tool can be used directly or through the specialized agent:

```python
from app.core.tools import FirebaseScraperTool

# Create tool instance
scraper = FirebaseScraperTool()

# Scrape a single URL
result = await scraper.execute(
    url="https://example.com",
    use_selenium=True,
    store_in_firestore=True,
    extract_images=False,
    extract_links=True,
    timeout=30,
    collection_name="scraped_data"
)

# Batch scrape multiple URLs
results = await scraper.batch_scrape(
    urls=["https://example1.com", "https://example2.com"],
    use_selenium=True,
    store_in_firestore=True,
    timeout=30
)
```

### Agent Usage

The Firebase Scraper Agent provides intelligent scraping with automatic analysis:

```python
from app.core.agents import FirebaseScraperAgent
from app.core.config import get_llm

# Create agent with LLM
llm = await get_llm()
agent = FirebaseScraperAgent(llm=llm)

# Execute scraping task
result = await agent.execute(
    "Scrape content from https://example.com and extract all links"
)

# Batch scraping
batch_result = await agent.batch_scrape(
    urls=["https://example1.com", "https://example2.com"],
    use_selenium=True
)
```

## Features

### Content Extraction

The tool extracts comprehensive content including:
- **Text Content**: Main article/content text with cleaning
- **Metadata**: Title, description, keywords
- **Links**: All hyperlinks with text and URLs
- **Images**: Image sources and alt text (optional)
- **Structure**: Content hierarchy and organization

### Selenium Rendering

For JavaScript-heavy websites, Selenium provides:
- **Dynamic Content**: Execute JavaScript and wait for content
- **SPA Support**: Single Page Application compatibility
- **Form Handling**: Interactive form submission
- **AJAX Support**: Asynchronous content loading

### Firebase Integration

- **Firestore Storage**: Structured data storage with timestamps
- **Document Management**: Automatic document ID generation
- **Query Support**: Easy data retrieval and filtering
- **Security**: Firebase security rules integration

## API Reference

### FirebaseScraperTool

#### `execute(url, **kwargs)`

Scrape a single URL with advanced options.

**Parameters:**
- `url` (str): URL to scrape (required)
- `use_selenium` (bool): Use Selenium rendering (default: True)
- `store_in_firestore` (bool): Store results in Firestore (default: True)
- `extract_images` (bool): Extract images (default: False)
- `extract_links` (bool): Extract links (default: True)
- `timeout` (int): Timeout in seconds (default: 30)
- `collection_name` (str): Firestore collection name (default: "scraped_data")

**Returns:** Dictionary with scraped data

#### `batch_scrape(urls, **kwargs)`

Scrape multiple URLs concurrently.

**Parameters:**
- `urls` (List[str]): List of URLs to scrape
- `use_selenium` (bool): Use Selenium rendering
- `store_in_firestore` (bool): Store results in Firestore
- `timeout` (int): Timeout per request

**Returns:** List of scraping results

### FirebaseScraperAgent

#### `execute(query, context=None)`

Intelligent scraping with automatic task analysis.

**Parameters:**
- `query` (str): Natural language scraping request
- `context` (dict): Additional context information

**Returns:** Comprehensive scraping results with analysis

#### `batch_scrape(urls, use_selenium=True)`

Batch scraping with intelligent error handling.

**Parameters:**
- `urls` (List[str]): URLs to scrape
- `use_selenium` (bool): Use Selenium rendering

**Returns:** Batch scraping results with success/failure tracking

## Examples

### Basic Scraping

```python
# Simple content extraction
result = await scraper.execute("https://news-site.com/article")

# Extract with Selenium for dynamic content
result = await scraper.execute(
    "https://spa-website.com",
    use_selenium=True,
    extract_links=True
)
```

### Advanced Usage

```python
# Scrape multiple sites with different strategies
urls = [
    "https://static-site.com",  # Simple HTTP
    "https://dynamic-app.com",  # Selenium required
]

# Use agent for intelligent scraping
agent_result = await agent.execute(
    "Scrape these websites and extract main content and links: " + ", ".join(urls)
)
```

### Firestore Integration

```python
# Store results with custom collection
result = await scraper.execute(
    "https://example.com",
    store_in_firestore=True,
    collection_name="research_data"
)

# Access stored data
doc_id = result.get('firestore_doc_id')
print(f"Data stored with ID: {doc_id}")
```

## Error Handling

The tool provides comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Timeout Handling**: Configurable timeouts with graceful failure
- **Content Validation**: Data quality checks and validation
- **Firebase Errors**: Proper error messages for Firebase issues

## Best Practices

### Performance Optimization

1. **Use Selenium Sparingly**: Only for JavaScript-heavy sites
2. **Batch Operations**: Use batch scraping for multiple URLs
3. **Caching**: Implement caching for repeated requests
4. **Rate Limiting**: Respect website rate limits

### Data Quality

1. **Content Validation**: Verify extracted content quality
2. **Error Handling**: Implement robust error handling
3. **Data Cleaning**: Clean and normalize extracted data
4. **Metadata Extraction**: Extract comprehensive metadata

### Security Considerations

1. **Firebase Security Rules**: Configure proper Firestore security
2. **Input Validation**: Validate URLs and parameters
3. **Error Reporting**: Secure error message handling
4. **Access Control**: Implement proper access controls

## Troubleshooting

### Common Issues

**Selenium Connection Problems:**
- Ensure Chrome/ChromeDriver is properly installed
- Check headless mode compatibility
- Verify browser timeout settings

**Firebase Authentication Errors:**
- Validate service account credentials
- Check Firebase project permissions
- Verify database URL configuration

**Content Extraction Issues:**
- Adjust content selectors for specific sites
- Increase timeout for slow-loading content
- Enable Selenium for dynamic content

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Tools

- [SearXNG Search Tool](../tools/searx.md) - Web search capabilities
- [Firecrawl Tool](../tools/firecrawl.md) - Alternative scraping service
- [Content Processing](../agents/content_processor.md) - Content analysis and processing

## Next Steps

- [Configure Firebase Security Rules](https://firebase.google.com/docs/firestore/security/get-started)
- [Set Up Monitoring](../deployment/monitoring.md) for scraping performance
- [Implement Caching](../architecture/caching.md) for improved performance