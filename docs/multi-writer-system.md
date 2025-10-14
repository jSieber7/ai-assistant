# Multi-Writer/Checker System

The multi-writer/checker system is an advanced content generation and validation pipeline that creates high-quality content through AI collaboration. This system is **disabled by default** and must be explicitly enabled in configuration.

## Overview

The system implements a sophisticated content generation workflow where:
- **Multiple AI writers** create different versions of content
- **AI checkers** validate and improve the content
- **Firecrawl API** extracts raw web content for processing
- **Jinja API** formats and templates the final output
- **Quality gates** ensure content meets standards before output
- **MongoDB** stores all workflow data and results

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Sources   │───▶│   Firecrawl API  │───▶│  Raw Content    │
│ (URLs, Docs)    │    │  (Scraping)      │    │   Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Writer Agents │◄───│   Content Hub    │───▶│  Checker Agents │
│ (Multiple AIs)  │    │  (Orchestration) │    │ (Validators)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                                               │
          │                                               ▼
          │                                      ┌─────────────────┐
          │                                      │  Quality Gates   │
          │                                      │ (Scoring &       │
          └──────────────────────────────────────▶│   Filtering)     │
                                                 └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jinja API     │◄───│  Best Content   │───▶│  Final Output   │
│ (Templating)    │    │   (Selected)     │    │ (Formatted)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration

### Enabling the System

The multi-writer system is disabled by default. To enable it, set the following environment variable:

```bash
MULTI_WRITER_ENABLED=true
```

### Required Configuration

To use the multi-writer system, you need to configure the following:

```bash
# Firecrawl API (for web scraping)
MULTI_WRITER_FIRECRAWL_API_KEY=your-firecrawl-api-key

# MongoDB (for storage)
MULTI_WRITER_MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MULTI_WRITER_MONGODB_DATABASE_NAME=multi_writer_system

# Optional: Template directory
MULTI_WRITER_TEMPLATE_DIR=templates
```

### Optional Configuration

```bash
# Quality settings
MULTI_WRITER_QUALITY_THRESHOLD=70.0
MULTI_WRITER_MAX_ITERATIONS=2

# Performance settings
MULTI_WRITER_MAX_CONCURRENT_WORKFLOWS=5
MULTI_WRITER_WORKFLOW_TIMEOUT=600

# API settings
MULTI_WRITER_API_PREFIX=/v1/multi-writer
MULTI_WRITER_ENABLE_ASYNC_EXECUTION=true
```

## Writers

The system includes multiple writer agents with different specialties:

### Available Writers

| Writer ID | Specialty | Model | Description |
|-----------|-----------|--------|-------------|
| technical_1 | technical | claude-3.5-sonnet | Focuses on accuracy, clarity, and technical precision |
| technical_2 | technical | gpt-4-turbo | Alternative technical writer |
| creative_1 | creative | claude-3.5-sonnet | Focuses on engaging storytelling and creative expression |
| analytical_1 | analytical | gpt-4-turbo | Focuses on data-driven insights and logical reasoning |

### Customizing Writers

You can customize the available writers in the configuration:

```python
# In your configuration
available_writers = {
    "custom_technical": {
        "specialty": "technical",
        "model": "your-preferred-model"
    }
}
```

## Checkers

The system includes multiple checker agents that validate and improve content:

### Available Checkers

| Checker ID | Focus Area | Model | Description |
|------------|------------|--------|-------------|
| factual_1 | factual | claude-3.5-sonnet | Verifies claims and checks for accuracy |
| style_1 | style | claude-3.5-sonnet | Improves writing quality, tone, and readability |
| structure_1 | structure | gpt-4-turbo | Ensures logical flow and proper formatting |
| seo_1 | seo | gpt-4-turbo | Optimizes content for search engines |

## Templates

The system uses Jinja2 templates for formatting the final output. Two default templates are provided:

### Article HTML Template
- File: `templates/article.html.jinja`
- Outputs: Complete HTML article with metadata
- Includes: Sources, editor notes, quality metrics

### Blog Post Markdown Template
- File: `templates/blog-post.md.jinja`
- Outputs: Markdown format for blog posts
- Includes: Writer info, quality score, sources, recommendations

### Creating Custom Templates

Create your own templates in the configured template directory. Templates receive the following context:

```jinja
{{ content.best_improved_version.content }}  # The final content
{{ content.overall_score }}                 # Quality score (0-100)
{{ content.original_content.writer_id }}    # Original writer
{{ content.original_content.specialty }}    # Writer specialty
{{ content.aggregated_feedback.recommendations }}  # Editor recommendations
```

## API Usage

### Create Content

```bash
curl -X POST "http://localhost:8000/v1/multi-writer/create" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write about renewable energy benefits",
    "sources": [
      {"url": "https://example.com/renewable-energy"}
    ],
    "template_name": "article.html.jinja",
    "quality_threshold": 75.0,
    "async_execution": false
  }'
```

### Check Workflow Status

```bash
curl "http://localhost:8000/v1/multi-writer/status/workflow_20231014_123456"
```

### List Workflows

```bash
curl "http://localhost:8000/v1/multi-writer/workflows?status=completed&limit=10"
```

### Get Statistics

```bash
curl "http://localhost:8000/v1/multi-writer/statistics"
```

## Workflow Stages

Each content generation workflow goes through four stages:

### Stage 1: Source Processing
- Extracts content from provided URLs using Firecrawl API
- Cleans and structures content for AI processing
- Extracts key points and metadata

### Stage 2: Content Generation
- Multiple writers create different versions of the content
- Each writer applies their specialty and style guide
- Confidence scores are calculated for each version

### Stage 3: Quality Checking
- Multiple checkers validate each content version
- Issues are identified and improvements are suggested
- Content is iteratively improved based on feedback

### Stage 4: Template Rendering
- Best content version is selected based on quality score
- Content is rendered using the specified Jinja template
- Final output is generated with metadata

## Quality Scoring

The system uses a comprehensive quality scoring mechanism:

- **Overall Score**: Average of all checker scores (0-100)
- **Writer Confidence**: Self-assessed quality by writers (0-1)
- **Threshold**: Minimum score required for content to pass (default: 70)

### Quality Factors

- **Factual Accuracy**: Claims verification and consistency
- **Style Quality**: Readability, tone, and engagement
- **Structure**: Logical flow and organization
- **SEO**: Search engine optimization factors

## Monitoring and Metrics

The system includes comprehensive monitoring:

### Available Metrics

- Workflow counts (started, completed, failed)
- Stage execution times
- Quality score distributions
- API request metrics
- Database operation metrics

### Accessing Metrics

Metrics are available through Prometheus endpoints:

```bash
curl "http://localhost:8000/metrics"
```

## Storage

All workflow data is stored in MongoDB:

### Collections

- **workflows**: Complete workflow data and status
- **content**: Generated content versions
- **check_results**: Detailed checker results

### Data Retention

Configure data retention based on your needs:

```python
# In your application
# Add logic to clean up old workflows
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Process multiple sources together
2. **Concurrent Execution**: Run writers and checkers in parallel
3. **Caching**: Cache frequently accessed content
4. **Quality Thresholds**: Set appropriate thresholds to avoid excessive iterations

### Resource Limits

- Maximum concurrent workflows: 5 (configurable)
- Workflow timeout: 10 minutes (configurable)
- Maximum iterations: 2 (configurable)

## Troubleshooting

### Common Issues

1. **System Disabled**: Ensure `MULTI_WRITER_ENABLED=true`
2. **Firecrawl API**: Check API key and connectivity
3. **MongoDB**: Verify connection string and database access
4. **Template Errors**: Validate Jinja template syntax

### Debug Mode

Enable debug logging:

```bash
# In your environment
LOG_LEVEL=DEBUG
```

### Health Checks

Check system health:

```bash
curl "http://localhost:8000/v1/multi-writer/config"
```

## Examples

### Basic Content Generation

```python
import asyncio
from app.core.agents.multi_content_orchestrator import create_multi_content_orchestrator

async def generate_article():
    orchestrator = await create_multi_content_orchestrator()
    
    result = await orchestrator.create_content(
        prompt="Write about AI ethics",
        sources=[
            {"url": "https://example.com/ai-ethics-article"}
        ],
        style_guide={
            "tone": "informative",
            "audience": "general",
            "length": "medium"
        }
    )
    
    return result

# Run the workflow
result = asyncio.run(generate_article())
print(f"Generated content with quality score: {result['stages']['quality_checking']['best_score']}")
```

### Batch Content Processing

```python
import asyncio
from app.core.agents.multi_content_orchestrator import create_multi_content_orchestrator

async def generate_multiple_articles():
    orchestrator = await create_multi_content_orchestrator()
    
    topics = [
        {"prompt": "Climate change impacts", "sources": [{"url": "https://example.com/climate"}]},
        {"prompt": "Renewable energy benefits", "sources": [{"url": "https://example.com/renewable"}]},
        {"prompt": "Sustainable technology", "sources": [{"url": "https://example.com/sustainable"}]}
    ]
    
    tasks = []
    for topic in topics:
        task = orchestrator.create_content(**topic, async_execution=False)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# Run batch processing
results = asyncio.run(generate_multiple_articles())
for result in results:
    print(f"Generated: {result['workflow_id']} - Score: {result['stages']['quality_checking']['best_score']}")
```

## Security Considerations

### API Keys

- Store API keys securely in environment variables
- Rotate keys regularly
- Monitor usage for unusual activity

### Content Validation

- Validate all input parameters
- Sanitize user-provided content
- Rate limit API endpoints

### Access Control

- Implement authentication for API endpoints
- Use appropriate authorization for sensitive operations
- Audit access to generated content

## Future Enhancements

Planned improvements to the multi-writer system:

1. **Additional Writers**: More specialized writers for different domains
2. **Custom Checkers**: User-defined validation rules
3. **Template Gallery**: Pre-built templates for different use cases
4. **Workflow Designer**: Visual workflow configuration
5. **Content Analytics**: Advanced analytics on generated content
6. **Integration Hub**: Connect to external content management systems