# Multi-Writer/Checker System - Quick Start Guide

This guide will help you get the multi-writer/checker system running in just a few minutes.

## Prerequisites

1. **Python 3.12** installed
2. **MongoDB** instance (local or cloud)
3. **Firecrawl API** key (for web scraping)
4. **LLM Provider** access (OpenAI, OpenRouter, etc.)

## Step 1: Set Up MongoDB

### Option A: Using Docker (Recommended)

1. Start MongoDB using the provided Docker Compose file:

```bash
docker-compose -f docker-compose.multi-writer.yml up -d
```

2. Verify MongoDB is running:

```bash
docker ps | grep mongo
```

3. (Optional) Access MongoDB web UI at http://localhost:8081
   - Username: `admin`
   - Password: `admin123`

### Option B: Using MongoDB Atlas (Cloud)

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster
3. Get your connection string
4. Add your IP address to the access list

### Option C: Local MongoDB Installation

1. Install MongoDB following the [official guide](https://www.mongodb.com/docs/manual/installation/)
2. Start MongoDB service

## Step 2: Configure Environment

1. Copy the environment template:

```bash
cp .env.multi-writer.template .env
```

2. Edit the `.env` file with your configuration:

```bash
# Enable the system
MULTI_WRITER_ENABLED=true

# Add your Firecrawl API key
MULTI_WRITER_FIRECRAWL_API_KEY=your-actual-api-key-here

# Add your MongoDB connection string
# For Docker setup:
MULTI_WRITER_MONGODB_CONNECTION_STRING=mongodb://admin:password123@localhost:27017/multi_writer_system?authSource=admin

# For MongoDB Atlas:
# MULTI_WRITER_MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/multi_writer_system
```

## Step 3: Install Dependencies

1. Install the required Python packages:

```bash
pip install -e .
```

2. Or if using uv:

```bash
uv sync
```

## Step 4: Start the Application

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Step 5: Verify Installation

1. Check the system status:

```bash
curl http://localhost:8000/
```

You should see a response indicating the multi-writer system is enabled:

```json
{
  "message": "AI Assistant Tool System is running!",
  "multi_writer_system": {
    "enabled": true,
    "api_prefix": "/v1/multi-writer"
  }
}
```

2. Check the multi-writer configuration:

```bash
curl http://localhost:8000/v1/multi-writer/config
```

## Step 6: Generate Your First Content

Create a simple test request:

```bash
curl -X POST "http://localhost:8000/v1/multi-writer/create" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a brief introduction to artificial intelligence",
    "sources": [
      {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}
    ],
    "template_name": "article.html.jinja",
    "quality_threshold": 70.0,
    "async_execution": false
  }'
```

## Step 7: Check the Results

1. Get the workflow status (using the workflow_id from the previous response):

```bash
curl "http://localhost:8000/v1/multi-writer/status/workflow_YYYYMMDD_HHMMSS"
```

2. View the generated content in the output directory:

```bash
ls -la generated_content/
```

## Step 8: Explore the API

### List All Workflows

```bash
curl "http://localhost:8000/v1/multi-writer/workflows"
```

### Get System Statistics

```bash
curl "http://localhost:8000/v1/multi-writer/statistics"
```

### View Metrics

```bash
curl "http://localhost:8000/metrics"
```

## Common Issues and Solutions

### Issue: "Multi-writer system is not enabled"

**Solution**: Make sure `MULTI_WRITER_ENABLED=true` is set in your `.env` file.

### Issue: "Firecrawl API key not configured"

**Solution**: Add your Firecrawl API key to the `.env` file:
```
MULTI_WRITER_FIRECRAWL_API_KEY=your-actual-api-key
```

### Issue: "MongoDB connection failed"

**Solution**: Verify your MongoDB connection string is correct and MongoDB is running.

### Issue: "Template not found"

**Solution**: The system creates default templates automatically. Check that the template directory exists and is writable.

## Next Steps

1. **Create Custom Templates**: Design your own Jinja templates for different output formats
2. **Configure Writers/Checkers**: Customize the available writers and checkers for your needs
3. **Set Up Monitoring**: Configure Prometheus and Grafana for advanced monitoring
4. **Integrate with Your Application**: Use the API endpoints in your own applications

## Example: Batch Content Generation

Create a Python script to generate multiple articles:

```python
import asyncio
import httpx

async def generate_multiple_articles():
    async with httpx.AsyncClient() as client:
        topics = [
            {
                "prompt": "The benefits of renewable energy",
                "sources": [{"url": "https://example.com/renewable-energy"}]
            },
            {
                "prompt": "Introduction to machine learning",
                "sources": [{"url": "https://example.com/machine-learning"}]
            },
            {
                "prompt": "The future of quantum computing",
                "sources": [{"url": "https://example.com/quantum-computing"}]
            }
        ]
        
        tasks = []
        for topic in topics:
            task = client.post(
                "http://localhost:8000/v1/multi-writer/create",
                json={
                    **topic,
                    "template_name": "article.html.jinja",
                    "quality_threshold": 75.0,
                    "async_execution": False
                }
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            if result.status_code == 200:
                data = result.json()
                print(f"Generated article {i+1}: {data['workflow_id']}")
                print(f"Quality score: {data['stages']['quality_checking']['best_score']}")
            else:
                print(f"Failed to generate article {i+1}: {result.text}")

if __name__ == "__main__":
    asyncio.run(generate_multiple_articles())
```

## Support

- **Documentation**: [Multi-Writer System Documentation](multi-writer-system.md)
- **API Reference**: Check the `/docs` endpoint when the application is running
- **Issues**: Report problems on the project's issue tracker

## Security Notes

1. **API Keys**: Never commit API keys to version control
2. **MongoDB**: Use strong passwords and enable authentication in production
3. **Network**: Consider using a reverse proxy with SSL/TLS in production
4. **Access**: Implement proper authentication for production use

Enjoy using the multi-writer/checker system!