# Quick Start Guide

Get your AI Assistant system running in minutes with this comprehensive quick start guide.

## 🚀 Option 1: Docker (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- API key from any OpenAI-compatible provider

### Step 1: Get Your API Key

Choose a provider and get your API key:

| Provider | Get Key | Best For |
|----------|---------|----------|
| [OpenRouter](https://openrouter.ai) | Sign up for free | Multiple models, good pricing |
| [OpenAI](https://platform.openai.com) | Create account | Highest reliability |
| [Together AI](https://together.ai) | Sign up | Open-source models |

### Step 2: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant

# Set up environment
cp .env.docker .env

# Configure your API key
echo "OPENAI_COMPATIBLE_API_KEY=your_api_key_here" >> .env
echo "SECRET_KEY=your_secret_key_here" >> .env
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Step 4: Verify Installation

```bash
# Check application health
curl http://localhost:8000/health

# Check available tools
curl http://localhost:8000/v1/tools

# Check provider status
curl http://localhost:8000/v1/providers
```

**🎉 You're running!** Access your AI Assistant at:
- **Main Application**: http://localhost (through Traefik)
- **API Documentation**: http://localhost/docs
- **Gradio Interface**: http://localhost/gradio
- **Traefik Dashboard**: http://localhost:8080
- **SearXNG Search**: http://localhost/search

## 🛠️ Option 2: Local Development

### Prerequisites
- Python 3.12
- UV package manager
- API key from any OpenAI-compatible provider

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant

# Copy environment template
cp .env.template .env

# Create virtual environment
uv venv .venv

# Install dependencies
uv sync --dev
```

### Step 2: Configure

```bash
# Add your API key to .env
echo "OPENAI_COMPATIBLE_API_KEY=your_api_key_here" >> .env

# Optional: Specify provider and model
echo "OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1" >> .env
echo "OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet" >> .env
```

### Step 3: Start Development Server

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start the server
uv run uvicorn app.main:app --reload
```

### Step 4: Test Your Setup

```bash
# Run tests to verify everything works
python run_tests.py --unit

# Test the API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello, can you help me?"}]
  }'
```

## 🧪 Your First API Call

### Using curl

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "What is 15 * 24?"}]
  }'
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What is 15 * 24?"}]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Using JavaScript

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'anthropic/claude-3.5-sonnet',
    messages: [
      { role: 'user', content: 'What is 15 * 24?' }
    ]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## 🔧 Using Tools

The AI Assistant comes with built-in tools for enhanced capabilities:

### Example: Calculator Tool

```python
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What is 153 * 42 + 17?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations"
                }
            }
        ]
    }
)
```

### Example: Time Tool

```python
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "What time is it now?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time"
                }
            }
        ]
    }
)
```

## 📊 Monitoring Your System

### Health Check

```bash
curl http://localhost:8000/health
```

### System Metrics

```bash
curl http://localhost:8000/metrics
```

### Available Tools

```bash
curl http://localhost:8000/v1/tools
```

## 🔍 Common Quick Start Issues

### API Key Problems
**Issue**: "Invalid API key" errors
**Solution**: 
1. Verify your API key is correct
2. Check you're using the right base URL
3. Ensure your account has credits

### Connection Issues
**Issue**: Can't connect to localhost:8000
**Solution**:
1. Check if the service is running: `docker-compose ps`
2. Look at logs: `docker-compose logs ai-assistant`
3. Try restarting: `docker-compose restart ai-assistant`

### Tool Not Working
**Issue**: Tools aren't being used
**Solution**:
1. Check if tools are enabled: `curl http://localhost:8000/v1/tools`
2. Verify your prompt is clear about using tools
3. Check the system logs for errors

## 🎯 Next Steps

Now that you're up and running:

1. **Explore the API documentation** at http://localhost:8000/docs
2. **Try different models** by changing the `model` parameter
3. **Enable additional tools** like web search
4. **Set up monitoring** with Prometheus and Grafana
5. **Deploy to production** using the deployment guides

## 📚 Further Reading

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Development Guide](development/development-guide.md) - Creating custom tools
- [Deployment Guide](deployment/production.md) - Production deployment
- [API Reference](api/endpoints.md) - Complete API documentation

## 🆘 Getting Help

If you run into issues:

1. **Check the troubleshooting guide** at [Troubleshooting](troubleshooting/common-issues.md)
2. **Look at GitHub Issues** for similar problems
3. **Create a new issue** with detailed error information
4. **Join our community** for support and discussions

---

🎉 **Congratulations!** You now have a fully functional AI Assistant system with tool-calling capabilities. Start building amazing AI-powered applications!