# Frequently Asked Questions (FAQ)

This FAQ covers common questions about the AI Assistant system, from basic setup to advanced configuration.

## 🚀 Getting Started

### Q: What do I need to run the AI Assistant?
**A:** You need:
- An API key from any OpenAI-compatible provider (OpenRouter, OpenAI, Together AI, etc.)
- Docker and Docker Compose (recommended) OR Python 3.12 with UV
- For local models: Ollama installed

### Q: Can I use this without Docker?
**A:** Yes! See the [Local Development Setup](../development/setup.md) guide. You'll need Python 3.12 and UV package manager.

### Q: How much does it cost to run?
**A:** Costs vary by provider:
- OpenAI GPT-4: ~$30 per 1M output tokens
- Claude 3.5 Sonnet: ~$15 per 1M output tokens  
- OpenRouter: Often 10-50% cheaper than direct provider pricing
- Local models: Free after hardware investment

### Q: What's the easiest way to get started?
**A:** The Docker quick start is recommended:
```bash
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.docker .env
# Add your API key to .env
docker-compose up -d
```

## 🔧 Configuration

### Q: How do I switch between different LLM providers?
**A:** Update your `.env` file:

**OpenRouter:**
```bash
OPENAI_COMPATIBLE_API_KEY=your_openrouter_key
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

**OpenAI:**
```bash
OPENAI_COMPATIBLE_API_KEY=sk-your-openai-key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=gpt-4-turbo
```

**Local Ollama:**
```bash
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
OLLAMA_SETTINGS_DEFAULT_MODEL=llama3.1:8b
```

### Q: Can I use multiple providers at once?
**A:** Yes! Enable fallback mode:
```bash
PREFERRED_PROVIDER=openai_compatible
ENABLE_FALLBACK=true
```

The system will automatically switch to backup providers if the primary fails.

### Q: How do I configure the system for production?
**A:** Key production settings:
```bash
ENVIRONMENT=production
DEBUG=false
REDIS_CACHE_ENABLED=true
MONITORING_ENABLED=true
LOG_LEVEL=INFO
```

See the [Production Deployment Guide](../deployment/production.md) for details.

## 🤖 AI & Models

### Q: Which model should I use?
**A:** It depends on your needs:
- **Claude 3.5 Sonnet**: Best for tool calling and complex reasoning
- **GPT-4 Turbo**: Most reliable, good for general tasks
- **GPT-3.5 Turbo**: Fast and cheap for simple queries
- **Local models**: Best for privacy and cost control

### Q: How do I improve AI response quality?
**A:** Try these strategies:
1. **Use clear, specific prompts**
2. **Enable tools** for relevant tasks
3. **Provide context** in system messages
4. **Choose appropriate models** for the task
5. **Enable caching** for consistent responses

### Q: Can the AI work without tools?
**A:** Yes! Tools are optional. Simply omit the `tools` parameter from your API call.

### Q: How do I make the AI use tools more reliably?
**A:** 
1. **Be explicit** in your prompts about using tools
2. **Provide clear tool descriptions**
3. **Use system messages** to encourage tool usage
4. **Test different models** (Claude 3.5 Sonnet is best for tools)

## 🔧 Tools & Features

### Q: What tools are built-in?
**A:** Default tools include:
- **Calculator**: Mathematical calculations
- **Time Tool**: Current time and date functions
- **Echo Tool**: Testing and debugging
- **SearXNG Search**: Web search capabilities
- **Custom Tools**: Framework for adding your own

### Q: How do I disable specific tools?
**A:** Set environment variables:
```bash
CALCULATOR_TOOL_ENABLED=false
TIME_TOOL_ENABLED=false
SEARXNG_TOOL_ENABLED=false
```

### Q: Can I create my own tools?
**A:** Yes! See the [Tool Development Guide](../tools/tool-development.md). Basic structure:
```python
from app.core.tools.base import BaseTool

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Description of what this tool does"
    
    async def execute(self, **kwargs):
        # Your tool logic here
        return {"result": "success"}
```

### Q: How do web search capabilities work?
**A:** The system includes SearXNG for privacy-focused web search:
```bash
# Enable in .env
SEARXNG_TOOL_ENABLED=true
SEARXNG_URL=http://localhost:8080
```

Then use in your API calls:
```python
"tools": [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information"
    }
}]
```

## 📊 Performance & Scaling

### Q: How can I improve response times?
**A:** Several optimization strategies:
1. **Enable caching**: `MEMORY_CACHE_ENABLED=true`
2. **Use faster models**: GPT-3.5 Turbo for simple tasks
3. **Optimize prompts**: Be specific and concise
4. **Use Redis**: `REDIS_CACHE_ENABLED=true`
5. **Parallel tool execution** for independent tools

### Q: How do I monitor system performance?
**A:** Built-in monitoring endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics

# Tool usage statistics
curl http://localhost:8000/v1/tools/stats
```

For advanced monitoring, set up Prometheus and Grafana.

### Q: Can I run this on multiple servers?
**A:** Yes! The system is designed for horizontal scaling:
- Use Redis for shared caching
- Deploy behind a load balancer
- Use environment variables for configuration
- Monitor with centralized metrics

## 🔒 Security

### Q: How are API keys handled?
**A:** Security best practices:
- **No hardcoded keys** in the codebase
- **Environment variables** for sensitive data
- **SecretStr type** for automatic protection
- **No logging** of API keys
- **Regular security scanning** in CI/CD

### Q: Is my data private?
**A:** It depends on your setup:
- **Cloud providers**: Data sent to provider servers
- **Local models**: Data stays on your machine
- **Self-hosted**: Complete control over data
- **SearXNG search**: Privacy-focused, no tracking

### Q: How do I secure the production deployment?
**A:** Key security measures:
```bash
# Use HTTPS
SSL_ENABLED=true

# Enable authentication (coming soon)
AUTH_ENABLED=true

# Secure headers
SECURITY_HEADERS_ENABLED=true

# Rate limiting
RATE_LIMIT_ENABLED=true
```

## 🐛 Troubleshooting

### Q: The AI isn't responding. What do I check?
**A:** Quick diagnostic steps:
1. **Check service status**: `curl http://localhost:8000/health`
2. **Verify API key**: Test with provider directly
3. **Check logs**: `docker-compose logs ai-assistant`
4. **Test simple query**: Try without tools first

### Q: Tools aren't working. How do I fix this?
**A:** Common solutions:
1. **Check tool status**: `curl http://localhost:8000/v1/tools`
2. **Verify tool enabled**: Check environment variables
3. **Test tool directly**: Use tool execution endpoint
4. **Check tool logs**: Look for execution errors

### Q: I'm getting rate limited. What can I do?
**A:** Rate limiting strategies:
1. **Use multiple API keys** (if available)
2. **Enable fallback providers**
3. **Implement caching** to reduce API calls
4. **Use local models** for non-sensitive tasks
5. **Optimize prompts** to reduce token usage

## 💰 Cost Management

### Q: How can I reduce costs?
**A:** Cost optimization strategies:
1. **Use appropriate models**: Don't use GPT-4 for simple tasks
2. **Enable caching**: Avoid repeat API calls
3. **Monitor usage**: Track token consumption
4. **Use local models**: For privacy-sensitive tasks
5. **Optimize prompts**: Be concise and specific

### Q: How do I monitor API costs?
**A:** Built-in cost tracking:
```bash
# Check usage statistics
curl http://localhost:8000/v1/usage

# Token usage by model
curl http://localhost:8000/v1/usage/tokens

# Cost breakdown
curl http://localhost:8000/v1/usage/costs
```

### Q: What's included in the free tier?
**A:** The AI Assistant system is **completely free and open source**. You only pay for:
- LLM provider API calls (OpenAI, OpenRouter, etc.)
- Hosting costs (if deploying to cloud)
- Optional services (monitoring, databases)

## 🔧 Advanced Usage

### Q: Can I integrate this with my existing application?
**A:** Yes! The system provides OpenAI-compatible APIs:
- **Drop-in replacement** for OpenAI API
- **Standard HTTP endpoints**
- **Multiple client libraries** available
- **WebSocket support** for streaming

### Q: How do I customize the AI behavior?
**A**: Customization options:
1. **System messages**: Set AI personality and behavior
2. **Tool selection**: Choose which tools are available
3. **Model configuration**: Optimize for specific tasks
4. **Prompt templates**: Standardize common interactions
5. **Custom tools**: Add domain-specific capabilities

### Q: Can I use this for commercial applications?
**A:** Yes! The project is MIT licensed, which means:
- ✅ **Commercial use** allowed
- ✅ **Modifications** allowed
- ✅ **Distribution** allowed
- ✅ **Private use** allowed
- ❌ **No warranty** provided
- ❌ **No liability** assumed

## 📚 Learning & Support

### Q: Where can I learn more?
**A:** Resources for learning:
- [Documentation](../index.md): Comprehensive guides
- [Development Notes](../../development_notes/README.md): Technical insights
- [AI Notes](../../ai_notes/README.md): AI-specific guidance
- [GitHub Issues](https://github.com/jSieber7/ai_assistant/issues): Community support
- [Examples](../examples/): Code samples and patterns

### Q: How do I contribute to the project?
**A:** Contribution guidelines:
1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation**
6. **Submit a pull request**

See the [Contributing Guide](../development/contributing.md) for details.

### Q: Who can I ask for help?
**A:** Support options:
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and community support
- **Documentation**: Check guides and troubleshooting sections
- **Development Notes**: Advanced technical insights

---

## 🎯 Still Have Questions?

If your question isn't answered here:

1. **Check the documentation** - most answers are in the guides
2. **Search GitHub Issues** - someone may have asked before
3. **Create a new issue** - we're happy to help!
4. **Join our community** - connect with other users

Remember: The best way to learn is to experiment with the system. Start with the [Quick Start Guide](../quick-start.md) and explore from there!