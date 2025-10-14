# Common Issues and Solutions

This guide covers the most common issues users encounter with the AI Assistant system and their solutions.

## 🔧 Quick Diagnosis Checklist

Before diving into specific issues, run these quick checks:

```bash
# 1. Check if the service is running
curl http://localhost:8000/health

# 2. Verify environment variables
docker-compose exec ai-assistant env | grep -E "(API_KEY|OPENAI|OLLAMA)"

# 3. Check logs for errors
docker-compose logs --tail=50 ai-assistant

# 4. Test basic API functionality
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'
```

## 🔑 API Key Issues

### Issue: "Invalid API key" or "Authentication failed"

**Symptoms:**
- `401 Unauthorized` responses
- Error messages about invalid API keys
- No response from LLM providers

**Common Causes:**
1. Incorrect API key
2. Wrong provider URL
3. Account suspended or out of credits
4. API key format issues

**Solutions:**

#### Check API Key Validity
```bash
# Test with OpenAI
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.openai.com/v1/models

# Test with OpenRouter
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://openrouter.ai/api/v1/models
```

#### Verify Configuration
```bash
# Check your .env file
cat .env | grep -E "(API_KEY|BASE_URL)"

# For Docker, check environment
docker-compose exec ai-assistant printenv | grep -E "(API_KEY|BASE_URL)"
```

#### Fix Common Configuration Errors

**Wrong provider URL:**
```bash
# Correct OpenRouter URL
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1

# Correct OpenAI URL  
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
```

**API key format:**
```bash
# Remove quotes if you added them
OPENAI_COMPATIBLE_API_KEY=sk-your-key-here  # NOT "sk-your-key-here"
```

### Issue: API key works in curl but not in application

**Cause**: Environment variable not loaded properly

**Solution:**
```bash
# Restart the service to reload environment
docker-compose restart ai-assistant

# Or for local development
source .env && uv run uvicorn app.main:app --reload
```

## 🌐 Connection Issues

### Issue: Cannot connect to localhost:8000

**Symptoms:**
- Connection refused errors
- Timeout errors
- Service not responding

**Diagnosis:**
```bash
# Check if port is in use
netstat -tlnp | grep :8000

# Check Docker status
docker-compose ps

# Check service logs
docker-compose logs ai-assistant
```

**Solutions:**

#### Port Already in Use
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
export PORT=8001
uvicorn app.main:app --port 8001
```

#### Docker Service Not Starting
```bash
# Check Docker logs
docker-compose logs ai-assistant

# Restart services
docker-compose down && docker-compose up -d

# Rebuild if needed
docker-compose build --no-cache ai-assistant
```

#### Firewall Issues
```bash
# Check if firewall is blocking
sudo ufw status

# Allow port 8000
sudo ufw allow 8000
```

## 🤖 LLM Provider Issues

### Issue: OpenRouter not working

**Symptoms:**
- Timeouts when calling OpenRouter
- Model not available errors
- Rate limiting issues

**Solutions:**

#### Check OpenRouter Status
```bash
# Test OpenRouter directly
curl -H "Authorization: Bearer YOUR_KEY" \
  https://openrouter.ai/api/v1/models

# Check specific model availability
curl -H "Authorization: Bearer YOUR_KEY" \
  https://openrouter.ai/api/v1/models/anthropic/claude-3.5-sonnet
```

#### Handle Rate Limits
```bash
# Add to .env for OpenRouter
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"HTTP-Referer": "https://your-site.com", "X-Title": "Your App"}
```

#### Alternative Models
```bash
# Try different models if your preferred one is unavailable
export OPENAI_COMPATIBLE_DEFAULT_MODEL="meta-llama/llama-3.1-70b-instruct"
```

### Issue: Ollama not responding

**Symptoms:**
- Connection refused to localhost:11434
- Model not found errors
- Slow responses

**Solutions:**

#### Check Ollama Status
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve

# Pull required models
ollama pull llama2
ollama pull llama3.1:8b
```

#### Docker Ollama Issues
```bash
# Check Ollama container
docker-compose ps searxng

# Restart Ollama
docker-compose restart searxng

# Check Ollama logs
docker-compose logs searxng
```

## 🔧 Tool Issues

### Issue: Tools not being used by AI

**Symptoms:**
- AI ignores tool capabilities
- Tools not in response
- Tool execution errors

**Diagnosis:**
```bash
# Check available tools
curl http://localhost:8000/v1/tools

# Check tool configuration
curl http://localhost:8000/v1/tools/calculator
```

**Solutions:**

#### Enable Tools in Configuration
```bash
# Add to .env
TOOL_CALLING_ENABLED=true
CALCULATOR_TOOL_ENABLED=true
TIME_TOOL_ENABLED=true
```

#### Improve Prompt for Tool Usage
```python
# Be explicit about tool usage
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[{
        "role": "user", 
        "content": "Use the calculator tool to compute 15 * 24"
    }],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations"
        }
    }]
)
```

#### Check Tool Logs
```bash
# Look for tool execution logs
docker-compose logs ai-assistant | grep -i tool
```

### Issue: Tool execution fails

**Symptoms:**
- Tool timeout errors
- Tool returns error messages
- Partial tool results

**Solutions:**

#### Increase Tool Timeout
```bash
# Add to .env
TOOL_TIMEOUT_SECONDS=60
```

#### Check Tool Dependencies
```bash
# Verify external services are running
docker-compose ps

# Check network connectivity
docker network ls
```

#### Debug Specific Tools
```bash
# Test calculator tool directly
curl -X POST "http://localhost:8000/v1/tools/calculator/execute" \
  -H "Content-Type: application/json" \
  -d '{"expression": "15 * 24"}'
```

## 🐳 Docker Issues

### Issue: Docker containers won't start

**Symptoms:**
- `docker-compose up` fails
- Container restart loops
- Port conflicts

**Solutions:**

#### Clean Docker State
```bash
# Stop all containers
docker-compose down

# Remove volumes (WARNING: This deletes data)
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

#### Check Docker Resources
```bash
# Check Docker disk usage
docker system df

# Clean up unused images
docker system prune -a

# Check memory usage
docker stats
```

#### Fix Permission Issues
```bash
# Fix Docker socket permissions
sudo usermod -aG docker $USER

# Restart Docker service
sudo systemctl restart docker
```

### Issue: Docker build fails

**Symptoms:**
- Build errors during `docker-compose build`
- Missing dependencies
- Python package installation failures

**Solutions:**

#### Update Base Image
```bash
# Pull latest base images
docker-compose pull

# Rebuild with fresh cache
docker-compose build --no-cache --pull
```

#### Fix Python Dependencies
```bash
# Clear UV cache
docker-compose exec ai-assistant uv cache clean

# Reinstall dependencies
docker-compose exec ai-assistant uv sync --reinstall
```

## 📊 Performance Issues

### Issue: Slow response times

**Symptoms:**
- Responses take >10 seconds
- Timeouts during tool usage
- High latency

**Diagnosis:**
```bash
# Check response times
time curl http://localhost:8000/health

# Monitor resource usage
docker stats

# Check API response times
curl -w "@curl-format.txt" -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'
```

**Solutions:**

#### Enable Caching
```bash
# Add to .env
MEMORY_CACHE_ENABLED=true
REDIS_CACHE_ENABLED=true
CACHE_COMPRESSION_ENABLED=true
```

#### Optimize Model Selection
```bash
# Use faster models for simple queries
export OPENAI_COMPATIBLE_DEFAULT_MODEL="gpt-3.5-turbo"

# Or use local models for privacy
export OLLAMA_SETTINGS_DEFAULT_MODEL="llama3.1:8b"
```

#### Increase Resources
```bash
# Give Docker more memory
# In docker-compose.yml:
services:
  ai-assistant:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Issue: High memory usage

**Symptoms:**
- Container crashes due to OOM
- System becomes slow
- Memory leaks

**Solutions:**

#### Monitor Memory Usage
```bash
# Check container memory
docker stats --no-stream

# Check system memory
free -h
```

#### Optimize Configuration
```bash
# Reduce cache sizes
MEMORY_CACHE_MAX_SIZE=100
REDIS_CACHE_MAX_MEMORY=256mb

# Enable compression
CACHE_COMPRESSION_ENABLED=true
```

## 🔍 Debugging Techniques

### Enable Debug Logging
```bash
# Set debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Or in .env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Use Health Endpoints
```bash
# Detailed health check
curl http://localhost:8000/health?detailed=true

# Check system metrics
curl http://localhost:8000/metrics
```

### Test Components Individually
```bash
# Test API without LLM
curl http://localhost:8000/v1/models

# Test tools without AI
curl http://localhost:8000/v1/tools

# Test LLM without tools
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}], "tools": []}'
```

## 🆘 Getting Additional Help

If you've tried these solutions and still have issues:

1. **Check GitHub Issues** for similar problems
2. **Create a detailed issue** with:
   - Your configuration (hide API keys)
   - Complete error messages
   - Steps to reproduce
   - System information (OS, Docker version, etc.)
3. **Join our community** for real-time help
4. **Check the development notes** for advanced debugging techniques

## 📝 Prevention Tips

- **Regular updates**: Keep your system updated
- **Monitor resources**: Use the built-in metrics
- **Test changes**: Test in development before production
- **Backup configuration**: Save working .env files
- **Document issues**: Keep track of problems and solutions

---

Remember: Most issues are configuration-related. Double-check your environment variables and API keys first!