# LangChain Migration Guide

This guide provides step-by-step instructions for migrating from the legacy AI Assistant system to the new LangChain and LangGraph integration.

## Overview

The migration to LangChain provides numerous benefits:
- Better performance and scalability
- More robust error handling
- Enhanced monitoring and observability
- Easier extension and customization
- Access to the LangChain ecosystem

## Prerequisites

Before starting the migration, ensure you have:

1. **Backup your data**: Create a full backup of your database and configuration
2. **Review dependencies**: Update your Python dependencies to include LangChain packages
3. **Test environment**: Set up a staging environment for testing the migration
4. **API keys**: Obtain necessary API keys for LLM providers
5. **Documentation**: Review the LangChain integration documentation

## Migration Phases

The migration is designed to be incremental, allowing you to migrate components gradually:

### Phase 1: Preparation and Setup

### Phase 2: Hybrid Mode Testing

### Phase 3: Gradual Component Migration

### Phase 4: Full LangChain Migration

### Phase 5: Legacy Cleanup (Optional)

## Phase 1: Preparation and Setup

### 1.1 Update Dependencies

Update your `pyproject.toml` to include LangChain dependencies:

```toml
[tool.poetry.dependencies]
python = "^3.9"
# ... existing dependencies ...

# LangChain dependencies
langchain = "^0.1.0"
langchain-community = "^0.0.20"
langchain-openai = "^0.0.5"
langchain-ollama = "^0.0.2"
langgraph = "^0.0.26"

# Additional dependencies
pydantic = "^2.0.0"
sqlalchemy = "^2.0.0"
asyncpg = "^0.29.0"
```

Install the updated dependencies:

```bash
poetry install
```

### 1.2 Configure Environment Variables

Add LangChain configuration to your `.env` file:

```bash
# LangChain Integration
LANGCHAIN_INTEGRATION_MODE=hybrid
LANGCHAIN_LLM_MANAGER_ENABLED=true
LANGCHAIN_TOOL_REGISTRY_ENABLED=true
LANGCHAIN_AGENT_MANAGER_ENABLED=true
LANGCHAIN_MEMORY_WORKFLOW_ENABLED=true
LANGCHAIN_MONITORING_ENABLED=true

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ai_assistant
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password

# LLM Provider Configuration
OPENAI_API_KEY=your-openai-key
OLLAMA_BASE_URL=http://localhost:11434

# Monitoring Configuration
MONITORING_METRICS_RETENTION_DAYS=30
MONITORING_PERFORMANCE_TRACKING_ENABLED=true
MONITORING_ERROR_TRACKING_ENABLED=true
```

### 1.3 Database Schema Updates

Run the LangChain database migrations:

```bash
# Apply LangChain schema
docker-compose exec app python -m app.core.langchain.database.migrations.apply_schema

# Verify schema
docker-compose exec app python -m app.core.langchain.database.migrations.verify_schema
```

### 1.4 Verify Setup

Verify the LangChain integration is working:

```bash
# Check health
curl http://localhost:8000/api/langchain/health

# Check status
curl http://localhost:8000/api/langchain/status
```

## Phase 2: Hybrid Mode Testing

In hybrid mode, you can test LangChain components alongside legacy components.

### 2.1 Enable Hybrid Mode

Set the integration mode to hybrid in your configuration:

```bash
LANGCHAIN_INTEGRATION_MODE=hybrid
```

### 2.2 Test LLM Manager

Test the new LLM manager:

```bash
# Test with curl
curl -X POST http://localhost:8000/api/langchain/llm/request \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Hello, LangChain!"
  }'
```

Or with Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/langchain/llm/request",
    json={
        "model": "gpt-3.5-turbo",
        "prompt": "Hello, LangChain!"
    }
)

print(response.json())
```

### 2.3 Test Tool Registry

Test the new tool registry:

```bash
# List tools
curl http://localhost:8000/api/langchain/tools

# Execute a tool
curl -X POST http://localhost:8000/api/langchain/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_tool",
    "parameters": {
      "query": "LangChain migration"
    }
  }'
```

### 2.4 Test Agent Manager

Test the new agent manager:

```bash
# List agents
curl http://localhost:8000/api/langchain/agents

# Invoke an agent
curl -X POST http://localhost:8000/api/langchain/agents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "research_agent",
    "input_data": {
      "messages": [{"role": "user", "content": "Test migration"}]
    }
  }'
```

### 2.5 Test Memory Manager

Test the new memory manager:

```bash
# Create conversation
curl -X POST http://localhost:8000/api/langchain/memory/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test_migration",
    "agent_name": "test_agent"
  }'

# Add message
curl -X POST http://localhost:8000/api/langchain/memory/conversations/test_migration/messages \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "Testing LangChain migration"
  }'
```

### 2.6 Compare Performance

Compare performance between legacy and LangChain components:

```python
import time
import requests

# Test legacy LLM
start = time.time()
response = requests.post(
    "http://localhost:8000/api/legacy/llm/request",
    json={"prompt": "Hello, legacy!"}
)
legacy_time = time.time() - start

# Test LangChain LLM
start = time.time()
response = requests.post(
    "http://localhost:8000/api/langchain/llm/request",
    json={"model": "gpt-3.5-turbo", "prompt": "Hello, LangChain!"}
)
langchain_time = time.time() - start

print(f"Legacy time: {legacy_time:.2f}s")
print(f"LangChain time: {langchain_time:.2f}s")
```

## Phase 3: Gradual Component Migration

Migrate components one by one, starting with the least critical ones.

### 3.1 Migrate LLM Providers

Update your LLM provider configuration to use LangChain:

```python
# Before (legacy)
from app.core.llm_providers import get_llm_provider
llm = get_llm_provider("openai")

# After (LangChain)
from app.core.langchain.integration import langchain_integration
llm = await langchain_integration.get_llm_manager().get_llm("gpt-3.5-turbo")
```

Update your API endpoints to use the new LLM manager:

```python
# Before
@router.post("/llm/request")
async def legacy_llm_request(request: LLMRequest):
    provider = get_llm_provider(request.provider)
    response = await provider.generate(request.prompt)
    return {"response": response}

# After
@router.post("/llm/request")
async def langchain_llm_request(request: LLMRequest):
    llm_manager = langchain_integration.get_llm_manager()
    llm = await llm_manager.get_llm(request.model)
    response = await llm.ainvoke(request.prompt)
    return {"response": response.content}
```

### 3.2 Migrate Tool System

Update your tool system to use LangChain:

```python
# Before (legacy)
from app.core.tools.base import BaseTool

class CustomTool(BaseTool):
    def execute(self, input_data):
        # Tool implementation
        return result

# After (LangChain)
from langchain.tools import tool

@tool
def custom_tool(input_data: str) -> str:
    """Tool description"""
    # Tool implementation
    return result
```

Register your tools with the LangChain tool registry:

```python
from app.core.langchain.integration import langchain_integration

tool_registry = langchain_integration.get_tool_registry()
await tool_registry.register_custom_tool(
    name="custom_tool",
    func=custom_tool,
    description="Custom tool description"
)
```

### 3.3 Migrate Agent System

Update your agent system to use LangGraph:

```python
# Before (legacy)
from app.core.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    async def process(self, input_data):
        # Agent implementation
        return result

# After (LangGraph)
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

def custom_agent_workflow(state):
    # Agent workflow implementation
    return {"result": result}

workflow = StateGraph(dict)
workflow.add_node("process", custom_agent_workflow)
workflow.set_entry_point("process")
workflow.add_edge("process", "end")

compiled_agent = workflow.compile(checkpointer=MemorySaver())
```

Register your agents with the LangGraph agent manager:

```python
from app.core.langchain.integration import langchain_integration

agent_manager = langchain_integration.get_agent_manager()
await agent_manager.register_agent(
    name="custom_agent",
    agent=compiled_agent,
    description="Custom agent description"
)
```

### 3.4 Migrate Memory System

Update your memory system to use LangChain:

```python
# Before (legacy)
from app.core.agents.conversation_manager import ConversationManager

conv_manager = ConversationManager()
await conv_manager.create_conversation("conv_id", "agent_name")
await conv_manager.add_message("conv_id", "user", "Hello")

# After (LangChain)
from app.core.langchain.integration import langchain_integration

memory_manager = langchain_integration.get_memory_manager()
await memory_manager.create_conversation("conv_id", "agent_name")
await memory_manager.add_message("conv_id", "user", "Hello")
```

### 3.5 Update API Routes

Update your API routes to use LangChain components:

```python
# Before
@router.post("/process")
async def process_data(request: ProcessRequest):
    # Legacy processing
    result = await legacy_process(request.data)
    return {"result": result}

# After
@router.post("/process")
async def process_data(request: ProcessRequest):
    # LangChain processing
    tool_registry = langchain_integration.get_tool_registry()
    result = await tool_registry.execute_tool("process_tool", request.data)
    return {"result": result}
```

## Phase 4: Full LangChain Migration

Once all components are migrated, switch to full LangChain mode.

### 4.1 Enable LangChain Mode

Set the integration mode to LangChain:

```bash
LANGCHAIN_INTEGRATION_MODE=langchain
```

### 4.2 Update Configuration

Update your configuration to enable all LangChain components:

```python
# app/core/config.py
LANGCHAIN_FEATURES = {
    "llm_manager": True,
    "tool_registry": True,
    "agent_manager": True,
    "memory_workflow": True,
    "monitoring": True
}
```

### 4.3 Verify Full Migration

Verify all components are using LangChain:

```bash
# Check system status
curl http://localhost:8000/api/langchain/status

# Check all components are enabled
curl http://localhost:8000/api/langchain/health
```

### 4.4 Performance Testing

Run comprehensive performance tests:

```python
import asyncio
import time
from app.core.langchain.integration import langchain_integration

async def performance_test():
    # Initialize integration
    await langchain_integration.initialize()
    
    # Test LLM performance
    llm_manager = langchain_integration.get_llm_manager()
    start = time.time()
    for i in range(100):
        llm = await llm_manager.get_llm("gpt-3.5-turbo")
        await llm.ainvoke(f"Test message {i}")
    llm_time = time.time() - start
    print(f"LLM performance: {100/llm_time:.2f} requests/second")
    
    # Test tool performance
    tool_registry = langchain_integration.get_tool_registry()
    start = time.time()
    for i in range(100):
        await tool_registry.execute_tool("test_tool", {"input": f"test {i}"})
    tool_time = time.time() - start
    print(f"Tool performance: {100/tool_time:.2f} requests/second")

asyncio.run(performance_test())
```

## Phase 5: Legacy Cleanup (Optional)

Once you're confident with the LangChain migration, you can remove legacy code.

### 5.1 Remove Legacy Components

Remove or archive legacy components:

```bash
# Archive legacy code
mv app/core/agents/legacy app/core/agents/legacy.backup
mv app/core/tools/legacy app/core/tools/legacy.backup
mv app/api/legacy_routes.py app/api/legacy_routes.py.backup
```

### 5.2 Update Imports

Update imports to remove legacy references:

```python
# Remove
# from app.core.legacy import LegacyComponent

# Keep
from app.core.langchain.integration import langchain_integration
```

### 5.3 Clean Up Configuration

Remove legacy configuration options:

```python
# Remove from config.py
# LEGACY_FEATURES = {...}
# LEGACY_SETTINGS = {...}
```

### 5.4 Update Documentation

Update your documentation to reflect the new architecture:

```markdown
# Update README.md
# Remove legacy references
# Add LangChain setup instructions
# Update API documentation
```

## Troubleshooting

### Common Issues

#### 1. Component Initialization Errors

**Problem**: Components fail to initialize

**Solution**:
- Check environment variables
- Verify database connection
- Review component configuration

```bash
# Check configuration
python -c "from app.core.langchain.integration import LangChainIntegration; print('Config OK')"

# Check database
docker-compose exec app python -c "from app.core.langchain.database import get_client; print('DB OK')"
```

#### 2. Performance Issues

**Problem**: LangChain components are slower than expected

**Solution**:
- Check for blocking operations
- Optimize database queries
- Implement caching
- Monitor metrics

```python
# Check metrics
curl http://localhost:8000/api/langchain/monitoring/performance

# Identify bottlenecks
curl http://localhost:8000/api/langchain/monitoring/metrics
```

#### 3. Memory Leaks

**Problem**: Memory usage increases over time

**Solution**:
- Implement proper cleanup
- Use connection pooling
- Monitor memory usage
- Implement garbage collection

```python
# Monitor memory
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

#### 4. API Compatibility Issues

**Problem**: API responses have changed

**Solution**:
- Update API clients
- Use backward compatibility layer
- Document breaking changes

```python
# Use compatibility layer
from app.core.langchain.integration import langchain_integration

# Check if component is enabled
if langchain_integration.is_component_enabled("llm_manager"):
    # Use LangChain
    response = await langchain_llm_request(request)
else:
    # Use legacy
    response = await legacy_llm_request(request)
```

### Debugging Tools

#### 1. Health Check

Use the health check endpoint to diagnose issues:

```bash
curl http://localhost:8000/api/langchain/health
```

#### 2. Metrics Dashboard

Monitor metrics to identify issues:

```bash
curl http://localhost:8000/api/langchain/monitoring/metrics
curl http://localhost:8000/api/langchain/monitoring/performance
```

#### 3. Debug Logging

Enable debug logging for detailed information:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Check logs
docker-compose logs app
```

## Best Practices

### 1. Gradual Migration

- Migrate one component at a time
- Test each component thoroughly
- Monitor performance during migration
- Have a rollback plan

### 2. Monitoring

- Set up comprehensive monitoring
- Track performance metrics
- Monitor error rates
- Set up alerts for issues

### 3. Testing

- Write comprehensive tests
- Test in staging environment
- Perform load testing
- Test error scenarios

### 4. Documentation

- Document migration process
- Update API documentation
- Create troubleshooting guides
- Document configuration changes

## Rollback Plan

If you encounter issues during migration, you can rollback:

### 1. Revert Integration Mode

```bash
LANGCHAIN_INTEGRATION_MODE=legacy
```

### 2. Restore Legacy Configuration

Restore your legacy configuration files:

```bash
cp app/core/config.py.backup app/core/config.py
cp .env.backup .env
```

### 3. Restart Services

```bash
docker-compose restart app
```

### 4. Verify Rollback

```bash
curl http://localhost:8000/api/health
```

## Support

For migration support:

- Documentation: [LangChain Integration Documentation](../architecture/langchain-integration.md)
- API Reference: [LangChain API Reference](../api/langchain-api-reference.md)
- Community: [AI Assistant Community Forum](https://community.ai-assistant.com)
- Support: support@ai-assistant.com

## Conclusion

The LangChain migration provides a more robust, scalable, and maintainable foundation for the AI Assistant. By following this guide, you can migrate your system incrementally with minimal disruption to your users.

Remember to:
- Test thoroughly at each phase
- Monitor performance closely
- Have a rollback plan ready
- Document your migration process

Good luck with your migration!