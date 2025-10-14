# Debugging Guide

This guide provides comprehensive debugging strategies for troubleshooting issues in the AI Assistant System.

## Overview

Debugging AI systems requires a systematic approach to identify and resolve issues. This guide covers tools, techniques, and best practices for effective debugging.

## Common Issues and Solutions

### 1. Model Not Responding

**Symptoms:**
- Timeouts when making requests
- Empty or null responses
- Connection errors

**Debugging Steps:**

1. Check API keys and configuration:
```python
import os
from app.core.config import get_settings

settings = get_settings()
print(f"API Key configured: {bool(settings.OPENAI_API_KEY)}")
print(f"Base URL: {settings.OPENAI_BASE_URL}")
```

2. Test connectivity:
```python
import httpx

async def test_connection():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OPENAI_BASE_URL}/models")
            print(f"Connection status: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
```

3. Check rate limits:
```python
from app.core.monitoring import RateLimitMonitor

monitor = RateLimitMonitor()
current_usage = monitor.get_current_usage()
print(f"Current usage: {current_usage}")
```

### 2. Poor Response Quality

**Symptoms:**
- Irrelevant responses
- Inconsistent output format
- Low-quality content

**Debugging Steps:**

1. Analyze the prompt:
```python
from app.core.debugging import PromptAnalyzer

analyzer = PromptAnalyzer()
prompt_analysis = analyzer.analyze_prompt(prompt)
print(f"Prompt clarity: {prompt_analysis.clarity}")
print(f"Prompt specificity: {prompt_analysis.specificity}")
```

2. Check model parameters:
```python
def debug_model_parameters():
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Top-p: {top_p}")
    print(f"Frequency penalty: {frequency_penalty}")
```

3. Test with different models:
```python
models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
for model in models:
    response = await generate_response(prompt, model=model)
    print(f"{model}: {response.text[:100]}...")
```

### 3. Performance Issues

**Symptoms:**
- Slow response times
- High resource usage
- Memory leaks

**Debugging Steps:**

1. Profile the request:
```python
import time
import psutil
from app.core.debugging import PerformanceProfiler

with PerformanceProfiler() as profiler:
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    response = await generate_response(prompt)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    print(f"Response time: {end_time - start_time:.2f}s")
    print(f"Memory used: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
```

2. Check token usage:
```python
def analyze_token_usage(response):
    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
```

3. Monitor system resources:
```python
import psutil

def check_system_resources():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"CPU usage: {cpu_percent}%")
    print(f"Memory usage: {memory.percent}%")
    print(f"Disk usage: {disk.percent}%")
```

## Debugging Tools

### 1. Logging

Configure comprehensive logging:

```python
import logging
from app.core.debugging import configure_debug_logging

# Configure debug logging
configure_debug_logging(level=logging.DEBUG)

# Create logger
logger = logging.getLogger(__name__)

async def debug_generate_response(prompt):
    logger.debug(f"Generating response for prompt: {prompt[:50]}...")
    
    try:
        response = await model.generate(prompt)
        logger.debug(f"Response generated: {response.text[:50]}...")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise
```

### 2. Request Tracing

Trace requests through the system:

```python
from app.core.debugging import RequestTracer

tracer = RequestTracer()

@tracer.trace_request
async def traced_generate_response(prompt):
    trace_id = tracer.start_trace("generate_response")
    
    try:
        # Log request details
        tracer.log_event(trace_id, "request_started", {"prompt_length": len(prompt)})
        
        # Generate response
        response = await model.generate(prompt)
        
        # Log response details
        tracer.log_event(trace_id, "response_received", {
            "response_length": len(response.text),
            "token_usage": response.usage.total_tokens
        })
        
        return response
    finally:
        tracer.end_trace(trace_id)
```

### 3. Error Analyzer

Analyze errors for patterns:

```python
from app.core.debugging import ErrorAnalyzer

error_analyzer = ErrorAnalyzer()

@error_analyzer.analyze_errors
async def debug_with_error_handling(prompt):
    try:
        return await model.generate(prompt)
    except Exception as e:
        error_analyzer.record_error(e, {
            "prompt": prompt,
            "model": model.name,
            "timestamp": time.time()
        })
        raise
```

## Advanced Debugging Techniques

### 1. Response Comparison

Compare responses across models:

```python
from app.core.debugging import ResponseComparator

comparator = ResponseComparator()

async def compare_responses(prompt):
    models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    responses = {}
    
    for model in models:
        response = await generate_response(prompt, model=model)
        responses[model] = response
    
    # Compare responses
    comparison = comparator.compare(responses)
    print(f"Quality scores: {comparison.quality_scores}")
    print(f"Response lengths: {comparison.lengths}")
    print(f"Similarity matrix: {comparison.similarity_matrix}")
```

### 2. A/B Testing

Debug by testing variations:

```python
from app.core.debugging import ABTestDebugger

ab_debugger = ABTestDebugger()

async def debug_prompt_variations(original_prompt):
    variations = [
        original_prompt,
        f"Please {original_prompt.lower()}",
        f"Could you {original_prompt.lower()}?",
        f"I need you to {original_prompt.lower()}."
    ]
    
    results = {}
    for i, variation in enumerate(variations):
        response = await generate_response(variation)
        results[f"variation_{i}"] = {
            "prompt": variation,
            "response": response.text,
            "quality": evaluate_quality(response.text)
        }
    
    return results
```

### 3. Memory Profiling

Profile memory usage:

```python
from app.core.debugging import MemoryProfiler

memory_profiler = MemoryProfiler()

@memory_profiler.profile_memory
async def profile_memory_usage(prompt):
    # Take memory snapshot
    snapshot1 = memory_profiler.take_snapshot()
    
    response = await model.generate(prompt)
    
    # Take another snapshot
    snapshot2 = memory_profiler.take_snapshot()
    
    # Compare snapshots
    diff = memory_profiler.compare_snapshots(snapshot1, snapshot2)
    print(f"Memory difference: {diff}")
    
    return response
```

## Debugging Checklist

### Before Debugging

1. **Reproduce the Issue**
   - Can you consistently reproduce the problem?
   - What are the exact steps to reproduce?
   - What are the expected vs actual results?

2. **Gather Information**
   - Error messages and stack traces
   - System logs and metrics
   - Configuration details
   - Recent changes

### During Debugging

1. **Isolate the Problem**
   - Narrow down the component causing the issue
   - Test with minimal inputs
   - Disable non-essential features

2. **Form Hypotheses**
   - What could be causing this issue?
   - How can you test each hypothesis?

3. **Test Systematically**
   - Change one variable at a time
   - Document each test and result
   - Use control cases for comparison

### After Debugging

1. **Verify the Fix**
   - Ensure the issue is resolved
   - Check for regression
   - Test edge cases

2. **Document the Solution**
   - What was the root cause?
   - How was it fixed?
   - How can it be prevented?

## Debugging Best Practices

### 1. Use Version Control

Track changes to isolate when issues were introduced:
```bash
git bisect start
git bisect bad  # Current version with issue
git bisect good [commit_before_issue]
# Git will checkout commits for testing
git bisect good  # or bad depending on test result
```

### 2. Implement Health Checks

Add health checks for early detection:
```python
from app.core.debugging import HealthChecker

health_checker = HealthChecker()

@health_checker.check_health
async def check_model_health():
    try:
        response = await model.generate("Test prompt", max_tokens=5)
        return True
    except Exception:
        return False
```

### 3. Create Debug Utilities

Build reusable debugging tools:
```python
class DebugUtils:
    @staticmethod
    def print_request_details(request):
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Headers: {request.headers}")
        print(f"Body: {request.body}")
    
    @staticmethod
    def print_response_details(response):
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Body: {response.text[:100]}...")
```

### 4. Use Automated Testing

Catch issues early with tests:
```python
import pytest

@pytest.mark.asyncio
async def test_model_response():
    response = await model.generate("Test prompt")
    assert response.text is not None
    assert len(response.text) > 0
    assert response.usage.total_tokens > 0
```

## Troubleshooting Specific Components

### 1. Tool Integration

Debug tool execution:
```python
from app.core.debugging import ToolDebugger

tool_debugger = ToolDebugger()

async def debug_tool_execution(tool_name, parameters):
    # Log tool call
    tool_debugger.log_tool_call(tool_name, parameters)
    
    try:
        result = await tool_registry.execute(tool_name, parameters)
        tool_debugger.log_tool_result(tool_name, result)
        return result
    except Exception as e:
        tool_debugger.log_tool_error(tool_name, e)
        raise
```

### 2. Caching Issues

Debug cache behavior:
```python
from app.core.debugging import CacheDebugger

cache_debugger = CacheDebugger()

async def debug_cache_behavior(key):
    # Check if key exists in cache
    exists = cache_debugger.key_exists(key)
    print(f"Key exists in cache: {exists}")
    
    if exists:
        # Get cache metadata
        metadata = cache_debugger.get_metadata(key)
        print(f"Cache metadata: {metadata}")
    
    # Monitor cache operations
    with cache_debugger.monitor_cache_operations():
        result = await cache.get_or_set(key, lambda: expensive_operation())
    
    return result
```

### 3. Agent Behavior

Debug agent decision-making:
```python
from app.core.debugging import AgentDebugger

agent_debugger = AgentDebugger()

async def debug_agent_decision(agent, input_data):
    # Log agent state
    agent_debugger.log_agent_state(agent)
    
    # Trace decision process
    with agent_debugger.trace_decision():
        decision = await agent.make_decision(input_data)
    
    # Log reasoning
    agent_debugger.log_reasoning(agent, decision)
    
    return decision
```

## Conclusion

Effective debugging is essential for maintaining a reliable AI system. By using the tools and techniques outlined in this guide, you can systematically identify and resolve issues.

Remember that debugging is an iterative process. Start with the most likely causes, test your hypotheses systematically, and document your findings. Over time, you'll build intuition for quickly identifying and resolving common issues.

The key to successful debugging is maintaining a curious, methodical mindset and leveraging the right tools for the job.