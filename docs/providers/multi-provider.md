# Multi-Provider Setup

This guide explains how to configure and use multiple LLM providers with the AI Assistant System.

## Overview

The AI Assistant System supports multiple LLM providers through a unified OpenAI-compatible interface, allowing you to:
- Use different models for different tasks
- Implement failover between providers
- Load balance requests across providers
- Optimize for cost and performance
- Mix cloud and local models

## Supported Providers

- **OpenAI** - GPT models (GPT-4, GPT-3.5 Turbo, etc.)
- **OpenRouter** - Multiple models from various providers
- **Anthropic Claude** - Claude models (Claude-3.5 Sonnet, Claude-3 Opus, etc.)
- **Together AI** - Open-source models
- **Azure OpenAI** - Enterprise OpenAI models
- **Ollama** - Local models (Llama, Mistral, etc.)
- **Custom OpenAI-compatible endpoints** - Any provider implementing the OpenAI API

## Configuration

### Environment Variables

Configure multiple providers using the OpenAI-compatible provider interface:

```env
# Primary OpenAI-Compatible Provider (Recommended)
OPENAI_COMPATIBLE_ENABLED=true
OPENAI_COMPATIBLE_API_KEY=your_primary_key
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
PREFERRED_PROVIDER=openai_compatible

# Enable automatic fallback
ENABLE_PROVIDER_FALLBACK=true

# Optional: Custom headers for specific providers
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"X-Custom-Header": "value"}
```

### Provider-Specific Configurations

#### OpenRouter (Recommended for variety)
```env
OPENAI_COMPATIBLE_API_KEY=your_openrouter_key
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

#### OpenAI
```env
OPENAI_COMPATIBLE_API_KEY=sk-your-openai-key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=gpt-4-turbo
```

#### Anthropic Claude
```env
OPENAI_COMPATIBLE_API_KEY=your_anthropic_key
OPENAI_COMPATIBLE_BASE_URL=https://api.anthropic.com
OPENAI_COMPATIBLE_DEFAULT_MODEL=claude-3-5-sonnet-20241022
```

#### Together AI
```env
OPENAI_COMPATIBLE_API_KEY=your_together_key
OPENAI_COMPATIBLE_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

#### Azure OpenAI
```env
OPENAI_COMPATIBLE_API_KEY=your_azure_key
OPENAI_COMPATIBLE_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_COMPATIBLE_CUSTOM_HEADERS={"api-key": "your_azure_key"}
OPENAI_COMPATIBLE_DEFAULT_MODEL=gpt-4
```

#### Ollama (Local Models)
```env
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
OLLAMA_SETTINGS_DEFAULT_MODEL=llama3.1:8b
```

#### Custom Provider
```env
OPENAI_COMPATIBLE_API_KEY=your_custom_key
OPENAI_COMPATIBLE_BASE_URL=https://api.custom-provider.com/v1
OPENAI_COMPATIBLE_PROVIDER_NAME=Custom Provider
OPENAI_COMPATIBLE_DEFAULT_MODEL=custom-model-name
```

### Provider Configuration File

Create a `providers.yaml` configuration file for advanced setups:

```yaml
providers:
  openai_compatible:
    type: openai_compatible
    api_key: ${OPENAI_COMPATIBLE_API_KEY}
    base_url: ${OPENAI_COMPATIBLE_BASE_URL}
    default_model: ${OPENAI_COMPATIBLE_DEFAULT_MODEL}
    provider_name: ${OPENAI_COMPATIBLE_PROVIDER_NAME}
    custom_headers: ${OPENAI_COMPATIBLE_CUSTOM_HEADERS}
    timeout: 30
    max_retries: 3
    models:
      - anthropic/claude-3.5-sonnet
      - gpt-4-turbo
      - gpt-3.5-turbo
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 90000
    retry:
      max_attempts: 3
      backoff_factor: 2

  ollama:
    type: ollama
    enabled: ${OLLAMA_SETTINGS_ENABLED}
    base_url: ${OLLAMA_SETTINGS_BASE_URL}
    default_model: ${OLLAMA_SETTINGS_DEFAULT_MODEL}
    models:
      - llama3.1:8b
      - codellama
      - mistral
    rate_limit:
      requests_per_minute: 30
    retry:
      max_attempts: 2
      backoff_factor: 1.5

strategies:
  default:
    provider: openai_compatible
    model: anthropic/claude-3.5-sonnet
    fallback_providers:
      - ollama

  code_generation:
    provider: openai_compatible
    model: claude-3-opus-20240229
    fallback_providers:
      - ollama

  local_development:
    provider: ollama
    model: llama3.1:8b
    fallback_providers:
      - openai_compatible

  cost_optimized:
    provider: openai_compatible
    model: gpt-3.5-turbo
    fallback_providers:
      - ollama

  creative_writing:
    provider: openai_compatible
    model: claude-3.5-sonnet
    fallback_providers:
      - gpt-4-turbo
```

## Provider Selection Strategies

### Automatic Fallback

The system automatically falls back to alternative providers when the primary provider fails:

```bash
# Enable automatic fallback
ENABLE_PROVIDER_FALLBACK=true

# Set preferred provider
PREFERRED_PROVIDER=openai_compatible

# Configure multiple providers in order of preference
OPENAI_COMPATIBLE_API_KEY=primary_key
OPENAI_COMPATIBLE_BASE_URL=https://primary-provider.com/v1

# Fallback provider
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
```

### Manual Provider Selection

Specify the provider in your API requests:

```python
import httpx

# Use specific provider
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "openai_compatible:anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)

# Use local model
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama:llama3.1:8b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

### Round Robin

Distribute requests evenly across providers:

```python
from app.core.llm_providers import ProviderManager, RoundRobinStrategy

provider_manager = ProviderManager()
strategy = RoundRobinStrategy(providers=["openai_compatible", "ollama"])
provider_manager.set_strategy(strategy)
```

### Weighted Round Robin

Distribute requests based on weights:

```python
from app.core.llm_providers import WeightedRoundRobinStrategy

strategy = WeightedRoundRobinStrategy(
    providers={
        "openai_compatible": 0.7,
        "ollama": 0.3
    }
)
provider_manager.set_strategy(strategy)
```

### Least Cost

Select provider based on cost:

```python
from app.core.llm_providers import LeastCostStrategy

strategy = LeastCostStrategy()
provider_manager.set_strategy(strategy)
```

### Performance Based

Select provider based on response time:

```python
from app.core.llm_providers import PerformanceBasedStrategy

strategy = PerformanceBasedStrategy()
provider_manager.set_strategy(strategy)
```

## Failover Configuration

### Automatic Failover

Configure automatic failover when a provider fails:

```python
from app.core.llm_providers import FailoverConfig

failover_config = FailoverConfig(
    primary_provider="openai_compatible",
    fallback_providers=["ollama"],
    health_check_interval=60,  # seconds
    max_failures=3,
    recovery_timeout=300  # seconds
)

provider_manager.configure_failover(failover_config)
```

### Health Checks

The system automatically performs health checks on all configured providers:

```python
from app.core.llm_providers import HealthChecker

health_checker = HealthChecker()

@health_checker.register("openai_compatible")
async def check_openai_compatible_health():
    try:
        # Simple health check
        response = await openai_client.completions.create(
            model="gpt-3.5-turbo",
            prompt="test",
            max_tokens=1
        )
        return True
    except Exception:
        return False

# Schedule health checks
health_checker.schedule_checks(interval=60)
```

### Manual Health Check

Check provider health manually:

```bash
# Check all providers
curl http://localhost:8000/v1/providers

# Check specific provider
curl http://localhost:8000/v1/providers/openai_compatible/health
```

## Load Balancing

### Request Load Balancing

Balance requests across multiple provider instances:

```python
from app.core.llm_providers import LoadBalancer

load_balancer = LoadBalancer(
    providers=["openai_compatible", "ollama"],
    algorithm="least_connections",
    health_checker=health_checker
)

# Use load balancer for requests
response = await load_balancer.complete(
    prompt="Explain quantum computing",
    max_tokens=1000
)
```

### Token Load Balancing

Balance based on token usage:

```python
from app.core.llm_providers import TokenLoadBalancer

token_balancer = TokenLoadBalancer(
    providers={
        "openai_compatible": {"max_tokens": 100000, "current_usage": 0},
        "ollama": {"max_tokens": 50000, "current_usage": 0}
    }
)
```

## Cost Management

### Cost Tracking

Track costs across providers:

```python
from app.core.llm_providers import CostTracker

cost_tracker = CostTracker()

@cost_tracker.track
async def generate_response(prompt, provider, model):
    response = await provider.complete(prompt=prompt, model=model)
    cost_tracker.add_cost(provider, model, response.usage)
    return response

# Get cost report
cost_report = cost_tracker.get_report()
print(f"Total cost: ${cost_report.total_cost}")
print(f"Cost by provider: {cost_report.by_provider}")
```

### Budget Alerts

Set budget alerts:

```python
from app.core.llm_providers import BudgetAlert

budget_alert = BudgetAlert(
    daily_budget=10.0,  # $10 per day
    monthly_budget=300.0,  # $300 per month
    alert_threshold=0.8  # Alert at 80% of budget
)

@budget_alert.check_budget
async def check_costs():
    current_cost = cost_tracker.get_daily_cost()
    if current_cost > budget_alert.daily_threshold:
        await send_alert(f"Daily budget threshold reached: ${current_cost}")
```

### Cost Optimization

Configure cost optimization strategies:

```bash
# Prefer local models for cost savings
PREFERRED_PROVIDER=ollama
ENABLE_PROVIDER_FALLBACK=true

# Set usage limits
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Enable caching to reduce API calls
REDIS_CACHE_ENABLED=true
CACHE_TTL=3600
```

## Monitoring Multi-Provider Setup

### Provider Metrics

Monitor metrics for each provider:

```python
from prometheus_client import Counter, Histogram

PROVIDER_REQUESTS = Counter(
    'provider_requests_total',
    'Total requests by provider',
    ['provider', 'status']
)

PROVIDER_RESPONSE_TIME = Histogram(
    'provider_response_time_seconds',
    'Response time by provider',
    ['provider']
)

PROVIDER_COST = Counter(
    'provider_cost_total',
    'Total cost by provider',
    ['provider']
)
```

### Dashboard

Create a Grafana dashboard to monitor:
- Request distribution across providers
- Response times by provider
- Error rates by provider
- Cost tracking by provider
- Provider health status

### API Endpoints for Monitoring

```bash
# Get provider status
curl http://localhost:8000/v1/providers

# Get detailed statistics
curl http://localhost:8000/api/v1/monitoring/stats

# Get Prometheus metrics
curl http://localhost:8000/metrics
```

## Best Practices

1. **Test Failover**: Regularly test failover mechanisms
2. **Monitor Costs**: Keep track of costs across providers
3. **Set Limits**: Configure appropriate rate limits
4. **Health Checks**: Implement regular health checks
5. **Document Setup**: Document your multi-provider configuration
6. **Start Small**: Begin with a simple setup and expand as needed
7. **Review Performance**: Regularly review provider performance
8. **Use Caching**: Enable caching to reduce API calls and costs
9. **Security**: Never commit API keys to version control
10. **Provider Selection**: Choose providers based on your specific needs

## Troubleshooting

### Common Issues

1. **Provider Unavailable**: Check health checks and failover configuration
2. **High Latency**: Consider using faster providers or load balancing
3. **Cost Overruns**: Monitor costs and set budget alerts
4. **Rate Limits**: Configure appropriate rate limits and retry strategies
5. **Authentication Issues**: Verify API keys and endpoints
6. **Model Not Available**: Check provider model availability

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Check provider status
curl http://localhost:8000/v1/providers

# Test specific provider
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai_compatible:gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Test"}]
  }'
```

### Migration from OpenRouter-Specific

If you're migrating from the old OpenRouter-specific configuration:

```bash
# Old configuration (still works)
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PREFERRED_PROVIDER=openrouter

# New configuration (recommended)
OPENAI_COMPATIBLE_API_KEY=your_key
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
PREFERRED_PROVIDER=openai_compatible
```

Both configurations work simultaneously for backward compatibility.