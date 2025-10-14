# Multi-Provider Setup

This guide explains how to configure and use multiple LLM providers with the AI Assistant System.

## Overview

The AI Assistant System supports multiple LLM providers, allowing you to:
- Use different models for different tasks
- Implement failover between providers
- Load balance requests across providers
- Optimize for cost and performance

## Supported Providers

- OpenAI
- Anthropic Claude
- Ollama (local models)
- Azure OpenAI
- Custom OpenAI-compatible endpoints

## Configuration

### Environment Variables

Configure multiple providers using environment variables:

```env
# Primary provider
PRIMARY_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Secondary provider
SECONDARY_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Local provider
LOCAL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Custom provider
CUSTOM_PROVIDER_NAME=myprovider
CUSTOM_PROVIDER_API_KEY=your_custom_key
CUSTOM_PROVIDER_BASE_URL=https://api.custom-provider.com/v1
```

### Provider Configuration File

Create a `providers.yaml` configuration file:

```yaml
providers:
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL}
    models:
      - gpt-4
      - gpt-3.5-turbo
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 90000
    retry:
      max_attempts: 3
      backoff_factor: 2

  anthropic:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    models:
      - claude-3-opus-20240229
      - claude-3-sonnet-20240229
    rate_limit:
      requests_per_minute: 50
      tokens_per_minute: 100000
    retry:
      max_attempts: 3
      backoff_factor: 2

  ollama:
    type: ollama
    base_url: ${OLLAMA_BASE_URL}
    models:
      - llama2
      - codellama
      - mistral
    rate_limit:
      requests_per_minute: 30
    retry:
      max_attempts: 2
      backoff_factor: 1.5

  azure_openai:
    type: azure_openai
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}
    models:
      - gpt-4
      - gpt-35-turbo
    rate_limit:
      requests_per_minute: 120
      tokens_per_minute: 120000
    retry:
      max_attempts: 3
      backoff_factor: 2

strategies:
  default:
    provider: openai
    model: gpt-4
    fallback_providers:
      - anthropic
      - ollama

  code_generation:
    provider: anthropic
    model: claude-3-opus-20240229
    fallback_providers:
      - openai

  local_development:
    provider: ollama
    model: llama2
    fallback_providers:
      - openai

  cost_optimized:
    provider: openai
    model: gpt-3.5-turbo
    fallback_providers:
      - ollama
```

## Provider Selection Strategies

### Round Robin

Distribute requests evenly across providers:

```python
from app.core.llm_providers import ProviderManager, RoundRobinStrategy

provider_manager = ProviderManager()
strategy = RoundRobinStrategy(providers=["openai", "anthropic", "ollama"])
provider_manager.set_strategy(strategy)
```

### Weighted Round Robin

Distribute requests based on weights:

```python
from app.core.llm_providers import WeightedRoundRobinStrategy

strategy = WeightedRoundRobinStrategy(
    providers={
        "openai": 0.5,
        "anthropic": 0.3,
        "ollama": 0.2
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
    primary_provider="openai",
    fallback_providers=["anthropic", "ollama"],
    health_check_interval=60,  # seconds
    max_failures=3,
    recovery_timeout=300  # seconds
)

provider_manager.configure_failover(failover_config)
```

### Health Checks

Implement health checks for providers:

```python
from app.core.llm_providers import HealthChecker

health_checker = HealthChecker()

@health_checker.register("openai")
async def check_openai_health():
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

## Load Balancing

### Request Load Balancing

Balance requests across multiple provider instances:

```python
from app.core.llm_providers import LoadBalancer

load_balancer = LoadBalancer(
    providers=["openai", "anthropic"],
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
        "openai": {"max_tokens": 100000, "current_usage": 0},
        "anthropic": {"max_tokens": 80000, "current_usage": 0},
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

## Best Practices

1. **Test Failover**: Regularly test failover mechanisms
2. **Monitor Costs**: Keep track of costs across providers
3. **Set Limits**: Configure appropriate rate limits
4. **Health Checks**: Implement regular health checks
5. **Document Setup**: Document your multi-provider configuration
6. **Start Small**: Begin with a simple setup and expand as needed
7. **Review Performance**: Regularly review provider performance

## Troubleshooting

### Common Issues

1. **Provider Unavailable**: Check health checks and failover configuration
2. **High Latency**: Consider using faster providers or load balancing
3. **Cost Overruns**: Monitor costs and set budget alerts
4. **Rate Limits**: Configure appropriate rate limits and retry strategies

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.getLogger("app.core.llm_providers").setLevel(logging.DEBUG)