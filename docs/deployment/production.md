# Production Deployment Guide

This comprehensive guide covers deploying the AI Assistant System to production environments with best practices for security, scalability, monitoring, and reliability.

## Overview

Production deployment of the AI Assistant System requires careful consideration of:

- **High Availability**: Multiple instances with load balancing
- **Security**: TLS/SSL, API authentication, network policies
- **Scalability**: Horizontal scaling with auto-scaling capabilities
- **Monitoring**: Comprehensive observability with Prometheus and Grafana
- **Performance**: Optimized caching, connection pooling, and resource management
- **Reliability**: Health checks, graceful degradation, and failover
- **Data Privacy**: Secure handling of API keys and user data

## Production Architecture

A complete production deployment includes:

- **Traefik Reverse Proxy**: Load balancing and SSL termination
- **AI Assistant Application**: Multiple instances with tool and agent systems
- **Redis Cluster**: High-performance caching and session storage
- **SearXNG Search**: Privacy-focused web search capabilities
- **Firecrawl**: Advanced web scraping capabilities
- **Jina Reranker**: Search result reranking
- **MongoDB**: Multi-writer system data storage
- **Milvus**: Vector storage for RAG capabilities
- **PostgreSQL**: Optional database for persistent storage
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting dashboards
- **Centralized Logging**: Log aggregation and analysis

## Docker Deployment

### Production Dockerfile

```dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Production

```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v3.0
    command:
      - "--api.dashboard=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=admin@example.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./letsencrypt:/letsencrypt
    restart: unless-stopped

  app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ai_assistant
      - REDIS_URL=redis://redis:6379/0
      - MONGODB_URL=mongodb://mongo:27017/ai_assistant
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
      - SECRET_KEY=${SECRET_KEY}
      - OPENAI_COMPATIBLE_API_KEY=${OPENAI_COMPATIBLE_API_KEY}
      - BEHIND_PROXY=true
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.app.rule=Host(`yourdomain.com`)"
      - "traefik.http.routers.app.entrypoints=websecure"
      - "traefik.http.routers.app.tls.certresolver=myresolver"
      - "traefik.http.services.app.loadbalancer.server.port=8000"
    depends_on:
      - postgres
      - redis
      - mongo
      - searxng
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_assistant
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  mongo:
    image: mongo:7
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_ROOT_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD}
      - MONGO_INITDB_DATABASE=ai_assistant
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

  searxng:
    image: searxng/searxng:latest
    environment:
      - SEARXNG_SECRET_KEY=${SEARXNG_SECRET_KEY}
    volumes:
      - ./docker-configs/searxng:/etc/searxng
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  firecrawl-api:
    image: mendable/firecrawl:latest
    environment:
      - REDIS_URL=redis://firecrawl-redis:6379
      - PORT=3002
      - BULL_AUTH_KEY=${FIRECRAWL_BULL_AUTH_KEY}
    depends_on:
      - firecrawl-redis
      - firecrawl-postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

  firecrawl-redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - firecrawl_redis_data:/data
    restart: unless-stopped

  firecrawl-postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=firecrawl
      - POSTGRES_USER=firecrawl
      - POSTGRES_PASSWORD=${FIRECRAWL_POSTGRES_PASSWORD}
    volumes:
      - firecrawl_postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  jina-reranker:
    build:
      context: ./docker-configs/jina-reranker
    environment:
      - MODEL_NAME=jina-reranker-v1-base-en
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker-configs/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker-configs/monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mongo_data:
  firecrawl_redis_data:
  firecrawl_postgres_data:
  prometheus_data:
  grafana_data:
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-assistant
  labels:
    app: ai-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-assistant
  template:
    metadata:
      labels:
        app: ai-assistant
    spec:
      containers:
      - name: ai-assistant
        image: your-registry/ai-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-assistant-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ai-assistant-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-assistant-service
spec:
  selector:
    app: ai-assistant
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-assistant-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - ai-assistant.example.com
    secretName: ai-assistant-tls
  rules:
  - host: ai-assistant.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-assistant-service
            port:
              number: 80
```

## Environment Configuration

### Production Environment Variables

```env
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-super-secret-key-here
BEHIND_PROXY=true
CORS_ORIGINS=https://yourdomain.com

# LLM Provider
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=anthropic/claude-3.5-sonnet
PREFERRED_PROVIDER=openai_compatible
ENABLE_PROVIDER_FALLBACK=true

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/ai_assistant
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://redis:6379/0
REDIS_CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_COMPRESSION=true

# MongoDB (for multi-writer)
MONGODB_URL=mongodb://mongo:27017/ai_assistant
MULTI_WRITER_ENABLED=false  # Enable if using multi-writer

# Tool Configuration
TOOL_CALLING_ENABLED=true
MAX_TOOLS_PER_QUERY=3
TOOL_EXECUTION_TIMEOUT=30

# Search and Scraping
SEARXNG_URL=http://searxng:8080
SEARXNG_TOOL_ENABLED=true
FIRECRAWL_DEPLOYMENT_MODE=docker
FIRECRAWL_DOCKER_URL=http://firecrawl-api:3002
FIRECRAWL_TOOL_ENABLED=false  # Enable if needed
JINA_RERANKER_URL=http://jina-reranker:8000
JINA_RERANKER_TOOL_ENABLED=false  # Enable if needed

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_COLLECTION_ENABLED=true

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Gradio
GRADIO_ENABLED=true
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
```

## Security Best Practices

1. **API Keys**: Use environment variables for sensitive data, never commit to version control
2. **HTTPS**: Enable TLS/SSL for all communications with automatic certificate management
3. **Network Policies**: Implement network restrictions and firewall rules
4. **Container Security**: Use non-root users, minimal images, and security scanning
5. **Secrets Management**: Use Kubernetes secrets, HashiCorp Vault, or AWS Secrets Manager
6. **Regular Updates**: Keep dependencies updated with automated security scanning
7. **Input Validation**: Validate all inputs and sanitize user data
8. **Rate Limiting**: Implement rate limiting to prevent abuse
9. **Access Control**: Implement proper authentication and authorization
10. **Audit Logging**: Log all access and modifications for security auditing

## Monitoring and Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response
```

### Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${DATE}.sql"

pg_dump $DATABASE_URL > $BACKUP_FILE

# Upload to cloud storage
aws s3 cp $BACKUP_FILE s3://your-backup-bucket/

# Clean up old backups (keep last 7 days)
find /backups -name "backup_*.sql" -mtime +7 -delete
```

### Disaster Recovery

1. **Regular Backups**: Automated daily backups
2. **Multi-Region**: Deploy across multiple regions
3. **Failover**: Implement automatic failover
4. **Testing**: Regular disaster recovery tests

## Monitoring and Observability

### Key Metrics to Monitor

- **Application Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk, network usage
- **Database Metrics**: Connection pool, query performance
- **Cache Metrics**: Hit/miss ratio, eviction rate
- **Tool Metrics**: Tool execution time, success rate
- **Provider Metrics**: API calls, response time, error rate

### Alerting Strategies

1. **Threshold Alerts**: Alert when metrics exceed thresholds
2. **Rate of Change Alerts**: Alert on rapid metric changes
3. **Anomaly Detection**: Use ML-based anomaly detection
4. **Multi-Metric Alerts**: Combine multiple metrics for alerts
5. **Escalation Policies**: Implement proper alert escalation

### Log Management

1. **Structured Logging**: Use JSON format for logs
2. **Log Aggregation**: Centralize logs with ELK or similar
3. **Log Retention**: Implement proper log retention policies
4. **Log Analysis**: Use tools for log analysis and debugging
5. **Security Logging**: Log security events and access

## Scaling

### Horizontal Scaling

- Use load balancers to distribute traffic
- Implement auto-scaling based on metrics
- Consider stateless design for easier scaling

### Vertical Scaling

- Monitor resource utilization
- Increase resources as needed
- Consider resource partitioning