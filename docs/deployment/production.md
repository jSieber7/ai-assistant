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

## Production Architecture

A complete production deployment includes:

- **Traefik Reverse Proxy**: Load balancing and SSL termination
- **AI Assistant Application**: Multiple instances with tool and agent systems
- **Redis Cluster**: High-performance caching and session storage
- **SearXNG Search**: Privacy-focused web search capabilities
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
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Production

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/ai_assistant
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      - db
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_assistant
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
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
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
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
APP_ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:password@db:5432/ai_assistant
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Security
SECRET_KEY=your-super-secret-key
API_KEY_HEADER=X-API-Key
CORS_ORIGINS=https://yourdomain.com

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## Security Best Practices

1. **API Keys**: Use environment variables for sensitive data
2. **HTTPS**: Enable TLS/SSL for all communications
3. **Network Policies**: Implement network restrictions
4. **Container Security**: Use non-root users and minimal images
5. **Secrets Management**: Use Kubernetes secrets or secret managers
6. **Regular Updates**: Keep dependencies updated

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

## Performance Optimization

1. **Caching**: Implement multiple cache layers
2. **Connection Pooling**: Use connection pools for databases
3. **Async Operations**: Use async/await for I/O operations
4. **Load Testing**: Regular performance testing
5. **Resource Limits**: Set appropriate resource limits

## Scaling

### Horizontal Scaling

- Use load balancers to distribute traffic
- Implement auto-scaling based on metrics
- Consider stateless design for easier scaling

### Vertical Scaling

- Monitor resource utilization
- Increase resources as needed
- Consider resource partitioning