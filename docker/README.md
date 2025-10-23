# Throwaway Dockers 2 - Integrated Development Stack

A powerful, Docker-based development stack that combines a web scraper, a privacy-respecting search engine, and a full-featured Backend-as-a-Service. This project is designed for rapid development and testing of modern web applications.

## 🚀 Quick Start

```bash
# Start all services with comprehensive setup
./run.sh

# Quick start (basic setup)
./start.sh

# Quick status check
./quick-test.sh

# Stop all services
docker compose down
```

## 📚 Documentation

For complete documentation, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 🌐 Access Points

| Service             | Description                                                    | URL                                      |
|---------------------|----------------------------------------------------------------|------------------------------------------|
| Traefik Dashboard   | Monitor and configure the reverse proxy                         | `http://traefik.localhost:8080`           |
| Supabase Studio     | Manage your Supabase database, authentication, and storage       | `http://supabase.localhost`               |
| Firecrawl API       | Access the web scraping API                                      | `http://firecrawl.localhost:3002`         |
| SearXNG             | Use the privacy-respecting search engine                         | `http://searxng.localhost`                |

## 🏗️ Architecture Overview

This stack uses a centralized Traefik reverse proxy to provide unified access to all services. The architecture is built around a separation of concerns, with each major service having its own dedicated database to ensure stability and avoid conflicts.

```
┌─────────────────────────────────────────────────────────────────┐
│                          Internet                                │
└─────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Traefik (v3)                             │
│                 Central Reverse Proxy                           │
└─────────────────────┬───────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  SearXNG    │ │  Firecrawl  │ │ Supabase    │
│  (8080)     │ │  (3002)     │ │ Studio      │
│             │ │             │ │ (3000)      │
└─────────────┘ └─────────────┘ └─────────────┘
         │             │             │
         └─────────────┼─────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Shared Services                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │    Redis    │  │ PostgreSQL  │  │ PostgreSQL  │           │
│  │   (6379)    │  │   (5432)    │  │   (5433)    │           │
│  │             │  │   Firecrawl │  │   Supabase  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Core Services

*   **Traefik:** Central reverse proxy and load balancer for routing traffic to all services.
*   **Supabase:** An open-source Backend-as-a-Service providing a PostgreSQL database, authentication, storage, and real-time capabilities.
*   **Firecrawl:** A web scraping service that can crawl and extract data from any website.
*   **SearXNG:** A privacy-respecting search engine that aggregates results from various sources.
*   **Redis:** A shared in-memory data store used for caching and rate limiting.

## 📁 Project Structure

```
/
├── .env                              # Shared infrastructure variables
├── docker-compose.yml                # Main compose file orchestrating all services
├── start.sh                          # Script to start the entire stack
├── run.sh                            # Complete setup runner script
├── test.sh                           # Comprehensive test suite
├── quick-test.sh                     # Quick status check
├── supabase/                         # Supabase configuration and volumes
│   ├── .env                          # Supabase environment variables
│   ├── docker-compose.supabase.yml  # Supabase service definitions
│   └── volumes/                      # Persistent data for Supabase services
├── firecrawl/                        # Firecrawl configuration
│   └── docker-compose.firecrawl.yml # Standalone Firecrawl compose file
├── searxng/                          # SearXNG configuration
│   └── searxng/settings.yml         # SearXNG settings
├── traefik/                          # Traefik configuration
│   └── traefik.yml                   # Traefik static configuration
├── env/                              # Service-specific environment files
│   ├── firecrawl.env                 # Firecrawl-specific variables
│   ├── searxng.env                   # SearXNG-specific variables
│   └── supabase.env.example          # Example Supabase configuration
└── SETUP_GUIDE.md                    # Detailed guide for management and troubleshooting
```

## 🔧 Configuration

The project uses several environment files for configuration. Ensure they are present and configured:
*   `.env`: Shared infrastructure variables.
*   `supabase/.env`: Supabase-specific configuration (database passwords, JWT secrets, etc.).
*   `env/firecrawl.env`: Firecrawl-specific configuration (API keys, etc.).
*   `env/searxng.env`: SearXNG-specific configuration.

### Shared Variables (.env)
- `REDIS_URL`: Redis connection URL
- `TRAEFIK_DASHBOARD_PORT`: Traefik dashboard port (default: 8080)
- `TRAEFIK_HTTP_PORT`: HTTP port (default: 80)
- `BASE_DOMAIN`: Base domain for services (default: localhost)

### Service-Specific Variables
- `supabase/.env`: Supabase database passwords, JWT secrets, and API keys
- `env/firecrawl.env`: Firecrawl API keys, database URLs, and configuration
- `env/searxng.env`: SearXNG hostname and SSL configuration

### Suppressing Warnings
All environment variables have sensible defaults to suppress Docker Compose warnings. The docker-compose.yml file uses the `${VAR:-default}` syntax to provide fallback values. You can override these defaults by setting values in the appropriate .env files.

## 🗄️ Database Integration

The project successfully integrates two distinct PostgreSQL databases:
1.  **Supabase Database:** Hosted on port `5432`, used for application data (user management, persistent storage, etc.).
2.  **Firecrawl Database:** Hosted on port `5433`, used internally by Firecrawl for managing crawl jobs and queues.

This separation ensures that the operational data of your tools does not interfere with your core application data.

## 🔄 Management Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Restart specific service
docker compose restart firecrawl-api

# View service status
docker compose ps
```

## 🌐 Networking

All services communicate through the `shared-network` bridge network. Internal services (like Redis and PostgreSQL) are not exposed directly through Traefik for security.

## 💾 Volumes

- `redis-data`: Redis persistence
- `traefik-logs`: Traefik log files
- `postgres-data`: PostgreSQL data persistence (Firecrawl)
- `searxng-data`: SearXNG cache
- `supabase/volumes`: Supabase persistent data

## 🚀 Development

To add a new service:

1. Create a service-specific .env file in the `env/` directory
2. Add the service definition to `docker-compose.yml`
3. Configure Traefik labels for routing
4. Update this README

## 🔍 Troubleshooting

1. **Port Conflicts**: Check if ports 80, 8080, 3002, 5432, 5433, and 6379 are available
2. **Environment Variables**: Ensure all required variables are set in the appropriate .env files
3. **Service Dependencies**: Services that depend on Redis will wait for it to start automatically
4. **PostgreSQL Service**: Uses separate databases for Supabase and Firecrawl to avoid conflicts

## 🔒 Security Notes

- Redis is not exposed directly through Traefik
- Database connections are internal to the Docker network
- API keys and sensitive data should be set in the appropriate .env files
- Consider using Docker Secrets for production environments

## ✨ Key Features

*   **Unified Stack:** All services are managed and started together with a single command.
*   **Conflict-Free Databases:** Supabase and Firecrawl run on separate, non-conflicting databases.
*   **Centralized Routing:** Traefik provides clean, subdomain-based access to all services.
*   **Shared Infrastructure:** Redis is shared across services for efficient caching and rate limiting.
*   **Developer-Friendly:** Easy to start, stop, and manage the entire development environment.

## 📖 Further Documentation

For detailed information on management commands, testing, troubleshooting, and service configuration, please refer to the comprehensive [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Future plans

Integrate React App
```
To route a React app through Traefik, you would treat it as another service in your Docker stack.

Create a Dockerfile for your React app. For development, this would typically be a simple Node.js image that runs npm start. For production, a multi-stage build that creates static assets and serves them with Nginx is best practice.

Add a service definition to your docker-compose.yml. This service would build your React app and include Traefik labels to route traffic. For example:

services:
  react-app:
    build: ./react-app
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.react-app.rule=Host(`app.localhost`)"
      - "traefik.http.services.react-app.loadbalancer.server.port=3000"
    networks:
      - shared-network

This configuration routes traffic from app.localhost to port 3000 of the React container.

Update run.sh (Optional but Recommended). You could add the react-app service to the run.sh script's health checks and status reporting, just like was done for the FastAPI app.

The core principle is that Traefik acts as a traffic cop, using the Host() rule in the labels to direct incoming requests to the correct container based on the requested subdomain.
```


# Hybrid Docker Compose Setup - Complete Guide

This guide provides comprehensive documentation for the hybrid Docker Compose setup with centralized Traefik reverse proxy and shared services.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Internet                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Traefik (v3)                             │
│                 Central Reverse Proxy                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  SearXNG    │ │  Firecrawl  │ │   Traefik   │
│  (8080)     │ │  (3002)     │ │ Dashboard   │
│             │ │             │ │  (8080)     │
└─────────────┘ └─────────────┘ └─────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Shared Services                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │    Redis    │  │ PostgreSQL  │  │  Playwright │           │
│  │   (6379)    │  │   (5432)    │  │   (3000)    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### Core Infrastructure
- **Traefik**: Central reverse proxy and load balancer (v3)
  - Dashboard: http://traefik.localhost:8080
  - API: http://traefik.localhost:8080/api

### Shared Services
- **Redis**: Shared caching and rate limiting
  - Database 0: Firecrawl rate limiting
  - Database 1: SearXNG caching
- **PostgreSQL**: Database for Firecrawl
  - Hostname: nuq-postgres

### Applications
- **Firecrawl**: Web scraping service
  - API: http://firecrawl.localhost
  - Playwright service: Internal (port 3000)
- **SearXNG**: Privacy-respecting search engine
  - Web interface: http://searxng.localhost

## File Structure

```
/
├── .env                    # Shared infrastructure variables
├── docker-compose.yml      # Main compose file with all services
├── env/                    # Service-specific environment files
│   ├── firecrawl.env       # Firecrawl-specific variables
│   └── searxng.env         # SearXNG-specific variables
├── traefik/                # Traefik configuration
│   └── traefik.yml
├── searxng/                # SearXNG configuration
│   └── searxng/settings.yml
├── run.sh                  # Complete setup runner script
├── test.sh                 # Comprehensive test suite
├── quick-test.sh           # Quick status check
└── SETUP_GUIDE.md          # This documentation
```

## Quick Start

### 1. Initial Setup
```bash
# Clone or download the project
cd hybrid-docker-compose

# Make scripts executable
chmod +x run.sh test.sh quick-test.sh
```

### 2. Start All Services
```bash
# Run the complete setup script
./run.sh

# Or start manually
docker compose up -d
```

### 3. Verify Setup
```bash
# Quick status check
./quick-test.sh

# Comprehensive testing
./test.sh
```

### 4. Access Services
- Traefik Dashboard: http://traefik.localhost:8080
- Firecrawl API: http://firecrawl.localhost
- SearXNG: http://searxng.localhost

## Configuration

### Environment Variables

#### Shared Variables (.env)
```bash
# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_RATE_LIMIT_URL=redis://redis:6379/0

# Traefik Configuration
TRAEFIK_DASHBOARD_PORT=8080
TRAEFIK_HTTP_PORT=80
TRAEFIK_VERSION=v3

# Network Configuration
NETWORK_NAME=shared-network

# Common Settings
LOG_LEVEL=info
COMPOSE_PROJECT_NAME=my-stack

# Domain Configuration
BASE_DOMAIN=localhost
```

#### Service-Specific Variables
- `env/firecrawl.env`: Firecrawl API keys, database URLs, and configuration
- `env/searxng.env`: SearXNG hostname and SSL configuration

### Traefik Configuration

The Traefik configuration (`traefik/traefik.yml`) includes:
- Docker provider with automatic service discovery
- API and dashboard configuration
- Entry points for HTTP and HTTPS
- Default routing rule for `.localhost` domains

## Service Configuration

#### Firecrawl
- Uses shared Redis for rate limiting and caching
- Connects to PostgreSQL via hostname `nuq-postgres`
- Requires API keys for AI services (OpenAI, etc.)

#### SearXNG
- Uses shared Redis (database 1) for caching
- Configured to work directly through Traefik
- No Caddy proxy needed

## Service Access: Internal vs External

### Internal Service Access (Docker-to-Docker)

When applications running inside Docker containers need to access other services, use the internal service names:

```python
# Python example accessing Firecrawl from another container
import requests

# Use service name:port format
response = requests.post("http://firecrawl-api:3002/v1/scrape", json={
    "url": "https://example.com"
})
```

**Available Internal Services:**
- Firecrawl API: `firecrawl-api:3002`
- Redis: `redis:6379`
- PostgreSQL: `nuq-postgres:5432`
- SearXNG: `searxng:8080`
- Playwright: `firecrawl-playwright:3000`

### External Service Access (Outside Docker)

When accessing services from outside Docker (e.g., from your host machine), use the external URLs through Traefik:

```python
# Python example accessing Firecrawl from host machine
import requests

# Use external URL through Traefik
response = requests.post("http://firecrawl.localhost/v1/scrape", json={
    "url": "https://example.com"
})
```

**Available External URLs:**
- Firecrawl API: `http://firecrawl.localhost`
- SearXNG: `http://searxng.localhost`
- Traefik Dashboard: `http://traefik.localhost:8080`

### Why Use Internal Service Names?

1. **Performance**: Direct container-to-container communication is faster
2. **Reliability**: Doesn't depend on Traefik being up
3. **Security**: Stays within the Docker network
4. **No Port Conflicts**: Uses internal ports instead of external ones

### Example: Adding a Python Application

To add a Python application that uses Firecrawl:

```yaml
# Add to docker-compose.yml
  python-app:
    image: your-python-app
    container_name: python-app
    networks:
      - shared-network
    environment:
      FIRECRAWL_URL: http://firecrawl-api:3002
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - firecrawl-api
      - redis
```

Then in your Python code:
```python
import os
import requests

firecrawl_url = os.getenv("FIRECRAWL_URL", "http://firecrawl-api:3002")

def scrape_url(url):
    response = requests.post(f"{firecrawl_url}/v1/scrape", json={"url": url})
    return response.json()
```

## Management Commands

### Starting and Stopping
```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Viewing Logs
```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f traefik
docker compose logs -f firecrawl-api
docker compose logs -f searxng
```

### Managing Services
```bash
# Restart specific service
docker compose restart firecrawl-api

# Update service
docker compose pull firecrawl-api
docker compose up -d firecrawl-api
```

## Testing

### Quick Test
```bash
./quick-test.sh
```
Checks:
- Container status
- Service accessibility through Traefik
- Internal service health
- Traefik route configuration

### Comprehensive Test
```bash
./test.sh
```
Tests:
- Docker connectivity
- Service health
- HTTP endpoints
- Network connectivity
- Service integration
- Response times

## Troubleshooting

### Common Issues

1. **Firecrawl API restarting**
   - Cause: Missing database tables
   - Solution: Wait for initial setup or check logs

2. **Service not accessible through Traefik**
   - Cause: Container restarting frequently
   - Solution: Check service logs and fix underlying issues

3. **Port conflicts**
   - Cause: Ports already in use
   - Solution: Check port availability with `netstat -tulpn`

4. **Environment variable warnings**
   - Cause: Missing variables in .env files
   - Solution: Copy from .example files and configure

### Debug Commands

```bash
# Check container status
docker compose ps

# Check service logs
docker compose logs [service-name]

# Check Traefik routes
curl http://traefik.localhost:8080/api/http/routers

# Test Redis connection
docker exec shared-redis redis-cli ping

# Test PostgreSQL connection
docker exec firecrawl-postgres pg_isready -U postgres
```

## Security Considerations

1. **Environment Variables**
   - Never commit actual .env files to version control
   - Use .env.example files as templates
   - Consider using Docker Secrets for production

2. **Network Security**
   - Internal services not exposed directly
   - Only Traefik exposed to external traffic
   - Use shared network for service communication

3. **Default Credentials**
   - Change PostgreSQL default password
   - Configure proper authentication for services

## Performance Optimization

1. **Resource Limits**
   - Configure memory limits for containers
   - Monitor resource usage

2. **Redis Optimization**
   - Use separate databases for different services
   - Configure appropriate eviction policies

3. **Traefik Optimization**
   - Configure proper timeouts
   - Enable caching where appropriate

## Backup and Recovery

1. **Volume Backups**
   ```bash
   # Backup volumes
   docker run --rm -v my-stack_redis-data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz -C /data .
   docker run --rm -v my-stack_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz -C /data .
   ```

2. **Configuration Backups**
   - Backup .env files
   - Backup docker-compose.yml
   - Backup Traefik configuration

## Production Deployment

For production deployment:

1. **Environment Variables**
   - Use proper domain names
   - Configure SSL certificates
   - Set strong passwords

2. **Scaling**
   - Consider using Docker Swarm or Kubernetes
   - Configure load balancing
   - Implement health checks

3. **Monitoring**
   - Set up log aggregation
   - Configure metrics collection
   - Implement alerting

## Support

For issues:
1. Check the troubleshooting section
2. Review service logs
3. Run the test scripts
4. Check the GitHub issues (if applicable)