# Docker Development Environment

This directory contains a complete, containerized development environment, including the application, frontend, and supporting infrastructure services like a database (Supabase), a search engine (SearXNG), and a web crawler (Firecrawl).

## 🚨 Important: Required Project Structure

This Docker setup is designed to work with a specific project structure. **It will only function correctly if your repository is organized as follows:**

```
.
├── app/                  # Your FastAPI/Python application source code
├── docker/               # This directory, containing all Docker configuration
├── frontend/             # Your React/Vue/etc. frontend application source code
├── pyproject.toml        # Python project dependencies for the `app` service
└── uv.lock               # Locked Python dependencies for the `app` service
```

The `docker/` directory must be a sibling of the `app/` and `frontend/` directories at the project root.

## Prerequisites

Before you begin, ensure you have the following installed:
- [Docker](https://www.docker.com/get-started/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) (Python package manager)

## Quick Start

1.  **Configure Environment Variables**:
    Copy the example environment files for each service and customize them as needed.
    ```bash
    # For the app service
    cp docker/app/example.env docker/app/dev.env
    cp docker/app/example.env docker/app/prod.env

    # For the frontend service
    cp docker/frontend/env/example.env docker/frontend/env/dev.env
    cp docker/frontend/env/example.env docker/frontend/env/prod.env

    # For other services (supabase, firecrawl, searxng)
    cp docker/supabase/env/example.env docker/supabase/env/dev.env
    cp docker/firecrawl/env/example.env docker/firecrawl/env/dev.env
    cp docker/searxng/env/example.env docker/searxng/env/dev.env
    ```
    Edit these files to set passwords, API keys, and other configurations.

2.  **Run the Management Script**:
    Use the provided `run_dockers.py` script to manage the services. It's a powerful tool that simplifies interactions with Docker Compose.

    To start all services in development mode:
    ```bash
    uv run docker/run_dockers.py
    ```

3.  **Access the Services**:
    Once the services are running, you can access them at the following URLs (assuming `BASE_DOMAIN=localhost` in your `.env` files):
    - **Frontend App**: http://frontend.localhost
    - **Backend App**: http://app.localhost
    - **Supabase Studio**: http://supabase.localhost
    - **Firecrawl API**: http://firecrawl.localhost
    - **SearXNG Search**: http://searxng.localhost
    - **Traefik Dashboard**: http://traefik.localhost

## Service Management with `run_dockers.py`

The `run_dockers.py` script in this directory is your primary tool for managing the Docker environment.

### Basic Usage

The general command structure is:
`uv run docker/run_dockers.py [service] [environment] [command] [options]`

- **service**: `all` (default), `app`, `frontend`, `supabase`, `firecrawl`, `searxng`
- **environment**: `dev` (default), `prod`
- **command**: `up` (default), `down`, `logs`, `status`, `reset`, `test`, `build`
- **options**: `-f` (foreground), `--no-cache` (for build)

### Common Commands

**Start all services in the background (detached mode):**
```bash
uv run docker/run_dockers.py
```
or
```bash
uv run docker/run_dockers.py all dev up
```

**Start a single service (e.g., the app) in the foreground to see logs:**
```bash
uv run docker/run_dockers.py app dev up -f
```

**View logs for a running service:**
```bash
uv run docker/run_dockers.py frontend dev logs
```

**Stop a service:**
```bash
uv run docker/run_dockers.py firecrawl dev down
```

**Stop all services:**
```bash
uv run docker/run_dockers.py all dev down
```

**Rebuild a service (useful after changing code or Dockerfile):**
```bash
uv run docker/run_dockers.py app dev build
```

**Rebuild all services without cache:**
```bash
uv run docker/run_dockers.py all dev build --no-cache
```

**Check the status of all services:**
```bash
uv run docker/run_dockers.py all dev status
```

**Reset a service's data (⚠️ This will delete all data for that service):**
```bash
uv run docker/run_dockers.py supabase dev reset
```

For more help, run:
```bash
uv run docker/run_dockers.py --help
```

## Service Overview

- **Traefik**: A reverse proxy and load balancer that routes traffic to your services and handles SSL/TLS termination.
- **Redis**: A shared in-memory key-value store, used for caching by other services.
- **App**: Your Python/FastAPI application.
- **Frontend**: Your frontend application (e.g., React, Vue).
- **Supabase**: An open-source Firebase alternative. Provides a Postgres database, authentication, storage, and more.
- **Firecrawl**: A web scraping and data extraction API.
- **SearXNG**: A privacy-respecting, hackable metasearch engine.

## Development Workflow

1.  Code changes in the `app/` directory will require a rebuild of the `app` service.
    ```bash
    uv run docker/run_dockers.py app dev build
    ```
2.  Code changes in the `frontend/` directory are automatically picked up in development mode due to the volume mount. No rebuild is needed.
3.  Configuration changes in `.env` files require restarting the affected service.
    ```bash
    uv run docker/run_dockers.py [service] dev down
    uv run docker/run_dockers.py [service] dev up
    ```

## Troubleshooting

- **Port conflicts**: If you get errors about ports being in use (e.g., port 80, 443, 5432), check your `.env` files and change the port settings (e.g., `TRAEFIK_HTTP_PORT`, `POSTGRES_PORT`).
- **Build fails**: Ensure your `pyproject.toml`, `uv.lock`, and source code directories (`app/`, `frontend/`) are in the correct locations at the project root.
- **`uv run docker/run_dockers.py` command not found**: Make sure you are running the command from the project root directory, not from inside the `docker/` directory.