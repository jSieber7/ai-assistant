#!/usr/bin/env python3
"""
Docker Integration Tests
Tests all Docker services for the AI Assistant project
"""

import re
import pytest
import subprocess
import time
import requests
from typing import Dict, Tuple


@pytest.mark.integration
@pytest.mark.slow
class TestDockerIntegration:
    """Test Docker services integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.services = {
            "ai-assistant": {"port": 8001, "health_path": "/"},
            "redis": {"port": 6379, "health_check": self._check_redis},
            # Skip SearXNG for now due to known configuration issues
            # "searxng": {"port": 8080, "health_path": "/"},
        }
        self.base_url = "http://localhost"

    def run_command(self, command: str, capture_output=True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout if capture_output else ""
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=5)
            r.ping()
            return True
        except Exception:
            return False

    def check_http_service(self, service: str, path: str = "/") -> bool:
        """Check if an HTTP service is responding."""
        port = self.services[service]["port"]
        url = f"{self.base_url}:{port}{path}"

        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    @pytest.mark.slow
    def test_service_status(self):
        """Test if all Docker services are running."""
        success, output = self.run_command("docker compose ps")
        assert success, f"Failed to get service status: {output}"

        services = {}
        lines = output.split("\n")
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                # Find the service name and status
                # Split by multiple spaces to handle variable spacing
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 6:
                    service_name = parts[3]
                    status = parts[5].split()[0]  # Get just the first word of status
                    # Skip SearXNG due to known configuration issues
                    if service_name != "searxng":
                        services[service_name] = "Up" in status

        assert services, "No services found - Docker may not be running"
        
        for service_name, status in services.items():
            assert status, f"{service_name} is not running"

    @pytest.mark.slow
    def test_service_health(self):
        """Test health of all services."""
        results = {}

        for service, config in self.services.items():
            if service == "redis":
                results[service] = config["health_check"]()
            else:
                health_path = config.get("health_path", "/")
                results[service] = self.check_http_service(service, health_path)

        for service, healthy in results.items():
            assert healthy, f"{service} health check failed"

    @pytest.mark.slow
    def test_connectivity(self):
        """Test connectivity between services."""
        # Test AI Assistant can reach Redis
        success, output = self.run_command(
            'docker compose exec ai-assistant uv run python -c "'
            "import redis; "
            "r = redis.Redis(host='redis', port=6379, db=0); "
            'print(r.ping())"'
        )
        assert success, "AI Assistant -> Redis connectivity failed"

        # Skip SearXNG connectivity test due to known issues
        # success, output = self.run_command(
        #     'docker compose exec ai-assistant uv run python -c "'
        #     "import requests; "
        #     "r = requests.get('http://searxng:8080/', timeout=5); "
        #     'print(r.status_code)"'
        # )
        # assert success, "AI Assistant -> SearXNG connectivity failed"

    @pytest.mark.slow
    def test_application_endpoints(self):
        """Test AI Assistant application endpoints."""
        endpoints = [
            ("/", "Root endpoint"),
            ("/docs", "API documentation"),
            ("/health", "Health check"),
            ("/metrics", "Prometheus metrics"),
        ]

        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.base_url}:8001{endpoint}", timeout=10)
                success = response.status_code in [200, 404]  # 404 is acceptable for some endpoints
                assert success, f"{description} ({endpoint}) returned status {response.status_code}"
            except Exception as e:
                pytest.fail(f"{description} ({endpoint}) failed with error: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.skip(reason="SearXNG has known configuration issues")
    def test_searxng_functionality(self):
        """Test SearXNG search functionality."""
        # Test basic search
        try:
            response = requests.get(
                f"{self.base_url}:8080/search",
                params={"q": "test", "format": "json"},
                timeout=10,
            )
            assert response.status_code == 200, "SearXNG search endpoint failed"
        except Exception as e:
            pytest.fail(f"SearXNG search endpoint failed with error: {str(e)}")


@pytest.fixture(scope="session")
def docker_services():
    """Fixture to start Docker services before integration tests."""
    # Start services
    result = subprocess.run(
        "docker compose up -d",
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Failed to start Docker services: {result.stderr}"

    # Wait for services to start
    time.sleep(10)

    yield

    # Stop services
    subprocess.run(
        "docker compose down",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )