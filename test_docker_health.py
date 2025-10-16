import requests


def check_http_service(port: int, path: str = "/") -> bool:
    """Check if an HTTP service is responding."""
    base_url = "http://localhost"
    url = f"{base_url}:{port}{path}"

    try:
        response = requests.get(url, timeout=10)
        # For health checks, accept 200 or 202 status codes
        # For other endpoints, only accept 200
        if path == "/monitoring/health":
            return response.status_code in [200, 202]
        return response.status_code == 200
    except Exception:
        return False


# Test AI Assistant service
ai_assistant_healthy = check_http_service(8000, "/monitoring/health")
print(f"AI Assistant health check: {'PASS' if ai_assistant_healthy else 'FAIL'}")

# Test SearXNG service
searxng_healthy = check_http_service(8080, "/")
print(f"SearXNG health check: {'PASS' if searxng_healthy else 'FAIL'}")
