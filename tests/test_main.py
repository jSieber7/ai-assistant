import pytest
from fastapi.testclient import TestClient


class TestMainEndpoints:
    """Test basic application endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "LangChain Agent Hub is running!"
        assert data["version"] == "0.1.0"
        assert data["status"] == "ready"
    
    def test_health_endpoint(self, client: TestClient):
        """Test the health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "langchain-agent-hub"
    
    def test_models_endpoint(self, client: TestClient):
        """Test the models endpoint returns correct model list."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        
        model_data = data["data"][0]
        assert model_data["id"] == "langchain-agent-hub"
        assert model_data["object"] == "model"
        assert model_data["owned_by"] == "langchain-agent-hub"
        assert model_data["root"] == "langchain-agent-hub"
        assert model_data["parent"] is None
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are properly set."""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # CORS preflight should return 200
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        # When allow_credentials=True, the origin is echoed back instead of "*"
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    
    def test_nonexistent_endpoint(self, client: TestClient):
        """Test that nonexistent endpoints return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestErrorScenarios:
    """Test error scenarios for core endpoints."""
    
    def test_invalid_http_method(self, client: TestClient):
        """Test that invalid HTTP methods return proper status codes."""
        # POST to GET-only endpoint
        response = client.post("/")
        assert response.status_code == 405  # Method Not Allowed
        
        # PUT to health endpoint
        response = client.put("/health")
        assert response.status_code == 405
    
    def test_malformed_requests(self, client: TestClient):
        """Test handling of malformed requests."""
        # Send invalid JSON to endpoints that expect JSON
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoint functionality."""
    
    async def test_async_root_endpoint(self, client: TestClient):
        """Test root endpoint works correctly in async context."""
        response = client.get("/")
        assert response.status_code == 200
        assert "version" in response.json()
    
    async def test_async_health_endpoint(self, client: TestClient):
        """Test health endpoint works correctly in async context."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"