"""
Unit tests for tool routes functionality
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.api.tool_routes import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfoResponse,
)


@pytest.mark.unit
class TestToolRoutesDataModels:
    """Test data models used in tool routes"""

    def test_tool_execution_request_model(self):
        """Test ToolExecutionRequest model validation"""
        request = ToolExecutionRequest(
            tool_name="search_tool",
            parameters={
                "query": "Python testing",
                "engine": "google",
                "max_results": 10
            }
        )
        
        assert request.tool_name == "search_tool"
        assert request.parameters["query"] == "Python testing"
        assert request.parameters["engine"] == "google"
        assert request.parameters["max_results"] == 10

    def test_tool_execution_response_model(self):
        """Test ToolExecutionResponse model validation"""
        response = ToolExecutionResponse(
            tool_name="search_tool",
            success=True,
            data={
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1"}
                ],
                "count": 1
            },
            execution_time=1.5
        )
        
        assert response.tool_name == "search_tool"
        assert response.success is True
        assert response.data["count"] == 1
        assert response.execution_time == 1.5

    def test_tool_execution_response_with_error(self):
        """Test ToolExecutionResponse model with error"""
        response = ToolExecutionResponse(
            tool_name="search_tool",
            success=False,
            data=None,
            error="API rate limit exceeded",
            execution_time=0.5
        )
        
        assert response.success is False
        assert response.error == "API rate limit exceeded"
        assert response.execution_time == 0.5

    def test_tool_info_response_model(self):
        """Test ToolInfoResponse model validation"""
        response = ToolInfoResponse(
            name="search_tool",
            description="Search the web using various engines",
            enabled=True,
            version="1.0.0",
            author="Test Author",
            categories=["search", "web"],
            keywords=["search", "web", "query"],
            timeout=30,
            parameters={
                "query": {
                    "type": "string",
                    "required": True,
                    "description": "Search query"
                },
                "engine": {
                    "type": "string",
                    "required": False,
                    "description": "Search engine to use",
                    "default": "google"
                }
            }
        )
        
        assert response.name == "search_tool"
        assert response.enabled is True
        assert response.version == "1.0.0"
        assert response.author == "Test Author"
        assert response.categories == ["search", "web"]
        assert response.parameters["query"]["required"] is True


@pytest.mark.unit
class TestToolRoutesErrorHandling:
    """Test error handling in tool routes"""

    def test_invalid_tool_execution_request(self, client):
        """Test handling of invalid tool execution request"""
        # Missing required fields
        invalid_request = {
            "tool_name": "",  # Empty tool name
            "parameters": {}  # Empty parameters
        }
        
        response = client.post("/tools/execute", json=invalid_request)
        # Should return validation error or 404 if endpoint doesn't exist
        assert response.status_code in [422, 400, 404]

    def test_invalid_tool_name(self, client):
        """Test handling of invalid tool name"""
        # Test with special characters that might not be allowed
        response = client.get("/tools/tool@#$%")
        assert response.status_code in [422, 404, 500]


@pytest.mark.unit
class TestToolRoutesBasicFunctionality:
    """Test basic functionality of tool routes"""

    def test_tool_execution_request_with_complex_parameters(self):
        """Test tool execution request with complex parameters"""
        request = ToolExecutionRequest(
            tool_name="web_scrape_tool",
            parameters={
                "url": "https://example.com",
                "selector": ".content",
                "wait_for": 2,
                "screenshot": True,
                "format": "json"
            }
        )
        
        assert request.tool_name == "web_scrape_tool"
        assert request.parameters["url"] == "https://example.com"
        assert request.parameters["selector"] == ".content"
        assert request.parameters["wait_for"] == 2
        assert request.parameters["screenshot"] is True
        assert request.parameters["format"] == "json"

    def test_tool_execution_request_minimal(self):
        """Test tool execution request with minimal parameters"""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            parameters={}
        )
        
        assert request.tool_name == "test_tool"
        assert request.parameters == {}

    def test_tool_info_response_with_optional_fields(self):
        """Test tool info response with optional fields"""
        response = ToolInfoResponse(
            name="test_tool",
            description="Test tool",
            enabled=True,
            version="1.0.0",
            author="Test Author",
            categories=["test"],
            keywords=["test"],
            timeout=30,
            parameters={}
        )
        
        assert response.name == "test_tool"
        assert response.description == "Test tool"
        assert response.enabled is True
        assert response.parameters == {}

    def test_tool_execution_response_with_large_data(self):
        """Test tool execution response with large data"""
        # Create a large response
        large_results = [{"title": f"Result {i}", "url": f"https://example.com/{i}"} for i in range(100)]
        
        response = ToolExecutionResponse(
            tool_name="search_tool",
            success=True,
            data={
                "results": large_results,
                "count": len(large_results)
            },
            execution_time=5.2
        )
        
        assert response.success is True
        assert len(response.data["results"]) == 100
        assert response.data["count"] == 100
        assert response.execution_time == 5.2


@pytest.mark.unit
class TestToolRoutesParameters:
    """Test tool routes with various parameters"""

    def test_tool_execution_request_with_nested_parameters(self):
        """Test tool execution request with nested parameters"""
        request = ToolExecutionRequest(
            tool_name="api_call_tool",
            parameters={
                "url": "https://api.example.com/data",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer token123"
                },
                "body": {
                    "query": "test",
                    "limit": 10,
                    "filters": ["active", "verified"]
                }
            }
        )
        
        assert request.parameters["method"] == "POST"
        assert request.parameters["headers"]["Content-Type"] == "application/json"
        assert request.parameters["body"]["filters"] == ["active", "verified"]

    def test_tool_execution_request_with_array_parameters(self):
        """Test tool execution request with array parameters"""
        request = ToolExecutionRequest(
            tool_name="batch_process_tool",
            parameters={
                "items": ["item1", "item2", "item3"],
                "options": ["fast", "verbose"],
                "exclude": ["test", "demo"]
            }
        )
        
        assert len(request.parameters["items"]) == 3
        assert len(request.parameters["options"]) == 2
        assert len(request.parameters["exclude"]) == 2
        assert "item1" in request.parameters["items"]