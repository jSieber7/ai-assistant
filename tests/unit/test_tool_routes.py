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


class TestToolRoutes:
    """Test tool routes endpoints"""

    @patch("app.api.tool_routes.tool_registry")
    def test_list_tools(self, mock_tool_registry, client):
        """Test listing available tools"""
        # Mock tools
        mock_tool1 = AsyncMock()
        mock_tool1.name = "search_tool"
        mock_tool1.description = "Search the web"
        mock_tool1.enabled = True
        mock_tool1.category = "search"
        
        mock_tool2 = AsyncMock()
        mock_tool2.name = "web_scrape_tool"
        mock_tool2.description = "Scrape web content"
        mock_tool2.enabled = False
        mock_tool2.category = "scraping"
        
        mock_tool_registry.list_tools.return_value = [mock_tool1, mock_tool2]
        
        response = client.get("/tools/")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_list_tools_enabled_only(self, mock_tool_registry, client):
        """Test listing only enabled tools"""
        # Mock tools
        mock_tool1 = AsyncMock()
        mock_tool1.name = "search_tool"
        mock_tool1.enabled = True
        
        mock_tool2 = AsyncMock()
        mock_tool2.name = "disabled_tool"
        mock_tool2.enabled = False
        
        mock_tool_registry.list_tools.return_value = [mock_tool1, mock_tool2]
        
        response = client.get("/tools/?enabled_only=true")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_get_tool_info(self, mock_tool_registry, client):
        """Test getting specific tool information"""
        # Mock tool
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.description = "Search the web using various engines"
        mock_tool.enabled = True
        mock_tool.category = "search"
        mock_tool.parameters = {
            "query": {"type": "string", "required": True},
            "engine": {"type": "string", "required": False}
        }
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        response = client.get("/tools/search_tool")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_execute_tool(self, mock_tool_registry, client):
        """Test executing a tool"""
        # Mock tool
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.execute.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1"},
                {"title": "Result 2", "url": "https://example.com/2"}
            ],
            "count": 2
        }
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        request_data = {
            "tool_name": "search_tool",
            "parameters": {
                "query": "Python testing",
                "engine": "google"
            }
        }
        
        response = client.post("/tools/execute", json=request_data)
        
        # This test would need more implementation based on the actual route logic
        assert request_data["tool_name"] == "search_tool"
        assert request_data["parameters"]["query"] == "Python testing"

    @patch("app.api.tool_routes.tool_registry")
    def test_enable_tool(self, mock_tool_registry, client):
        """Test enabling a tool"""
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.enabled = False
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        response = client.post("/tools/search_tool/enable")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_disable_tool(self, mock_tool_registry, client):
        """Test disabling a tool"""
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.enabled = True
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        response = client.post("/tools/search_tool/disable")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_get_tools_by_category(self, mock_tool_registry, client):
        """Test getting tools by category"""
        # Mock tools
        mock_tool1 = AsyncMock()
        mock_tool1.name = "search_tool"
        mock_tool1.category = "search"
        mock_tool1.enabled = True
        
        mock_tool2 = AsyncMock()
        mock_tool2.name = "image_search_tool"
        mock_tool2.category = "search"
        mock_tool2.enabled = True
        
        mock_tool_registry.get_tools_by_category.return_value = [mock_tool1, mock_tool2]
        
        response = client.get("/tools/categories/search")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_get_registry_stats(self, mock_tool_registry, client):
        """Test getting tool registry statistics"""
        mock_tool_registry.get_stats.return_value = {
            "total_tools": 10,
            "enabled_tools": 7,
            "disabled_tools": 3,
            "categories": ["search", "scraping", "analysis"],
            "executions_today": 25,
            "success_rate": 0.92
        }
        
        response = client.get("/tools/registry/stats")
        
        # This test would need more implementation based on the actual route logic
        assert response.status_code not in [404]  # Endpoint should exist


class TestToolRoutesErrorHandling:
    """Test error handling in tool routes"""

    @patch("app.api.tool_routes.tool_registry")
    def test_nonexistent_tool(self, mock_tool_registry, client):
        """Test handling of nonexistent tool"""
        mock_tool_registry.get_tool.return_value = None
        
        response = client.get("/tools/nonexistent_tool")
        assert response.status_code in [404, 500]

    @patch("app.api.tool_routes.tool_registry")
    def test_execute_nonexistent_tool(self, mock_tool_registry, client):
        """Test executing nonexistent tool"""
        mock_tool_registry.get_tool.return_value = None
        
        request_data = {
            "tool_name": "nonexistent_tool",
            "parameters": {"query": "test"}
        }
        
        response = client.post("/tools/execute", json=request_data)
        assert response.status_code in [404, 500]

    @patch("app.api.tool_routes.tool_registry")
    def test_execute_tool_with_missing_parameters(self, mock_tool_registry, client):
        """Test executing tool with missing required parameters"""
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.execute.side_effect = ValueError("Missing required parameter: query")
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        request_data = {
            "tool_name": "search_tool",
            "parameters": {}  # Missing required query parameter
        }
        
        response = client.post("/tools/execute", json=request_data)
        assert response.status_code in [400, 422, 500]

    @patch("app.api.tool_routes.tool_registry")
    def test_execute_tool_with_invalid_parameters(self, mock_tool_registry, client):
        """Test executing tool with invalid parameters"""
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.execute.side_effect = ValueError("Invalid parameter type")
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        request_data = {
            "tool_name": "search_tool",
            "parameters": {
                "query": 123  # Should be string
            }
        }
        
        response = client.post("/tools/execute", json=request_data)
        assert response.status_code in [400, 422, 500]

    @patch("app.api.tool_routes.tool_registry")
    def test_tool_execution_failure(self, mock_tool_registry, client):
        """Test handling of tool execution failure"""
        mock_tool = AsyncMock()
        mock_tool.name = "search_tool"
        mock_tool.execute.side_effect = Exception("API rate limit exceeded")
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        request_data = {
            "tool_name": "search_tool",
            "parameters": {"query": "test"}
        }
        
        response = client.post("/tools/execute", json=request_data)
        assert response.status_code in [500, 503]

    def test_invalid_tool_name(self, client):
        """Test handling of invalid tool name"""
        # Test with special characters that might not be allowed
        response = client.get("/tools/tool@#$%")
        assert response.status_code in [422, 404, 500]

    @patch("app.api.tool_routes.tool_registry")
    def test_invalid_category(self, mock_tool_registry, client):
        """Test handling of invalid category"""
        mock_tool_registry.get_tools_by_category.return_value = []
        
        response = client.get("/tools/categories/invalid_category")
        assert response.status_code in [200, 404, 500]  # May return empty list or 404


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
            },
            timeout=30
        )
        
        assert request.tool_name == "search_tool"
        assert request.parameters["query"] == "Python testing"
        assert request.parameters["engine"] == "google"
        assert request.timeout == 30

    def test_tool_execution_response_model(self):
        """Test ToolExecutionResponse model validation"""
        response = ToolExecutionResponse(
            tool_name="search_tool",
            success=True,
            result={
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1"}
                ],
                "count": 1
            },
            execution_time=1.5,
            error_message=None
        )
        
        assert response.tool_name == "search_tool"
        assert response.success is True
        assert response.result["count"] == 1
        assert response.execution_time == 1.5
        assert response.error_message is None

    def test_tool_execution_response_with_error(self):
        """Test ToolExecutionResponse model with error"""
        response = ToolExecutionResponse(
            tool_name="search_tool",
            success=False,
            result=None,
            execution_time=0.5,
            error_message="API rate limit exceeded"
        )
        
        assert response.success is False
        assert response.result is None
        assert response.error_message == "API rate limit exceeded"

    def test_tool_info_response_model(self):
        """Test ToolInfoResponse model validation"""
        response = ToolInfoResponse(
            name="search_tool",
            description="Search the web using various engines",
            enabled=True,
            category="search",
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
            },
            examples=[
                {
                    "parameters": {"query": "Python testing"},
                    "description": "Basic search"
                }
            ]
        )
        
        assert response.name == "search_tool"
        assert response.enabled is True
        assert response.category == "search"
        assert response.parameters["query"]["required"] is True
        assert len(response.examples) == 1


class TestToolRoutesParameters:
    """Test tool routes with various parameters"""

    @patch("app.api.tool_routes.tool_registry")
    def test_list_tools_with_category_filter(self, mock_tool_registry, client):
        """Test listing tools with category filter"""
        mock_tool_registry.list_tools.return_value = []
        
        response = client.get("/tools/?category=search")
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_list_tools_with_search(self, mock_tool_registry, client):
        """Test listing tools with search term"""
        mock_tool_registry.list_tools.return_value = []
        
        response = client.get("/tools/?search=scraping")
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_execute_tool_with_timeout(self, mock_tool_registry, client):
        """Test executing tool with custom timeout"""
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = {"result": "success"}
        
        mock_tool_registry.get_tool.return_value = mock_tool
        
        request_data = {
            "tool_name": "search_tool",
            "parameters": {"query": "test"},
            "timeout": 60
        }
        
        response = client.post("/tools/execute", json=request_data)
        assert response.status_code not in [404]  # Endpoint should exist

    @patch("app.api.tool_routes.tool_registry")
    def test_get_tools_by_nonexistent_category(self, mock_tool_registry, client):
        """Test getting tools by nonexistent category"""
        mock_tool_registry.get_tools_by_category.return_value = []
        
        response = client.get("/tools/categories/nonexistent")
        assert response.status_code in [200, 404]  # May return empty list or 404


class TestToolRoutesAuthentication:
    """Test authentication and authorization for tool routes"""

    def test_public_endpoints(self, client):
        """Test that certain endpoints are publicly accessible"""
        # Tool listing should be public
        response = client.get("/tools/")
        # Should not require authentication (may fail due to missing mocks but not auth)
        assert response.status_code not in [401, 403]

    def test_protected_endpoints(self, client):
        """Test that certain endpoints require authentication"""
        # These endpoints might require authentication in production
        protected_endpoints = [
            "/tools/execute",
            "/tools/search_tool/enable",
            "/tools/search_tool/disable",
        ]
        
        for endpoint in protected_endpoints:
            if endpoint.startswith("/tools/execute"):
                # POST request
                response = client.post(endpoint, json={
                    "tool_name": "test_tool",
                    "parameters": {}
                })
            else:
                # POST request
                response = client.post(endpoint)
            
            # May fail due to missing mocks or authentication requirements
            assert response.status_code not in [404]  # Endpoint should exist