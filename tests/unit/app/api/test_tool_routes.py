"""
Unit tests for tool API routes.

Tests for the tool management endpoints including tool listing,
execution, enable/disable, and category management.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app


class TestToolListing:
    """Test cases for tool listing endpoints"""

    @pytest.mark.asyncio
    async def test_list_tools_legacy(self):
        """Test listing tools with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                # Mock tool
                mock_tool = MagicMock()
                mock_tool.name = "calculator"
                mock_tool.description = "Performs calculations"
                mock_tool.categories = ["utility"]
                mock_tool.enabled = True
                mock_registry.list_tools.return_value = [mock_tool]
                
                client = TestClient(app)
                response = client.get("/v1/tools/")
                
                assert response.status_code == 200
                data = response.json()
                assert "tools" in data
                assert len(data["tools"]) == 1
                assert data["tools"][0]["name"] == "calculator"
                assert data["total"] == 1
                assert data["enabled_only"] is True
                assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_list_tools_langchain(self):
        """Test listing tools with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_tool = MagicMock()
            mock_tool.name = "calculator"
            mock_tool.description = "Performs calculations"
            mock_integration.get_tools.return_value = [mock_tool]
            mock_integration.get_tool_categories.return_value = ["utility"]
            mock_integration.is_tool_enabled.return_value = True
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.get("/v1/tools/")
            
            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert len(data["tools"]) == 1
            assert data["tools"][0]["name"] == "calculator"
            assert data["total"] == 1
            assert data["enabled_only"] is True
            assert data["registry"] == "langchain"

    @pytest.mark.asyncio
    async def test_list_tools_include_disabled(self):
        """Test listing tools including disabled ones"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                # Mock tools
                mock_tool1 = MagicMock()
                mock_tool1.name = "calculator"
                mock_tool1.description = "Performs calculations"
                mock_tool1.categories = ["utility"]
                mock_tool1.enabled = True
                
                mock_tool2 = MagicMock()
                mock_tool2.name = "disabled-tool"
                mock_tool2.description = "Disabled tool"
                mock_tool2.categories = ["testing"]
                mock_tool2.enabled = False
                
                mock_registry.list_tools.return_value = [mock_tool1, mock_tool2]
                
                client = TestClient(app)
                response = client.get("/v1/tools/?enabled_only=false")
                
                assert response.status_code == 200
                data = response.json()
                assert len(data["tools"]) == 2
                assert data["enabled_only"] is False


class TestToolInfo:
    """Test cases for tool info endpoint"""

    @pytest.mark.asyncio
    async def test_get_tool_info_legacy(self):
        """Test getting tool info with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                # Mock tool
                mock_tool = MagicMock()
                mock_tool.name = "calculator"
                mock_tool.description = "Performs calculations"
                mock_tool.version = "1.0.0"
                mock_tool.author = "AI Assistant Team"
                mock_tool.categories = ["utility"]
                mock_tool.keywords = ["math", "calculation"]
                mock_tool.parameters = {"expression": {"type": "string"}}
                mock_tool.enabled = True
                mock_tool.timeout = 30
                mock_registry.get_tool.return_value = mock_tool
                
                client = TestClient(app)
                response = client.get("/v1/tools/calculator")
                
                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "calculator"
                assert data["description"] == "Performs calculations"
                assert data["version"] == "1.0.0"
                assert data["author"] == "AI Assistant Team"
                assert "utility" in data["categories"]
                assert data["enabled"] is True
                assert data["timeout"] == 30

    @pytest.mark.asyncio
    async def test_get_tool_info_langchain(self):
        """Test getting tool info with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_tool = MagicMock()
            mock_tool.name = "calculator"
            mock_tool.description = "Performs calculations"
            mock_integration.get_tool.return_value = mock_tool
            mock_integration.get_tool_categories.return_value = ["utility"]
            mock_integration.is_tool_enabled.return_value = True
            mock_integration.get_tool_metadata.return_value = {
                "version": "1.0.0",
                "author": "AI Assistant Team",
                "keywords": ["math", "calculation"],
                "parameters": {"expression": {"type": "string"}},
                "timeout": 30
            }
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.get("/v1/tools/calculator")
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "calculator"
            assert data["description"] == "Performs calculations"
            assert data["version"] == "1.0.0"
            assert data["author"] == "AI Assistant Team"
            assert "utility" in data["categories"]
            assert data["enabled"] is True
            assert data["timeout"] == 30

    @pytest.mark.asyncio
    async def test_get_tool_info_not_found(self):
        """Test getting info for non-existent tool"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.get_tool.return_value = None
                
                client = TestClient(app)
                response = client.get("/v1/tools/non-existent-tool")
                
                assert response.status_code == 404
                assert "not found" in response.json()["detail"]


class TestToolExecution:
    """Test cases for tool execution endpoint"""

    @pytest.mark.asyncio
    async def test_execute_tool_legacy_success(self):
        """Test successful tool execution with legacy registry"""
        request_data = {
            "tool_name": "calculator",
            "parameters": {"expression": "2+2"}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.tool_routes.tool_registry') as mock_registry:
                    # Mock tool
                    mock_tool = MagicMock()
                    mock_tool.enabled = True
                    mock_tool.validate_parameters.return_value = True
                    mock_result = MagicMock()
                    mock_result.dict.return_value = {
                        "success": True,
                        "data": {"result": 4},
                        "error": None,
                        "tool_name": "calculator",
                        "execution_time": 0.1,
                        "metadata": {}
                    }
                    mock_tool.execute_with_timeout.return_value = mock_result
                    mock_registry.get_tool.return_value = mock_tool
                    
                    client = TestClient(app)
                    response = client.post("/v1/tools/execute", json=request_data)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["data"]["result"] == 4
                    assert data["tool_name"] == "calculator"

    @pytest.mark.asyncio
    async def test_execute_tool_langchain_success(self):
        """Test successful tool execution with LangChain integration"""
        request_data = {
            "tool_name": "calculator",
            "parameters": {"expression": "2+2"}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_tools = True
                mock_tool = MagicMock()
                mock_tool._run.return_value = {"result": 4}
                mock_integration.get_tool.return_value = mock_tool
                mock_integration.is_tool_enabled.return_value = True
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/v1/tools/execute", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["data"]["result"] == 4
                assert data["tool_name"] == "calculator"
                assert data["metadata"]["execution_method"] == "langchain"

    @pytest.mark.asyncio
    async def test_execute_tool_langchain_async(self):
        """Test async tool execution with LangChain integration"""
        request_data = {
            "tool_name": "async-tool",
            "parameters": {"query": "test"}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration enabled
                mock_integration = MagicMock()
                mock_integration.feature_flags.use_langchain_tools = True
                mock_tool = MagicMock()
                mock_tool._arun = AsyncMock(return_value={"result": "async result"})
                mock_integration.get_tool.return_value = mock_tool
                mock_integration.is_tool_enabled.return_value = True
                mock_get_integration.return_value = mock_integration
                
                client = TestClient(app)
                response = client.post("/v1/tools/execute", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["data"]["result"] == "async result"

    @pytest.mark.asyncio
    async def test_execute_tool_disabled(self):
        """Test executing a disabled tool"""
        request_data = {
            "tool_name": "disabled-tool",
            "parameters": {}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.tool_routes.tool_registry') as mock_registry:
                    # Mock disabled tool
                    mock_tool = MagicMock()
                    mock_tool.enabled = False
                    mock_registry.get_tool.return_value = mock_tool
                    
                    client = TestClient(app)
                    response = client.post("/v1/tools/execute", json=request_data)
                    
                    assert response.status_code == 403
                    assert "is disabled" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a non-existent tool"""
        request_data = {
            "tool_name": "non-existent-tool",
            "parameters": {}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.tool_routes.tool_registry') as mock_registry:
                    mock_registry.get_tool.return_value = None
                    
                    client = TestClient(app)
                    response = client.post("/v1/tools/execute", json=request_data)
                    
                    assert response.status_code == 404
                    assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_parameters(self):
        """Test executing tool with invalid parameters"""
        request_data = {
            "tool_name": "calculator",
            "parameters": {"invalid_param": "value"}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.tool_routes.tool_registry') as mock_registry:
                    # Mock tool
                    mock_tool = MagicMock()
                    mock_tool.enabled = True
                    mock_tool.validate_parameters.return_value = False
                    mock_registry.get_tool.return_value = mock_tool
                    
                    client = TestClient(app)
                    response = client.post("/v1/tools/execute", json=request_data)
                    
                    assert response.status_code == 400
                    assert "Invalid parameters" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_execute_tool_system_disabled(self):
        """Test executing tool when tool system is disabled"""
        request_data = {
            "tool_name": "calculator",
            "parameters": {}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = False
            
            client = TestClient(app)
            response = client.post("/v1/tools/execute", json=request_data)
            
            assert response.status_code == 503
            assert "Tool system is disabled" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_execute_tool_execution_error(self):
        """Test tool execution with error"""
        request_data = {
            "tool_name": "calculator",
            "parameters": {"expression": "invalid"}
        }
        
        with patch('app.api.tool_routes.settings') as mock_settings:
            mock_settings.tool_system_enabled = True
            
            with patch('app.api.tool_routes.get_integration') as mock_get_integration:
                # Mock integration disabled
                mock_get_integration.return_value = None
                
                with patch('app.api.tool_routes.tool_registry') as mock_registry:
                    # Mock tool
                    mock_tool = MagicMock()
                    mock_tool.enabled = True
                    mock_tool.validate_parameters.return_value = True
                    mock_tool.execute_with_timeout.side_effect = Exception("Calculation error")
                    mock_registry.get_tool.return_value = mock_tool
                    
                    client = TestClient(app)
                    response = client.post("/v1/tools/execute", json=request_data)
                    
                    assert response.status_code == 500
                    assert "Tool execution failed" in response.json()["detail"]


class TestToolRegistryStats:
    """Test cases for tool registry stats endpoint"""

    @pytest.mark.asyncio
    async def test_get_registry_stats_legacy(self):
        """Test getting registry stats with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.get_registry_stats.return_value = {
                    "total_tools": 10,
                    "enabled_tools": 8,
                    "categories": ["utility", "search", "testing"]
                }
                
                client = TestClient(app)
                response = client.get("/v1/tools/registry/stats")
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_tools"] == 10
                assert data["enabled_tools"] == 8
                assert "utility" in data["categories"]

    @pytest.mark.asyncio
    async def test_get_registry_stats_langchain(self):
        """Test getting registry stats with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_integration.get_tool_registry_stats.return_value = {
                "total_tools": 10,
                "enabled_tools": 8,
                "categories": ["utility", "search", "testing"]
            }
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.get("/v1/tools/registry/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_tools"] == 10
            assert data["enabled_tools"] == 8


class TestToolEnableDisable:
    """Test cases for tool enable/disable endpoints"""

    @pytest.mark.asyncio
    async def test_enable_tool_legacy(self):
        """Test enabling a tool with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.enable_tool.return_value = True
                
                client = TestClient(app)
                response = client.post("/v1/tools/calculator/enable")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "enabled"
                assert data["tool"] == "calculator"
                assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_enable_tool_langchain(self):
        """Test enabling a tool with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_integration.enable_tool.return_value = True
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.post("/v1/tools/calculator/enable")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "enabled"
            assert data["tool"] == "calculator"
            assert data["registry"] == "langchain"

    @pytest.mark.asyncio
    async def test_enable_tool_not_found(self):
        """Test enabling a non-existent tool"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.enable_tool.return_value = False
                
                client = TestClient(app)
                response = client.post("/v1/tools/non-existent-tool/enable")
                
                assert response.status_code == 404
                assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_disable_tool_legacy(self):
        """Test disabling a tool with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.disable_tool.return_value = True
                
                client = TestClient(app)
                response = client.post("/v1/tools/calculator/disable")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "disabled"
                assert data["tool"] == "calculator"
                assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_disable_tool_langchain(self):
        """Test disabling a tool with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_integration.disable_tool.return_value = True
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.post("/v1/tools/calculator/disable")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"
            assert data["tool"] == "calculator"
            assert data["registry"] == "langchain"


class TestToolCategories:
    """Test cases for tool categories endpoint"""

    @pytest.mark.asyncio
    async def test_get_tools_by_category_legacy(self):
        """Test getting tools by category with legacy registry"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                # Mock tool
                mock_tool = MagicMock()
                mock_tool.name = "calculator"
                mock_tool.description = "Performs calculations"
                mock_tool.enabled = True
                mock_registry.get_tools_by_category.return_value = [mock_tool]
                
                client = TestClient(app)
                response = client.get("/v1/tools/categories/utility")
                
                assert response.status_code == 200
                data = response.json()
                assert data["category"] == "utility"
                assert len(data["tools"]) == 1
                assert data["tools"][0]["name"] == "calculator"
                assert data["total"] == 1
                assert data["registry"] == "legacy"

    @pytest.mark.asyncio
    async def test_get_tools_by_category_langchain(self):
        """Test getting tools by category with LangChain integration"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration enabled
            mock_integration = MagicMock()
            mock_integration.feature_flags.use_langchain_tools = True
            mock_tool = MagicMock()
            mock_tool.name = "calculator"
            mock_tool.description = "Performs calculations"
            mock_integration.get_tools_by_category.return_value = [mock_tool]
            mock_integration.is_tool_enabled.return_value = True
            mock_get_integration.return_value = mock_integration
            
            client = TestClient(app)
            response = client.get("/v1/tools/categories/utility")
            
            assert response.status_code == 200
            data = response.json()
            assert data["category"] == "utility"
            assert len(data["tools"]) == 1
            assert data["tools"][0]["name"] == "calculator"
            assert data["total"] == 1
            assert data["registry"] == "langchain"

    @pytest.mark.asyncio
    async def test_get_tools_by_empty_category(self):
        """Test getting tools from empty category"""
        with patch('app.api.tool_routes.get_integration') as mock_get_integration:
            # Mock integration disabled
            mock_get_integration.return_value = None
            
            with patch('app.api.tool_routes.tool_registry') as mock_registry:
                mock_registry.get_tools_by_category.return_value = []
                
                client = TestClient(app)
                response = client.get("/v1/tools/categories/empty-category")
                
                assert response.status_code == 200
                data = response.json()
                assert data["category"] == "empty-category"
                assert len(data["tools"]) == 0
                assert data["total"] == 0