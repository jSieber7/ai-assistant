"""
Tool management and execution routes for the AI Assistant.

This module provides API endpoints for managing and executing tools
within the AI Assistant system.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from ..core.tools import tool_registry, CalculatorTool, TimeTool, EchoTool, SearXNGTool
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/tools", tags=["tools"])

# Register example tools
tool_registry.register(CalculatorTool(), "utility")
tool_registry.register(TimeTool(), "utility")
tool_registry.register(EchoTool(), "testing")

# Register SearXNG search tool if URL is configured

if settings.searxng_url:
    tool_registry.register(SearXNGTool(), "search")


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""

    tool_name: str
    parameters: Dict[str, Any] = {}


class ToolExecutionResponse(BaseModel):
    """Response model for tool execution"""

    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str
    execution_time: float
    metadata: Dict[str, Any] = {}


class ToolInfoResponse(BaseModel):
    """Response model for tool information"""

    name: str
    description: str
    version: str
    author: str
    categories: List[str]
    keywords: List[str]
    parameters: Dict[str, Dict[str, Any]]
    enabled: bool
    timeout: int


@router.get("/")
async def list_tools(enabled_only: bool = True) -> Dict[str, Any]:
    """List all available tools"""
    tools = tool_registry.list_tools(enabled_only=enabled_only)

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "categories": tool.categories,
                "enabled": tool.enabled,
            }
            for tool in tools
        ],
        "total": len(tools),
        "enabled_only": enabled_only,
    }


@router.get("/{tool_name}")
async def get_tool_info(tool_name: str) -> ToolInfoResponse:
    """Get detailed information about a specific tool"""
    tool = tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return ToolInfoResponse(
        name=tool.name,
        description=tool.description,
        version=tool.version,
        author=tool.author,
        categories=tool.categories,
        keywords=tool.keywords,
        parameters=tool.parameters,
        enabled=tool.enabled,
        timeout=tool.timeout,
    )


@router.post("/execute")
async def execute_tool(request: ToolExecutionRequest) -> ToolExecutionResponse:
    """Execute a specific tool with given parameters"""
    if not settings.tool_system_enabled:
        raise HTTPException(status_code=503, detail="Tool system is disabled")

    tool = tool_registry.get_tool(request.tool_name)
    if not tool:
        raise HTTPException(
            status_code=404, detail=f"Tool '{request.tool_name}' not found"
        )

    if not tool.enabled:
        raise HTTPException(
            status_code=403, detail=f"Tool '{request.tool_name}' is disabled"
        )

    # Validate parameters
    if not tool.validate_parameters(**request.parameters):
        raise HTTPException(
            status_code=400, detail=f"Invalid parameters for tool '{request.tool_name}'"
        )

    try:
        # Execute tool with timeout protection
        result = await tool.execute_with_timeout(**request.parameters)
        return ToolExecutionResponse(**result.dict())

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@router.get("/registry/stats")
async def get_registry_stats() -> Dict[str, Any]:
    """Get tool registry statistics"""
    return tool_registry.get_registry_stats()


@router.post("/{tool_name}/enable")
async def enable_tool(tool_name: str) -> Dict[str, Any]:
    """Enable a specific tool"""
    success = tool_registry.enable_tool(tool_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return {"status": "enabled", "tool": tool_name}


@router.post("/{tool_name}/disable")
async def disable_tool(tool_name: str) -> Dict[str, Any]:
    """Disable a specific tool"""
    success = tool_registry.disable_tool(tool_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return {"status": "disabled", "tool": tool_name}


@router.get("/categories/{category}")
async def get_tools_by_category(category: str) -> Dict[str, Any]:
    """Get all tools in a specific category"""
    tools = tool_registry.get_tools_by_category(category)

    return {
        "category": category,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "enabled": tool.enabled,
            }
            for tool in tools
        ],
        "total": len(tools),
    }
