"""
Tool registry for managing and discovering tools in the AI Assistant system.

This module provides the ToolRegistry class which serves as a central registry
for tool registration, discovery, and management.
"""

from typing import Dict, List, Optional, Set, Any
from app.core.tools.base.base import BaseTool
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for managing and discovering tools"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._enabled_tools: Set[str] = set()

    def register(self, tool: BaseTool, category: str = "general") -> bool:
        """
        Register a tool with the registry

        Args:
            tool: Tool instance to register
            category: Tool category for organization

        Returns:
            True if registration successful, False otherwise
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' is already registered")
            return False

        self._tools[tool.name] = tool

        # Add to category
        if category not in self._categories:
            self._categories[category] = set()
        self._categories[category].add(tool.name)

        # Enable by default
        self._enabled_tools.add(tool.name)

        logger.info(f"Registered tool '{tool.name}' in category '{category}'")
        return True

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            return False

        # Remove from categories
        for category, tools in self._categories.items():
            if tool_name in tools:
                tools.remove(tool_name)
                if not tools:  # Remove empty categories
                    del self._categories[category]

        # Remove from enabled tools
        if tool_name in self._enabled_tools:
            self._enabled_tools.remove(tool_name)

        del self._tools[tool_name]
        logger.info(f"Unregistered tool '{tool_name}'")
        return True

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(tool_name)

    def list_tools(self, enabled_only: bool = True) -> List[BaseTool]:
        """List all registered tools"""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [tool for tool in tools if tool.name in self._enabled_tools]
        return tools

    def list_tool_names(self, enabled_only: bool = True) -> List[str]:
        """List names of all registered tools"""
        tool_names = list(self._tools.keys())
        if enabled_only:
            tool_names = [name for name in tool_names if name in self._enabled_tools]
        return tool_names

    def find_relevant_tools(
        self, query: str, context: Dict[str, Any] = None, max_results: int = 5
    ) -> List[BaseTool]:
        """
        Find tools relevant to the given query

        Args:
            query: User query
            context: Additional context
            max_results: Maximum number of tools to return

        Returns:
            List of relevant tools, sorted by relevance
        """
        relevant_tools = []

        for tool in self.list_tools(enabled_only=True):
            if tool.should_use(query, context):
                relevant_tools.append(tool)

        # Sort by potential relevance (simple implementation)
        # In future, this could use more sophisticated ranking
        relevant_tools.sort(key=lambda t: len(t.keywords), reverse=True)

        return relevant_tools[:max_results]

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a specific category"""
        if category not in self._categories:
            return []

        tools = []
        for tool_name in self._categories[category]:
            tool = self.get_tool(tool_name)
            if tool and tool.enabled:
                tools.append(tool)

        return tools

    def enable_tool(self, tool_name: str) -> bool:
        """Enable a specific tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False

        tool.enabled = True
        self._enabled_tools.add(tool_name)
        logger.info(f"Enabled tool '{tool_name}'")
        return True

    def disable_tool(self, tool_name: str) -> bool:
        """Disable a specific tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False

        tool.enabled = False
        if tool_name in self._enabled_tools:
            self._enabled_tools.remove(tool_name)
        logger.info(f"Disabled tool '{tool_name}'")
        return True

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self._tools),
            "enabled_tools": len(self._enabled_tools),
            "categories": list(self._categories.keys()),
            "tools_by_category": {
                category: len(tools) for category, tools in self._categories.items()
            },
        }


# Global tool registry instance
tool_registry = ToolRegistry()
