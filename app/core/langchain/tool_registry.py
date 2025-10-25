"""
LangChain-based Tool Registry for AI Assistant.

This module provides comprehensive tool management using LangChain's tool framework,
supporting both native LangChain tools and custom tools with seamless integration.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum

from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import BaseToolkit
from langchain_core.tools import tool as langchain_tool_decorator

from app.core.tools.base.base import BaseTool
from app.core.tools.execution.registry import tool_registry as legacy_tool_registry
from app.core.langchain.monitoring import LangChainMonitor

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools supported by the registry"""
    
    LANGCHAIN_NATIVE = "langchain_native"
    CUSTOM_WRAPPED = "custom_wrapped"
    LEGACY_COMPATIBLE = "legacy_compatible"


class LangChainToolRegistry:
    """
    Enhanced tool registry using LangChain's tool framework.
    
    This registry supports:
    - Native LangChain tools
    - Wrapped custom tools
    - Legacy tool compatibility
    - Dynamic tool discovery
    - Tool categorization and filtering
    """
    
    def __init__(self):
        self._langchain_tools: Dict[str, LangChainBaseTool] = {}
        self._custom_tools: Dict[str, BaseTool] = {}
        self._toolkits: Dict[str, BaseToolkit] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._enabled_tools: Set[str] = set()
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._monitoring = LangChainMonitor()
        
    async def initialize(self):
        """Initialize the tool registry with existing tools"""
        if self._initialized:
            return
            
        logger.info("Initializing LangChain Tool Registry...")
        
        # Initialize monitoring
        await self._monitoring.initialize()
        
        # Migrate existing tools from legacy registry
        await self._migrate_legacy_tools()
        
        # Register built-in LangChain tools
        await self._register_builtin_tools()
        
        # Register toolkits
        await self._register_toolkits()
        
        self._initialized = True
        logger.info("LangChain Tool Registry initialized successfully")
        
    async def _migrate_legacy_tools(self):
        """Migrate tools from legacy registry"""
        try:
            legacy_tools = legacy_tool_registry.list_tools(enabled_only=True)
            for tool in legacy_tools:
                await self.register_custom_tool(tool)
            logger.info(f"Migrated {len(legacy_tools)} tools from legacy registry")
        except Exception as e:
            logger.error(f"Failed to migrate legacy tools: {str(e)}")
            
    async def _register_builtin_tools(self):
        """Register built-in LangChain tools"""
        try:
            # Import and register common LangChain tools
            from langchain_community.tools import (
                DuckDuckGoSearchRun,
                WikipediaQueryRun,
                ShellTool,
                PythonREPLTool,
            )
            
            builtin_tools = [
                DuckDuckGoSearchRun(name="duckduckgo_search"),
                WikipediaQueryRun(name="wikipedia_search"),
                ShellTool(name="shell_tool"),
                PythonREPLTool(name="python_repl"),
            ]
            
            for tool in builtin_tools:
                await self.register_langchain_tool(tool, category="builtin")
                
            logger.info(f"Registered {len(builtin_tools)} built-in LangChain tools")
        except ImportError as e:
            logger.warning(f"Some built-in tools not available: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to register built-in tools: {str(e)}")
            
    async def _register_toolkits(self):
        """Register LangChain toolkits"""
        try:
            from langchain_community.agent_toolkits import (
                FileManagementToolkit,
                GmailToolkit,
                SlackToolkit,
            )
            
            # Note: Toolkits require additional configuration
            # For now, just register FileManagementToolkit
            try:
                file_toolkit = FileManagementToolkit(
                    root_dir="/tmp",  # Configure appropriately
                )
                await self.register_toolkit(file_toolkit, category="file_management")
                logger.info("Registered FileManagementToolkit")
            except Exception as e:
                logger.warning(f"Failed to register FileManagementToolkit: {str(e)}")
                
        except ImportError as e:
            logger.warning(f"Some toolkits not available: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to register toolkits: {str(e)}")
            
    async def register_langchain_tool(
        self,
        tool: LangChainBaseTool,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a native LangChain tool.
        
        Args:
            tool: LangChain BaseTool instance
            category: Tool category for organization
            metadata: Additional metadata for the tool
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Track tool registration
            start_time = time.time()
            
            if tool.name in self._langchain_tools:
                logger.warning(f"LangChain tool '{tool.name}' is already registered")
                await self._monitoring.track_tool_registration(
                    tool_name=tool.name,
                    tool_type="langchain_native",
                    category=category,
                    success=False,
                    error="Tool already registered",
                    duration=time.time() - start_time
                )
                return False
                
            self._langchain_tools[tool.name] = tool
            
            # Add to category
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(tool.name)
            
            # Store metadata
            self._tool_metadata[tool.name] = {
                "type": ToolType.LANGCHAIN_NATIVE,
                "category": category,
                "enabled": True,
                **(metadata or {})
            }
            
            # Enable by default
            self._enabled_tools.add(tool.name)
            
            duration = time.time() - start_time
            await self._monitoring.track_tool_registration(
                tool_name=tool.name,
                tool_type="langchain_native",
                category=category,
                success=True,
                duration=duration
            )
            
            logger.info(f"Registered LangChain tool '{tool.name}' in category '{category}'")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            await self._monitoring.track_tool_registration(
                tool_name=tool.name,
                tool_type="langchain_native",
                category=category,
                success=False,
                error=str(e),
                duration=duration
            )
            logger.error(f"Failed to register LangChain tool '{tool.name}': {str(e)}")
            return False
            
    async def register_custom_tool(
        self,
        tool: BaseTool,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a custom tool with LangChain wrapper.
        
        Args:
            tool: Custom BaseTool instance
            category: Tool category for organization
            metadata: Additional metadata for the tool
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Track tool registration
            start_time = time.time()
            
            if tool.name in self._custom_tools:
                logger.warning(f"Custom tool '{tool.name}' is already registered")
                await self._monitoring.track_tool_registration(
                    tool_name=tool.name,
                    tool_type="custom_wrapped",
                    category=category,
                    success=False,
                    error="Tool already registered",
                    duration=time.time() - start_time
                )
                return False
                
            # Create LangChain wrapper
            wrapped_tool = self._create_langchain_wrapper(tool)
            
            self._custom_tools[tool.name] = tool
            self._langchain_tools[tool.name] = wrapped_tool
            
            # Add to category
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(tool.name)
            
            # Store metadata
            self._tool_metadata[tool.name] = {
                "type": ToolType.CUSTOM_WRAPPED,
                "category": category,
                "enabled": True,
                "original_tool": tool,
                **(metadata or {})
            }
            
            # Enable by default
            self._enabled_tools.add(tool.name)
            
            duration = time.time() - start_time
            await self._monitoring.track_tool_registration(
                tool_name=tool.name,
                tool_type="custom_wrapped",
                category=category,
                success=True,
                duration=duration
            )
            
            logger.info(f"Registered custom tool '{tool.name}' in category '{category}'")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            await self._monitoring.track_tool_registration(
                tool_name=tool.name,
                tool_type="custom_wrapped",
                category=category,
                success=False,
                error=str(e),
                duration=duration
            )
            logger.error(f"Failed to register custom tool '{tool.name}': {str(e)}")
            return False
            
    def _create_langchain_wrapper(self, tool: BaseTool) -> LangChainBaseTool:
        """Create a LangChain wrapper for custom tool"""
        
        class CustomToolWrapper(LangChainBaseTool):
            name: str = tool.name
            description: str = tool.description
            
            def _run(self, *args, **kwargs) -> Any:
                """Synchronous run method"""
                import asyncio
                
                # Get monitoring instance
                monitoring = getattr(self, '_monitoring', None)
                if not monitoring:
                    # Fallback to creating new instance
                    monitoring = LangChainMonitor()
                    asyncio.run(monitoring.initialize())
                
                # Track tool execution
                start_time = time.time()
                execution_id = None
                
                try:
                    # Start tracking
                    execution_id = asyncio.run(monitoring.start_tool_execution(
                        tool_name=tool.name,
                        tool_type="custom_wrapped",
                        parameters=kwargs
                    ))
                    
                    # Execute tool
                    result = asyncio.run(tool.execute_with_timeout(**kwargs))
                    
                    # Complete tracking
                    duration = time.time() - start_time
                    if result.success:
                        asyncio.run(monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=True,
                            result=result.data,
                            duration=duration
                        ))
                        return result.data
                    else:
                        asyncio.run(monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=False,
                            error=result.error,
                            duration=duration
                        ))
                        raise Exception(f"Tool '{tool.name}' failed: {result.error}")
                        
                except Exception as e:
                    # Track error
                    duration = time.time() - start_time
                    if execution_id:
                        asyncio.run(monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=False,
                            error=str(e),
                            duration=duration
                        ))
                    raise
                
            async def _arun(self, *args, **kwargs) -> Any:
                """Asynchronous run method"""
                # Get monitoring instance
                monitoring = getattr(self, '_monitoring', None)
                if not monitoring:
                    # Fallback to creating new instance
                    monitoring = LangChainMonitor()
                    await monitoring.initialize()
                
                # Track tool execution
                start_time = time.time()
                execution_id = None
                
                try:
                    # Start tracking
                    execution_id = await monitoring.start_tool_execution(
                        tool_name=tool.name,
                        tool_type="custom_wrapped",
                        parameters=kwargs
                    )
                    
                    # Execute tool
                    result = await tool.execute_with_timeout(**kwargs)
                    
                    # Complete tracking
                    duration = time.time() - start_time
                    if result.success:
                        await monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=True,
                            result=result.data,
                            duration=duration
                        )
                        return result.data
                    else:
                        await monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=False,
                            error=result.error,
                            duration=duration
                        )
                        raise Exception(f"Tool '{tool.name}' failed: {result.error}")
                        
                except Exception as e:
                    # Track error
                    duration = time.time() - start_time
                    if execution_id:
                        await monitoring.complete_tool_execution(
                            execution_id=execution_id,
                            success=False,
                            error=str(e),
                            duration=duration
                        )
                    raise
                    
            @property
            def args_schema(self):
                """Return the tool's argument schema"""
                # Convert tool parameters to LangChain schema
                return self._convert_parameters_to_schema(tool.parameters)
                
            def _convert_parameters_to_schema(self, parameters: Dict[str, Dict[str, Any]]) -> Any:
                """Convert tool parameters to LangChain schema format"""
                # This is a simplified conversion
                # In practice, you'd want more sophisticated schema conversion
                from pydantic import BaseModel, Field
                
                fields = {}
                for param_name, param_config in parameters.items():
                    field_type = param_config.get("type", str)
                    required = param_config.get("required", False)
                    description = param_config.get("description", "")
                    
                    if field_type == str:
                        fields[param_name] = Field(... if required else None, description=description)
                    elif field_type == int:
                        fields[param_name] = Field(... if required else None, description=description)
                    elif field_type == bool:
                        fields[param_name] = Field(... if required else None, description=description)
                    # Add more types as needed
                    
                return type("ToolSchema", (BaseModel,), fields)
        
        # Set monitoring instance on wrapper
        wrapper = CustomToolWrapper()
        wrapper._monitoring = self._monitoring
        return wrapper
        
    async def register_toolkit(
        self, 
        toolkit: BaseToolkit, 
        category: str = "toolkit",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a LangChain toolkit.
        
        Args:
            toolkit: LangChain BaseToolkit instance
            category: Toolkit category for organization
            metadata: Additional metadata for the toolkit
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            toolkit_name = toolkit.__class__.__name__
            if toolkit_name in self._toolkits:
                logger.warning(f"Toolkit '{toolkit_name}' is already registered")
                return False
                
            self._toolkits[toolkit_name] = toolkit
            
            # Register all tools from the toolkit
            tools = toolkit.get_tools()
            for tool in tools:
                await self.register_langchain_tool(tool, category=category)
                
            # Store metadata
            self._tool_metadata[toolkit_name] = {
                "type": "toolkit",
                "category": category,
                "enabled": True,
                "tool_count": len(tools),
                **(metadata or {})
            }
            
            logger.info(f"Registered toolkit '{toolkit_name}' with {len(tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register toolkit '{toolkit_name}': {str(e)}")
            return False
            
    def get_tool(self, tool_name: str) -> Optional[LangChainBaseTool]:
        """
        Get a LangChain tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            LangChain BaseTool instance or None if not found
        """
        return self._langchain_tools.get(tool_name)
        
    def get_custom_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a custom tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Custom BaseTool instance or None if not found
        """
        return self._custom_tools.get(tool_name)
        
    def list_tools(
        self, 
        enabled_only: bool = True,
        tool_type: Optional[ToolType] = None,
        category: Optional[str] = None
    ) -> List[LangChainBaseTool]:
        """
        List registered tools with filtering options.
        
        Args:
            enabled_only: Only return enabled tools
            tool_type: Filter by tool type
            category: Filter by category
            
        Returns:
            List of LangChain BaseTool instances
        """
        tools = []
        
        for tool_name, tool in self._langchain_tools.items():
            # Check if enabled
            if enabled_only and tool_name not in self._enabled_tools:
                continue
                
            # Check tool type
            if tool_type:
                metadata = self._tool_metadata.get(tool_name, {})
                if metadata.get("type") != tool_type:
                    continue
                    
            # Check category
            if category:
                metadata = self._tool_metadata.get(tool_name, {})
                if metadata.get("category") != category:
                    continue
                    
            tools.append(tool)
            
        return tools
        
    def list_tool_names(
        self, 
        enabled_only: bool = True,
        tool_type: Optional[ToolType] = None,
        category: Optional[str] = None
    ) -> List[str]:
        """List names of registered tools with filtering options"""
        tools = self.list_tools(enabled_only, tool_type, category)
        return [tool.name for tool in tools]
        
    def get_tools_by_category(self, category: str) -> List[LangChainBaseTool]:
        """Get all tools in a specific category"""
        return self.list_tools(enabled_only=True, category=category)
        
    def get_toolkit(self, toolkit_name: str) -> Optional[BaseToolkit]:
        """Get a toolkit by name"""
        return self._toolkits.get(toolkit_name)
        
    def find_relevant_tools(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 5,
        category: Optional[str] = None
    ) -> List[LangChainBaseTool]:
        """
        Find tools relevant to a given query.
        
        Args:
            query: User query
            context: Additional context information
            max_results: Maximum number of tools to return
            category: Filter by category
            
        Returns:
            List of relevant tools, sorted by relevance
        """
        relevant_tools = []
        query_lower = query.lower()
        
        for tool in self.list_tools(enabled_only=True, category=category):
            # Check tool name and description
            tool_text = f"{tool.name} {tool.description}".lower()
            
            # Simple keyword matching
            if any(keyword in tool_text for keyword in query_lower.split()):
                relevant_tools.append(tool)
                continue
                
            # Check custom tool keywords
            if tool.name in self._custom_tools:
                custom_tool = self._custom_tools[tool.name]
                if any(keyword.lower() in query_lower for keyword in custom_tool.keywords):
                    relevant_tools.append(tool)
                    
        # Sort by relevance (simple implementation)
        # In future, could use more sophisticated ranking
        relevant_tools.sort(key=lambda t: len(t.description), reverse=True)
        
        return relevant_tools[:max_results]
        
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a specific tool"""
        if tool_name in self._langchain_tools:
            self._enabled_tools.add(tool_name)
            
            # Update metadata
            if tool_name in self._tool_metadata:
                self._tool_metadata[tool_name]["enabled"] = True
                
            logger.info(f"Enabled tool '{tool_name}'")
            return True
        return False
        
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a specific tool"""
        if tool_name in self._langchain_tools:
            if tool_name in self._enabled_tools:
                self._enabled_tools.remove(tool_name)
                
            # Update metadata
            if tool_name in self._tool_metadata:
                self._tool_metadata[tool_name]["enabled"] = False
                
            logger.info(f"Disabled tool '{tool_name}'")
            return True
        return False
        
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool"""
        return self._tool_metadata.get(tool_name)
        
    def get_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(self._categories.keys())
        
    def get_tools_in_category(self, category: str) -> List[str]:
        """Get list of tool names in a specific category"""
        return list(self._categories.get(category, set()))
        
    def get_toolkit_for_tools(self, tool_names: List[str]) -> Optional[BaseToolkit]:
        """Find a toolkit that contains the specified tools"""
        for toolkit in self._toolkits.values():
            toolkit_tool_names = [tool.name for tool in toolkit.get_tools()]
            if all(name in toolkit_tool_names for name in tool_names):
                return toolkit
        return None
        
    def create_langchain_toolkit(self, category: Optional[str] = None) -> BaseToolkit:
        """
        Create a LangChain toolkit with registered tools.
        
        Args:
            category: Filter tools by category
            
        Returns:
            LangChain BaseToolkit instance
        """
        from langchain_core.tools import BaseToolkit
        
        class DynamicToolkit(BaseToolkit):
            def get_tools(self):
                return self.list_tools(enabled_only=True, category=category)
                
        return DynamicToolkit()
        
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        enabled_tools = len(self._enabled_tools)
        total_langchain_tools = len(self._langchain_tools)
        total_custom_tools = len(self._custom_tools)
        total_toolkits = len(self._toolkits)
        
        return {
            "total_tools": total_langchain_tools,
            "enabled_tools": enabled_tools,
            "custom_tools": total_custom_tools,
            "toolkits": total_toolkits,
            "categories": list(self._categories.keys()),
            "tools_by_category": {
                category: len(tools) for category, tools in self._categories.items()
            },
            "tool_types": {
                "langchain_native": len([
                    t for t in self._tool_metadata.values() 
                    if t.get("type") == ToolType.LANGCHAIN_NATIVE
                ]),
                "custom_wrapped": len([
                    t for t in self._tool_metadata.values() 
                    if t.get("type") == ToolType.CUSTOM_WRAPPED
                ]),
            }
        }
        
    def clear_cache(self):
        """Clear any cached tool instances"""
        # LangChain tools don't typically cache, but we can clear our metadata cache
        logger.info("Tool registry cache cleared")
        
    async def refresh_tools(self):
        """Refresh tool registrations"""
        logger.info("Refreshing tool registrations...")
        
        # Clear current registrations
        self._langchain_tools.clear()
        self._custom_tools.clear()
        self._toolkits.clear()
        self._categories.clear()
        self._enabled_tools.clear()
        self._tool_metadata.clear()
        
        # Re-initialize
        await self.initialize()
        
        logger.info("Tool registry refreshed")


# Global tool registry instance
tool_registry = LangChainToolRegistry()