"""
LangChain integration for the AI Assistant tool system.

This module provides compatibility layers to make the tool system
work seamlessly with LangChain agents and tools.
"""

from typing import List, Dict, Any
from langchain.schema import BaseOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool as LangChainCoreTool

from .base import BaseTool
from .registry import ToolRegistry
import logging


# We'll use a lazy import pattern to avoid circular imports
def get_tool_registry():
    """Get the global tool registry instance"""
    from . import tool_registry

    return tool_registry


logger = logging.getLogger(__name__)


class LangChainToolWrapper(LangChainCoreTool):
    """Wrapper to convert our BaseTool to LangChain BaseTool"""

    def __init__(self, tool: BaseTool):
        self._tool = tool
        super().__init__(
            name=tool.name,
            description=tool.description,
        )

    def _run(self, *args, **kwargs) -> Any:
        """Synchronous run method for LangChain compatibility"""
        import asyncio

        # Convert sync call to async
        return asyncio.run(self._tool.execute_with_timeout(**kwargs))

    async def _arun(self, *args, **kwargs) -> Any:
        """Async run method for LangChain compatibility"""
        result = await self._tool.execute_with_timeout(**kwargs)
        return result.data if result.success else result.error


class LangChainToolkit(BaseToolkit):
    """Toolkit for LangChain integration"""

    def __init__(self, registry: ToolRegistry = None):
        self.registry = registry or get_tool_registry()

    def get_tools(self) -> List[LangChainCoreTool]:
        """Get all tools as LangChain tools"""
        tools = []
        for tool in self.registry.list_tools(enabled_only=True):
            try:
                langchain_tool = LangChainToolWrapper(tool)
                tools.append(langchain_tool)
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool.name} to LangChain: {e}")

        return tools

    @classmethod
    def from_registry(cls, registry: ToolRegistry) -> "LangChainToolkit":
        """Create toolkit from existing registry"""
        return cls(registry)


class ToolOutputParser(BaseOutputParser):
    """Custom output parser for tool results"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse tool output"""
        # For now, return as simple dict
        # In future, could parse structured tool responses
        return {"output": text}

    @property
    def _type(self) -> str:
        return "tool_output_parser"


def create_agent_with_tools(llm, registry: ToolRegistry = None) -> AgentExecutor:
    """
    Create a LangChain agent that can use our tool system

    Args:
        llm: LangChain LLM instance
        registry: Tool registry (uses default if None)

    Returns:
        AgentExecutor configured with our tools
    """
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    registry = registry or get_tool_registry()
    toolkit = LangChainToolkit(registry)
    tools = toolkit.get_tools()

    if not tools:
        logger.warning("No tools available for agent creation")
        # Return a simple agent without tools
        from langchain.agents import create_openai_functions_agent

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                ("user", "{input}"),
            ]
        )
        agent = create_openai_functions_agent(llm, [], prompt)
        return AgentExecutor(agent=agent, tools=[], verbose=True)

    # Create tool-calling agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant with access to tools.
        Use the available tools when appropriate to help answer questions.
        
        Available tools: {tool_names}
        
        When using tools, be precise with your inputs and provide clear reasoning.""",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    tool_names = ", ".join([tool.name for tool in tools])
    prompt = prompt.partial(tool_names=tool_names)

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def get_tool_descriptions(registry: ToolRegistry = None) -> List[Dict[str, Any]]:
    """
    Get tool descriptions in LangChain-compatible format

    Args:
        registry: Tool registry (uses default if None)

    Returns:
        List of tool descriptions
    """
    registry = registry or get_tool_registry()
    tools = registry.list_tools(enabled_only=True)

    descriptions = []
    for tool in tools:
        descriptions.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "categories": tool.categories,
            }
        )

    return descriptions


def tool_selection_prompt(tool_descriptions: List[Dict[str, Any]]) -> str:
    """
    Generate a prompt for tool selection based on available tools

    Args:
        tool_descriptions: List of tool descriptions

    Returns:
        Formatted prompt string
    """
    if not tool_descriptions:
        return "No tools available."

    prompt_lines = ["Available tools:"]
    for tool in tool_descriptions:
        param_desc = ", ".join(
            [
                f"{name} ({config.get('type', 'any')})"
                for name, config in tool["parameters"].items()
            ]
        )
        prompt_lines.append(
            f"- {tool['name']}: {tool['description']} [Parameters: {param_desc}]"
        )

    return "\n".join(prompt_lines)


# Compatibility functions for existing LangChain code
def get_langchain_tools(registry: ToolRegistry = None) -> List[LangChainCoreTool]:
    """Get tools in LangChain format"""
    registry = registry or get_tool_registry()
    toolkit = LangChainToolkit(registry)
    return toolkit.get_tools()


def is_tool_available(tool_name: str, registry: ToolRegistry = None) -> bool:
    """Check if a specific tool is available and enabled"""
    registry = registry or get_tool_registry()
    tool = registry.get_tool(tool_name)
    return tool is not None and tool.enabled
