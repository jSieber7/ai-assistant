"""
LangChain integration for the agent system.

This module provides integration between the agent system and LangChain's
tool calling API, allowing agents to be used as LangChain tools and
vice versa.
"""

from typing import Any, List, Optional
from langchain.tools import BaseTool as LangChainBaseTool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.tools.render import render_text_description
import logging

from .base import BaseAgent
from .registry import agent_registry
from ..tools.base import BaseTool
from ..tools.registry import tool_registry

logger = logging.getLogger(__name__)


class LangChainAgentTool(LangChainBaseTool):
    """
    LangChain tool wrapper for our agent system.

    This allows LangChain agents to use our custom agents as tools.
    """

    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    def _run(self, query: str) -> str:
        """Synchronous run method (not recommended for async environments)"""
        import asyncio

        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Asynchronous run method"""
        try:
            result = await self.agent.process_message(query)
            if result.success:
                return result.response
            else:
                return f"Agent error: {result.error}"
        except Exception as e:
            logger.error(f"LangChain agent tool failed: {str(e)}")
            return f"Tool execution failed: {str(e)}"

    @property
    def name(self) -> str:
        return f"agent_{self.agent.name}"

    @property
    def description(self) -> str:
        return f"Use the {self.agent.name} agent to help with: {self.agent.description}"


class AgentAsLangChainTool:
    """
    Adapter to use our agents as LangChain tools.
    """

    @staticmethod
    def create_tool(agent: BaseAgent) -> LangChainAgentTool:
        """Create a LangChain tool from an agent"""
        return LangChainAgentTool(
            agent=agent, name=f"agent_{agent.name}", description=agent.description
        )

    @staticmethod
    def create_tools_from_registry(
        agent_names: List[str] = None,
    ) -> List[LangChainAgentTool]:
        """Create LangChain tools from agents in the registry"""
        tools = []

        if agent_names:
            # Create tools for specific agents
            for agent_name in agent_names:
                agent = agent_registry.get_agent(agent_name)
                if agent:
                    tools.append(AgentAsLangChainTool.create_tool(agent))
        else:
            # Create tools for all active agents
            active_agents = agent_registry.list_agents(active_only=True)
            for agent in active_agents:
                tools.append(AgentAsLangChainTool.create_tool(agent))

        return tools


class LangChainToolAsAgentTool(BaseTool):
    """
    Adapter to use LangChain tools in our agent system.
    """

    def __init__(self, langchain_tool: LangChainBaseTool):
        super().__init__()
        self._langchain_tool = langchain_tool
        self._name = langchain_tool.name
        self._description = langchain_tool.description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def keywords(self) -> List[str]:
        # Extract keywords from description or use tool name
        return [self.name] + self.description.lower().split()[:5]

    async def execute(self, **kwargs) -> Any:
        """Execute the LangChain tool"""
        try:
            # LangChain tools typically expect a string input
            if "query" in kwargs:
                result = await self._langchain_tool.arun(kwargs["query"])
            elif "input" in kwargs:
                result = await self._langchain_tool.arun(kwargs["input"])
            else:
                # Try to convert kwargs to a query string
                query = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
                result = await self._langchain_tool.arun(query)

            return result
        except Exception as e:
            logger.error(f"LangChain tool execution failed: {str(e)}")
            raise


class LangChainAgentExecutor:
    """
    Advanced agent executor that integrates LangChain's agent capabilities
    with our custom agent system.
    """

    def __init__(
        self,
        llm,
        tools: List[LangChainBaseTool] = None,
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    ):
        self.llm = llm
        self.tools = tools or []
        self.agent_type = agent_type
        self.agent_executor: Optional[AgentExecutor] = None

    def initialize_agent(self):
        """Initialize the LangChain agent executor"""
        if not self.tools:
            logger.warning("No tools provided for LangChain agent")
            return

        try:
            self.agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=self.agent_type,
                verbose=True,
                handle_parsing_errors=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agent: {str(e)}")
            raise

    async def run(self, query: str, **kwargs) -> str:
        """Run the LangChain agent"""
        if not self.agent_executor:
            self.initialize_agent()

        try:
            result = await self.agent_executor.arun(input=query, **kwargs)
            return result
        except Exception as e:
            logger.error(f"LangChain agent execution failed: {str(e)}")
            return f"Agent execution failed: {str(e)}"

    def add_tool(self, tool: LangChainBaseTool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        # Reinitialize agent with new tools
        if self.agent_executor:
            self.initialize_agent()


class HybridAgentSystem:
    """
    Hybrid system that combines our agent system with LangChain's capabilities.

    This allows for flexible tool usage between both systems.
    """

    def __init__(self, llm, tool_registry, agent_registry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.langchain_agent: Optional[LangChainAgentExecutor] = None

        # Create LangChain tools from our agents
        self._setup_hybrid_system()

    def _setup_hybrid_system(self):
        """Setup the hybrid system by creating LangChain tools from our agents"""
        # Create LangChain tools from our agents
        agent_tools = AgentAsLangChainTool.create_tools_from_registry()

        # Create LangChain agent executor
        self.langchain_agent = LangChainAgentExecutor(
            llm=self.llm,
            tools=agent_tools,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    async def process_message(self, message: str, use_hybrid: bool = True) -> str:
        """
        Process a message using the hybrid system.

        Args:
            message: User message
            use_hybrid: Whether to use hybrid approach or fallback to our agent system

        Returns:
            Response text
        """
        if not use_hybrid or not self.langchain_agent:
            # Fallback to our agent system
            default_agent = self.agent_registry.get_default_agent()
            if default_agent:
                result = await default_agent.process_message(message)
                return result.response
            else:
                return "No agents available to process your request."

        try:
            # Use LangChain agent with our agents as tools
            return await self.langchain_agent.run(message)
        except Exception as e:
            logger.error(f"Hybrid agent system failed: {str(e)}")
            # Fallback to our agent system
            default_agent = self.agent_registry.get_default_agent()
            if default_agent:
                result = await default_agent.process_message(message)
                return result.response
            else:
                return "I encountered an error while processing your request."


def create_langchain_tool_from_agent(agent_name: str) -> Optional[LangChainAgentTool]:
    """Convenience function to create a LangChain tool from an agent"""
    agent = agent_registry.get_agent(agent_name)
    if agent:
        return AgentAsLangChainTool.create_tool(agent)
    return None


def register_langchain_tools_as_agent_tools(langchain_tools: List[LangChainBaseTool]):
    """
    Register LangChain tools as agent tools in our system.

    This allows our agents to use LangChain tools.
    """
    for tool in langchain_tools:
        agent_tool = LangChainToolAsAgentTool(tool)
        tool_registry.register(agent_tool, category="langchain")
        logger.info(f"Registered LangChain tool '{tool.name}' as agent tool")


def create_react_agent_prompt(tools: List[LangChainBaseTool]) -> str:
    """
    Create a ReAct-style prompt for tool-using agents.

    This is useful for creating custom agent prompts that integrate with our tool system.
    """
    tool_descriptions = render_text_description(tools)

    prompt = f"""You are an AI assistant with access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join([tool.name for tool in tools])}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

    return prompt


class ToolCallingLLM:
    """
    LLM wrapper that integrates tool calling capabilities.

    This can be used to create LLMs that are aware of our tool system.
    """

    def __init__(self, llm, tools: List[LangChainBaseTool] = None):
        self.llm = llm
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}

    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        """Invoke the LLM with tool calling awareness"""
        # Check if the prompt indicates tool usage
        if any(tool.name in prompt for tool in self.tools):
            # This is a simplified implementation
            # In a real scenario, you'd use LangChain's tool calling API
            response = await self.llm.ainvoke(prompt, **kwargs)
            return response

        # Regular LLM call
        return await self.llm.ainvoke(prompt, **kwargs)

    def add_tool(self, tool: LangChainBaseTool):
        """Add a tool to the LLM"""
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
