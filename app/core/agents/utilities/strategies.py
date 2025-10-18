"""
Tool selection strategies for intelligent agent decision making.

This module provides various strategies for selecting which tools to use
based on the user's query and context.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import re

from app.core.tools.base.base import BaseTool
from app.core.tools.execution.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolSelectionStrategy(ABC):
    """Abstract base class for tool selection strategies"""

    @abstractmethod
    async def select_tools(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_registry: ToolRegistry = None,
    ) -> List[BaseTool]:
        """
        Select tools that are relevant to the given query

        Args:
            query: User query
            context: Additional context information
            tool_registry: Tool registry to search from

        Returns:
            List of relevant tools, sorted by relevance
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy"""
        pass


class KeywordStrategy(ToolSelectionStrategy):
    """
    Simple keyword-based tool selection strategy.

    This strategy matches tools based on keyword presence in the query.
    """

    def __init__(self, min_keyword_matches: int = 1):
        self.min_keyword_matches = min_keyword_matches

    async def select_tools(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_registry: ToolRegistry = None,
    ) -> List[BaseTool]:
        """Select tools based on keyword matching"""
        if not tool_registry:
            return []

        query_lower = query.lower()
        relevant_tools = []

        for tool in tool_registry.list_tools(enabled_only=True):
            # Check if tool should be used based on its own logic
            if tool.should_use(query, context):
                relevant_tools.append(tool)
                continue

            # Additional keyword matching
            keyword_matches = 0
            for keyword in tool.keywords:
                if keyword.lower() in query_lower:
                    keyword_matches += 1

            if keyword_matches >= self.min_keyword_matches:
                relevant_tools.append(tool)

        # Sort by number of keyword matches (simple relevance scoring)
        relevant_tools.sort(
            key=lambda t: self._score_tool(t, query_lower), reverse=True
        )
        return relevant_tools

    def _score_tool(self, tool: BaseTool, query: str) -> int:
        """Score a tool based on keyword matches"""
        score = 0
        for keyword in tool.keywords:
            if keyword.lower() in query:
                score += 1
        return score

    def get_strategy_name(self) -> str:
        return "keyword_strategy"


class LLMStrategy(ToolSelectionStrategy):
    """
    LLM-based tool selection strategy.

    This strategy uses an LLM to intelligently select tools based on
    the query and available tool descriptions.
    """

    def __init__(self, llm, max_tools: int = 3):
        self.llm = llm
        self.max_tools = max_tools

    async def select_tools(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_registry: ToolRegistry = None,
    ) -> List[BaseTool]:
        """Use LLM to intelligently select tools"""
        if not tool_registry:
            return []

        # Get available tools
        available_tools = tool_registry.list_tools(enabled_only=True)
        if not available_tools:
            return []

        # Create tool descriptions for the LLM
        tool_descriptions = self._create_tool_descriptions(available_tools)

        # Use LLM to select tools
        selected_tool_names = await self._llm_select_tools(query, tool_descriptions)

        # Map back to tool objects
        selected_tools = []
        for tool_name in selected_tool_names:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                selected_tools.append(tool)

        return selected_tools[: self.max_tools]

    def _create_tool_descriptions(self, tools: List[BaseTool]) -> Dict[str, str]:
        """Create descriptions of tools for the LLM"""
        descriptions = {}
        for tool in tools:
            descriptions[tool.name] = (
                f"{tool.description}. Keywords: {', '.join(tool.keywords)}"
            )
        return descriptions

    async def _llm_select_tools(
        self, query: str, tool_descriptions: Dict[str, str]
    ) -> List[str]:
        """Use LLM to select tools based on query"""
        # Create prompt for tool selection
        prompt = self._create_selection_prompt(query, tool_descriptions)

        try:
            # Use LLM to get tool selection
            response = await self.llm.ainvoke(prompt)

            # Parse the response to extract tool names
            selected_tools = self._parse_llm_response(response.content)
            return selected_tools
        except Exception as e:
            logger.error(f"LLM tool selection failed: {str(e)}")
            return []

    def _create_selection_prompt(
        self, query: str, tool_descriptions: Dict[str, str]
    ) -> str:
        """Create prompt for LLM tool selection"""
        tools_text = "\n".join(
            [f"- {name}: {desc}" for name, desc in tool_descriptions.items()]
        )

        # Use string formatting to avoid potential SQL injection
        # The query is treated as a string literal, not as executable code
        # Sanitize the query to prevent injection attacks
        sanitized_query = (
            query.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        )

        # Use string formatting to avoid potential SQL injection
        # The query is treated as a string literal, not as executable code
        # Using f-string here is safe as we're not executing the query as code
        # nosec B608 - This is not a SQL query, it's a prompt template
        # nosec B608 - This is not a SQL query, it's a prompt template
        prompt = f"""Based on the user's query, select the most relevant tools from the available options.  # nosec B608

User Query: "{sanitized_query}"

Available Tools:
{tools_text}

Instructions:
1. Analyze the user's query and determine which tools would be most helpful
2. Select up to {self.max_tools} tools that are most relevant
3. Return only the tool names as a comma-separated list
4. If no tools are relevant, return an empty list

Selected tools:"""

        return prompt

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract tool names"""
        # Simple parsing - look for comma-separated tool names
        tool_names = []
        lines = response.strip().split("\n")

        for line in lines:
            # Look for tool names in the response
            if ":" in line or "-" in line:
                continue  # Skip lines that look like descriptions

            # Extract potential tool names
            parts = re.split(r"[,\s]+", line.strip())
            for part in parts:
                part = part.strip(".,;")
                if part and len(part) > 2:  # Basic validation
                    tool_names.append(part)

        return tool_names

    def get_strategy_name(self) -> str:
        return "llm_strategy"


class HybridStrategy(ToolSelectionStrategy):
    """
    Hybrid tool selection strategy combining multiple approaches.

    This strategy uses both keyword matching and LLM-based selection
    to get the best of both worlds.
    """

    def __init__(
        self,
        llm=None,
        keyword_strategy: KeywordStrategy = None,
        llm_strategy: LLMStrategy = None,
    ):
        self.keyword_strategy = keyword_strategy or KeywordStrategy()
        self.llm_strategy = llm_strategy
        if llm and not llm_strategy:
            self.llm_strategy = LLMStrategy(llm)

    async def select_tools(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_registry: ToolRegistry = None,
    ) -> List[BaseTool]:
        """Use hybrid approach to select tools"""
        if not tool_registry:
            return []

        # Get tools from keyword strategy
        keyword_tools = await self.keyword_strategy.select_tools(
            query, context, tool_registry
        )

        # Get tools from LLM strategy if available
        llm_tools = []
        if self.llm_strategy:
            llm_tools = await self.llm_strategy.select_tools(
                query, context, tool_registry
            )

        # Combine and deduplicate
        all_tools = {}
        for tool in keyword_tools + llm_tools:
            all_tools[tool.name] = tool

        # Return combined list, prioritizing tools that appear in both strategies
        combined_tools = list(all_tools.values())

        # Score tools based on strategy agreement
        scored_tools = []
        for tool in combined_tools:
            score = 0
            if tool in keyword_tools:
                score += 1
            if tool in llm_tools:
                score += 2  # LLM selection gets higher weight

            scored_tools.append((tool, score))

        # Sort by score and return
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in scored_tools]

    def get_strategy_name(self) -> str:
        return "hybrid_strategy"


class ToolSelectionManager:
    """Manager for coordinating multiple tool selection strategies"""

    def __init__(self, strategies: List[ToolSelectionStrategy] = None):
        self.strategies = strategies or []
        self.default_strategy = KeywordStrategy()

    def add_strategy(self, strategy: ToolSelectionStrategy):
        """Add a strategy to the manager"""
        self.strategies.append(strategy)

    async def select_tools(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_registry: ToolRegistry = None,
    ) -> List[BaseTool]:
        """Use all strategies to select tools and combine results"""
        if not self.strategies:
            return await self.default_strategy.select_tools(
                query, context, tool_registry
            )

        all_tools = {}

        for strategy in self.strategies:
            try:
                tools = await strategy.select_tools(query, context, tool_registry)
                for tool in tools:
                    if tool.name not in all_tools:
                        all_tools[tool.name] = {
                            "tool": tool,
                            "strategies": [strategy.get_strategy_name()],
                        }
                    else:
                        all_tools[tool.name]["strategies"].append(
                            strategy.get_strategy_name()
                        )
            except Exception as e:
                logger.error(
                    f"Strategy {strategy.get_strategy_name()} failed: {str(e)}"
                )

        # Sort by number of strategies that selected the tool
        sorted_tools = sorted(
            all_tools.values(), key=lambda x: len(x["strategies"]), reverse=True
        )

        return [item["tool"] for item in sorted_tools]
