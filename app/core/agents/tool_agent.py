"""
Intelligent agent that can dynamically select and use tools based on context.

This module provides the ToolAgent class which integrates LLM reasoning with
tool execution capabilities, enabling intelligent tool selection and usage.
"""

from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
import json

from .base import ToolCallingAgent, AgentResult, AgentState
from .strategies import ToolSelectionStrategy, KeywordStrategy
from ..tools.base import ToolResult
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolAgent(ToolCallingAgent):
    """
    Intelligent agent that can dynamically select and use tools.

    This agent uses a combination of strategies to decide when and which tools
    to use, then executes them and incorporates the results into its response.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm,
        selection_strategy: ToolSelectionStrategy = None,
        max_iterations: int = 5,
        max_tool_calls_per_message: int = 3,
    ):
        super().__init__(tool_registry, max_iterations)
        self.llm = llm
        self.selection_strategy = selection_strategy or KeywordStrategy()
        self._max_tool_calls_per_message = max_tool_calls_per_message

        # Agent configuration
        self._enable_tool_fallback = True
        self._tool_timeout = 30
        self._thinking_timeout = 60

    @property
    def name(self) -> str:
        return "tool_agent"

    @property
    def description(self) -> str:
        return "Intelligent agent that can dynamically select and use tools based on context"

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Process a message using intelligent tool selection and execution
        """
        start_time = time.time()
        self.state = AgentState.THINKING

        if conversation_id:
            self._current_conversation_id = conversation_id
        else:
            self.start_conversation()

        # Add user message to conversation history
        self.add_to_conversation("user", message, {"timestamp": start_time})

        try:
            # Decide whether to use tools
            tool_decision = await self.decide_tool_usage(message, context)

            tool_results = []
            final_response = ""

            if tool_decision.get("use_tools", False):
                # Execute tools and collect results
                tool_results = await self._execute_tools(
                    tool_decision["selected_tools"], message, context
                )

                # Generate response incorporating tool results
                final_response = await self.generate_response(
                    message, tool_results, context
                )
            else:
                # Generate response without tools
                final_response = await self._generate_direct_response(message, context)

            execution_time = time.time() - start_time

            # Add assistant response to conversation history
            self.add_to_conversation(
                "assistant",
                final_response,
                {
                    "execution_time": execution_time,
                    "tool_results_count": len(tool_results),
                    "tools_used": [tr.tool_name for tr in tool_results],
                },
            )

            self._usage_count += 1
            self._last_used = time.time()
            self.state = AgentState.IDLE

            return AgentResult(
                success=True,
                response=final_response,
                tool_results=tool_results,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={
                    "tool_decision": tool_decision,
                    "iterations": 1,  # For future multi-iteration support
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent processing failed: {str(e)}"
            logger.error(error_msg)
            self.state = AgentState.ERROR

            return AgentResult(
                success=False,
                response="I encountered an error while processing your request.",
                tool_results=[],
                error=error_msg,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={"error_type": type(e).__name__},
            )

    async def decide_tool_usage(
        self, message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Decide whether and which tools to use for the given message
        """
        context = context or {}

        # Check if tools are available
        available_tools = self.tool_registry.list_tools(enabled_only=True)
        if not available_tools:
            return {
                "use_tools": False,
                "selected_tools": [],
                "reason": "No tools available",
            }

        # Use selection strategy to choose tools
        selected_tools = await self.selection_strategy.select_tools(
            message, context, self.tool_registry
        )

        if not selected_tools:
            return {
                "use_tools": False,
                "selected_tools": [],
                "reason": "No relevant tools found",
            }

        # Limit the number of tools to avoid excessive calls
        selected_tools = selected_tools[: self._max_tool_calls_per_message]

        return {
            "use_tools": True,
            "selected_tools": [tool.name for tool in selected_tools],
            "reason": f"Found {len(selected_tools)} relevant tools",
            "strategy": self.selection_strategy.get_strategy_name(),
        }

    async def _execute_tools(
        self, tool_names: List[str], message: str, context: Dict[str, Any] = None
    ) -> List[ToolResult]:
        """
        Execute the selected tools and return their results
        """
        tool_results = []
        context = context or {}

        for tool_name in tool_names:
            try:
                # Prepare parameters for the tool
                tool_params = await self._prepare_tool_parameters(
                    tool_name, message, context
                )

                # Execute the tool
                result = await self.execute_tool(tool_name, **tool_params)
                tool_results.append(result)

                # If tool failed and we have fallback enabled, consider alternatives
                if not result.success and self._enable_tool_fallback:
                    logger.warning(f"Tool {tool_name} failed, considering fallback")
                    # Could implement fallback logic here

            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                tool_results.append(
                    ToolResult(
                        success=False,
                        data=None,
                        error=f"Execution error: {str(e)}",
                        tool_name=tool_name,
                        execution_time=0.0,
                    )
                )

        return tool_results

    async def _prepare_tool_parameters(
        self, tool_name: str, message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Prepare parameters for tool execution
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"query": message}  # Default parameter

        # Use tool's default parameters or customize based on context
        tool_params = {}

        # For most tools, the primary parameter will be the query/message
        expected_params = tool.parameters
        if "query" in expected_params:
            tool_params["query"] = message
        elif "input" in expected_params:
            tool_params["input"] = message
        else:
            # Use the first required parameter or default to "query"
            required_params = [
                p
                for p, config in expected_params.items()
                if config.get("required", False)
            ]
            if required_params:
                tool_params[required_params[0]] = message
            else:
                tool_params["query"] = message

        # Add context if the tool expects it
        if "context" in expected_params:
            tool_params["context"] = context

        return tool_params

    async def generate_response(
        self,
        message: str,
        tool_results: List[ToolResult] = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate final response after tool execution
        """
        tool_results = tool_results or []
        context = context or {}

        # Filter successful tool results
        successful_results = [tr for tr in tool_results if tr.success]

        if not successful_results:
            # If no tools were successful, generate a direct response
            return await self._generate_direct_response(message, context)

        # Create prompt that incorporates tool results
        prompt = self._create_tool_integration_prompt(
            message, successful_results, context
        )

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt), timeout=self._thinking_timeout
            )
            return response.content.strip()
        except asyncio.TimeoutError:
            logger.warning("LLM response generation timed out")
            return "I'm having trouble processing your request right now. Please try again."
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return await self._generate_fallback_response(message, successful_results)

    async def _generate_direct_response(
        self, message: str, context: Dict[str, Any] = None
    ) -> str:
        """Generate response without using tools"""
        prompt = f"""Please respond to the following user message:

User: {message}

Provide a helpful and informative response."""

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt), timeout=self._thinking_timeout
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Direct response generation failed: {str(e)}")
            return "I'm here to help! How can I assist you today?"

    def _create_tool_integration_prompt(
        self,
        message: str,
        tool_results: List[ToolResult],
        context: Dict[str, Any] = None,
    ) -> str:
        """Create prompt that incorporates tool results"""
        tool_info = []

        for i, result in enumerate(tool_results):
            tool_data = result.data
            if isinstance(tool_data, dict):
                # Format dictionary data nicely
                formatted_data = json.dumps(tool_data, indent=2)
            elif isinstance(tool_data, (list, tuple)):
                # Format list data
                formatted_data = json.dumps(tool_data, indent=2)
            else:
                formatted_data = str(tool_data)

            tool_info.append(
                f"""
Tool {i+1}: {result.tool_name}
Execution Time: {result.execution_time:.2f}s
Result: {formatted_data}
"""
            )

        tool_info_text = "\n".join(tool_info)

        prompt = f"""You are an AI assistant that has access to various tools. 
I have executed some tools based on the user's request and gathered the following information:

User's original message: "{message}"

Tool Results:
{tool_info_text}

Based on the tool results above, please provide a comprehensive and helpful response to the user. 
Incorporate the information from the tools naturally into your response. 
If the tool results don't fully answer the question, acknowledge what you found and suggest next steps.

Your response:"""

        return prompt

    async def _generate_fallback_response(
        self, message: str, tool_results: List[ToolResult]
    ) -> str:
        """Generate a fallback response when LLM fails"""
        if tool_results:
            # We have tool results but LLM failed - create a simple response
            tool_names = [tr.tool_name for tr in tool_results]
            return f"I've gathered some information using tools ({', '.join(tool_names)}), but I'm having trouble formulating a complete response. The data is available, but I need to process it differently."
        else:
            return "I'm experiencing some technical difficulties. Please try your request again."


class AdvancedToolAgent(ToolAgent):
    """
    Advanced agent with multi-iteration tool usage and reasoning capabilities.

    This agent can perform multiple iterations of tool usage and reasoning
    to solve complex problems.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm,
        selection_strategy: ToolSelectionStrategy = None,
        max_iterations: int = 3,
    ):
        super().__init__(tool_registry, llm, selection_strategy, max_iterations)
        self._reasoning_enabled = True

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Multi-iteration processing with reasoning between tool calls
        """
        start_time = time.time()
        self.state = AgentState.THINKING

        if conversation_id:
            self._current_conversation_id = conversation_id
        else:
            self.start_conversation()

        self.add_to_conversation("user", message, {"timestamp": start_time})

        try:
            all_tool_results = []
            current_context = context or {}
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1
                logger.debug(f"Agent iteration {iteration}")

                # Decide tool usage for this iteration
                tool_decision = await self.decide_tool_usage(message, current_context)

                if not tool_decision.get("use_tools", False):
                    break  # No more tools needed

                # Execute tools for this iteration
                tool_results = await self._execute_tools(
                    tool_decision["selected_tools"], message, current_context
                )
                all_tool_results.extend(tool_results)

                # Update context with tool results
                current_context.update(
                    {
                        f"tool_result_{iteration}": tool_results,
                        "previous_tool_results": all_tool_results,
                    }
                )

                # Check if we should continue
                if not self._should_continue_iteration(tool_results, iteration):
                    break

            # Generate final response
            final_response = await self.generate_response(
                message, all_tool_results, current_context
            )

            execution_time = time.time() - start_time

            self.add_to_conversation(
                "assistant",
                final_response,
                {
                    "execution_time": execution_time,
                    "tool_results_count": len(all_tool_results),
                    "iterations": iteration,
                    "tools_used": [tr.tool_name for tr in all_tool_results],
                },
            )

            self._usage_count += 1
            self._last_used = time.time()
            self.state = AgentState.IDLE

            return AgentResult(
                success=True,
                response=final_response,
                tool_results=all_tool_results,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={
                    "iterations": iteration,
                    "max_iterations_reached": iteration >= self.max_iterations,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Advanced agent processing failed: {str(e)}"
            logger.error(error_msg)
            self.state = AgentState.ERROR

            return AgentResult(
                success=False,
                response="I encountered an error while processing your complex request.",
                tool_results=[],
                error=error_msg,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
            )

    def _should_continue_iteration(
        self, tool_results: List[ToolResult], iteration: int
    ) -> bool:
        """
        Determine if we should continue with another iteration
        """
        if iteration >= self.max_iterations:
            return False

        # If no tools were successful in this iteration, stop
        successful_results = [tr for tr in tool_results if tr.success]
        if not successful_results:
            return False

        # Additional logic for determining continuation
        # For example, check if tools produced actionable results
        actionable_results = any(
            tr.data is not None and len(str(tr.data)) > 10 for tr in successful_results
        )

        return actionable_results
