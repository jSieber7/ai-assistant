"""
Manager for Deep Agents integration.

This module provides the DeepAgentManager class, which wraps the deepagents
library and integrates it with the existing tool and LLM provider systems.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.llm_providers import provider_registry
from app.core.tools.execution.registry import tool_registry
from app.core.tools.base.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# In-memory store for conversation states.
# TODO: Replace with persistent storage (e.g., database)
_conversation_states: Dict[str, Dict[str, Any]] = {}


def adapt_tool_for_deepagents(tool: BaseTool) -> Callable:
    """
    Adapts a BaseTool instance to be compatible with the deepagents library.

    The deepagents library expects simple Python functions with type hints
    and docstrings for its tools. This adapter wraps the execute method
    of our BaseTool class.

    Args:
        tool: The BaseTool instance to adapt.

    Returns:
        A callable function compatible with deepagents.
    """

    async def adapted_tool(**kwargs: Any) -> Any:
        """
        Wrapper function for executing the tool.

        This function calls the original tool's execute_with_timeout method
        and handles the result.
        """
        try:
            result: ToolResult = await tool.execute_with_timeout(**kwargs)
            if result.success:
                return result.data
            else:
                # Raise an exception to signal failure to the agent
                raise Exception(f"Tool '{tool.name}' failed: {result.error}")
        except Exception as e:
            logger.error(f"Error executing adapted tool '{tool.name}': {str(e)}")
            raise

    # Set the name and docstring for the function to help the agent understand its purpose
    adapted_tool.__name__ = tool.name
    adapted_tool.__doc__ = tool.description

    return adapted_tool


class DeepAgentManager:
    """
    Manages the lifecycle and execution of a deepagents instance.

    This class is responsible for initializing the deepagents agent with the
    appropriate LLM and tools, and for invoking it while managing conversation state.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initializes the DeepAgentManager.

        Args:
            system_prompt: An optional system prompt to guide the agent's behavior.
        """
        self.agent: Any = None
        self.llm: Optional[BaseChatModel] = None
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with access to a variety of tools. "
            "Use them to accomplish complex, multi-step tasks."
        )
        self._initialized = False

    async def initialize(self):
        """
        Initializes the deepagents agent.

        This method should be called once during application startup. It fetches
        the default LLM, adapts the registered tools, and creates the agent.
        """
        if self._initialized:
            return

        logger.info("Initializing DeepAgentManager...")

        # 1. Get the default LLM
        default_provider = provider_registry.get_default_provider()
        if not default_provider:
            raise RuntimeError("No default LLM provider configured. Cannot initialize DeepAgentManager.")
        
        # We assume a default model for now, this could be made configurable
        model_info_list = await default_provider.list_models()
        if not model_info_list:
            raise RuntimeError(f"No models available for provider '{default_provider.name}'.")
        
        default_model_name = model_info_list[0].name
        self.llm = await default_provider.create_llm(default_model_name)
        logger.info(f"Using LLM: {default_model_name} from provider '{default_provider.name}'")

        # 2. Adapt the registered tools
        tools_to_register = []
        for tool in tool_registry.list_tools(enabled_only=True):
            try:
                adapted_tool = adapt_tool_for_deepagents(tool)
                tools_to_register.append(adapted_tool)
                logger.info(f"Adapted tool '{tool.name}' for deepagents.")
            except Exception as e:
                logger.error(f"Failed to adapt tool '{tool.name}': {str(e)}")
        
        if not tools_to_register:
            logger.warning("No tools were adapted for deepagents. The agent will have no tools.")

        # 3. Create the deepagents agent
        try:
            self.agent = create_deep_agent(
                llm=self.llm,
                tools=tools_to_register,
                system_prompt=self.system_prompt,
            )
            self._initialized = True
            logger.info("DeepAgentManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to create deepagents agent: {str(e)}")
            raise

    async def invoke(
        self,
        message: str,
        conversation_id: str,
    ) -> str:
        """
        Invokes the deepagents agent with a message.

        Args:
            message: The user's message.
            conversation_id: The ID of the conversation.

        Returns:
            The agent's text response.
        
        Raises:
            RuntimeError: If the manager has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeepAgentManager is not initialized. Call initialize() first.")

        logger.info(f"Invoking deepagent for conversation '{conversation_id}'")

        # 1. Load previous state if it exists
        current_state = _conversation_states.get(conversation_id, {"messages": []})

        # 2. Add the new user message to the state
        current_state["messages"].append({"role": "user", "content": message})

        # 3. Invoke the agent with the current state
        try:
            # The deepagents library expects the full state dictionary
            new_state = await self.agent.ainvoke(current_state)
            
            # 4. Save the new state
            _conversation_states[conversation_id] = new_state
            
            # 5. Extract and return the last assistant message
            last_message = new_state["messages"][-1]
            if last_message.get("role") == "assistant":
                return last_message.get("content", "Agent produced an empty response.")
            else:
                # This case should ideally not happen in a well-behaved agent
                logger.warning("Agent did not produce a final assistant message in the state.")
                return "Agent finished without a final response."

        except Exception as e:
            logger.error(f"Error during deepagent invocation: {str(e)}")
            # Re-raise the exception to be handled by the API layer
            raise

    def clear_conversation_state(self, conversation_id: str):
        """Clears the stored state for a given conversation."""
        if conversation_id in _conversation_states:
            del _conversation_states[conversation_id]
            logger.info(f"Cleared state for conversation '{conversation_id}'")


# Global instance of the DeepAgentManager
deep_agent_manager = DeepAgentManager()