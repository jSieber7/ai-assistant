"""
Conversational agent workflow using LangGraph.

This module implements a conversational agent workflow that handles
natural conversation with optional tool usage.
"""

import logging
from typing import Dict, Any, List, Optional, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool as LangChainBaseTool
from langgraph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from ..agent_manager import AgentState
from ..tool_registry import tool_registry

logger = logging.getLogger(__name__)


async def understand_intent(state: AgentState) -> AgentState:
    """
    Understand user intent and determine if tools are needed.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with intent analysis
    """
    try:
        # Get the last human message
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            return state
            
        last_message = human_messages[-1].content
        
        # Simple intent analysis (could be enhanced with LLM)
        intent_analysis = {
            "needs_search": any(keyword in last_message.lower() for keyword in 
                          ["search", "find", "look up", "what is", "tell me about"]),
            "needs_calculation": any(keyword in last_message.lower() for keyword in 
                               ["calculate", "compute", "math", "solve"]),
            "needs_code": any(keyword in last_message.lower() for keyword in 
                          ["code", "program", "script", "implement"]),
            "is_question": "?" in last_message or any(keyword in last_message.lower() for keyword in 
                                          ["what", "how", "why", "when", "where"]),
        }
        
        # Update state with intent analysis
        state.context["intent_analysis"] = intent_analysis
        
        logger.debug(f"Intent analysis: {intent_analysis}")
        return state
        
    except Exception as e:
        logger.error(f"Error in understand_intent: {str(e)}")
        state.context["error"] = str(e)
        return state


async def select_tools(state: AgentState) -> AgentState:
    """
    Select appropriate tools based on intent analysis.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with selected tools
    """
    try:
        intent_analysis = state.context.get("intent_analysis", {})
        
        # Get available tools
        available_tools = tool_registry.list_tools(enabled_only=True)
        
        selected_tools = []
        
        # Select tools based on intent
        if intent_analysis.get("needs_search"):
            search_tools = tool_registry.find_relevant_tools(
                state.messages[-1].content if state.messages else "",
                max_results=2,
                category="search"
            )
            selected_tools.extend(search_tools)
            
        if intent_analysis.get("needs_calculation"):
            calc_tools = tool_registry.find_relevant_tools(
                state.messages[-1].content if state.messages else "",
                max_results=1,
                category="calculation"
            )
            selected_tools.extend(calc_tools)
            
        if intent_analysis.get("needs_code"):
            code_tools = tool_registry.find_relevant_tools(
                state.messages[-1].content if state.messages else "",
                max_results=2,
                category="code"
            )
            selected_tools.extend(code_tools)
            
        # Remove duplicates
        selected_tool_names = list({tool.name for tool in selected_tools})
        selected_tools = [tool for tool in selected_tools if tool.name in selected_tool_names]
        
        # Update state
        state.context["selected_tools"] = selected_tools
        state.context["tool_count"] = len(selected_tools)
        
        logger.debug(f"Selected {len(selected_tools)} tools: {[t.name for t in selected_tools]}")
        return state
        
    except Exception as e:
        logger.error(f"Error in select_tools: {str(e)}")
        state.context["error"] = str(e)
        return state


async def should_use_tools(state: AgentState) -> Literal["use_tools", "generate_response"]:
    """
    Decide whether to use tools or generate direct response.
    
    Args:
        state: Current agent state
        
    Returns:
        Decision for next step
    """
    try:
        # Check if tools were selected
        selected_tools = state.context.get("selected_tools", [])
        
        if selected_tools and state.context.get("intent_analysis", {}):
            intent_analysis = state.context["intent_analysis"]
            
            # Use tools if any intent indicates need
            if (intent_analysis.get("needs_search") or 
                intent_analysis.get("needs_calculation") or 
                intent_analysis.get("needs_code")):
                return "use_tools"
                
        return "generate_response"
        
    except Exception as e:
        logger.error(f"Error in should_use_tools: {str(e)}")
        return "generate_response"


async def execute_tools(state: AgentState) -> AgentState:
    """
    Execute selected tools and collect results.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool results
    """
    try:
        selected_tools = state.context.get("selected_tools", [])
        
        if not selected_tools:
            state.context["tool_results"] = []
            return state
            
        tool_results = []
        
        # Get the last human message for tool execution
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            state.context["tool_results"] = []
            return state
            
        last_message = human_messages[-1].content
        
        # Execute tools (in parallel if possible)
        import asyncio
        
        async def execute_single_tool(tool):
            try:
                # Extract tool parameters from message
                # This is a simplified approach - in practice, you'd use
                # more sophisticated parameter extraction
                tool_input = {"query": last_message}
                
                result = await tool.ainvoke(tool_input)
                
                return {
                    "tool_name": tool.name,
                    "success": True,
                    "result": result,
                    "error": None,
                }
            except Exception as e:
                return {
                    "tool_name": tool.name,
                    "success": False,
                    "result": None,
                    "error": str(e),
                }
                
        # Execute tools with limited parallelism
        max_parallel = min(3, len(selected_tools))
        tasks = [execute_single_tool(tool) for tool in selected_tools[:max_parallel]]
        
        if len(selected_tools) > max_parallel:
            # Execute remaining tools sequentially
            sequential_results = []
            for tool in selected_tools[max_parallel:]:
                result = await execute_single_tool(tool)
                sequential_results.append(result)
        else:
            sequential_results = []
            
        # Wait for parallel execution
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        
        for result in parallel_results:
            if isinstance(result, Exception):
                logger.error(f"Tool execution exception: {str(result)}")
                continue
            all_results.append(result)
            
        all_results.extend(sequential_results)
        
        # Update state
        state.context["tool_results"] = all_results
        state.context["successful_tools"] = len([r for r in all_results if r["success"]])
        state.context["failed_tools"] = len([r for r in all_results if not r["success"]])
        
        logger.info(f"Executed {len(all_results)} tools, {len([r for r in all_results if r['success']])} successful")
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_tools: {str(e)}")
        state.context["error"] = str(e)
        state.context["tool_results"] = []
        return state


async def generate_response(state: AgentState) -> AgentState:
    """
    Generate response based on conversation context and tool results.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with generated response
    """
    try:
        # Get LLM from context or use default
        llm = state.context.get("llm")
        if not llm:
            from ..llm_manager import llm_manager
            llm = await llm_manager.get_llm("gpt-4")
            
        # Prepare context for response generation
        tool_results = state.context.get("tool_results", [])
        intent_analysis = state.context.get("intent_analysis", {})
        
        # Build system prompt
        system_prompt = """You are a helpful AI assistant engaging in natural conversation.
        
        Guidelines:
        - Be conversational and friendly
        - Provide thoughtful, detailed responses
        - Use tool results when available to inform your answer
        - If tools failed, acknowledge the limitation
        - Maintain context of the ongoing conversation
        """
        
        # Add tool results context
        if tool_results:
            tool_summary = "\n\n".join([
                f"Tool: {result['tool_name']}\n"
                f"Success: {result['success']}\n"
                f"Result: {result.get('result', 'No result')}\n"
                f"Error: {result.get('error', 'None')}"
                for result in tool_results
            ])
            system_prompt += f"\n\nTool Results:\n{tool_summary}"
            
        # Create messages for LLM
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state.messages)
        
        # Generate response
        response = await llm.ainvoke(messages)
        
        # Add AI message to state
        ai_message = AIMessage(content=response.content)
        state.messages.append(ai_message)
        
        # Update context
        state.context["response_generated"] = True
        state.context["response_length"] = len(response.content)
        
        logger.debug(f"Generated response of length {len(response.content)}")
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        state.context["error"] = str(e)
        return state


async def create_conversational_workflow() -> StateGraph:
    """
    Create a conversational agent workflow using LangGraph.
    
    Returns:
        Compiled StateGraph workflow
    """
    try:
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand_intent", understand_intent)
        workflow.add_node("select_tools", select_tools)
        workflow.add_node("execute_tools", execute_tools)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge(START, "understand_intent")
        workflow.add_edge("understand_intent", "select_tools")
        workflow.add_conditional_edges(
            "select_tools",
            should_use_tools,
            {
                "use_tools": "execute_tools",
                "generate_response": "generate_response"
            }
        )
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info("Created conversational workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create conversational workflow: {str(e)}")
        raise


# Additional helper functions for enhanced conversational capabilities

async def handle_follow_up(state: AgentState) -> AgentState:
    """
    Handle follow-up questions and clarification.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with follow-up handling
    """
    try:
        # Check if response needs clarification
        last_ai_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
                
        if last_ai_message:
            # Simple heuristic for clarification needs
            clarification_indicators = [
                "Did you mean", "Would you like", "Are you looking for",
                "Could you clarify", "What specifically"
            ]
            
            needs_clarification = any(
                indicator in last_ai_message.content 
                for indicator in clarification_indicators
            )
            
            state.context["needs_clarification"] = needs_clarification
            
        return state
        
    except Exception as e:
        logger.error(f"Error in handle_follow_up: {str(e)}")
        state.context["error"] = str(e)
        return state


async def maintain_context(state: AgentState) -> AgentState:
    """
    Maintain conversation context and coherence.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with context maintenance
    """
    try:
        # Extract key topics and entities from conversation
        messages_text = " ".join([msg.content for msg in state.messages[-5:]])
        
        # Simple topic extraction (could be enhanced with NLP)
        topics = []
        entities = []
        
        # Update context
        state.context["recent_topics"] = topics
        state.context["recent_entities"] = entities
        state.context["conversation_length"] = len(state.messages)
        
        return state
        
    except Exception as e:
        logger.error(f"Error in maintain_context: {str(e)}")
        state.context["error"] = str(e)
        return state