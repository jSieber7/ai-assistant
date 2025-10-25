"""
Tool-heavy agent workflow using LangGraph.

This module implements a tool-intensive agent workflow that
extensively uses tools to gather and process information.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool as LangChainBaseTool
from langgraph import StateGraph, START, END

from ..agent_manager import AgentState
from ..tool_registry import tool_registry

logger = logging.getLogger(__name__)


async def plan_tool_usage(state: AgentState) -> AgentState:
    """
    Plan extensive tool usage for the task.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool usage plan
    """
    try:
        # Get the task
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            return state
            
        task = human_messages[-1].content
        
        # Get all available tools
        available_tools = tool_registry.list_tools(enabled_only=True)
        
        # Plan tool usage based on task analysis
        tool_plan = {
            "search_tools": [],
            "analysis_tools": [],
            "computation_tools": [],
            "content_tools": [],
            "utility_tools": [],
        }
        
        # Categorize tools
        for tool in available_tools:
            tool_categories = getattr(tool, 'categories', ['general'])
            
            if any(cat in ['search', 'web', 'information'] for cat in tool_categories):
                tool_plan["search_tools"].append(tool.name)
            elif any(cat in ['analysis', 'data', 'research'] for cat in tool_categories):
                tool_plan["analysis_tools"].append(tool.name)
            elif any(cat in ['calculation', 'math', 'computation'] for cat in tool_categories):
                tool_plan["computation_tools"].append(tool.name)
            elif any(cat in ['content', 'text', 'document'] for cat in tool_categories):
                tool_plan["content_tools"].append(tool.name)
            else:
                tool_plan["utility_tools"].append(tool.name)
        
        # Determine which tool categories are needed
        task_lower = task.lower()
        needed_categories = []
        
        if any(keyword in task_lower for keyword in 
               ["search", "find", "look up", "information", "data"]):
            needed_categories.extend(["search_tools", "analysis_tools"])
            
        if any(keyword in task_lower for keyword in 
               ["calculate", "compute", "math", "analyze", "process"]):
            needed_categories.extend(["analysis_tools", "computation_tools"])
            
        if any(keyword in task_lower for keyword in 
               ["generate", "create", "write", "content"]):
            needed_categories.extend(["content_tools", "utility_tools"])
            
        # Create execution plan
        execution_plan = []
        for category in needed_categories:
            if category in tool_plan:
                execution_plan.extend(tool_plan[category])
                
        # Remove duplicates and limit
        planned_tools = list(dict.fromkeys(execution_plan))[:8]  # Max 8 tools
        
        # Update state
        state.context["tool_plan"] = tool_plan
        state.context["planned_tools"] = planned_tools
        state.context["tool_categories_needed"] = needed_categories
        
        logger.info(f"Planned {len(planned_tools)} tools: {planned_tools}")
        return state
        
    except Exception as e:
        logger.error(f"Error in plan_tool_usage: {str(e)}")
        state.context["error"] = str(e)
        return state


async def execute_tools_parallel(state: AgentState) -> AgentState:
    """
    Execute planned tools in parallel batches.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool execution results
    """
    try:
        planned_tools = state.context.get("planned_tools", [])
        
        if not planned_tools:
            state.context["tool_results"] = []
            return state
            
        # Get tool instances
        tool_instances = []
        for tool_name in planned_tools:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                tool_instances.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found")
                
        if not tool_instances:
            state.context["tool_results"] = []
            return state
            
        # Get the task for tool execution
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        task = human_messages[-1].content if human_messages else ""
        
        # Execute tools in batches
        batch_size = 3  # Execute up to 3 tools in parallel
        all_results = []
        
        for i in range(0, len(tool_instances), batch_size):
            batch = tool_instances[i:i + batch_size]
            
            # Execute batch in parallel
            async def execute_single_tool(tool):
                try:
                    # Prepare tool input based on tool type
                    tool_input = await _prepare_tool_input(tool, task, state)
                    
                    result = await tool.ainvoke(tool_input)
                    
                    return {
                        "tool_name": tool.name,
                        "success": True,
                        "result": result,
                        "error": None,
                        "execution_time": 0.0,  # Would be measured in real implementation
                    }
                except Exception as e:
                    return {
                        "tool_name": tool.name,
                        "success": False,
                        "result": None,
                        "error": str(e),
                        "execution_time": 0.0,
                    }
                    
            # Execute batch
            tasks = [execute_single_tool(tool) for tool in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Tool execution exception: {str(result)}")
                    continue
                all_results.append(result)
                
            # Small delay between batches
            if i + batch_size < len(tool_instances):
                await asyncio.sleep(0.1)
                
        # Update state
        state.context["tool_results"] = all_results
        state.context["successful_tools"] = len([r for r in all_results if r["success"]])
        state.context["failed_tools"] = len([r for r in all_results if not r["success"]])
        
        logger.info(f"Executed {len(all_results)} tools, {len([r for r in all_results if r['success']])} successful")
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_tools_parallel: {str(e)}")
        state.context["error"] = str(e)
        state.context["tool_results"] = []
        return state


async def aggregate_results(state: AgentState) -> AgentState:
    """
    Aggregate and analyze results from multiple tools.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with aggregated results
    """
    try:
        tool_results = state.context.get("tool_results", [])
        
        if not tool_results:
            state.context["aggregated_results"] = {}
            return state
            
        # Separate successful and failed results
        successful_results = [r for r in tool_results if r["success"]]
        failed_results = [r for r in tool_results if not r["success"]]
        
        # Aggregate by tool category
        aggregated = {
            "successful_tools": successful_results,
            "failed_tools": failed_results,
            "search_results": [],
            "analysis_results": [],
            "computation_results": [],
            "content_results": [],
            "utility_results": [],
            "insights": [],
            "recommendations": [],
        }
        
        # Categorize successful results
        for result in successful_results:
            tool_name = result["tool_name"]
            tool = tool_registry.get_tool(tool_name)
            
            if tool:
                tool_categories = getattr(tool, 'categories', ['general'])
                
                if any(cat in ['search', 'web', 'information'] for cat in tool_categories):
                    aggregated["search_results"].append(result)
                elif any(cat in ['analysis', 'data', 'research'] for cat in tool_categories):
                    aggregated["analysis_results"].append(result)
                elif any(cat in ['calculation', 'math', 'computation'] for cat in tool_categories):
                    aggregated["computation_results"].append(result)
                elif any(cat in ['content', 'text', 'document'] for cat in tool_categories):
                    aggregated["content_results"].append(result)
                else:
                    aggregated["utility_results"].append(result)
                    
        # Generate insights from results
        insights = []
        
        if aggregated["search_results"]:
            insights.append(f"Found {len(aggregated['search_results'])} search results")
            
        if aggregated["analysis_results"]:
            insights.append(f"Completed {len(aggregated['analysis_results'])} analyses")
            
        if aggregated["computation_results"]:
            insights.append(f"Performed {len(aggregated['computation_results'])} computations")
            
        if failed_results:
            insights.append(f"{len(failed_results)} tools failed to execute")
            
        aggregated["insights"] = insights
        
        # Generate recommendations
        recommendations = []
        
        if len(successful_results) < len(tool_results):
            recommendations.append("Consider retrying failed tools or using alternatives")
            
        if not aggregated["search_results"] and "search" in state.context.get("tool_categories_needed", []):
            recommendations.append("Consider using search tools to gather more information")
            
        aggregated["recommendations"] = recommendations
        
        # Update state
        state.context["aggregated_results"] = aggregated
        
        logger.info(f"Aggregated {len(successful_results)} successful tool results")
        return state
        
    except Exception as e:
        logger.error(f"Error in aggregate_results: {str(e)}")
        state.context["error"] = str(e)
        return state


async def generate_comprehensive_response(state: AgentState) -> AgentState:
    """
    Generate comprehensive response based on all tool results.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with generated response
    """
    try:
        # Get LLM
        from ..llm_manager import llm_manager
        llm = state.context.get("llm") or await llm_manager.get_llm("gpt-4")
        
        # Get aggregated results
        aggregated = state.context.get("aggregated_results", {})
        
        # Build comprehensive system prompt
        system_prompt = """You are a comprehensive AI assistant that extensively uses tools to provide detailed, thorough responses.
        
        Guidelines:
        1. Synthesize information from all tool results
        2. Provide detailed, structured responses
        3. Acknowledge tool limitations or failures
        4. Offer follow-up suggestions when appropriate
        5. Organize information clearly with headings and bullet points
        6. Include specific data and sources from tools
        7. Provide actionable insights and recommendations
        """
        
        # Add tool results context
        if aggregated.get("successful_tools"):
            tool_summary = _format_tool_results(aggregated["successful_tools"])
            system_prompt += f"\n\nSuccessful Tool Results:\n{tool_summary}"
            
        if aggregated.get("failed_tools"):
            failed_summary = _format_tool_results(aggregated["failed_tools"])
            system_prompt += f"\n\nFailed Tool Results:\n{failed_summary}"
            
        if aggregated.get("insights"):
            insights_text = "\n".join([f"- {insight}" for insight in aggregated["insights"]])
            system_prompt += f"\n\nKey Insights:\n{insights_text}"
            
        if aggregated.get("recommendations"):
            recommendations_text = "\n".join([f"- {rec}" for rec in aggregated["recommendations"]])
            system_prompt += f"\n\nRecommendations:\n{recommendations_text}"
            
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
        state.context["response_comprehensive"] = True
        state.context["tools_used_count"] = len(aggregated.get("successful_tools", []))
        
        logger.debug(f"Generated comprehensive response of length {len(response.content)}")
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_comprehensive_response: {str(e)}")
        state.context["error"] = str(e)
        return state


async def should_continue_tools(state: AgentState) -> Literal["continue", "finalize"]:
    """
    Decide whether to continue using more tools or finalize response.
    
    Args:
        state: Current agent state
        
    Returns:
        Decision for next step
    """
    try:
        # Check iteration count
        if state.iteration_count >= state.max_iterations:
            return "finalize"
            
        # Check if we have enough information
        aggregated = state.context.get("aggregated_results", {})
        successful_tools = len(aggregated.get("successful_tools", []))
        
        # Continue if few tools succeeded and we have iterations left
        if successful_tools < 3 and state.iteration_count < state.max_iterations - 1:
            return "continue"
            
        return "finalize"
        
    except Exception as e:
        logger.error(f"Error in should_continue_tools: {str(e)}")
        return "finalize"


async def create_tool_heavy_workflow() -> StateGraph:
    """
    Create a tool-heavy agent workflow using LangGraph.
    
    Returns:
        Compiled StateGraph workflow
    """
    try:
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_tool_usage", plan_tool_usage)
        workflow.add_node("execute_tools_parallel", execute_tools_parallel)
        workflow.add_node("aggregate_results", aggregate_results)
        workflow.add_node("generate_comprehensive_response", generate_comprehensive_response)
        
        # Add edges
        workflow.add_edge(START, "plan_tool_usage")
        workflow.add_edge("plan_tool_usage", "execute_tools_parallel")
        workflow.add_edge("execute_tools_parallel", "aggregate_results")
        workflow.add_conditional_edges(
            "aggregate_results",
            should_continue_tools,
            {
                "continue": "plan_tool_usage",
                "finalize": "generate_comprehensive_response"
            }
        )
        workflow.add_edge("generate_comprehensive_response", END)
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info("Created tool-heavy workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create tool-heavy workflow: {str(e)}")
        raise


# Helper functions

async def _prepare_tool_input(tool: LangChainBaseTool, task: str, state: AgentState) -> Dict[str, Any]:
    """Prepare appropriate input for a specific tool"""
    # This is a simplified implementation
    # In practice, you'd want more sophisticated input preparation
    
    tool_name = tool.name.lower()
    task_lower = task.lower()
    
    # Basic input preparation based on tool name and task
    if "search" in tool_name:
        return {"query": task}
    elif "calculate" in tool_name or "math" in tool_name:
        # Try to extract mathematical expressions
        import re
        math_expr = re.search(r'[\d+\-\*/\(\)\s]+', task)
        if math_expr:
            return {"expression": math_expr.group()}
        else:
            return {"query": task}
    elif "shell" in tool_name:
        # Extract commands (be careful with security)
        return {"command": task}
    else:
        return {"query": task}


def _format_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    """Format tool results for display"""
    if not tool_results:
        return "No tool results"
        
    formatted = []
    for result in tool_results:
        tool_name = result["tool_name"]
        success = result["success"]
        
        if success:
            formatted.append(f"✓ {tool_name}: {result.get('result', 'Success')}")
        else:
            formatted.append(f"✗ {tool_name}: {result.get('error', 'Failed')}")
            
    return "\n".join(formatted)