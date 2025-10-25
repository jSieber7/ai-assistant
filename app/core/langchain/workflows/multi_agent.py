"""
Multi-agent collaboration workflow using LangGraph.

This module implements a multi-agent workflow that coordinates
different specialized agents to work on complex tasks.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool as LangChainBaseTool
from langgraph import StateGraph, START, END

from ..agent_manager import MultiAgentState, AgentType
from ..tool_registry import tool_registry

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for different agents in multi-agent system"""
    
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"


async def analyze_task(state: MultiAgentState) -> MultiAgentState:
    """
    Analyze the task and determine required agents.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with task analysis
    """
    try:
        # Get the task from messages
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            state.coordinator_decision = "no_task"
            return state
            
        task = human_messages[-1].content
        
        # Analyze task complexity and requirements
        task_analysis = {
            "complexity": _assess_complexity(task),
            "requires_research": _requires_research(task),
            "requires_analysis": _requires_analysis(task),
            "requires_synthesis": _requires_synthesis(task),
            "requires_validation": _requires_validation(task),
            "estimated_agents": _estimate_required_agents(task),
        }
        
        # Update state
        state.current_task = task
        state.context["task_analysis"] = task_analysis
        
        logger.debug(f"Task analysis: {task_analysis}")
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_task: {str(e)}")
        state.context["error"] = str(e)
        return state


async def coordinate_agents(state: MultiAgentState) -> MultiAgentState:
    """
    Coordinate which agents should work on the task.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with agent coordination
    """
    try:
        task_analysis = state.context.get("task_analysis", {})
        
        # Determine which agents to activate
        active_agents = []
        
        if task_analysis.get("requires_research"):
            active_agents.append(AgentRole.RESEARCHER.value)
            
        if task_analysis.get("requires_analysis"):
            active_agents.append(AgentRole.ANALYST.value)
            
        if task_analysis.get("requires_synthesis"):
            active_agents.append(AgentRole.SYNTHESIZER.value)
            
        if task_analysis.get("requires_validation"):
            active_agents.append(AgentRole.VALIDATOR.value)
            
        # Always include coordinator
        if AgentRole.COORDINATOR.value not in active_agents:
            active_agents.insert(0, AgentRole.COORDINATOR.value)
            
        # Update state
        state.active_agents = active_agents
        state.context["coordination_complete"] = True
        
        logger.info(f"Coordinated agents: {active_agents}")
        return state
        
    except Exception as e:
        logger.error(f"Error in coordinate_agents: {str(e)}")
        state.context["error"] = str(e)
        return state


async def execute_researcher(state: MultiAgentState) -> MultiAgentState:
    """
    Execute researcher agent to gather information.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with research results
    """
    try:
        # Get researcher agent
        from ..agent_manager import agent_manager
        researcher = await agent_manager.get_agent("researcher_agent")
        
        if not researcher:
            state.context["research_error"] = "Researcher agent not available"
            return state
            
        # Execute researcher with the task
        result = await researcher.ainvoke({
            "messages": state.messages,
            "current_task": state.current_task,
            "context": state.context,
        })
        
        # Store research results
        state.agent_results[AgentRole.RESEARCHER.value] = {
            "success": True,
            "data": result,
            "execution_time": result.get("execution_time", 0),
        }
        
        logger.info("Researcher agent execution completed")
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_researcher: {str(e)}")
        state.agent_results[AgentRole.RESEARCHER.value] = {
            "success": False,
            "error": str(e),
        }
        return state


async def execute_analyst(state: MultiAgentState) -> MultiAgentState:
    """
    Execute analyst agent to analyze information.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with analysis results
    """
    try:
        # Get analyst agent
        from ..agent_manager import agent_manager
        analyst = await agent_manager.get_agent("analyst_agent")
        
        if not analyst:
            state.context["analysis_error"] = "Analyst agent not available"
            return state
            
        # Prepare analysis context with research results
        analysis_context = state.context.copy()
        if AgentRole.RESEARCHER.value in state.agent_results:
            research_data = state.agent_results[AgentRole.RESEARCHER.value]
            if research_data.get("success"):
                analysis_context["research_results"] = research_data.get("data")
                
        # Execute analyst
        result = await analyst.ainvoke({
            "messages": state.messages,
            "current_task": state.current_task,
            "context": analysis_context,
        })
        
        # Store analysis results
        state.agent_results[AgentRole.ANALYST.value] = {
            "success": True,
            "data": result,
            "execution_time": result.get("execution_time", 0),
        }
        
        logger.info("Analyst agent execution completed")
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_analyst: {str(e)}")
        state.agent_results[AgentRole.ANALYST.value] = {
            "success": False,
            "error": str(e),
        }
        return state


async def execute_synthesizer(state: MultiAgentState) -> MultiAgentState:
    """
    Execute synthesizer agent to combine results.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with synthesis results
    """
    try:
        # Get synthesizer agent
        from ..agent_manager import agent_manager
        synthesizer = await agent_manager.get_agent("synthesizer_agent")
        
        if not synthesizer:
            state.context["synthesis_error"] = "Synthesizer agent not available"
            return state
            
        # Prepare synthesis context with all previous results
        synthesis_context = state.context.copy()
        synthesis_context["agent_results"] = state.agent_results
        
        # Execute synthesizer
        result = await synthesizer.ainvoke({
            "messages": state.messages,
            "current_task": state.current_task,
            "context": synthesis_context,
        })
        
        # Store synthesis results
        state.agent_results[AgentRole.SYNTHESIZER.value] = {
            "success": True,
            "data": result,
            "execution_time": result.get("execution_time", 0),
        }
        
        logger.info("Synthesizer agent execution completed")
        return state
        
    except Exception as e:
        logger.error(f"Error in execute_synthesizer: {str(e)}")
        state.agent_results[AgentRole.SYNTHESIZER.value] = {
            "success": False,
            "error": str(e),
        }
        return state


async def should_continue_collaboration(state: MultiAgentState) -> Literal["continue", "finalize"]:
    """
    Decide whether to continue collaboration or finalize.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Decision for next step
    """
    try:
        # Check if all active agents have completed
        active_agents = set(state.active_agents)
        completed_agents = set(state.agent_results.keys())
        
        # Add coordinator if not already completed
        if AgentRole.COORDINATOR.value in active_agents:
            completed_agents.add(AgentRole.COORDINATOR.value)
            
        # Check if all required agents completed successfully
        successful_agents = set([
            agent for agent, result in state.agent_results.items()
            if result.get("success", False)
        ])
        
        # Continue if some agents failed or haven't completed
        if len(completed_agents) < len(active_agents):
            return "continue"
            
        # Continue if any agents failed and retry is available
        failed_agents = active_agents - successful_agents
        if failed_agents and state.iteration_count < state.max_iterations:
            logger.info(f"Retrying failed agents: {failed_agents}")
            return "continue"
            
        return "finalize"
        
    except Exception as e:
        logger.error(f"Error in should_continue_collaboration: {str(e)}")
        return "finalize"


async def finalize_results(state: MultiAgentState) -> MultiAgentState:
    """
    Finalize multi-agent results into coherent response.
    
    Args:
        state: Current multi-agent state
        
    Returns:
        Updated state with final response
    """
    try:
        # Get LLM for finalization
        from ..llm_manager import llm_manager
        llm = await llm_manager.get_llm("gpt-4")
        
        # Prepare finalization context
        finalization_context = {
            "task": state.current_task,
            "agent_results": state.agent_results,
            "active_agents": state.active_agents,
            "iterations": state.iteration_count,
        }
        
        # Create system prompt for finalization
        system_prompt = """You are a coordinator agent synthesizing results from multiple specialized agents.
        
        Your task is to:
        1. Review all agent results
        2. Identify key insights and findings
        3. Resolve any conflicts or inconsistencies
        4. Create a comprehensive, coherent response
        5. Acknowledge any limitations or uncertainties
        
        Agent Roles:
        - Researcher: Gathers information and data
        - Analyst: Provides critical analysis and insights
        - Synthesizer: Combines information into coherent responses
        - Validator: Checks accuracy and completeness
        
        Provide a well-structured response that addresses the original task comprehensively."""
        
        # Create messages for LLM
        messages = [SystemMessage(content=system_prompt)]
        
        # Add task context
        if state.current_task:
            messages.append(HumanMessage(content=f"Task: {state.current_task}"))
            
        # Add agent results
        for agent_role, result in state.agent_results.items():
            if result.get("success"):
                messages.append(SystemMessage(content=f"{agent_role} result: {result.get('data', 'No data')}"))
            else:
                messages.append(SystemMessage(content=f"{agent_role} failed: {result.get('error', 'Unknown error')}"))
                
        # Generate final response
        response = await llm.ainvoke(messages)
        
        # Add final AI message to state
        ai_message = AIMessage(content=response.content)
        state.messages.append(ai_message)
        
        # Update context
        state.context["finalization_complete"] = True
        state.context["final_response"] = response.content
        state.context["successful_agents"] = len([
            r for r in state.agent_results.values() if r.get("success", False)
        ])
        
        logger.info("Multi-agent collaboration finalized")
        return state
        
    except Exception as e:
        logger.error(f"Error in finalize_results: {str(e)}")
        state.context["error"] = str(e)
        return state


async def create_multi_agent_workflow() -> StateGraph:
    """
    Create a multi-agent collaboration workflow using LangGraph.
    
    Returns:
        Compiled StateGraph workflow
    """
    try:
        # Create workflow graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("analyze_task", analyze_task)
        workflow.add_node("coordinate_agents", coordinate_agents)
        workflow.add_node("execute_researcher", execute_researcher)
        workflow.add_node("execute_analyst", execute_analyst)
        workflow.add_node("execute_synthesizer", execute_synthesizer)
        workflow.add_node("finalize_results", finalize_results)
        
        # Add edges
        workflow.add_edge(START, "analyze_task")
        workflow.add_edge("analyze_task", "coordinate_agents")
        
        # Add conditional edges for agent execution
        workflow.add_conditional_edges(
            "coordinate_agents",
            lambda state: "execute_researcher" if AgentRole.RESEARCHER.value in state.active_agents else "skip_researcher",
            {
                "execute_researcher": "execute_researcher",
                "skip_researcher": "execute_analyst"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_researcher",
            lambda state: "execute_analyst" if AgentRole.ANALYST.value in state.active_agents else "execute_synthesizer",
            {
                "execute_analyst": "execute_analyst",
                "execute_synthesizer": "execute_synthesizer"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_analyst",
            lambda state: "execute_synthesizer" if AgentRole.SYNTHESIZER.value in state.active_agents else "should_continue",
            {
                "execute_synthesizer": "execute_synthesizer",
                "should_continue": "should_continue"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_synthesizer",
            should_continue_collaboration,
            {
                "continue": "coordinate_agents",
                "finalize": "finalize_results"
            }
        )
        
        workflow.add_edge("finalize_results", END)
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info("Created multi-agent collaboration workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create multi-agent workflow: {str(e)}")
        raise


# Helper functions for task analysis

def _assess_complexity(task: str) -> str:
    """Assess the complexity of a task"""
    complexity_indicators = {
        "high": ["analyze", "research", "compare", "evaluate", "synthesize", "comprehensive"],
        "medium": ["explain", "describe", "summarize", "outline"],
        "low": ["what is", "define", "simple", "basic"],
    }
    
    task_lower = task.lower()
    
    for level, indicators in complexity_indicators.items():
        if any(indicator in task_lower for indicator in indicators):
            return level
            
    return "medium"


def _requires_research(task: str) -> bool:
    """Check if task requires research"""
    research_keywords = [
        "research", "find", "search", "investigate", "gather information",
        "what is", "tell me about", "compare", "analyze data"
    ]
    
    task_lower = task.lower()
    return any(keyword in task_lower for keyword in research_keywords)


def _requires_analysis(task: str) -> bool:
    """Check if task requires analysis"""
    analysis_keywords = [
        "analyze", "evaluate", "critique", "assess", "examine",
        "pros and cons", "advantages", "disadvantages", "implications"
    ]
    
    task_lower = task.lower()
    return any(keyword in task_lower for keyword in analysis_keywords)


def _requires_synthesis(task: str) -> bool:
    """Check if task requires synthesis"""
    synthesis_keywords = [
        "synthesize", "combine", "integrate", "merge", "summarize",
        "comprehensive", "overview", "holistic", "complete picture"
    ]
    
    task_lower = task.lower()
    return any(keyword in task_lower for keyword in synthesis_keywords)


def _requires_validation(task: str) -> bool:
    """Check if task requires validation"""
    validation_keywords = [
        "validate", "verify", "confirm", "check", "ensure accuracy",
        "fact check", "review", "quality check"
    ]
    
    task_lower = task.lower()
    return any(keyword in task_lower for keyword in validation_keywords)


def _estimate_required_agents(task: str) -> int:
    """Estimate number of agents required for task"""
    # Count different types of requirements
    requirements = 0
    
    if _requires_research(task):
        requirements += 1
    if _requires_analysis(task):
        requirements += 1
    if _requires_synthesis(task):
        requirements += 1
    if _requires_validation(task):
        requirements += 1
        
    # Always need coordinator for multi-agent tasks
    if requirements > 1:
        requirements += 1
        
    return requirements