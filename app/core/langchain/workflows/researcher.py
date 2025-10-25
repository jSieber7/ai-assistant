"""
Researcher agent workflow using LangGraph.

This module implements a researcher agent workflow that specializes
in gathering information and conducting research.
"""

import logging
from typing import Dict, Any, List, Optional, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph import StateGraph, START, END

from ..agent_manager import AgentState
from ..tool_registry import tool_registry

logger = logging.getLogger(__name__)


async def analyze_research_needs(state: AgentState) -> AgentState:
    """Analyze what research is needed for the task"""
    try:
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            return state
            
        task = human_messages[-1].content
        
        # Identify research requirements
        research_needs = {
            "web_search": any(keyword in task.lower() for keyword in 
                          ["search", "find", "look up", "information about", "what is"]),
            "deep_dive": any(keyword in task.lower() for keyword in 
                         ["analyze", "investigate", "research", "study"]),
            "multiple_sources": any(keyword in task.lower() for keyword in 
                               ["compare", "versus", "vs", "alternative", "options"]),
            "recent_info": any(keyword in task.lower() for keyword in 
                            ["latest", "recent", "current", "news", "updates"]),
        }
        
        state.context["research_needs"] = research_needs
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_research_needs: {str(e)}")
        state.context["error"] = str(e)
        return state


async def gather_information(state: AgentState) -> AgentState:
    """Gather information using appropriate research tools"""
    try:
        research_needs = state.context.get("research_needs", {})
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        task = human_messages[-1].content if human_messages else ""
        
        # Get relevant search tools
        search_tools = tool_registry.find_relevant_tools(
            task,
            max_results=3,
            category="search"
        )
        
        # Execute search tools
        tool_results = []
        for tool in search_tools:
            try:
                result = await tool.ainvoke({"query": task})
                tool_results.append({
                    "tool_name": tool.name,
                    "success": True,
                    "data": result,
                    "source": tool.name,
                })
            except Exception as e:
                tool_results.append({
                    "tool_name": tool.name,
                    "success": False,
                    "error": str(e),
                    "source": tool.name,
                })
        
        state.context["search_results"] = tool_results
        return state
        
    except Exception as e:
        logger.error(f"Error in gather_information: {str(e)}")
        state.context["error"] = str(e)
        return state


async def synthesize_findings(state: AgentState) -> AgentState:
    """Synthesize research findings into coherent information"""
    try:
        from ..llm_manager import llm_manager
        llm = await llm_manager.get_llm("gpt-4")
        
        search_results = state.context.get("search_results", [])
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        task = human_messages[-1].content if human_messages else ""
        
        # Build synthesis prompt
        system_prompt = """You are a research assistant. Synthesize search results into comprehensive, accurate information.
        
        Guidelines:
        1. Combine information from multiple sources
        2. Identify key facts and insights
        3. Note any contradictions or uncertainties
        4. Organize information logically
        5. Cite sources when possible
        6. Distinguish between facts and opinions
        """
        
        # Add search results to prompt
        if search_results:
            results_text = "\n\n".join([
                f"Source: {result.get('source', 'Unknown')}\n"
                f"Success: {result.get('success', False)}\n"
                f"Information: {result.get('data', 'No data')}\n"
                f"Error: {result.get('error', 'None')}"
                for result in search_results
            ])
            system_prompt += f"\n\nSearch Results:\n{results_text}"
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=task)]
        response = await llm.ainvoke(messages)
        
        # Store synthesized findings
        state.context["synthesized_findings"] = response.content
        return state
        
    except Exception as e:
        logger.error(f"Error in synthesize_findings: {str(e)}")
        state.context["error"] = str(e)
        return state


async def create_researcher_workflow() -> StateGraph:
    """Create a researcher agent workflow using LangGraph"""
    try:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_research_needs", analyze_research_needs)
        workflow.add_node("gather_information", gather_information)
        workflow.add_node("synthesize_findings", synthesize_findings)
        
        # Add edges
        workflow.add_edge(START, "analyze_research_needs")
        workflow.add_edge("analyze_research_needs", "gather_information")
        workflow.add_edge("gather_information", "synthesize_findings")
        workflow.add_edge("synthesize_findings", END)
        
        compiled_workflow = workflow.compile()
        logger.info("Created researcher workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create researcher workflow: {str(e)}")
        raise