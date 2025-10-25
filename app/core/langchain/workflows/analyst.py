"""
Analyst agent workflow using LangGraph.

This module implements an analyst agent workflow that specializes
in critical analysis and evaluation of information.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph import StateGraph, START, END

from ..agent_manager import AgentState

logger = logging.getLogger(__name__)


async def analyze_information(state: AgentState) -> AgentState:
    """Analyze information critically and provide insights"""
    try:
        from ..llm_manager import llm_manager
        llm = await llm_manager.get_llm("gpt-4")
        
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            return state
            
        task = human_messages[-1].content
        
        # Get research results if available
        research_data = state.context.get("research_results", "")
        
        # Build analysis prompt
        system_prompt = """You are a critical analyst. Analyze the provided information thoroughly.
        
        Analysis Framework:
        1. Evaluate accuracy and reliability
        2. Identify biases and assumptions
        3. Assess completeness and gaps
        4. Consider implications and consequences
        5. Provide balanced perspective
        6. Highlight uncertainties and limitations
        
        Provide structured analysis with clear reasoning."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        if research_data:
            messages.append(SystemMessage(content=f"Research Data:\n{research_data}"))
            
        messages.append(HumanMessage(content=task))
        
        response = await llm.ainvoke(messages)
        
        state.context["analysis"] = response.content
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_information: {str(e)}")
        state.context["error"] = str(e)
        return state


async def create_analyst_workflow() -> StateGraph:
    """Create an analyst agent workflow using LangGraph"""
    try:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_information", analyze_information)
        
        workflow.add_edge(START, "analyze_information")
        workflow.add_edge("analyze_information", END)
        
        compiled_workflow = workflow.compile()
        logger.info("Created analyst workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create analyst workflow: {str(e)}")
        raise