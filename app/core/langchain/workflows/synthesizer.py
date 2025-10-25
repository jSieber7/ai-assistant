"""
Synthesizer agent workflow using LangGraph.

This module implements a synthesizer agent workflow that specializes
in combining multiple sources of information into coherent responses.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph import StateGraph, START, END

from ..agent_manager import AgentState

logger = logging.getLogger(__name__)


async def synthesize_information(state: AgentState) -> AgentState:
    """Synthesize multiple information sources into coherent response"""
    try:
        from ..llm_manager import llm_manager
        llm = await llm_manager.get_llm("gpt-4")
        
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            return state
            
        task = human_messages[-1].content
        
        # Get agent results if available
        agent_results = state.context.get("agent_results", {})
        
        # Build synthesis prompt
        system_prompt = """You are a synthesis expert. Combine multiple information sources into a coherent, comprehensive response.
        
        Synthesis Guidelines:
        1. Identify key themes and patterns
        2. Resolve conflicts and contradictions
        3. Create logical flow and structure
        4. Highlight important insights
        5. Maintain accuracy and proper attribution
        6. Provide balanced perspective
        7. Address the original task comprehensively
        
        Structure your response clearly with headings and bullet points."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        if agent_results:
            results_text = "\n\n".join([
                f"Source: {source}\nData: {data}"
                for source, data in agent_results.items()
            ])
            messages.append(SystemMessage(content=f"Agent Results:\n{results_text}"))
            
        messages.append(HumanMessage(content=task))
        
        response = await llm.ainvoke(messages)
        
        state.context["synthesis"] = response.content
        return state
        
    except Exception as e:
        logger.error(f"Error in synthesize_information: {str(e)}")
        state.context["error"] = str(e)
        return state


async def create_synthesizer_workflow() -> StateGraph:
    """Create a synthesizer agent workflow using LangGraph"""
    try:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("synthesize_information", synthesize_information)
        
        workflow.add_edge(START, "synthesize_information")
        workflow.add_edge("synthesize_information", END)
        
        compiled_workflow = workflow.compile()
        logger.info("Created synthesizer workflow")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create synthesizer workflow: {str(e)}")
        raise