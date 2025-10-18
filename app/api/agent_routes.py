"""
Agent system API routes for the AI Assistant.

This module provides FastAPI routes for interacting with the agent system,
including agent-based chat completions and agent management endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import logging

from ..core.config import settings
from ..core.agents.management.registry import agent_registry
from ..core.tools.execution.registry import tool_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


class AgentChatMessage(BaseModel):
    """Chat message for agent-based conversations"""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AgentChatRequest(BaseModel):
    """Request for agent-based chat completion"""

    messages: List[AgentChatMessage]
    agent_name: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentChatResponse(BaseModel):
    """Response from agent-based chat completion"""

    id: str
    object: str = "agent.chat.completion"
    model: str
    agent_name: str
    choices: List[dict]
    tool_results: Optional[List[dict]] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentInfo(BaseModel):
    """Information about an agent"""

    name: str
    description: str
    version: str
    state: str
    usage_count: int
    last_used: Optional[float] = None
    categories: List[str]


class AgentRegistryInfo(BaseModel):
    """Information about the agent registry"""

    total_agents: int
    active_agents: int
    default_agent: Optional[str] = None
    categories: List[str]
    agents_by_category: Dict[str, int]


@router.post("/chat/completions")
async def agent_chat_completions(request: AgentChatRequest):
    """
    Agent-based chat completion endpoint.

    This endpoint uses the agent system to intelligently select and use tools
    based on the conversation context.
    """
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    try:
        # Extract the last user message (most recent)
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user messages found in request"
            )

        last_user_message = user_messages[-1]

        # Use the agent system to process the message
        result = await agent_registry.process_message(
            message=last_user_message.content,
            agent_name=request.agent_name,
            conversation_id=request.conversation_id,
            context=request.context or {},
        )

        # Format tool results for response
        tool_results = None
        if result.tool_results:
            tool_results = [
                {
                    "tool_name": tr.tool_name,
                    "success": tr.success,
                    "execution_time": tr.execution_time,
                    "data": tr.data,
                    "error": tr.error,
                    "metadata": tr.metadata,
                }
                for tr in result.tool_results
            ]

        # Create response in OpenAI-compatible format
        response_id = f"agentchat-{uuid.uuid4()}"

        return AgentChatResponse(
            id=response_id,
            model=request.model or "agent-system",
            agent_name=result.agent_name,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.response,
                        "metadata": {
                            "agent_name": result.agent_name,
                            "execution_time": result.execution_time,
                            "tool_count": len(result.tool_results),
                        },
                    },
                    "finish_reason": "stop",
                }
            ],
            tool_results=tool_results,
            conversation_id=result.conversation_id,
            metadata=result.metadata,
        )

    except Exception as e:
        logger.error(f"Agent chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_agents():
    """List all available agents"""
    if not settings.agent_system_enabled:
        return {"agents": [], "message": "Agent system is disabled"}

    agents = agent_registry.list_agents(active_only=True)
    agent_info_list = []

    for agent in agents:
        stats = agent.get_usage_stats()
        agent_info_list.append(
            AgentInfo(
                name=agent.name,
                description=agent.description,
                version=agent.version,
                state=agent.state.value,
                usage_count=stats["usage_count"],
                last_used=stats["last_used"],
                categories=(
                    agent.categories if hasattr(agent, "categories") else ["general"]
                ),
            )
        )

    return {"agents": agent_info_list, "total_count": len(agent_info_list)}


@router.get("/{agent_name}")
async def get_agent_info(agent_name: str):
    """Get detailed information about a specific agent"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    stats = agent.get_usage_stats()
    registry_stats = agent_registry.get_registry_stats()

    return {
        "agent": AgentInfo(
            name=agent.name,
            description=agent.description,
            version=agent.version,
            state=agent.state.value,
            usage_count=stats["usage_count"],
            last_used=stats["last_used"],
            categories=(
                agent.categories if hasattr(agent, "categories") else ["general"]
            ),
        ),
        "registry_info": {
            "is_default": registry_stats["default_agent"] == agent_name,
            "is_active": agent_name in registry_stats.get("active_agents", []),
        },
        "conversation_info": {
            "current_conversation_id": stats.get("current_conversation_id"),
            "conversation_history_length": stats.get("conversation_history_length", 0),
        },
    }


@router.post("/{agent_name}/activate")
async def activate_agent(agent_name: str):
    """Activate a specific agent"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    success = agent_registry.activate_agent(agent_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {"message": f"Agent '{agent_name}' activated successfully"}


@router.post("/{agent_name}/deactivate")
async def deactivate_agent(agent_name: str):
    """Deactivate a specific agent"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    success = agent_registry.deactivate_agent(agent_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {"message": f"Agent '{agent_name}' deactivated successfully"}


@router.post("/{agent_name}/set-default")
async def set_default_agent(agent_name: str):
    """Set an agent as the default agent"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    success = agent_registry.set_default_agent(agent_name)
    if not success:
        raise HTTPException(
            status_code=400, detail=f"Cannot set '{agent_name}' as default agent"
        )

    return {"message": f"Agent '{agent_name}' set as default successfully"}


@router.get("/{agent_name}/conversation/{conversation_id}")
async def get_conversation_history(agent_name: str, conversation_id: str):
    """Get conversation history for a specific agent and conversation"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    history = agent.get_conversation_history(conversation_id)
    return {
        "agent_name": agent_name,
        "conversation_id": conversation_id,
        "history": history,
        "message_count": len(history),
    }


@router.post("/{agent_name}/reset")
async def reset_agent(agent_name: str):
    """Reset an agent's state and clear conversation history"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent.reset()
    return {"message": f"Agent '{agent_name}' reset successfully"}


@router.get("/registry/info")
async def get_registry_info():
    """Get information about the agent registry"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    stats = agent_registry.get_registry_stats()
    return AgentRegistryInfo(
        total_agents=stats["total_agents"],
        active_agents=stats["active_agents"],
        default_agent=stats["default_agent"],
        categories=stats["categories"],
        agents_by_category=stats["agents_by_category"],
    )


@router.post("/registry/reset")
async def reset_all_agents():
    """Reset all agents in the registry"""
    if not settings.agent_system_enabled:
        raise HTTPException(status_code=503, detail="Agent system is disabled")

    agent_registry.reset_all_agents()
    return {"message": "All agents reset successfully"}


@router.get("/tools/available")
async def get_available_tools():
    """Get information about tools available to agents"""
    tool_stats = tool_registry.get_registry_stats()
    available_tools = tool_registry.list_tools(enabled_only=True)

    tools_info = []
    for tool in available_tools:
        tool_stats = tool.get_usage_stats()
        tools_info.append(
            {
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
                "author": tool.author,
                "enabled": tool.enabled,
                "keywords": tool.keywords,
                "categories": tool.categories,
                "usage_count": tool_stats["usage_count"],
                "last_used": tool_stats["last_used"],
                "timeout": tool.timeout,
            }
        )

    return {
        "tools": tools_info,
        "total_tools": tool_stats["total_tools"],
        "enabled_tools": tool_stats["enabled_tools"],
        "categories": tool_stats["categories"],
    }


@router.get("/health")
async def agent_system_health():
    """Health check for the agent system"""
    agent_stats = (
        agent_registry.get_registry_stats() if settings.agent_system_enabled else {}
    )
    tool_stats = tool_registry.get_registry_stats()

    return {
        "agent_system_enabled": settings.agent_system_enabled,
        "agent_registry_healthy": (
            len(agent_stats) > 0 if settings.agent_system_enabled else True
        ),
        "tool_registry_healthy": len(tool_stats) > 0,
        "active_agents": agent_stats.get("active_agents", 0),
        "enabled_tools": tool_stats["enabled_tools"],
        "status": "healthy" if settings.agent_system_enabled else "disabled",
    }
