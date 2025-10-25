"""
LangGraph-based Agent Manager for AI Assistant.

This module provides comprehensive agent management using LangGraph workflows,
supporting multiple agent types and seamless integration with LangChain components.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentExecutor
from langgraph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_agent_executor

from .llm_manager import llm_manager
from .tool_registry import tool_registry
from .memory_manager import LangChainMemoryManager

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Supported agent types"""
    
    CONVERSATIONAL = "conversational"
    TOOL_HEAVY = "tool_heavy"
    MULTI_AGENT = "multi_agent"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


@dataclass
class AgentState:
    """Base state for LangGraph workflows"""
    
    messages: List[BaseMessage] = field(default_factory=list)
    current_task: Optional[str] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    agent_name: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 5


@dataclass
class MultiAgentState:
    """State for multi-agent workflows"""
    
    messages: List[BaseMessage] = field(default_factory=list)
    current_task: Optional[str] = None
    agent_results: Dict[str, Any] = field(default_factory=dict)
    active_agents: List[str] = field(default_factory=list)
    coordinator_decision: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 10


@dataclass
class AgentConfig:
    """Configuration for agent creation"""
    
    name: str
    agent_type: AgentType
    llm_model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_iterations: int = 5
    enable_streaming: bool = False
    memory_backend: str = "conversation"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangGraphAgentManager:
    """
    Comprehensive agent manager using LangGraph workflows.
    
    This manager provides:
    - Multiple agent types with different capabilities
    - LangGraph workflow creation and management
    - Agent state management and persistence
    - Tool integration and orchestration
    - Multi-agent collaboration workflows
    """
    
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._workflows: Dict[str, StateGraph] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._memory_manager: Optional[LangChainMemoryManager] = None
        self._checkpoint_saver = MemorySaver()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the agent manager"""
        if self._initialized:
            return
            
        logger.info("Initializing LangGraph Agent Manager...")
        
        # Initialize memory manager
        self._memory_manager = LangChainMemoryManager()
        await self._memory_manager.initialize()
        
        # Create default workflows
        await self._create_default_workflows()
        
        # Register default agents
        await self._register_default_agents()
        
        self._initialized = True
        logger.info("LangGraph Agent Manager initialized successfully")
        
    async def _create_default_workflows(self):
        """Create default LangGraph workflows"""
        try:
            # Conversational workflow
            self._workflows[AgentType.CONVERSATIONAL.value] = await self._create_conversational_workflow()
            
            # Tool-heavy workflow
            self._workflows[AgentType.TOOL_HEAVY.value] = await self._create_tool_heavy_workflow()
            
            # Multi-agent workflow
            self._workflows[AgentType.MULTI_AGENT.value] = await self._create_multi_agent_workflow()
            
            # Researcher workflow
            self._workflows[AgentType.RESEARCHER.value] = await self._create_researcher_workflow()
            
            # Analyst workflow
            self._workflows[AgentType.ANALYST.value] = await self._create_analyst_workflow()
            
            # Synthesizer workflow
            self._workflows[AgentType.SYNTHESIZER.value] = await self._create_synthesizer_workflow()
            
            logger.info(f"Created {len(self._workflows)} default workflows")
        except Exception as e:
            logger.error(f"Failed to create default workflows: {str(e)}")
            
    async def _register_default_agents(self):
        """Register default agent configurations"""
        try:
            default_configs = [
                AgentConfig(
                    name="conversational_agent",
                    agent_type=AgentType.CONVERSATIONAL,
                    llm_model="gpt-4",
                    system_prompt="You are a helpful AI assistant. Engage in natural conversation and provide thoughtful responses.",
                    tools=["duckduckgo_search", "wikipedia_search"],
                ),
                AgentConfig(
                    name="tool_heavy_agent",
                    agent_type=AgentType.TOOL_HEAVY,
                    llm_model="gpt-4",
                    system_prompt="You are an AI assistant with access to many tools. Use tools extensively to provide comprehensive answers.",
                    tools=["duckduckgo_search", "wikipedia_search", "shell_tool", "python_repl"],
                    max_iterations=10,
                ),
                AgentConfig(
                    name="researcher_agent",
                    agent_type=AgentType.RESEARCHER,
                    llm_model="gpt-4",
                    system_prompt="You are a research assistant. Use search tools to gather comprehensive information on topics.",
                    tools=["duckduckgo_search", "wikipedia_search"],
                    max_iterations=8,
                ),
                AgentConfig(
                    name="analyst_agent",
                    agent_type=AgentType.ANALYST,
                    llm_model="gpt-4",
                    system_prompt="You are an analytical assistant. Analyze information critically and provide insights.",
                    max_iterations=6,
                ),
                AgentConfig(
                    name="synthesizer_agent",
                    agent_type=AgentType.SYNTHESIZER,
                    llm_model="gpt-4",
                    system_prompt="You are a synthesis assistant. Combine multiple sources of information into coherent responses.",
                    max_iterations=4,
                ),
            ]
            
            for config in default_configs:
                await self.register_agent(config)
                
            logger.info(f"Registered {len(default_configs)} default agents")
        except Exception as e:
            logger.error(f"Failed to register default agents: {str(e)}")
            
    async def register_agent(self, config: AgentConfig) -> bool:
        """
        Register an agent with the manager.
        
        Args:
            config: Agent configuration
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if config.name in self._agent_configs:
                logger.warning(f"Agent '{config.name}' is already registered")
                return False
                
            self._agent_configs[config.name] = config
            
            # Create agent instance
            agent = await self._create_agent_instance(config)
            if agent:
                self._agents[config.name] = agent
                
                # Initialize agent state
                self._agent_states[config.name] = {
                    "created_at": datetime.now().isoformat(),
                    "last_used": None,
                    "usage_count": 0,
                    "conversations": {},
                }
                
                logger.info(f"Registered agent '{config.name}' of type '{config.agent_type.value}'")
                return True
            else:
                logger.error(f"Failed to create agent instance for '{config.name}'")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register agent '{config.name}': {str(e)}")
            return False
            
    async def _create_agent_instance(self, config: AgentConfig) -> Optional[Any]:
        """Create agent instance based on configuration"""
        try:
            # Get LLM
            llm = await llm_manager.get_llm(
                config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=config.enable_streaming
            )
            
            # Get tools
            tools = []
            for tool_name in config.tools:
                tool = tool_registry.get_tool(tool_name)
                if tool:
                    tools.append(tool)
                else:
                    logger.warning(f"Tool '{tool_name}' not found for agent '{config.name}'")
                    
            # Get workflow
            workflow = self._workflows.get(config.agent_type.value)
            if not workflow:
                logger.error(f"Workflow not found for agent type '{config.agent_type.value}'")
                return None
                
            # Create agent with workflow
            if config.agent_type == AgentType.MULTI_AGENT:
                # Multi-agent workflow is special
                agent = workflow.compile(checkpointer=self._checkpoint_saver)
            else:
                # Single agent workflow
                agent = workflow.compile(checkpointer=self._checkpoint_saver)
                
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent instance: {str(e)}")
            return None
            
    async def get_agent(self, agent_name: str) -> Optional[Any]:
        """
        Get an agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(agent_name)
        
    async def list_agents(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List registered agents.
        
        Args:
            active_only: Only return active agents
            
        Returns:
            List of agent information
        """
        agents = []
        
        for name, config in self._agent_configs.items():
            if active_only and name not in self._agents:
                continue
                
            agent_info = {
                "name": name,
                "type": config.agent_type.value,
                "llm_model": config.llm_model,
                "description": config.system_prompt or f"{config.agent_type.value} agent",
                "tools": config.tools,
                "enabled": name in self._agents,
                "created_at": self._agent_states.get(name, {}).get("created_at"),
                "usage_count": self._agent_states.get(name, {}).get("usage_count", 0),
                "last_used": self._agent_states.get(name, {}).get("last_used"),
            }
            agents.append(agent_info)
            
        return agents
        
    async def invoke_agent(
        self,
        agent_name: str,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke an agent with a message.
        
        Args:
            agent_name: Name of the agent
            message: User message
            conversation_id: Conversation ID for context
            context: Additional context
            stream: Whether to stream the response
            
        Returns:
            Agent execution result
        """
        try:
            agent = await self.get_agent(agent_name)
            if not agent:
                return {
                    "success": False,
                    "error": f"Agent '{agent_name}' not found",
                    "agent_name": agent_name,
                }
                
            config = self._agent_configs[agent_name]
            
            # Update agent state
            if agent_name in self._agent_states:
                self._agent_states[agent_name]["last_used"] = datetime.now().isoformat()
                self._agent_states[agent_name]["usage_count"] += 1
                
            # Create conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                
            # Load conversation history
            messages = []
            if self._memory_manager:
                messages = await self._memory_manager.get_conversation_messages(conversation_id)
                
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Prepare state
            if config.agent_type == AgentType.MULTI_AGENT:
                state = MultiAgentState(
                    messages=messages,
                    current_task=message,
                    context=context or {},
                    conversation_id=conversation_id,
                    agent_name=agent_name,
                    max_iterations=config.max_iterations
                )
            else:
                state = AgentState(
                    messages=messages,
                    current_task=message,
                    context=context or {},
                    conversation_id=conversation_id,
                    agent_name=agent_name,
                    max_iterations=config.max_iterations
                )
                
            # Execute agent
            if stream:
                return await self._invoke_agent_stream(agent, state, config)
            else:
                return await self._invoke_agent_sync(agent, state, config)
                
        except Exception as e:
            logger.error(f"Failed to invoke agent '{agent_name}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_name,
            }
            
    async def _invoke_agent_sync(
        self, 
        agent: Any, 
        state: Union[AgentState, MultiAgentState], 
        config: AgentConfig
    ) -> Dict[str, Any]:
        """Invoke agent synchronously"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute workflow
            result = await agent.ainvoke(state)
            
            # Extract response
            if isinstance(result, dict):
                if "messages" in result:
                    messages = result["messages"]
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, AIMessage):
                            response = last_message.content
                        else:
                            response = str(last_message)
                    else:
                        response = "No response generated"
                else:
                    response = str(result)
            else:
                response = str(result)
                
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Save to memory
            if self._memory_manager and state.conversation_id:
                await self._memory_manager.add_message(
                    conversation_id=state.conversation_id,
                    role="assistant",
                    content=response,
                    metadata={
                        "agent_name": state.agent_name,
                        "agent_type": config.agent_type.value,
                        "execution_time": execution_time,
                        "tool_results": result.get("tool_results", []),
                    }
                )
                
            return {
                "success": True,
                "response": response,
                "agent_name": state.agent_name,
                "conversation_id": state.conversation_id,
                "execution_time": execution_time,
                "tool_results": result.get("tool_results", []),
                "metadata": result.get("metadata", {}),
            }
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Agent execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "agent_name": state.agent_name,
                "conversation_id": state.conversation_id,
                "execution_time": execution_time,
            }
            
    async def _invoke_agent_stream(
        self, 
        agent: Any, 
        state: Union[AgentState, MultiAgentState], 
        config: AgentConfig
    ) -> Dict[str, Any]:
        """Invoke agent with streaming"""
        # For now, return sync result
        # In future, implement proper streaming
        return await self._invoke_agent_sync(agent, state, config)
        
    async def create_conversation(
        self,
        agent_name: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation for an agent.
        
        Args:
            agent_name: Name of the agent
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        if self._memory_manager:
            await self._memory_manager.create_conversation(
                conversation_id=conversation_id,
                agent_name=agent_name,
                title=title or f"Conversation with {agent_name}",
                metadata=metadata or {}
            )
            
        # Update agent state
        if agent_name in self._agent_states:
            if "conversations" not in self._agent_states[agent_name]:
                self._agent_states[agent_name]["conversations"] = {}
            self._agent_states[agent_name]["conversations"][conversation_id] = {
                "created_at": datetime.now().isoformat(),
                "title": title,
            }
            
        return conversation_id
        
    async def get_conversation_history(
        self,
        agent_name: str,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for an agent.
        
        Args:
            agent_name: Name of the agent
            conversation_id: Conversation ID
            
        Returns:
            List of messages
        """
        if self._memory_manager:
            return await self._memory_manager.get_conversation_messages(conversation_id)
        return []
        
    async def reset_agent(self, agent_name: str) -> bool:
        """
        Reset an agent's state.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_name in self._agent_states:
                self._agent_states[agent_name]["conversations"] = {}
                self._agent_states[agent_name]["usage_count"] = 0
                self._agent_states[agent_name]["last_used"] = None
                
                logger.info(f"Reset agent '{agent_name}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reset agent '{agent_name}': {str(e)}")
            return False
            
    def get_agent_stats(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent statistics or None if not found
        """
        return self._agent_states.get(agent_name)
        
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_agents = len(self._agent_configs)
        active_agents = len(self._agents)
        total_workflows = len(self._workflows)
        
        agent_types = {}
        for config in self._agent_configs.values():
            agent_type = config.agent_type.value
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_workflows": total_workflows,
            "agent_types": agent_types,
            "available_workflows": list(self._workflows.keys()),
        }
        
    # Workflow creation methods (to be implemented in separate workflow modules)
    async def _create_conversational_workflow(self) -> StateGraph:
        """Create conversational agent workflow"""
        # This will be implemented in workflows/conversational.py
        from .workflows.conversational import create_conversational_workflow
        return await create_conversational_workflow()
        
    async def _create_tool_heavy_workflow(self) -> StateGraph:
        """Create tool-heavy agent workflow"""
        # This will be implemented in workflows/tool_heavy.py
        from .workflows.tool_heavy import create_tool_heavy_workflow
        return await create_tool_heavy_workflow()
        
    async def _create_multi_agent_workflow(self) -> StateGraph:
        """Create multi-agent collaboration workflow"""
        # This will be implemented in workflows/multi_agent.py
        from .workflows.multi_agent import create_multi_agent_workflow
        return await create_multi_agent_workflow()
        
    async def _create_researcher_workflow(self) -> StateGraph:
        """Create researcher agent workflow"""
        # This will be implemented in workflows/researcher.py
        from .workflows.researcher import create_researcher_workflow
        return await create_researcher_workflow()
        
    async def _create_analyst_workflow(self) -> StateGraph:
        """Create analyst agent workflow"""
        # This will be implemented in workflows/analyst.py
        from .workflows.analyst import create_analyst_workflow
        return await create_analyst_workflow()
        
    async def _create_synthesizer_workflow(self) -> StateGraph:
        """Create synthesizer agent workflow"""
        # This will be implemented in workflows/synthesizer.py
        from .workflows.synthesizer import create_synthesizer_workflow
        return await create_synthesizer_workflow()


# Global agent manager instance
agent_manager = LangGraphAgentManager()