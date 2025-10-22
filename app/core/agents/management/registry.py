"""
Agent registry for managing and discovering agents in the AI Assistant system.

This module provides the AgentRegistry class which serves as a central registry
for agent registration, discovery, and management.
"""

from typing import Dict, List, Optional, Set, Any
from app.core.agents.base.base import BaseAgent, AgentResult
import logging

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for managing and discovering agents"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._active_agents: Set[str] = set()
        self._default_agent: Optional[str] = None

    def register(self, agent: BaseAgent, category: str = "general") -> bool:
        """
        Register an agent with the registry

        Args:
            agent: Agent instance to register
            category: Agent category for organization

        Returns:
            True if registration successful, False otherwise
        """
        if agent.name in self._agents:
            logger.warning(f"Agent '{agent.name}' is already registered")
            return False

        self._agents[agent.name] = agent

        # Add to category
        if category not in self._categories:
            self._categories[category] = set()
        self._categories[category].add(agent.name)

        # Activate by default
        self._active_agents.add(agent.name)

        # Set as default if no default exists
        if self._default_agent is None:
            self._default_agent = agent.name

        logger.info(f"Registered agent '{agent.name}' in category '{category}'")
        return True

    def unregister(self, agent_name: str) -> bool:
        """
        Unregister an agent from the registry

        Args:
            agent_name: Name of agent to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        if agent_name not in self._agents:
            logger.warning(f"Agent '{agent_name}' not found in registry")
            return False

        # Remove from categories
        for category, agents in self._categories.items():
            if agent_name in agents:
                agents.remove(agent_name)
                if not agents:  # Remove empty categories
                    del self._categories[category]

        # Remove from active agents
        if agent_name in self._active_agents:
            self._active_agents.remove(agent_name)

        # Update default agent if needed
        if self._default_agent == agent_name:
            self._default_agent = None
            # Set a new default from remaining agents
            if self._active_agents:
                self._default_agent = next(iter(self._active_agents))

        del self._agents[agent_name]
        logger.info(f"Unregistered agent '{agent_name}'")
        return True

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self._agents.get(agent_name)

    def list_agents(self, active_only: bool = True) -> List[BaseAgent]:
        """List all registered agents"""
        agents = list(self._agents.values())
        if active_only:
            agents = [agent for agent in agents if agent.name in self._active_agents]
        return agents

    def list_agent_names(self, active_only: bool = True) -> List[str]:
        """List names of all registered agents"""
        agent_names = list(self._agents.keys())
        if active_only:
            agent_names = [name for name in agent_names if name in self._active_agents]
        return agent_names

    def get_agents_by_category(self, category: str) -> List[BaseAgent]:
        """Get all agents in a specific category"""
        if category not in self._categories:
            return []

        agents = []
        for agent_name in self._categories[category]:
            agent = self.get_agent(agent_name)
            if agent and agent.name in self._active_agents:
                agents.append(agent)

        return agents

    def activate_agent(self, agent_name: str) -> bool:
        """Activate a specific agent"""
        agent = self.get_agent(agent_name)
        if not agent:
            return False

        self._active_agents.add(agent_name)
        logger.info(f"Activated agent '{agent_name}'")
        return True

    def deactivate_agent(self, agent_name: str) -> bool:
        """Deactivate a specific agent"""
        agent = self.get_agent(agent_name)
        if not agent:
            return False

        if agent_name in self._active_agents:
            self._active_agents.remove(agent_name)

        # Update default agent if needed
        if self._default_agent == agent_name:
            self._default_agent = None
            if self._active_agents:
                self._default_agent = next(iter(self._active_agents))

        logger.info(f"Deactivated agent '{agent_name}'")
        return True

    def set_default_agent(self, agent_name: str) -> bool:
        """Set the default agent"""
        if agent_name not in self._agents:
            logger.warning(f"Agent '{agent_name}' not found, cannot set as default")
            return False

        if agent_name not in self._active_agents:
            logger.warning(f"Agent '{agent_name}' is not active, cannot set as default")
            return False

        self._default_agent = agent_name
        logger.info(f"Set '{agent_name}' as default agent")
        return True

    def get_default_agent(self) -> Optional[BaseAgent]:
        """Get the default agent"""
        if self._default_agent is None:
            return None
        return self.get_agent(self._default_agent)

    def find_relevant_agent(
        self, query: str, context: Dict[str, Any] = None
    ) -> Optional[BaseAgent]:
        """
        Find the most relevant agent for the given query

        Args:
            query: User query
            context: Additional context

        Returns:
            Most relevant agent, or default agent if none found
        """
        # Simple implementation - return default agent
        # In future, this could use more sophisticated agent selection logic
        default_agent = self.get_default_agent()
        if default_agent:
            return default_agent

        # Fallback to first active agent
        active_agents = self.list_agents(active_only=True)
        if active_agents:
            return active_agents[0]

        return None

    async def process_message(
        self,
        message: str,
        agent_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Process a message using the specified agent or the most relevant one

        Args:
            message: User message to process
            agent_name: Specific agent to use (optional)
            conversation_id: Conversation ID for context
            context: Additional context information

        Returns:
            AgentResult with response
        """
        agent = None

        if agent_name:
            agent = self.get_agent(agent_name)
            if not agent:
                error_msg = f"Agent '{agent_name}' not found"
                logger.error(error_msg)
                return AgentResult(
                    success=False,
                    response="Requested agent is not available.",
                    error=error_msg,
                    agent_name="unknown",
                    execution_time=0.0,
                    conversation_id=conversation_id,
                )
        else:
            agent = self.find_relevant_agent(message, context)
            if not agent:
                error_msg = "No suitable agent found"
                logger.error(error_msg)
                return AgentResult(
                    success=False,
                    response="No agents are currently available.",
                    error=error_msg,
                    agent_name="unknown",
                    execution_time=0.0,
                    conversation_id=conversation_id,
                )

        # Process the message with the selected agent
        return await agent.process_message(message, conversation_id, context)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_agents": len(self._agents),
            "active_agents": len(self._active_agents),
            "default_agent": self._default_agent,
            "categories": list(self._categories.keys()),
            "agents_by_category": {
                category: len(agents) for category, agents in self._categories.items()
            },
        }

    async def reset_all_agents(self):
        """Reset all registered agents"""
        for agent in self._agents.values():
            await agent.reset()
        logger.info("Reset all agents")


# Global agent registry instance
agent_registry = AgentRegistry()
