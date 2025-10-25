"""
LangChain integration layer for AI Assistant.

This module provides compatibility and integration between the new LangChain
components and the existing API structure, ensuring backward compatibility
while enabling new features.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .llm_manager import llm_manager
from .tool_registry import tool_registry
from .agent_manager import agent_manager
from .memory_manager import memory_manager

# Legacy imports for compatibility
from ..llm_providers import provider_registry as legacy_provider_registry
from ..tools.execution.registry import tool_registry as legacy_tool_registry
from ..agents.management.registry import agent_registry as legacy_agent_registry

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for backward compatibility"""
    
    LEGACY = "legacy"          # Use old systems
    LANGCHAIN = "langchain"      # Use new LangChain systems
    HYBRID = "hybrid"           # Mix of old and new
    MIGRATION = "migration"        # Migrating from old to new


class LangChainIntegrationLayer:
    """
    Integration layer that bridges new LangChain components with existing APIs.
    
    This layer provides:
    - Backward compatibility with existing endpoints
    - Gradual migration from old to new systems
    - Feature flags for controlled rollout
    - Seamless switching between implementations
    - Monitoring and observability
    """
    
    def __init__(self):
        self._mode = IntegrationMode.LEGACY  # Default to legacy for safety
        self._feature_flags = {
            "use_langchain_llm": False,
            "use_langchain_tools": False,
            "use_langchain_agents": False,
            "use_langchain_memory": False,
            "use_langgraph_workflows": False,
        }
        self._migration_stats = {
            "started_at": None,
            "components_migrated": [],
            "legacy_components_retired": [],
            "migration_complete": False,
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize the integration layer"""
        if self._initialized:
            return
            
        logger.info("Initializing LangChain Integration Layer...")
        
        # Load feature flags from configuration
        await self._load_feature_flags()
        
        # Initialize LangChain components if enabled
        if self._feature_flags["use_langchain_llm"]:
            await llm_manager.initialize()
            
        if self._feature_flags["use_langchain_tools"]:
            await tool_registry.initialize()
            
        if self._feature_flags["use_langchain_agents"]:
            await agent_manager.initialize()
            
        if self._feature_flags["use_langchain_memory"]:
            await memory_manager.initialize()
            
        self._initialized = True
        logger.info(f"LangChain Integration Layer initialized in {self._mode.value} mode")
        
    async def _load_feature_flags(self):
        """Load feature flags from configuration"""
        try:
            from app.core.secure_settings import secure_settings
            
            # Load individual feature flags
            self._feature_flags["use_langchain_llm"] = secure_settings.get_setting(
                "langchain_integration", "use_langchain_llm", False
            )
            self._feature_flags["use_langchain_tools"] = secure_settings.get_setting(
                "langchain_integration", "use_langchain_tools", False
            )
            self._feature_flags["use_langchain_agents"] = secure_settings.get_setting(
                "langchain_integration", "use_langchain_agents", False
            )
            self._feature_flags["use_langchain_memory"] = secure_settings.get_setting(
                "langchain_integration", "use_langchain_memory", False
            )
            self._feature_flags["use_langgraph_workflows"] = secure_settings.get_setting(
                "langchain_integration", "use_langgraph_workflows", False
            )
            
            # Determine integration mode
            enabled_features = sum(self._feature_flags.values())
            
            if enabled_features == 0:
                self._mode = IntegrationMode.LEGACY
            elif enabled_features == len(self._feature_flags):
                self._mode = IntegrationMode.LANGCHAIN
            else:
                self._mode = IntegrationMode.HYBRID
                
            logger.info(f"Loaded feature flags: {self._feature_flags}")
            logger.info(f"Integration mode: {self._mode.value}")
            
        except Exception as e:
            logger.error(f"Failed to load feature flags: {str(e)}")
            
    async def get_llm(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get LLM instance with backward compatibility.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional LLM parameters
            
        Returns:
            LLM instance (legacy or LangChain)
        """
        if self._feature_flags["use_langchain_llm"]:
            # Use new LangChain LLM manager
            return await llm_manager.get_llm(model_name, **kwargs)
        else:
            # Use legacy provider registry
            provider, actual_model = await legacy_provider_registry.resolve_model(model_name)
            if provider:
                return await provider.create_llm(actual_model, **kwargs)
            else:
                raise ValueError(f"Model '{model_name}' not found in legacy providers")
                
    async def get_tools(self, enabled_only: bool = True) -> List[Any]:
        """
        Get tools with backward compatibility.
        
        Args:
            enabled_only: Only return enabled tools
            
        Returns:
            List of tools (legacy or LangChain)
        """
        if self._feature_flags["use_langchain_tools"]:
            # Use new LangChain tool registry
            return tool_registry.list_tools(enabled_only=enabled_only)
        else:
            # Use legacy tool registry
            return legacy_tool_registry.list_tools(enabled_only=enabled_only)
            
    async def get_agent(self, agent_name: str) -> Optional[Any]:
        """
        Get agent instance with backward compatibility.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance (legacy or LangChain)
        """
        if self._feature_flags["use_langchain_agents"]:
            # Use new LangChain agent manager
            return await agent_manager.get_agent(agent_name)
        else:
            # Use legacy agent registry
            return legacy_agent_registry.get_agent(agent_name)
            
    async def invoke_agent(
        self,
        agent_name: str,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke agent with backward compatibility.
        
        Args:
            agent_name: Name of the agent
            message: User message
            conversation_id: Conversation ID
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Agent execution result
        """
        if self._feature_flags["use_langchain_agents"]:
            # Use new LangChain agent manager
            return await agent_manager.invoke_agent(
                agent_name, message, conversation_id, context, **kwargs
            )
        else:
            # Use legacy agent registry
            result = await legacy_agent_registry.process_message(
                message, agent_name, conversation_id, context
            )
            
            # Convert to new format for compatibility
            return {
                "success": result.success,
                "response": result.response,
                "agent_name": result.agent_name,
                "conversation_id": result.conversation_id,
                "execution_time": result.execution_time,
                "tool_results": [tr.dict() for tr in result.tool_results] if result.tool_results else [],
                "metadata": result.metadata,
            }
            
    async def create_conversation(
        self,
        agent_name: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create conversation with backward compatibility.
        
        Args:
            agent_name: Name of the agent
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            Conversation ID
        """
        if self._feature_flags["use_langchain_memory"]:
            # Use new LangChain memory manager
            return await agent_manager.create_conversation(agent_name, title, metadata)
        else:
            # Use legacy conversation creation
            conversation_id = str(datetime.now().timestamp())
            
            # Store in legacy system (simplified)
            logger.info(f"Created legacy conversation {conversation_id}")
            return conversation_id
            
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add message to conversation with backward compatibility.
        
        Args:
            conversation_id: Conversation ID
            role: Message role
            content: Message content
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if self._feature_flags["use_langchain_memory"]:
            # Use new LangChain memory manager
            return await memory_manager.add_message(conversation_id, role, content, metadata)
        else:
            # Use legacy message storage (simplified)
            logger.debug(f"Added legacy message to {conversation_id}: {role}")
            return True
            
    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages with backward compatibility.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages
            
        Returns:
            List of message dictionaries
        """
        if self._feature_flags["use_langchain_memory"]:
            # Use new LangChain memory manager
            return await memory_manager.get_conversation_messages(conversation_id, limit)
        else:
            # Return empty list for legacy mode
            logger.debug(f"Retrieved legacy messages for {conversation_id}")
            return []
            
    def set_integration_mode(self, mode: IntegrationMode) -> bool:
        """
        Set the integration mode.
        
        Args:
            mode: Integration mode to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            old_mode = self._mode
            self._mode = mode
            
            # Update feature flags based on mode
            if mode == IntegrationMode.LEGACY:
                self._disable_all_langchain_features()
            elif mode == IntegrationMode.LANGCHAIN:
                self._enable_all_langchain_features()
            elif mode == IntegrationMode.HYBRID:
                # Keep current feature flags
                pass
            elif mode == IntegrationMode.MIGRATION:
                self._enable_migration_mode()
                
            logger.info(f"Changed integration mode from {old_mode.value} to {mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set integration mode: {str(e)}")
            return False
            
    def _disable_all_langchain_features(self):
        """Disable all LangChain features"""
        for flag in self._feature_flags:
            self._feature_flags[flag] = False
            
    def _enable_all_langchain_features(self):
        """Enable all LangChain features"""
        for flag in self._feature_flags:
            self._feature_flags[flag] = True
            
    def _enable_migration_mode(self):
        """Enable migration mode with gradual rollout"""
        # Start with LLM and tools, then gradually enable others
        self._feature_flags["use_langchain_llm"] = True
        self._feature_flags["use_langchain_tools"] = True
        # Keep others disabled for now
        self._feature_flags["use_langchain_agents"] = False
        self._feature_flags["use_langchain_memory"] = False
        self._feature_flags["use_langgraph_workflows"] = False
        
        # Start migration tracking
        self._migration_stats["started_at"] = datetime.now().isoformat()
        self._migration_stats["components_migrated"] = ["llm", "tools"]
        
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get current feature flags"""
        return self._feature_flags.copy()
        
    def get_integration_mode(self) -> IntegrationMode:
        """Get current integration mode"""
        return self._mode
        
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get migration statistics"""
        return self._migration_stats.copy()
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all integrated systems.
        
        Returns:
            Health check results
        """
        health_results = {
            "integration_mode": self._mode.value,
            "feature_flags": self._feature_flags,
            "components": {},
            "overall_status": "healthy",
        }
        
        # Check LangChain components
        if self._feature_flags["use_langchain_llm"]:
            try:
                models = await llm_manager.list_models()
                health_results["components"]["llm_manager"] = {
                    "status": "healthy",
                    "available_models": len(models),
                    "providers": list(set(m.provider.value for m in models)),
                }
            except Exception as e:
                health_results["components"]["llm_manager"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_results["overall_status"] = "degraded"
                
        if self._feature_flags["use_langchain_tools"]:
            try:
                tool_stats = tool_registry.get_registry_stats()
                health_results["components"]["tool_registry"] = {
                    "status": "healthy",
                    "total_tools": tool_stats["total_tools"],
                    "enabled_tools": tool_stats["enabled_tools"],
                }
            except Exception as e:
                health_results["components"]["tool_registry"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_results["overall_status"] = "degraded"
                
        if self._feature_flags["use_langchain_agents"]:
            try:
                agent_stats = agent_manager.get_registry_stats()
                health_results["components"]["agent_manager"] = {
                    "status": "healthy",
                    "total_agents": agent_stats["total_agents"],
                    "active_agents": agent_stats["active_agents"],
                }
            except Exception as e:
                health_results["components"]["agent_manager"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_results["overall_status"] = "degraded"
                
        if self._feature_flags["use_langchain_memory"]:
            try:
                memory_stats = await memory_manager.get_memory_stats()
                health_results["components"]["memory_manager"] = {
                    "status": "healthy",
                    "total_conversations": memory_stats["total_conversations"],
                    "total_messages": memory_stats["total_messages"],
                }
            except Exception as e:
                health_results["components"]["memory_manager"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_results["overall_status"] = "degraded"
                
        return health_results


# Global integration layer instance
integration_layer = LangChainIntegrationLayer()