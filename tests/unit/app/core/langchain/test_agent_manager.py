"""
Unit tests for LangGraph Agent Manager.

This module tests agent registration, invocation, and integration
with LangGraph agent components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Callable

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.core.langchain.agent_manager import LangGraphAgentManager
from app.core.secure_settings import secure_settings


class TestLangGraphAgentManager:
    """Test cases for LangGraph Agent Manager"""
    
    @pytest.fixture
    async def agent_manager(self):
        """Create a LangGraph Agent Manager instance for testing"""
        manager = LangGraphAgentManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'agent_manager_enabled'): True,
            ('langchain', 'max_concurrent_agents'): 5,
            ('agents', 'execution_timeout'): 60,
        }.get((section, key), default)
        return mock_settings
    
    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing"""
        async def sample_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:
            """A sample agent function"""
            return {
                "messages": [{"role": "assistant", "content": "Sample response"}],
                "next": "end"
            }
        
        # Create a simple state graph
        workflow = StateGraph(dict)
        workflow.add_node("process", sample_agent_function)
        workflow.set_entry_point("process")
        workflow.add_edge("process", "end")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of agent manager"""
        with patch('app.core.langchain.agent_manager.secure_settings', mock_settings):
            manager = LangGraphAgentManager()
            
            # Test initialization
            await manager.initialize()
            
            # Verify initialization
            assert manager._initialized is True
            assert manager._monitoring is not None
            assert isinstance(manager._agents, dict)
    
    async def test_register_agent(self, agent_manager, sample_agent):
        """Test registering an agent"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Verify registration
        assert "sample_agent" in agent_manager._agents
        assert agent_manager._agents["sample_agent"]["agent"] is sample_agent
        assert agent_manager._agents["sample_agent"]["description"] == "A sample agent for testing"
    
    async def test_register_agent_with_config(self, agent_manager, sample_agent):
        """Test registering an agent with configuration"""
        config = {
            "max_iterations": 10,
            "timeout": 30,
            "retry_attempts": 3
        }
        
        # Register agent with config
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing",
            config=config
        )
        
        # Verify registration with config
        assert "sample_agent" in agent_manager._agents
        assert agent_manager._agents["sample_agent"]["config"] == config
    
    async def test_get_agent(self, agent_manager, sample_agent):
        """Test getting a registered agent"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Get agent
        retrieved_agent = agent_manager.get_agent("sample_agent")
        
        assert retrieved_agent is sample_agent
    
    async def test_get_agent_not_found(self, agent_manager):
        """Test getting an agent that doesn't exist"""
        agent = agent_manager.get_agent("non_existent_agent")
        assert agent is None
    
    async def test_list_agents(self, agent_manager, sample_agent):
        """Test listing all registered agents"""
        # Register some agents
        await agent_manager.register_agent(
            name="sample_agent_1",
            agent=sample_agent,
            description="First sample agent"
        )
        
        await agent_manager.register_agent(
            name="sample_agent_2",
            agent=sample_agent,
            description="Second sample agent"
        )
        
        # List agents
        agents = agent_manager.list_agents()
        
        assert isinstance(agents, list)
        assert len(agents) == 2
        agent_names = [agent["name"] for agent in agents]
        assert "sample_agent_1" in agent_names
        assert "sample_agent_2" in agent_names
    
    async def test_invoke_agent_sync(self, agent_manager, sample_agent):
        """Test synchronous agent invocation"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Invoke agent
        result = await agent_manager.invoke_agent(
            agent_name="sample_agent",
            input_data={"messages": [{"role": "user", "content": "Hello"}]},
            config={"thread_id": "test_thread"}
        )
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) > 0
    
    async def test_invoke_agent_stream(self, agent_manager, sample_agent):
        """Test streaming agent invocation"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Invoke agent with streaming
        result_chunks = []
        async for chunk in agent_manager.invoke_agent_stream(
            agent_name="sample_agent",
            input_data={"messages": [{"role": "user", "content": "Hello"}]},
            config={"thread_id": "test_thread"}
        ):
            result_chunks.append(chunk)
        
        assert len(result_chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in result_chunks)
    
    async def test_invoke_agent_not_found(self, agent_manager):
        """Test invoking an agent that doesn't exist"""
        with pytest.raises(ValueError, match="Agent 'non_existent_agent' not found"):
            await agent_manager.invoke_agent("non_existent_agent", {})
    
    async def test_unregister_agent(self, agent_manager, sample_agent):
        """Test unregistering an agent"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Verify it's registered
        assert "sample_agent" in agent_manager._agents
        
        # Unregister it
        result = await agent_manager.unregister_agent("sample_agent")
        
        assert result is True
        assert "sample_agent" not in agent_manager._agents
    
    async def test_unregister_agent_not_found(self, agent_manager):
        """Test unregistering an agent that doesn't exist"""
        result = await agent_manager.unregister_agent("non_existent_agent")
        assert result is False
    
    async def test_get_agent_info(self, agent_manager, sample_agent):
        """Test getting information about an agent"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Get agent info
        info = agent_manager.get_agent_info("sample_agent")
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "registered_at" in info
        assert info["name"] == "sample_agent"
    
    async def test_get_agent_info_not_found(self, agent_manager):
        """Test getting info for an agent that doesn't exist"""
        info = agent_manager.get_agent_info("non_existent_agent")
        assert info is None
    
    async def test_health_check(self, agent_manager, sample_agent):
        """Test health check functionality"""
        # Register some agents
        await agent_manager.register_agent(
            name="sample_agent_1",
            agent=sample_agent,
            description="First sample agent"
        )
        
        await agent_manager.register_agent(
            name="sample_agent_2",
            agent=sample_agent,
            description="Second sample agent"
        )
        
        # Perform health check
        health = await agent_manager.health_check()
        
        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "agents" in health
        assert "timestamp" in health
        assert "sample_agent_1" in health["agents"]
        assert "sample_agent_2" in health["agents"]
    
    async def test_shutdown(self, agent_manager):
        """Test shutdown functionality"""
        # Mock monitoring shutdown
        agent_manager._monitoring.shutdown = AsyncMock()
        
        await agent_manager.shutdown()
        
        # Verify shutdown was called
        agent_manager._monitoring.shutdown.assert_called_once()
        
        # Verify manager is marked as not initialized
        assert agent_manager._initialized is False
        
        # Verify agents are cleared
        assert len(agent_manager._agents) == 0
    
    async def test_concurrent_agent_invocations(self, agent_manager, sample_agent):
        """Test concurrent agent invocations"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Invoke agent concurrently
        tasks = [
            agent_manager.invoke_agent(
                agent_name="sample_agent",
                input_data={"messages": [{"role": "user", "content": f"Hello {i}"}]},
                config={"thread_id": f"test_thread_{i}"}
            )
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all invocations succeeded
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
    
    async def test_agent_execution_timeout(self, agent_manager):
        """Test agent execution timeout"""
        async def slow_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(2)  # Simulate slow operation
            return {"messages": [{"role": "assistant", "content": "Slow response"}]}
        
        # Create slow agent
        workflow = StateGraph(dict)
        workflow.add_node("process", slow_agent)
        workflow.set_entry_point("process")
        workflow.add_edge("process", "end")
        slow_agent_compiled = workflow.compile(checkpointer=MemorySaver())
        
        # Register agent
        await agent_manager.register_agent(
            name="slow_agent",
            agent=slow_agent_compiled,
            description="A slow agent"
        )
        
        # Set a short timeout
        agent_manager._execution_timeout = 1.0
        
        # Invoke should timeout
        with pytest.raises(asyncio.TimeoutError):
            await agent_manager.invoke_agent("slow_agent", {})
    
    async def test_agent_error_handling(self, agent_manager):
        """Test error handling during agent execution"""
        async def error_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("Agent execution error")
        
        # Create error agent
        workflow = StateGraph(dict)
        workflow.add_node("process", error_agent)
        workflow.set_entry_point("process")
        workflow.add_edge("process", "end")
        error_agent_compiled = workflow.compile(checkpointer=MemorySaver())
        
        # Register agent
        await agent_manager.register_agent(
            name="error_agent",
            agent=error_agent_compiled,
            description="An agent that errors"
        )
        
        # Invoke should handle error gracefully
        with pytest.raises(RuntimeError, match="Agent execution error"):
            await agent_manager.invoke_agent("error_agent", {})
    
    async def test_monitoring_integration(self, agent_manager, sample_agent):
        """Test that monitoring is properly integrated"""
        # Mock monitoring to track calls
        agent_manager._monitoring.record_metric = AsyncMock()
        
        # Register and invoke an agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        await agent_manager.invoke_agent(
            agent_name="sample_agent",
            input_data={"messages": [{"role": "user", "content": "Hello"}]}
        )
        
        # Verify monitoring was called
        agent_manager._monitoring.record_metric.assert_called()
    
    async def test_agent_config_validation(self, agent_manager, sample_agent):
        """Test agent configuration validation"""
        # Test invalid config
        with pytest.raises(ValueError, match="Invalid agent configuration"):
            await agent_manager.register_agent(
                name="sample_agent",
                agent=sample_agent,
                description="A sample agent",
                config="invalid_config"  # Should be dict
            )
    
    async def test_agent_caching(self, agent_manager, sample_agent):
        """Test that agents are properly cached"""
        # Register an agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Get agent multiple times
        agent1 = agent_manager.get_agent("sample_agent")
        agent2 = agent_manager.get_agent("sample_agent")
        agent3 = agent_manager.get_agent("sample_agent")
        
        # All should be same instance
        assert agent1 is agent2
        assert agent2 is agent3
    
    async def test_get_statistics(self, agent_manager, sample_agent):
        """Test getting agent manager statistics"""
        # Register some agents
        await agent_manager.register_agent(
            name="sample_agent_1",
            agent=sample_agent,
            description="First sample agent"
        )
        
        await agent_manager.register_agent(
            name="sample_agent_2",
            agent=sample_agent,
            description="Second sample agent"
        )
        
        # Get statistics
        stats = agent_manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_agents_registered" in stats
        assert "initialized" in stats
        assert stats["total_agents_registered"] == 2
    
    async def test_agent_thread_management(self, agent_manager, sample_agent):
        """Test agent thread management"""
        # Register agent
        await agent_manager.register_agent(
            name="sample_agent",
            agent=sample_agent,
            description="A sample agent for testing"
        )
        
        # Invoke agent with different thread IDs
        result1 = await agent_manager.invoke_agent(
            agent_name="sample_agent",
            input_data={"messages": [{"role": "user", "content": "Hello 1"}]},
            config={"thread_id": "thread_1"}
        )
        
        result2 = await agent_manager.invoke_agent(
            agent_name="sample_agent",
            input_data={"messages": [{"role": "user", "content": "Hello 2"}]},
            config={"thread_id": "thread_2"}
        )
        
        # Results should be independent
        assert result1 != result2


if __name__ == "__main__":
    pytest.main([__file__])