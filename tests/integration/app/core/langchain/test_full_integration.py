"""
Integration tests for complete LangChain system.

This module tests end-to-end integration between all LangChain components
including LLM Manager, Tool Registry, Agent Manager, Memory Manager,
and Monitoring System.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.core.langchain.integration import LangChainIntegration
from app.core.langchain.llm_manager import LangChainLLMManager
from app.core.langchain.tool_registry import LangChainToolRegistry
from app.core.langchain.agent_manager import LangGraphAgentManager
from app.core.langchain.memory_manager import LangChainMemoryManager
from app.core.langchain.monitoring import LangChainMonitoring
from app.core.secure_settings import secure_settings


class TestLangChainFullIntegration:
    """Test cases for complete LangChain system integration"""
    
    @pytest.fixture
    async def integration(self):
        """Create a fully initialized LangChain integration"""
        integration = LangChainIntegration()
        await integration.initialize()
        return integration
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'integration_mode'): 'langchain',
            ('langchain', 'llm_manager_enabled'): True,
            ('langchain', 'tool_registry_enabled'): True,
            ('langchain', 'agent_manager_enabled'): True,
            ('langchain', 'memory_workflow_enabled'): True,
            ('langchain', 'monitoring_enabled'): True,
            ('llm_providers', 'openai', 'api_key'): 'test-openai-key',
            ('llm_providers', 'openai', 'model'): 'gpt-3.5-turbo',
            ('database', 'host'): 'localhost',
            ('database', 'port'): '5432',
            ('database', 'name'): 'test_db',
            ('database', 'user'): 'test_user',
            ('database', 'password'): 'test_password',
        }.get((section, key), default)
        return mock_settings
    
    async def test_full_system_initialization(self, mock_settings):
        """Test complete system initialization"""
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            with patch('app.core.langchain.integration.get_langchain_client') as mock_db_client:
                mock_db_client.return_value = Mock()
                
                integration = LangChainIntegration()
                await integration.initialize()
                
                # Verify all components are initialized
                assert integration._initialized is True
                assert isinstance(integration._llm_manager, LangChainLLMManager)
                assert isinstance(integration._tool_registry, LangChainToolRegistry)
                assert isinstance(integration._agent_manager, LangGraphAgentManager)
                assert isinstance(integration._memory_manager, LangChainMemoryManager)
                assert isinstance(integration._monitoring, LangChainMonitoring)
    
    async def test_component_communication(self, integration):
        """Test communication between components"""
        # Get component instances
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        agent_manager = integration.get_agent_manager()
        memory_manager = integration.get_memory_manager()
        
        # Verify components can interact
        # Get LLM from manager
        llm = await llm_manager.get_llm("gpt-3.5-turbo")
        assert llm is not None
        
        # Create a tool that uses LLM
        def llm_tool(input_text: str) -> str:
            return f"LLM processed: {input_text}"
        
        await tool_registry.register_custom_tool(
            name="llm_tool",
            func=llm_tool,
            description="Tool that uses LLM"
        )
        
        # Get tool from registry
        tool = tool_registry.get_tool("llm_tool")
        assert tool is not None
        
        # Create conversation in memory
        await memory_manager.create_conversation(
            conversation_id="test_conv",
            agent_name="test_agent"
        )
        
        # Add message to conversation
        await memory_manager.add_message(
            conversation_id="test_conv",
            role="human",
            content="Hello, world!"
        )
        
        # Verify all components are working together
        conv_info = await memory_manager.get_conversation_info("test_conv")
        assert conv_info is not None
        assert conv_info.conversation_id == "test_conv"
    
    async def test_end_to_end_agent_workflow(self, integration):
        """Test complete agent workflow with all components"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        agent_manager = integration.get_agent_manager()
        memory_manager = integration.get_memory_manager()
        
        # Create a tool
        def search_tool(query: str) -> Dict[str, Any]:
            return {"results": [f"Result for {query}"]}
        
        await tool_registry.register_custom_tool(
            name="search_tool",
            func=search_tool,
            description="Search tool"
        )
        
        # Create a simple agent workflow
        async def agent_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
            """Simple agent workflow"""
            messages = state.get("messages", [])
            
            # Get LLM
            llm = await llm_manager.get_llm("gpt-3.5-turbo")
            
            # Process with LLM
            if messages:
                last_message = messages[-1]["content"]
                response = await llm.ainvoke(f"Respond to: {last_message}")
                return {
                    "messages": messages + [{"role": "assistant", "content": response.content}],
                    "next": "end"
                }
            
            return {"messages": messages, "next": "end"}
        
        # Create and register agent
        from langgraph.graph import StateGraph
        from langgraph.checkpoint.memory import MemorySaver
        
        workflow = StateGraph(dict)
        workflow.add_node("process", agent_workflow)
        workflow.set_entry_point("process")
        workflow.add_edge("process", "end")
        
        compiled_agent = workflow.compile(checkpointer=MemorySaver())
        
        await agent_manager.register_agent(
            name="test_agent",
            agent=compiled_agent,
            description="Test agent for integration"
        )
        
        # Create conversation
        await memory_manager.create_conversation(
            conversation_id="integration_test",
            agent_name="test_agent"
        )
        
        # Add user message
        await memory_manager.add_message(
            conversation_id="integration_test",
            role="human",
            content="Hello, agent!"
        )
        
        # Invoke agent
        result = await agent_manager.invoke_agent(
            agent_name="test_agent",
            input_data={
                "messages": [{"role": "human", "content": "Hello, agent!"}]
            },
            config={"thread_id": "integration_test"}
        )
        
        # Verify workflow completed
        assert "messages" in result
        assert len(result["messages"]) >= 2  # User + assistant
        assert result["messages"][-1]["role"] == "assistant"
    
    async def test_monitoring_across_components(self, integration):
        """Test monitoring across all components"""
        # Get monitoring system
        monitoring = integration.get_monitoring()
        
        # Mock monitoring to track calls
        monitoring.record_metric = AsyncMock()
        
        # Get components and perform operations
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        agent_manager = integration.get_agent_manager()
        memory_manager = integration.get_memory_manager()
        
        # Perform operations that should generate metrics
        await llm_manager.get_llm("gpt-3.5-turbo")
        
        def test_tool(input_data: str) -> str:
            return f"Processed: {input_data}"
        
        await tool_registry.register_custom_tool(
            name="test_tool",
            func=test_tool,
            description="Test tool"
        )
        
        await tool_registry.execute_tool("test_tool", "test input")
        
        await memory_manager.create_conversation("test_conv")
        await memory_manager.add_message("test_conv", "human", "Hello")
        
        # Verify monitoring was called for each component
        assert monitoring.record_metric.call_count >= 4
    
    async def test_error_propagation_between_components(self, integration):
        """Test error handling between components"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        
        # Create a tool that errors
        def error_tool(input_data: str) -> str:
            raise ValueError("Tool error")
        
        await tool_registry.register_custom_tool(
            name="error_tool",
            func=error_tool,
            description="Tool that errors"
        )
        
        # Execute tool should handle error gracefully
        with pytest.raises(ValueError, match="Tool error"):
            await tool_registry.execute_tool("error_tool", "test input")
        
        # System should remain stable after error
        health = await integration.health_check()
        assert "overall_status" in health
        assert "components" in health
    
    async def test_configuration_changes(self, integration):
        """Test system behavior with configuration changes"""
        # Test initial configuration
        mode = integration.get_integration_mode()
        assert mode in ["legacy", "langchain", "hybrid", "migration"]
        
        # Test component status
        llm_enabled = integration.is_component_enabled("llm_manager")
        tool_enabled = integration.is_component_enabled("tool_registry")
        agent_enabled = integration.is_component_enabled("agent_manager")
        memory_enabled = integration.is_component_enabled("memory_workflow")
        
        # All should be enabled in langchain mode
        if mode == "langchain":
            assert llm_enabled is True
            assert tool_enabled is True
            assert agent_enabled is True
            assert memory_enabled is True
    
    async def test_concurrent_operations(self, integration):
        """Test concurrent operations across components"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        memory_manager = integration.get_memory_manager()
        
        # Create multiple tools
        async def create_tool(name: str) -> str:
            await asyncio.sleep(0.1)  # Simulate work
            return f"Tool {name} created"
        
        # Register tools concurrently
        tool_tasks = [
            tool_registry.register_custom_tool(
                name=f"tool_{i}",
                func=lambda x, i=i: f"Result from tool {i}",
                description=f"Tool {i}"
            )
            for i in range(5)
        ]
        
        await asyncio.gather(*tool_tasks)
        
        # Verify all tools were registered
        tools = tool_registry.list_tools()
        assert len(tools) >= 5
        
        # Create conversations concurrently
        conv_tasks = [
            memory_manager.create_conversation(f"conv_{i}")
            for i in range(3)
        ]
        
        await asyncio.gather(*conv_tasks)
        
        # Verify all conversations were created
        convs = await memory_manager.list_conversations()
        assert len(convs) >= 3
    
    async def test_memory_persistence_integration(self, integration):
        """Test memory persistence across components"""
        # Get components
        memory_manager = integration.get_memory_manager()
        
        # Create conversation with messages
        await memory_manager.create_conversation(
            conversation_id="persistence_test",
            agent_name="test_agent"
        )
        
        await memory_manager.add_message(
            conversation_id="persistence_test",
            role="human",
            content="First message"
        )
        
        await memory_manager.add_message(
            conversation_id="persistence_test",
            role="ai",
            content="First response"
        )
        
        # Get messages back
        messages = await memory_manager.get_conversation_messages("persistence_test")
        
        # Verify messages were persisted
        assert len(messages) == 2
        assert messages[0]["role"] == "human"
        assert messages[0]["content"] == "First message"
        assert messages[1]["role"] == "ai"
        assert messages[1]["content"] == "First response"
    
    async def test_system_shutdown(self, integration):
        """Test complete system shutdown"""
        # Mock component shutdown methods
        integration._llm_manager.shutdown = AsyncMock()
        integration._tool_registry.shutdown = AsyncMock()
        integration._agent_manager.shutdown = AsyncMock()
        integration._memory_manager.shutdown = AsyncMock()
        integration._monitoring.shutdown = AsyncMock()
        
        # Shutdown the system
        await integration.shutdown()
        
        # Verify all components were shut down
        integration._llm_manager.shutdown.assert_called_once()
        integration._tool_registry.shutdown.assert_called_once()
        integration._agent_manager.shutdown.assert_called_once()
        integration._memory_manager.shutdown.assert_called_once()
        integration._monitoring.shutdown.assert_called_once()
        
        # Verify system is marked as not initialized
        assert integration._initialized is False
    
    async def test_performance_under_load(self, integration):
        """Test system performance under load"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        
        # Create multiple tools
        for i in range(10):
            await tool_registry.register_custom_tool(
                name=f"load_tool_{i}",
                func=lambda x, i=i: f"Result {i}: {x}",
                description=f"Load test tool {i}"
            )
        
        # Execute tools concurrently under load
        start_time = asyncio.get_event_loop().time()
        
        tasks = [
            tool_registry.execute_tool(f"load_tool_{i % 10}", f"input_{j}")
            for i in range(50)
            for j in range(1)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Verify performance
        assert duration < 10.0  # Should complete within 10 seconds
        assert all(not isinstance(r, Exception) for r in results)
        assert len(results) == 50
    
    async def test_feature_flag_transitions(self, integration):
        """Test behavior during feature flag transitions"""
        # Test legacy mode
        with patch.object(integration, '_get_integration_mode', return_value='legacy'):
            mode = integration.get_integration_mode()
            assert mode == 'legacy'
        
        # Test langchain mode
        with patch.object(integration, '_get_integration_mode', return_value='langchain'):
            mode = integration.get_integration_mode()
            assert mode == 'langchain'
        
        # Test hybrid mode
        with patch.object(integration, '_get_integration_mode', return_value='hybrid'):
            mode = integration.get_integration_mode()
            assert mode == 'hybrid'
    
    async def test_system_recovery(self, integration):
        """Test system recovery from component failures"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        
        # Simulate component failure
        original_llm_method = llm_manager.get_llm
        
        # Mock LLM to fail temporarily
        async def failing_get_llm(model_name: str):
            if model_name == "failing_model":
                raise Exception("LLM unavailable")
            return await original_llm_method(model_name)
        
        llm_manager.get_llm = failing_get_llm
        
        # Try to get failing model
        with pytest.raises(Exception, match="LLM unavailable"):
            await llm_manager.get_llm("failing_model")
        
        # Verify system still works with other models
        working_llm = await llm_manager.get_llm("gpt-3.5-turbo")
        assert working_llm is not None
        
        # Restore original method
        llm_manager.get_llm = original_llm_method
        
        # Verify recovery
        recovered_llm = await llm_manager.get_llm("failing_model")
        assert recovered_llm is not None
    
    async def test_data_flow_integrity(self, integration):
        """Test data flow integrity between components"""
        # Get components
        llm_manager = integration.get_llm_manager()
        tool_registry = integration.get_tool_registry()
        agent_manager = integration.get_agent_manager()
        memory_manager = integration.get_memory_manager()
        
        # Create data flow: LLM -> Tool -> Agent -> Memory
        llm = await llm_manager.get_llm("gpt-3.5-turbo")
        
        def data_processing_tool(data: str) -> Dict[str, Any]:
            return {"processed": f"LLM: {llm.model_name} processed: {data}"}
        
        await tool_registry.register_custom_tool(
            name="data_processing_tool",
            func=data_processing_tool,
            description="Data processing tool"
        )
        
        # Create agent that uses the tool
        async def data_flow_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            input_data = state.get("input_data", "")
            
            # Use tool
            tool_result = await tool_registry.execute_tool(
                "data_processing_tool", 
                input_data
            )
            
            return {
                "result": tool_result,
                "processed_at": asyncio.get_event_loop().time()
            }
        
        # Create and register agent
        from langgraph.graph import StateGraph
        from langgraph.checkpoint.memory import MemorySaver
        
        workflow = StateGraph(dict)
        workflow.add_node("process", data_flow_agent)
        workflow.set_entry_point("process")
        workflow.add_edge("process", "end")
        
        compiled_agent = workflow.compile(checkpointer=MemorySaver())
        
        await agent_manager.register_agent(
            name="data_flow_agent",
            agent=compiled_agent,
            description="Data flow agent"
        )
        
        # Execute data flow
        test_data = "Test data flow"
        result = await agent_manager.invoke_agent(
            agent_name="data_flow_agent",
            input_data={"input_data": test_data},
            config={"thread_id": "data_flow_test"}
        )
        
        # Verify data integrity
        assert "result" in result
        assert "processed" in result["result"]
        assert "LLM: gpt-3.5-turbo processed: Test data flow" in result["result"]["processed"]
        
        # Store result in memory
        await memory_manager.create_conversation("data_flow_test")
        await memory_manager.add_message(
            conversation_id="data_flow_test",
            role="system",
            content=str(result["result"])
        )
        
        # Verify data was stored correctly
        messages = await memory_manager.get_conversation_messages("data_flow_test")
        assert len(messages) == 1
        assert "processed" in messages[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__])