"""
Integration tests for LangChain components.

This module tests integration between LangChain components and the existing system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass

from app.core.langchain.agent_manager import LangChainAgentManager
from app.core.langchain.llm_manager import LangChainLLMManager
from app.core.langchain.memory_manager import LangChainMemoryManager
from app.core.langchain.tool_registry import LangChainToolRegistry
from app.core.langchain.integration.langchain_integration import LangChainIntegration
from app.core.tools.execution.dynamic_executor import DynamicExecutor
from app.core.tools.execution.registry import ToolRegistry
from app.core.tools.web.searxng_tool import SearXNGTool
from app.core.tools.web.firecrawl_tool import FirecrawlTool


@dataclass
class MockLLMConfig:
    """Mock LLM configuration"""
    model_name: str = "gpt-3.5-turbo"
    api_key: str = "test-api-key"
    temperature: float = 0.7
    max_tokens: int = 1000


class TestLangChainIntegration:
    """Test LangChain integration with existing system"""

    @pytest.fixture
    def mock_llm_config(self):
        """Create mock LLM configuration"""
        return MockLLMConfig()

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with mock tools"""
        registry = ToolRegistry()
        
        # Register mock tools
        searxng_tool = SearXNGTool()
        firecrawl_tool = FirecrawlTool(api_key="test-key")
        
        registry.register_tool(searxng_tool)
        registry.register_tool(firecrawl_tool)
        
        return registry

    @pytest.fixture
    def dynamic_executor(self, tool_registry):
        """Create a dynamic executor with tools"""
        executor = DynamicExecutor()
        
        # Register tools from registry
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            executor.register_tool(tool)
        
        return executor

    @pytest.fixture
    def langchain_tool_registry(self, tool_registry):
        """Create a LangChain tool registry"""
        lc_registry = LangChainToolRegistry()
        
        # Convert existing tools to LangChain tools
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            lc_registry.register_tool(tool)
        
        return lc_registry

    @pytest.fixture
    def llm_manager(self, mock_llm_config):
        """Create a LangChain LLM manager"""
        manager = LangChainLLMManager()
        manager.configure_providers({
            "openai": {
                "api_key": mock_llm_config.api_key,
                "models": [mock_llm_config.model_name]
            }
        })
        return manager

    @pytest.fixture
    def memory_manager(self):
        """Create a LangChain memory manager"""
        manager = LangChainMemoryManager()
        manager.configure_backends({
            "conversation": {
                "type": "conversation",
                "max_messages": 10
            },
            "summary": {
                "type": "summary",
                "max_tokens": 500
            }
        })
        return manager

    @pytest.fixture
    def agent_manager(self, llm_manager, langchain_tool_registry, memory_manager):
        """Create a LangChain agent manager"""
        manager = LangChainAgentManager()
        manager.configure(
            llm_manager=llm_manager,
            tool_registry=langchain_tool_registry,
            memory_manager=memory_manager
        )
        return manager

    @pytest.fixture
    def langchain_integration(self, agent_manager, dynamic_executor):
        """Create a LangChain integration instance"""
        integration = LangChainIntegration()
        integration.configure(
            agent_manager=agent_manager,
            executor=dynamic_executor,
            mode="hybrid"  # Use both LangChain and existing system
        )
        return integration

    @pytest.mark.asyncio
    async def test_langchain_tool_conversion(self, tool_registry, langchain_tool_registry):
        """Test conversion of existing tools to LangChain tools"""
        # Get original tool
        original_tool = tool_registry.get_tool("searxng_search")
        
        # Get LangChain tool
        lc_tool = langchain_tool_registry.get_tool("searxng_search")
        
        assert lc_tool is not None
        assert lc_tool.name == "searxng_search"
        assert lc_tool.description == original_tool.description
        
        # Test LangChain tool execution
        with patch.object(original_tool, 'execute') as mock_execute:
            mock_execute.return_value = Mock(
                success=True,
                data={"results": [{"title": "Test Result"}]},
                message="Success"
            )
            
            result = await lc_tool.arun(query="test query")
            
            assert "Test Result" in result
            mock_execute.assert_called_once_with(query="test query")

    @pytest.mark.asyncio
    async def test_llm_manager_integration(self, llm_manager, mock_llm_config):
        """Test LLM manager integration"""
        # Get LLM instance
        llm = llm_manager.get_llm("openai", mock_llm_config.model_name)
        
        assert llm is not None
        
        # Mock LLM response
        with patch('langchain_openai.ChatOpenAI.acomplete') as mock_complete:
            mock_complete.return_value = Mock(
                content="Test response from LLM",
                additional_kwargs={},
                usage={"total_tokens": 50}
            )
            
            result = await llm.acomplete("Test prompt")
            
            assert "Test response from LLM" in result.content

    @pytest.mark.asyncio
    async def test_memory_manager_integration(self, memory_manager):
        """Test memory manager integration"""
        # Get memory instance
        memory = memory_manager.get_memory("conversation")
        
        assert memory is not None
        
        # Add messages to memory
        await memory.aadd_messages([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ])
        
        # Get messages from memory
        messages = await memory.aget_messages()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_agent_manager_integration(self, agent_manager, mock_llm_config):
        """Test agent manager integration"""
        # Create agent
        agent = agent_manager.create_agent(
            agent_type="conversational",
            model_name=mock_llm_config.model_name,
            provider="openai",
            tools_enabled=True
        )
        
        assert agent is not None
        assert agent.agent_type == "conversational"
        
        # Mock agent execution
        with patch.object(agent, 'arun') as mock_run:
            mock_run.return_value = {
                "output": "Agent response",
                "intermediate_steps": [],
                "tool_calls": []
            }
            
            result = await agent.arun("Hello, agent!")
            
            assert result["output"] == "Agent response"

    @pytest.mark.asyncio
    async def test_hybrid_mode_integration(self, langchain_integration):
        """Test hybrid mode integration"""
        # Test with LangChain mode
        result = await langchain_integration.process_request(
            message="Hello",
            mode="langchain",
            agent_type="conversational"
        )
        
        assert "response" in result
        assert "mode" in result
        assert result["mode"] == "langchain"
        
        # Test with legacy mode
        result = await langchain_integration.process_request(
            message="Hello",
            mode="legacy"
        )
        
        assert "response" in result
        assert result["mode"] == "legacy"

    @pytest.mark.asyncio
    async def test_tool_execution_integration(self, langchain_integration):
        """Test tool execution integration"""
        # Mock tool execution
        with patch.object(langchain_integration.executor, 'execute_tool') as mock_execute:
            mock_execute.return_value = Mock(
                success=True,
                data={"results": [{"title": "Tool result"}]},
                message="Tool executed successfully"
            )
            
            result = await langchain_integration.execute_tool(
                tool_name="searxng_search",
                parameters={"query": "test query"}
            )
            
            assert result.success is True
            assert "Tool result" in result.data["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_agent_with_tools_integration(self, langchain_integration):
        """Test agent with tools integration"""
        # Mock agent execution with tool calls
        with patch.object(langchain_integration.agent_manager, 'create_agent') as mock_create:
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = {
                "output": "I'll search for that information.",
                "tool_calls": [
                    {
                        "name": "searxng_search",
                        "arguments": {"query": "search query"}
                    }
                ]
            }
            mock_create.return_value = mock_agent
            
            result = await langchain_integration.process_request(
                message="Search for information",
                mode="langchain",
                agent_type="tool_heavy",
                tools_enabled=True
            )
            
            assert "response" in result
            assert "tool_calls" in result
            assert len(result["tool_calls"]) > 0

    @pytest.mark.asyncio
    async def test_memory_persistence_integration(self, langchain_integration, memory_manager):
        """Test memory persistence integration"""
        # Create conversation with memory
        conversation_id = str(uuid.uuid4())
        
        # Add messages to conversation
        await langchain_integration.add_conversation_message(
            conversation_id=conversation_id,
            message="Hello",
            role="user"
        )
        
        await langchain_integration.add_conversation_message(
            conversation_id=conversation_id,
            message="Hi there!",
            role="assistant"
        )
        
        # Get conversation history
        history = await langchain_integration.get_conversation_history(conversation_id)
        
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_mode_switching_integration(self, langchain_integration):
        """Test mode switching integration"""
        # Start with legacy mode
        result1 = await langchain_integration.process_request(
            message="Test message 1",
            mode="legacy"
        )
        
        assert result1["mode"] == "legacy"
        
        # Switch to LangChain mode
        result2 = await langchain_integration.process_request(
            message="Test message 2",
            mode="langchain"
        )
        
        assert result2["mode"] == "langchain"
        
        # Switch to hybrid mode
        result3 = await langchain_integration.process_request(
            message="Test message 3",
            mode="hybrid"
        )
        
        assert result3["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, langchain_integration):
        """Test error handling integration"""
        # Mock tool execution error
        with patch.object(langchain_integration.executor, 'execute_tool') as mock_execute:
            mock_execute.side_effect = Exception("Tool execution failed")
            
            result = await langchain_integration.execute_tool(
                tool_name="searxng_search",
                parameters={"query": "test query"}
            )
            
            assert result.success is False
            assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, langchain_integration):
        """Test performance monitoring integration"""
        # Process some requests
        await langchain_integration.process_request(
            message="Test message 1",
            mode="legacy"
        )
        
        await langchain_integration.process_request(
            message="Test message 2",
            mode="langchain"
        )
        
        # Get performance metrics
        metrics = await langchain_integration.get_performance_metrics()
        
        assert "total_requests" in metrics
        assert "avg_response_time" in metrics
        assert "mode_distribution" in metrics
        assert metrics["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, langchain_integration):
        """Test concurrent request handling"""
        # Send multiple concurrent requests
        async def send_request(message):
            return await langchain_integration.process_request(
                message=message,
                mode="legacy"
            )
        
        tasks = [
            send_request(f"Concurrent message {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all("response" in result for result in results)

    @pytest.mark.asyncio
    async def test_configuration_validation_integration(self, langchain_integration):
        """Test configuration validation integration"""
        # Test valid configuration
        valid_config = {
            "llm_provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "agent_type": "conversational",
            "tools_enabled": True
        }
        
        is_valid = await langchain_integration.validate_configuration(valid_config)
        assert is_valid is True
        
        # Test invalid configuration
        invalid_config = {
            "llm_provider": "invalid_provider",
            "model_name": "invalid_model"
        }
        
        is_valid = await langchain_integration.validate_configuration(invalid_config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_migration_mode_integration(self, langchain_integration):
        """Test migration mode integration"""
        # Configure migration mode
        langchain_integration.set_mode("migration")
        
        # Process request in migration mode
        result = await langchain_integration.process_request(
            message="Test migration",
            mode="migration"
        )
        
        assert result["mode"] == "migration"
        assert "legacy_response" in result
        assert "langchain_response" in result
        assert "comparison" in result

    @pytest.mark.asyncio
    async def test_tool_registry_synchronization(self, langchain_integration, tool_registry):
        """Test tool registry synchronization"""
        # Add new tool to original registry
        new_tool = MockTool("new_integration_tool")
        tool_registry.register_tool(new_tool)
        
        # Sync with LangChain registry
        await langchain_integration.sync_tools()
        
        # Check if tool is available in LangChain
        lc_tool = langchain_integration.langchain_tool_registry.get_tool("new_integration_tool")
        assert lc_tool is not None

    @pytest.mark.asyncio
    async def test_fallback_mechanism_integration(self, langchain_integration):
        """Test fallback mechanism integration"""
        # Configure fallback
        langchain_integration.configure_fallback(
            primary_mode="langchain",
            fallback_mode="legacy",
            fallback_triggers=["timeout", "error"]
        )
        
        # Mock LangChain failure
        with patch.object(langchain_integration.agent_manager, 'create_agent') as mock_create:
            mock_create.side_effect = Exception("LangChain failed")
            
            result = await langchain_integration.process_request(
                message="Test fallback",
                mode="langchain"
            )
            
            assert result["mode"] == "legacy"  # Should fall back to legacy
            assert "fallback_triggered" in result
            assert result["fallback_triggered"] is True


@dataclass
class MockTool:
    """Mock tool for testing"""
    name: str
    description: str = "Mock tool for testing"
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = [self.name]
    
    async def execute(self, **kwargs):
        return Mock(
            success=True,
            data={"result": f"Mock result for {self.name}"},
            message="Success"
        )