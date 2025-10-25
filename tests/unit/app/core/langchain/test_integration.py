"""
Unit tests for LangChain Integration Layer.

This module tests the integration layer that bridges LangChain
with existing APIs and components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

from app.core.langchain.integration import (
    LangChainIntegration,
    IntegrationMode,
    IntegrationConfig,
    BridgeConfig,
    IntegrationStats,
    langchain_integration
)


class TestIntegrationMode:
    """Test IntegrationMode enum"""

    def test_integration_mode_values(self):
        """Test that IntegrationMode has expected values"""
        expected_modes = [
            "legacy",
            "langchain",
            "hybrid",
            "migration"
        ]
        
        actual_modes = [mode.value for mode in IntegrationMode]
        assert actual_modes == expected_modes


class TestIntegrationConfig:
    """Test IntegrationConfig dataclass"""

    def test_integration_config_defaults(self):
        """Test IntegrationConfig default values"""
        config = IntegrationConfig(
            mode=IntegrationMode.LEGACY
        )
        
        assert config.mode == IntegrationMode.LEGACY
        assert config.enable_monitoring is True
        assert config.enable_caching is True
        assert config.enable_fallback is True
        assert config.fallback_threshold == 0.8
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.metadata == {}

    def test_integration_config_with_values(self):
        """Test IntegrationConfig with provided values"""
        metadata = {"version": "1.0", "custom": True}
        
        config = IntegrationConfig(
            mode=IntegrationMode.HYBRID,
            enable_monitoring=False,
            enable_caching=False,
            enable_fallback=False,
            fallback_threshold=0.5,
            max_retries=5,
            timeout=60,
            metadata=metadata
        )
        
        assert config.mode == IntegrationMode.HYBRID
        assert config.enable_monitoring is False
        assert config.enable_caching is False
        assert config.enable_fallback is False
        assert config.fallback_threshold == 0.5
        assert config.max_retries == 5
        assert config.timeout == 60
        assert config.metadata == metadata


class TestBridgeConfig:
    """Test BridgeConfig dataclass"""

    def test_bridge_config_defaults(self):
        """Test BridgeConfig default values"""
        config = BridgeConfig(
            name="test_bridge",
            source="legacy",
            target="langchain"
        )
        
        assert config.name == "test_bridge"
        assert config.source == "legacy"
        assert config.target == "langchain"
        assert config.enabled is True
        assert config.transformations == []
        assert config.filters == []
        assert config.metadata == {}

    def test_bridge_config_with_values(self):
        """Test BridgeConfig with provided values"""
        transformations = ["normalize_input", "format_output"]
        filters = ["remove_sensitive_data"]
        metadata = {"version": "1.0"}
        
        config = BridgeConfig(
            name="advanced_bridge",
            source="legacy",
            target="langchain",
            enabled=False,
            transformations=transformations,
            filters=filters,
            metadata=metadata
        )
        
        assert config.name == "advanced_bridge"
        assert config.source == "legacy"
        assert config.target == "langchain"
        assert config.enabled is False
        assert config.transformations == transformations
        assert config.filters == filters
        assert config.metadata == metadata


class TestIntegrationStats:
    """Test IntegrationStats dataclass"""

    def test_integration_stats_defaults(self):
        """Test IntegrationStats default values"""
        stats = IntegrationStats(
            mode="legacy"
        )
        
        assert stats.mode == "legacy"
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.fallback_requests == 0
        assert stats.total_response_time == 0.0
        assert stats.average_response_time == 0.0
        assert stats.last_used is None
        assert stats.created_at is not None

    def test_integration_stats_with_values(self):
        """Test IntegrationStats with provided values"""
        created_at = datetime.now()
        last_used = datetime.now()
        
        stats = IntegrationStats(
            mode="hybrid",
            total_requests=10,
            successful_requests=8,
            failed_requests=1,
            fallback_requests=1,
            total_response_time=45.5,
            average_response_time=4.55,
            last_used=last_used,
            created_at=created_at
        )
        
        assert stats.mode == "hybrid"
        assert stats.total_requests == 10
        assert stats.successful_requests == 8
        assert stats.failed_requests == 1
        assert stats.fallback_requests == 1
        assert stats.total_response_time == 45.5
        assert stats.average_response_time == 4.55
        assert stats.last_used == last_used
        assert stats.created_at == created_at


class TestLangChainIntegration:
    """Test LangChainIntegration class"""

    @pytest.fixture
    def integration_instance(self):
        """Create a fresh integration instance for testing"""
        return LangChainIntegration()

    @pytest.fixture
    def mock_agent_manager(self):
        """Mock agent manager"""
        with patch('app.core.langchain.integration.agent_manager') as mock:
            mock.initialize = AsyncMock()
            mock.invoke_agent = AsyncMock(return_value={"success": True, "response": "Test response"})
            mock.create_conversation = AsyncMock(return_value="conv-123")
            yield mock

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        with patch('app.core.langchain.integration.llm_manager') as mock:
            mock.initialize = AsyncMock()
            mock.get_llm = AsyncMock(return_value=Mock())
            yield mock

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager"""
        with patch('app.core.langchain.integration.memory_manager') as mock:
            mock.initialize = AsyncMock()
            mock.create_conversation = AsyncMock(return_value=True)
            mock.add_message = AsyncMock(return_value=True)
            mock.get_conversation_messages = AsyncMock(return_value=[])
            yield mock

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        with patch('app.core.langchain.integration.tool_registry') as mock:
            mock.initialize = AsyncMock()
            mock.list_tools = AsyncMock(return_value=[])
            mock.execute_tool = AsyncMock(return_value={"success": True, "result": "Tool result"})
            yield mock

    @pytest.fixture
    def mock_monitoring(self):
        """Mock monitoring system"""
        with patch('app.core.langchain.integration.LangChainMonitoring') as mock:
            mock_instance = Mock()
            mock_instance.initialize = AsyncMock()
            mock_instance.track_integration_request = AsyncMock()
            mock.return_value = mock_instance
            yield mock

    @pytest.mark.asyncio
    async def test_initialize(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_llm_manager, 
        mock_memory_manager, 
        mock_tool_registry, 
        mock_monitoring
    ):
        """Test integration initialization"""
        config = IntegrationConfig(mode=IntegrationMode.LANGCHAIN)
        
        await integration_instance.initialize(config)
        
        assert integration_instance._initialized is True
        assert integration_instance._config == config
        mock_agent_manager.initialize.assert_called_once()
        mock_llm_manager.initialize.assert_called_once()
        mock_memory_manager.initialize.assert_called_once()
        mock_tool_registry.initialize.assert_called_once()
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self, 
        integration_instance, 
        mock_agent_manager
    ):
        """Test that initialize is idempotent"""
        config = IntegrationConfig(mode=IntegrationMode.LEGACY)
        
        await integration_instance.initialize(config)
        await integration_instance.initialize(config)
        
        assert integration_instance._initialized is True
        # Should only initialize once
        mock_agent_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_bridge(
        self, 
        integration_instance
    ):
        """Test registering a bridge"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        bridge_config = BridgeConfig(
            name="legacy_to_langchain",
            source="legacy",
            target="langchain",
            transformations=["normalize_input"],
            filters=["remove_sensitive_data"]
        )
        
        result = await integration_instance.register_bridge(bridge_config)
        
        assert result is True
        assert "legacy_to_langchain" in integration_instance._bridges
        assert integration_instance._bridges["legacy_to_langchain"] == bridge_config

    @pytest.mark.asyncio
    async def test_register_bridge_duplicate(
        self, 
        integration_instance
    ):
        """Test registering duplicate bridge"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        bridge_config = BridgeConfig(
            name="duplicate_bridge",
            source="legacy",
            target="langchain"
        )
        
        # Register first time
        result1 = await integration_instance.register_bridge(bridge_config)
        assert result1 is True
        
        # Register second time
        result2 = await integration_instance.register_bridge(bridge_config)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_process_request_legacy_mode(
        self, 
        integration_instance, 
        mock_monitoring
    ):
        """Test processing request in legacy mode"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LEGACY))
        
        # Mock legacy processor
        with patch('app.core.langchain.integration.legacy_processor') as mock_legacy:
            mock_legacy.process = AsyncMock(return_value={"success": True, "response": "Legacy response"})
            
            result = await integration_instance.process_request(
                request_type="chat",
                data={"message": "Hello, world!"},
                conversation_id="conv-123"
            )
            
            assert result["success"] is True
            assert result["response"] == "Legacy response"
            assert result["mode"] == "legacy"
            mock_legacy.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_langchain_mode(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_monitoring
    ):
        """Test processing request in LangChain mode"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        result = await integration_instance.process_request(
            request_type="chat",
            data={"message": "Hello, world!"},
            conversation_id="conv-123"
        )
        
        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["mode"] == "langchain"
        mock_agent_manager.invoke_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_hybrid_mode_success(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_monitoring
    ):
        """Test processing request in hybrid mode with LangChain success"""
        await integration_instance.initialize(IntegrationConfig(
            mode=IntegrationMode.HYBRID,
            fallback_threshold=0.8
        ))
        
        result = await integration_instance.process_request(
            request_type="chat",
            data={"message": "Hello, world!"},
            conversation_id="conv-123"
        )
        
        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["mode"] == "langchain"
        assert result["fallback_used"] is False
        mock_agent_manager.invoke_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_hybrid_mode_fallback(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_monitoring
    ):
        """Test processing request in hybrid mode with fallback to legacy"""
        await integration_instance.initialize(IntegrationConfig(
            mode=IntegrationMode.HYBRID,
            fallback_threshold=0.8
        ))
        
        # Mock LangChain failure
        mock_agent_manager.invoke_agent = AsyncMock(return_value={
            "success": False,
            "error": "LangChain failed"
        })
        
        # Mock legacy processor
        with patch('app.core.langchain.integration.legacy_processor') as mock_legacy:
            mock_legacy.process = AsyncMock(return_value={"success": True, "response": "Legacy response"})
            
            result = await integration_instance.process_request(
                request_type="chat",
                data={"message": "Hello, world!"},
                conversation_id="conv-123"
            )
            
            assert result["success"] is True
            assert result["response"] == "Legacy response"
            assert result["mode"] == "legacy"
            assert result["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_process_request_with_bridge(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_monitoring
    ):
        """Test processing request with bridge transformation"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Register bridge
        bridge_config = BridgeConfig(
            name="input_transformer",
            source="legacy",
            target="langchain",
            transformations=["normalize_input", "add_context"]
        )
        await integration_instance.register_bridge(bridge_config)
        
        # Mock transformation functions
        with patch('app.core.langchain.integration.apply_transformations') as mock_transform:
            mock_transform.return_value = {"message": "Transformed: Hello, world!"}
            
            result = await integration_instance.process_request(
                request_type="chat",
                data={"message": "Hello, world!"},
                conversation_id="conv-123"
            )
            
            assert result["success"] is True
            mock_transform.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation(
        self, 
        integration_instance, 
        mock_agent_manager, 
        mock_memory_manager
    ):
        """Test creating a conversation"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        result = await integration_instance.create_conversation(
            agent_name="test_agent",
            title="Test Conversation",
            metadata={"user_id": "user-123"}
        )
        
        assert result["success"] is True
        assert result["conversation_id"] == "conv-123"
        mock_agent_manager.create_conversation.assert_called_once()
        mock_memory_manager.create_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_history(
        self, 
        integration_instance, 
        mock_memory_manager
    ):
        """Test getting conversation history"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        # Mock messages
        mock_messages = [
            {"role": "human", "content": "Hello"},
            {"role": "ai", "content": "Hi there!"}
        ]
        mock_memory_manager.get_conversation_messages.return_value = mock_messages
        
        result = await integration_instance.get_conversation_history(
            conversation_id="conv-123",
            limit=10
        )
        
        assert result == mock_messages
        mock_memory_manager.get_conversation_messages.assert_called_once_with("conv-123", limit=10)

    @pytest.mark.asyncio
    async def test_execute_tool(
        self, 
        integration_instance, 
        mock_tool_registry
    ):
        """Test executing a tool"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        result = await integration_instance.execute_tool(
            tool_name="search_tool",
            input_data={"query": "test query"}
        )
        
        assert result["success"] is True
        assert result["result"] == "Tool result"
        mock_tool_registry.execute_tool.assert_called_once_with(
            "search_tool", {"query": "test query"}
        )

    @pytest.mark.asyncio
    async def test_list_available_tools(
        self, 
        integration_instance, 
        mock_tool_registry
    ):
        """Test listing available tools"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        # Mock tools
        mock_tools = [
            {"name": "search_tool", "description": "Search tool", "enabled": True},
            {"name": "calculator_tool", "description": "Calculator tool", "enabled": True}
        ]
        mock_tool_registry.list_tools.return_value = mock_tools
        
        result = await integration_instance.list_available_tools()
        
        assert len(result) == 2
        assert result[0]["name"] == "search_tool"
        assert result[1]["name"] == "calculator_tool"
        mock_tool_registry.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_integration_stats(
        self, 
        integration_instance
    ):
        """Test getting integration statistics"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Add some stats
        integration_instance._stats.total_requests = 10
        integration_instance._stats.successful_requests = 8
        integration_instance._stats.failed_requests = 1
        integration_instance._stats.fallback_requests = 1
        integration_instance._stats.total_response_time = 45.5
        integration_instance._stats.average_response_time = 4.55
        
        result = await integration_instance.get_integration_stats()
        
        assert result["mode"] == "hybrid"
        assert result["total_requests"] == 10
        assert result["successful_requests"] == 8
        assert result["failed_requests"] == 1
        assert result["fallback_requests"] == 1
        assert result["total_response_time"] == 45.5
        assert result["average_response_time"] == 4.55

    @pytest.mark.asyncio
    async def test_reset_stats(self, integration_instance):
        """Test resetting integration statistics"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        # Add some stats
        integration_instance._stats.total_requests = 10
        integration_instance._stats.successful_requests = 8
        integration_instance._stats.failed_requests = 2
        
        result = await integration_instance.reset_stats()
        
        assert result is True
        assert integration_instance._stats.total_requests == 0
        assert integration_instance._stats.successful_requests == 0
        assert integration_instance._stats.failed_requests == 0
        assert integration_instance._stats.fallback_requests == 0
        assert integration_instance._stats.total_response_time == 0.0
        assert integration_instance._stats.average_response_time == 0.0

    @pytest.mark.asyncio
    async def test_get_bridge(self, integration_instance):
        """Test getting a bridge"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Register bridge
        bridge_config = BridgeConfig(
            name="test_bridge",
            source="legacy",
            target="langchain"
        )
        await integration_instance.register_bridge(bridge_config)
        
        # Get existing bridge
        result = integration_instance.get_bridge("test_bridge")
        assert result == bridge_config
        
        # Get non-existent bridge
        result = integration_instance.get_bridge("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_bridges(self, integration_instance):
        """Test listing bridges"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Register bridges
        bridges = [
            BridgeConfig(name="bridge1", source="legacy", target="langchain"),
            BridgeConfig(name="bridge2", source="api", target="langchain"),
            BridgeConfig(name="bridge3", source="langchain", target="legacy")
        ]
        
        for bridge in bridges:
            await integration_instance.register_bridge(bridge)
        
        result = await integration_instance.list_bridges()
        
        assert len(result) == 3
        bridge_names = [bridge.name for bridge in result]
        assert "bridge1" in bridge_names
        assert "bridge2" in bridge_names
        assert "bridge3" in bridge_names

    @pytest.mark.asyncio
    async def test_enable_disable_bridge(self, integration_instance):
        """Test enabling and disabling bridges"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Register bridge
        bridge_config = BridgeConfig(
            name="test_bridge",
            source="legacy",
            target="langchain",
            enabled=True
        )
        await integration_instance.register_bridge(bridge_config)
        
        # Disable bridge
        result = await integration_instance.disable_bridge("test_bridge")
        assert result is True
        assert integration_instance._bridges["test_bridge"].enabled is False
        
        # Enable bridge
        result = await integration_instance.enable_bridge("test_bridge")
        assert result is True
        assert integration_instance._bridges["test_bridge"].enabled is True

    @pytest.mark.asyncio
    async def test_apply_transformations(self, integration_instance):
        """Test applying transformations to data"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Mock transformation functions
        def normalize_input(data):
            return {**data, "message": data["message"].strip().lower()}
        
        def add_context(data):
            return {**data, "context": "test_context"}
        
        # Register transformations
        integration_instance._transformations = {
            "normalize_input": normalize_input,
            "add_context": add_context
        }
        
        # Apply transformations
        data = {"message": "  Hello, World!  "}
        result = integration_instance._apply_transformations(
            data, ["normalize_input", "add_context"]
        )
        
        assert result["message"] == "hello, world!"
        assert result["context"] == "test_context"

    @pytest.mark.asyncio
    async def test_apply_filters(self, integration_instance):
        """Test applying filters to data"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Mock filter functions
        def remove_sensitive_data(data):
            filtered = data.copy()
            if "password" in filtered:
                filtered["password"] = "[REDACTED]"
            return filtered
        
        def sanitize_output(data):
            filtered = data.copy()
            if "response" in filtered:
                filtered["response"] = filtered["response"].replace("secret", "[REDACTED]")
            return filtered
        
        # Register filters
        integration_instance._filters = {
            "remove_sensitive_data": remove_sensitive_data,
            "sanitize_output": sanitize_output
        }
        
        # Apply filters
        data = {
            "message": "Here's a secret",
            "password": "secret123",
            "response": "This response contains a secret"
        }
        result = integration_instance._apply_filters(
            data, ["remove_sensitive_data", "sanitize_output"]
        )
        
        assert result["message"] == "Here's a secret"
        assert result["password"] == "[REDACTED]"
        assert result["response"] == "This response contains a [REDACTED]"

    @pytest.mark.asyncio
    async def test_track_request(
        self, 
        integration_instance, 
        mock_monitoring
    ):
        """Test tracking integration requests"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.LANGCHAIN))
        
        await integration_instance._track_request(
            request_type="chat",
            mode="langchain",
            success=True,
            response_time=2.5,
            fallback_used=False,
            metadata={"conversation_id": "conv-123"}
        )
        
        # Update stats
        assert integration_instance._stats.total_requests == 1
        assert integration_instance._stats.successful_requests == 1
        assert integration_instance._stats.failed_requests == 0
        assert integration_instance._stats.fallback_requests == 0
        assert integration_instance._stats.total_response_time == 2.5
        assert integration_instance._stats.average_response_time == 2.5
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_integration_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_use_fallback(self, integration_instance):
        """Test fallback decision logic"""
        await integration_instance.initialize(IntegrationConfig(
            mode=IntegrationMode.HYBRID,
            fallback_threshold=0.8
        ))
        
        # Should not fallback (high confidence)
        result = integration_instance._should_use_fallback(
            langchain_result={"success": True, "confidence": 0.9}
        )
        assert result is False
        
        # Should fallback (low confidence)
        result = integration_instance._should_use_fallback(
            langchain_result={"success": True, "confidence": 0.7}
        )
        assert result is True
        
        # Should fallback (failure)
        result = integration_instance._should_use_fallback(
            langchain_result={"success": False}
        )
        assert result is True

    def test_get_mode_for_request_type(self, integration_instance):
        """Test getting mode for request type"""
        # Test with legacy mode
        integration_instance._config = IntegrationConfig(mode=IntegrationMode.LEGACY)
        result = integration_instance._get_mode_for_request_type("chat")
        assert result == "legacy"
        
        # Test with langchain mode
        integration_instance._config = IntegrationConfig(mode=IntegrationMode.LANGCHAIN)
        result = integration_instance._get_mode_for_request_type("chat")
        assert result == "langchain"
        
        # Test with hybrid mode (should default to langchain)
        integration_instance._config = IntegrationConfig(mode=IntegrationMode.HYBRID)
        result = integration_instance._get_mode_for_request_type("chat")
        assert result == "langchain"

    @pytest.mark.asyncio
    async def test_process_legacy_request(
        self, 
        integration_instance
    ):
        """Test processing legacy request"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        # Mock legacy processor
        with patch('app.core.langchain.integration.legacy_processor') as mock_legacy:
            mock_legacy.process = AsyncMock(return_value={"success": True, "response": "Legacy response"})
            
            result = await integration_instance._process_legacy_request(
                request_type="chat",
                data={"message": "Hello, world!"},
                conversation_id="conv-123"
            )
            
            assert result["success"] is True
            assert result["response"] == "Legacy response"
            assert result["mode"] == "legacy"

    @pytest.mark.asyncio
    async def test_process_langchain_request(
        self, 
        integration_instance, 
        mock_agent_manager
    ):
        """Test processing LangChain request"""
        await integration_instance.initialize(IntegrationConfig(mode=IntegrationMode.HYBRID))
        
        result = await integration_instance._process_langchain_request(
            request_type="chat",
            data={"message": "Hello, world!"},
            conversation_id="conv-123"
        )
        
        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["mode"] == "langchain"
        mock_agent_manager.invoke_agent.assert_called_once()


class TestGlobalLangChainIntegration:
    """Test global LangChain integration instance"""

    def test_global_instance(self):
        """Test that global LangChain integration instance exists"""
        from app.core.langchain.integration import langchain_integration
        assert langchain_integration is not None
        assert isinstance(langchain_integration, LangChainIntegration)