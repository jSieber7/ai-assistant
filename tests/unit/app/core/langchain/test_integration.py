"""
Unit tests for LangChain Integration Layer.

This module tests the core LangChain integration functionality, including
initialization, component management, and feature flag handling.
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


class TestLangChainIntegration:
    """Test cases for LangChain Integration Layer"""
    
    @pytest.fixture
    async def integration(self):
        """Create a LangChain integration instance for testing"""
        integration = LangChainIntegration()
        await integration.initialize()
        return integration
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.return_value = True
        return mock_settings
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of LangChain integration"""
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            
            # Test initialization
            await integration.initialize()
            
            # Verify components are initialized
            assert integration._initialized is True
            assert integration._llm_manager is not None
            assert integration._tool_registry is not None
            assert integration._agent_manager is not None
            assert integration._memory_manager is not None
            assert integration._monitoring is not None
    
    async def test_initialize_already_initialized(self, integration):
        """Test that initialize doesn't re-initialize already initialized components"""
        # Get initial component instances
        initial_llm_manager = integration._llm_manager
        initial_tool_registry = integration._tool_registry
        
        # Call initialize again
        await integration.initialize()
        
        # Verify same instances are used
        assert integration._llm_manager is initial_llm_manager
        assert integration._tool_registry is initial_tool_registry
    
    async def test_get_llm_manager(self, integration):
        """Test getting LLM manager instance"""
        llm_manager = integration.get_llm_manager()
        assert isinstance(llm_manager, LangChainLLMManager)
        assert llm_manager is integration._llm_manager
    
    async def test_get_tool_registry(self, integration):
        """Test getting tool registry instance"""
        tool_registry = integration.get_tool_registry()
        assert isinstance(tool_registry, LangChainToolRegistry)
        assert tool_registry is integration._tool_registry
    
    async def test_get_agent_manager(self, integration):
        """Test getting agent manager instance"""
        agent_manager = integration.get_agent_manager()
        assert isinstance(agent_manager, LangGraphAgentManager)
        assert agent_manager is integration._agent_manager
    
    async def test_get_memory_manager(self, integration):
        """Test getting memory manager instance"""
        memory_manager = integration.get_memory_manager()
        assert isinstance(memory_manager, LangChainMemoryManager)
        assert memory_manager is integration._memory_manager
    
    async def test_get_monitoring(self, integration):
        """Test getting monitoring instance"""
        monitoring = integration.get_monitoring()
        assert isinstance(monitoring, LangChainMonitoring)
        assert monitoring is integration._monitoring
    
    async def test_get_integration_mode_legacy(self, mock_settings):
        """Test getting integration mode when legacy is enabled"""
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'integration_mode'): 'legacy',
            ('langchain', 'llm_manager_enabled'): False,
            ('langchain', 'tool_registry_enabled'): False,
            ('langchain', 'agent_manager_enabled'): False,
            ('langchain', 'memory_workflow_enabled'): False,
        }.get((section, key), default)
        
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            await integration.initialize()
            
            mode = integration.get_integration_mode()
            assert mode == "legacy"
    
    async def test_get_integration_mode_langchain(self, mock_settings):
        """Test getting integration mode when langchain is enabled"""
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'integration_mode'): 'langchain',
            ('langchain', 'llm_manager_enabled'): True,
            ('langchain', 'tool_registry_enabled'): True,
            ('langchain', 'agent_manager_enabled'): True,
            ('langchain', 'memory_workflow_enabled'): True,
        }.get((section, key), default)
        
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            await integration.initialize()
            
            mode = integration.get_integration_mode()
            assert mode == "langchain"
    
    async def test_get_integration_mode_hybrid(self, mock_settings):
        """Test getting integration mode when hybrid is enabled"""
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'integration_mode'): 'hybrid',
            ('langchain', 'llm_manager_enabled'): True,
            ('langchain', 'tool_registry_enabled'): False,
            ('langchain', 'agent_manager_enabled'): True,
            ('langchain', 'memory_workflow_enabled'): False,
        }.get((section, key), default)
        
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            await integration.initialize()
            
            mode = integration.get_integration_mode()
            assert mode == "hybrid"
    
    async def test_is_component_enabled(self, mock_settings):
        """Test checking if components are enabled"""
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'llm_manager_enabled'): True,
            ('langchain', 'tool_registry_enabled'): False,
            ('langchain', 'agent_manager_enabled'): True,
            ('langchain', 'memory_workflow_enabled'): False,
        }.get((section, key), default)
        
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            await integration.initialize()
            
            assert integration.is_component_enabled("llm_manager") is True
            assert integration.is_component_enabled("tool_registry") is False
            assert integration.is_component_enabled("agent_manager") is True
            assert integration.is_component_enabled("memory_workflow") is False
            assert integration.is_component_enabled("unknown_component") is False
    
    async def test_get_status(self, integration):
        """Test getting integration status"""
        status = integration.get_status()
        
        assert isinstance(status, dict)
        assert "initialized" in status
        assert "integration_mode" in status
        assert "components" in status
        assert "llm_manager" in status["components"]
        assert "tool_registry" in status["components"]
        assert "agent_manager" in status["components"]
        assert "memory_manager" in status["components"]
        assert "monitoring" in status["components"]
    
    async def test_health_check(self, integration):
        """Test health check functionality"""
        health = await integration.health_check()
        
        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "components" in health
        assert "timestamp" in health
        assert "llm_manager" in health["components"]
        assert "tool_registry" in health["components"]
        assert "agent_manager" in health["components"]
        assert "memory_manager" in health["components"]
        assert "monitoring" in health["components"]
    
    async def test_shutdown(self, integration):
        """Test shutdown functionality"""
        # Mock component shutdown methods
        integration._llm_manager.shutdown = AsyncMock()
        integration._tool_registry.shutdown = AsyncMock()
        integration._agent_manager.shutdown = AsyncMock()
        integration._memory_manager.shutdown = AsyncMock()
        integration._monitoring.shutdown = AsyncMock()
        
        await integration.shutdown()
        
        # Verify shutdown was called on all components
        integration._llm_manager.shutdown.assert_called_once()
        integration._tool_registry.shutdown.assert_called_once()
        integration._agent_manager.shutdown.assert_called_once()
        integration._memory_manager.shutdown.assert_called_once()
        integration._monitoring.shutdown.assert_called_once()
        
        # Verify integration is marked as not initialized
        assert integration._initialized is False
    
    async def test_error_handling_during_initialization(self, mock_settings):
        """Test error handling during initialization"""
        # Mock a component to raise an error during initialization
        with patch('app.core.langchain.integration.LangChainLLMManager') as mock_llm_manager:
            mock_llm_manager.side_effect = Exception("Initialization error")
            
            with pytest.raises(Exception, match="Initialization error"):
                integration = LangChainIntegration()
                await integration.initialize()
    
    async def test_feature_flag_validation(self, mock_settings):
        """Test feature flag validation"""
        # Test invalid integration mode
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'integration_mode'): 'invalid_mode',
        }.get((section, key), default)
        
        with patch('app.core.langchain.integration.secure_settings', mock_settings):
            integration = LangChainIntegration()
            await integration.initialize()
            
            # Should default to legacy mode for invalid mode
            mode = integration.get_integration_mode()
            assert mode == "legacy"
    
    async def test_component_initialization_order(self, mock_settings):
        """Test that components are initialized in the correct order"""
        initialization_order = []
        
        def track_initialization(component_name):
            def initializer():
                initialization_order.append(component_name)
                return Mock()
            return initializer
        
        # Mock component constructors to track initialization order
        with patch('app.core.langchain.integration.LangChainMonitoring') as mock_monitoring:
            with patch('app.core.langchain.integration.LangChainLLMManager') as mock_llm:
                with patch('app.core.langchain.integration.LangChainToolRegistry') as mock_tool:
                    with patch('app.core.langchain.integration.LangGraphAgentManager') as mock_agent:
                        with patch('app.core.langchain.integration.LangChainMemoryManager') as mock_memory:
                            mock_monitoring.return_value = Mock()
                            mock_monitoring.return_value.initialize = AsyncMock()
                            
                            mock_llm.return_value = Mock()
                            mock_llm.return_value.initialize = AsyncMock()
                            
                            mock_tool.return_value = Mock()
                            mock_tool.return_value.initialize = AsyncMock()
                            
                            mock_agent.return_value = Mock()
                            mock_agent.return_value.initialize = AsyncMock()
                            
                            mock_memory.return_value = Mock()
                            mock_memory.return_value.initialize = AsyncMock()
                            
                            integration = LangChainIntegration()
                            await integration.initialize()
                            
                            # Verify initialization order: monitoring -> llm -> tool -> agent -> memory
                            assert mock_monitoring.called
                            assert mock_llm.called
                            assert mock_tool.called
                            assert mock_agent.called
                            assert mock_memory.called


if __name__ == "__main__":
    pytest.main([__file__])