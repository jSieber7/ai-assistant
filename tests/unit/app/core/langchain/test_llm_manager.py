"""
Unit tests for LangChain LLM Manager.

This module tests LLM provider management, request handling,
and integration with LangChain LLM components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLLM

from app.core.langchain.llm_manager import LangChainLLMManager
from app.core.secure_settings import secure_settings


class TestLangChainLLMManager:
    """Test cases for LangChain LLM Manager"""
    
    @pytest.fixture
    async def llm_manager(self):
        """Create a LangChain LLM Manager instance for testing"""
        manager = LangChainLLMManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('llm_providers', 'openai', 'api_key'): 'test-openai-key',
            ('llm_providers', 'openai', 'model'): 'gpt-3.5-turbo',
            ('llm_providers', 'openai', 'temperature'): 0.7,
            ('llm_providers', 'openai', 'max_tokens'): 1000,
            ('llm_providers', 'ollama', 'base_url'): 'http://localhost:11434',
            ('llm_providers', 'ollama', 'model'): 'llama2',
            ('llm_providers', 'ollama', 'temperature'): 0.5,
            ('llm_providers', 'ollama', 'max_tokens'): 2000,
        }.get((section, key), default)
        return mock_settings
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of LLM manager"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            
            # Test initialization
            await manager.initialize()
            
            # Verify initialization
            assert manager._initialized is True
            assert manager._monitoring is not None
            assert len(manager._llm_instances) > 0
    
    async def test_register_openai_provider(self, mock_settings):
        """Test registering OpenAI provider"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            await manager.initialize()
            
            # Test OpenAI registration
            llm = await manager.get_llm("gpt-3.5-turbo")
            
            assert isinstance(llm, ChatOpenAI)
            assert llm.model_name == "gpt-3.5-turbo"
            assert llm.temperature == 0.7
            assert llm.max_tokens == 1000
    
    async def test_register_ollama_provider(self, mock_settings):
        """Test registering Ollama provider"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            await manager.initialize()
            
            # Test Ollama registration
            llm = await manager.get_llm("llama2")
            
            assert isinstance(llm, Ollama)
            assert llm.base_url == "http://localhost:11434"
            assert llm.model == "llama2"
    
    async def test_get_existing_llm(self, llm_manager):
        """Test getting an already registered LLM"""
        # Register an LLM first
        first_llm = await llm_manager.get_llm("test-model")
        
        # Get the same LLM again
        second_llm = await llm_manager.get_llm("test-model")
        
        # Should return the same instance
        assert first_llm is second_llm
    
    async def test_get_llm_with_config(self, mock_settings):
        """Test getting LLM with custom configuration"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            await manager.initialize()
            
            # Test with custom config
            config = {
                "temperature": 0.9,
                "max_tokens": 1500,
                "top_p": 0.95
            }
            
            llm = await manager.get_llm("gpt-3.5-turbo", **config)
            
            assert isinstance(llm, ChatOpenAI)
            assert llm.temperature == 0.9
            assert llm.max_tokens == 1500
            assert llm.top_p == 0.95
    
    async def test_get_llm_not_found(self, llm_manager):
        """Test getting an LLM that doesn't exist"""
        with pytest.raises(ValueError, match="No configuration found for LLM"):
            await llm_manager.get_llm("non-existent-model")
    
    async def test_list_available_llms(self, llm_manager):
        """Test listing available LLMs"""
        # Register some LLMs
        await llm_manager.get_llm("test-model-1")
        await llm_manager.get_llm("test-model-2")
        
        # List available LLMs
        available_llms = llm_manager.list_available_llms()
        
        assert isinstance(available_llms, list)
        assert "test-model-1" in available_llms
        assert "test-model-2" in available_llms
    
    async def test_get_llm_info(self, llm_manager):
        """Test getting information about an LLM"""
        # Register an LLM
        await llm_manager.get_llm("test-model")
        
        # Get LLM info
        info = llm_manager.get_llm_info("test-model")
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "provider" in info
        assert "config" in info
        assert "registered_at" in info
    
    async def test_get_llm_info_not_found(self, llm_manager):
        """Test getting info for an LLM that doesn't exist"""
        info = llm_manager.get_llm_info("non-existent-model")
        assert info is None
    
    async def test_unregister_llm(self, llm_manager):
        """Test unregistering an LLM"""
        # Register an LLM
        await llm_manager.get_llm("test-model")
        
        # Verify it's registered
        assert "test-model" in llm_manager.list_available_llms()
        
        # Unregister it
        result = await llm_manager.unregister_llm("test-model")
        
        assert result is True
        assert "test-model" not in llm_manager.list_available_llms()
    
    async def test_unregister_llm_not_found(self, llm_manager):
        """Test unregistering an LLM that doesn't exist"""
        result = await llm_manager.unregister_llm("non-existent-model")
        assert result is False
    
    async def test_health_check(self, llm_manager):
        """Test health check functionality"""
        # Register some LLMs
        await llm_manager.get_llm("test-model-1")
        await llm_manager.get_llm("test-model-2")
        
        # Perform health check
        health = await llm_manager.health_check()
        
        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "llms" in health
        assert "timestamp" in health
        assert "test-model-1" in health["llms"]
        assert "test-model-2" in health["llms"]
    
    async def test_shutdown(self, llm_manager):
        """Test shutdown functionality"""
        # Mock monitoring shutdown
        llm_manager._monitoring.shutdown = AsyncMock()
        
        await llm_manager.shutdown()
        
        # Verify shutdown was called
        llm_manager._monitoring.shutdown.assert_called_once()
        
        # Verify manager is marked as not initialized
        assert llm_manager._initialized is False
        
        # Verify LLM instances are cleared
        assert len(llm_manager._llm_instances) == 0
    
    async def test_error_handling_during_registration(self, mock_settings):
        """Test error handling during LLM registration"""
        # Mock settings to cause an error
        mock_settings.get_setting.side_effect = Exception("Configuration error")
        
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            
            with pytest.raises(Exception, match="Configuration error"):
                await manager.get_llm("test-model")
    
    async def test_monitoring_integration(self, mock_settings):
        """Test that monitoring is properly integrated"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            await manager.initialize()
            
            # Mock monitoring to track calls
            manager._monitoring.record_metric = AsyncMock()
            
            # Make an LLM request
            await manager.get_llm("gpt-3.5-turbo")
            
            # Verify monitoring was called
            manager._monitoring.record_metric.assert_called()
    
    async def test_concurrent_requests(self, llm_manager):
        """Test handling concurrent LLM requests"""
        # Make concurrent requests for the same LLM
        tasks = [
            llm_manager.get_llm("test-model"),
            llm_manager.get_llm("test-model"),
            llm_manager.get_llm("test-model")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should return the same instance
        assert all(result is results[0] for result in results)
    
    async def test_llm_configuration_validation(self, mock_settings):
        """Test LLM configuration validation"""
        # Test invalid temperature
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('llm_providers', 'openai', 'api_key'): 'test-openai-key',
            ('llm_providers', 'openai', 'model'): 'gpt-3.5-turbo',
            ('llm_providers', 'openai', 'temperature'): 2.0,  # Invalid: > 1.0
        }.get((section, key), default)
        
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            
            # Should handle invalid configuration gracefully
            with pytest.raises(ValueError, match="Invalid temperature"):
                await manager.get_llm("gpt-3.5-turbo")
    
    async def test_provider_detection(self, mock_settings):
        """Test automatic provider detection based on model name"""
        with patch('app.core.langchain.llm_manager.secure_settings', mock_settings):
            manager = LangChainLLMManager()
            await manager.initialize()
            
            # Test different model patterns
            openai_llm = await manager.get_llm("gpt-4")
            assert isinstance(openai_llm, ChatOpenAI)
            
            ollama_llm = await manager.get_llm("llama2")
            assert isinstance(ollama_llm, Ollama)
    
    async def test_custom_provider_registration(self, llm_manager):
        """Test registering a custom LLM provider"""
        # Create a custom LLM
        custom_llm = Mock(spec=BaseLLM)
        
        # Register it
        await llm_manager.register_custom_llm("custom-model", custom_llm)
        
        # Verify registration
        assert "custom-model" in llm_manager.list_available_llms()
        
        # Get the custom LLM
        retrieved_llm = await llm_manager.get_llm("custom-model")
        assert retrieved_llm is custom_llm
    
    async def test_llm_caching(self, llm_manager):
        """Test that LLM instances are properly cached"""
        # Get the same LLM multiple times
        llm1 = await llm_manager.get_llm("test-model")
        llm2 = await llm_manager.get_llm("test-model")
        llm3 = await llm_manager.get_llm("test-model")
        
        # All should be the same instance
        assert llm1 is llm2
        assert llm2 is llm3
        
        # Verify only one instance in cache
        assert len(llm_manager._llm_instances) == 1
    
    async def test_get_statistics(self, llm_manager):
        """Test getting LLM manager statistics"""
        # Register some LLMs
        await llm_manager.get_llm("test-model-1")
        await llm_manager.get_llm("test-model-2")
        
        # Get statistics
        stats = llm_manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_llms_registered" in stats
        assert "llms_by_provider" in stats
        assert "cache_size" in stats
        assert "initialized" in stats
        assert stats["total_llms_registered"] == 2


if __name__ == "__main__":
    pytest.main([__file__])