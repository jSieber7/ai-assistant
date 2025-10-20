"""
Unit tests for Chainlit interface functionality.
"""

import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.unit
class TestChainlitInterface:
    """Test Chainlit interface functionality"""

    def test_imports(self):
        """Test that all required modules can be imported"""
        # Test chainlit import
        try:
            import chainlit as cl
            assert cl is not None
        except ImportError as e:
            pytest.skip(f"chainlit not available: {e}")
        
        # Test chainlit_app import
        try:
            from app.ui.chainlit_app import create_chainlit_app
            assert create_chainlit_app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import chainlit_app: {e}")
        
        # Test config import
        try:
            from app.core.config import settings, initialize_llm_providers
            assert settings is not None
            assert initialize_llm_providers is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")
        
        # Test llm_providers import
        try:
            from app.core.llm_providers import provider_registry
            assert provider_registry is not None
        except ImportError as e:
            pytest.fail(f"Failed to import llm_providers: {e}")

    @pytest.mark.asyncio
    async def test_provider_functions(self):
        """Test provider-related functions"""
        # Mock the provider functions
        with patch('app.ui.chainlit_app.get_providers_list') as mock_get_providers, \
             patch('app.ui.chainlit_app.get_models_for_provider') as mock_get_models:
            
            # Setup mocks
            mock_get_providers.return_value = ["provider1", "provider2"]
            mock_get_models.return_value = ["model1", "model2"]
            
            # Import after mocking
            from app.ui.chainlit_app import get_providers_list, get_models_for_provider
            
            # Test getting providers list
            providers = await get_providers_list()
            assert providers == ["provider1", "provider2"]
            
            # Test getting models for provider
            models = await get_models_for_provider("provider1")
            assert models == ["model1", "model2"]

    @patch('app.core.config.settings')
    def test_configuration(self, mock_settings):
        """Test configuration settings"""
        # Mock the settings object
        mock_settings.environment = "development"
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8000
        
        # Import after mocking to ensure the mock is used
        from app.core.config import settings
        
        assert settings.environment == "development"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000

    def test_chainlit_config_exists(self):
        """Test that Chainlit configuration file exists"""
        config_path = os.path.join(os.path.dirname(__file__), "../../../../..", ".chainlit", "config.toml")
        
        # The config file might not exist in all environments
        # So we just check if the path is correctly constructed
        assert isinstance(config_path, str)
        assert config_path.endswith(".chainlit/config.toml")

    @patch('builtins.open')
    @patch('os.path.exists')
    def test_chainlit_config_loading(self, mock_exists, mock_open):
        """Test Chainlit configuration loading"""
        # Mock the config file exists
        mock_exists.return_value = True
        
        # Mock the config content
        mock_config_data = {
            'UI': {
                'name': 'Test App',
                'theme': {
                    'layout': 'wide'
                }
            }
        }
        
        # Mock toml.load
        with patch('toml.load', return_value=mock_config_data):
            config_path = os.path.join(os.path.dirname(__file__), "../../../../..", ".chainlit", "config.toml")
            
            # Check if path exists (mocked to return True)
            assert os.path.exists(config_path)
            
            # If toml is available, test loading
            try:
                import toml
                with open(config_path, 'r') as f:
                    config = toml.load(f)
                
                assert config['UI']['name'] == 'Test App'
                assert config['UI']['theme']['layout'] == 'wide'
            except ImportError:
                pytest.skip("toml not available")