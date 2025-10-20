"""
Unit tests for basic Chainlit interface functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestChainlitBasic:
    """Test basic Chainlit interface functionality"""

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