"""
Unit tests for the Gradio interface integration.

This module tests if the Gradio interface is properly integrated with the FastAPI application.
"""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestGradioInterface:
    """Test class for Gradio interface functionality."""

    def test_gradio_app_import(self):
        """Test if Gradio app can be imported"""
        from app.ui import create_gradio_app

        assert create_gradio_app is not None, (
            "Failed to import create_gradio_app function"
        )

    def test_gradio_mount_function_import(self):
        """Test if mount_gradio_app function can be imported"""
        from app.ui import mount_gradio_app

        assert mount_gradio_app is not None, (
            "Failed to import mount_gradio_app function"
        )

    def test_gradio_dependency_installed(self):
        """Test if Gradio is installed"""
        try:
            import gradio as gr

            assert hasattr(gr, "Blocks"), "Gradio Blocks class not found"
            assert hasattr(gr, "mount_gradio_app"), (
                "Gradio mount_gradio_app function not found"
            )
        except ImportError:
            pytest.fail("Gradio is not installed")

    @patch("app.ui.gradio_app.get_available_models")
    @patch("app.ui.gradio_app.provider_registry")
    @patch("app.ui.gradio_app.tool_registry")
    @patch("app.ui.gradio_app.settings")
    def test_gradio_app_creation(
        self, mock_settings, mock_tool_registry, mock_provider_registry, mock_get_models
    ):
        """Test if Gradio app can be created without errors"""
        # Mock the dependencies
        mock_settings.tool_system_enabled = True
        mock_settings.agent_system_enabled = True
        mock_settings.debug = True
        mock_settings.preferred_provider = "openai_compatible"
        mock_settings.enable_fallback = True
        mock_settings.default_model = "test-model"
        mock_settings.host = "localhost"
        mock_settings.port = 8000

        # Mock models
        from unittest.mock import Mock

        mock_model = Mock()
        mock_model.provider.value = "test_provider"
        mock_model.name = "test_model"
        mock_get_models.return_value = [mock_model]

        # Mock providers
        mock_provider = Mock()
        mock_provider.is_configured = True
        mock_provider.is_healthy.return_value = True
        mock_provider.name = "Test Provider"
        mock_provider.provider_type.value = "test_provider"
        mock_provider_registry.list_providers.return_value = [mock_provider]
        mock_provider_registry._default_provider = "test_provider"

        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.enabled = True
        mock_tool_registry.list_tools.return_value = [mock_tool]

        # Test app creation
        from app.ui import create_gradio_app

        try:
            app = create_gradio_app()
            assert app is not None, "Gradio app creation failed"
        except Exception as e:
            pytest.fail(f"Gradio app creation failed with error: {str(e)}")

    @patch("app.ui.gradio_app.get_available_models")
    @patch("app.ui.gradio_app.provider_registry")
    @patch("app.ui.gradio_app.tool_registry")
    @patch("app.ui.gradio_app.settings")
    def test_gradio_mount_function(
        self, mock_settings, mock_tool_registry, mock_provider_registry, mock_get_models
    ):
        """Test if mount_gradio_app function works correctly"""
        # Mock the dependencies
        mock_settings.tool_system_enabled = True
        mock_settings.agent_system_enabled = True
        mock_settings.debug = True
        mock_settings.preferred_provider = "openai_compatible"
        mock_settings.enable_fallback = True
        mock_settings.default_model = "test-model"
        mock_settings.host = "localhost"
        mock_settings.port = 8000

        # Mock models
        from unittest.mock import Mock

        mock_model = Mock()
        mock_model.provider.value = "test_provider"
        mock_model.name = "test_model"
        mock_get_models.return_value = [mock_model]

        # Mock providers
        mock_provider = Mock()
        mock_provider.is_configured = True
        mock_provider.is_healthy.return_value = True
        mock_provider.name = "Test Provider"
        mock_provider.provider_type.value = "test_provider"
        mock_provider_registry.list_providers.return_value = [mock_provider]
        mock_provider_registry._default_provider = "test_provider"

        # Mock tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.enabled = True
        mock_tool_registry.list_tools.return_value = [mock_tool]

        # Create mock FastAPI app
        mock_fastapi_app = Mock()

        # Test mounting
        from app.ui import create_gradio_app, mount_gradio_app

        try:
            gradio_app = create_gradio_app()
            mounted_app = mount_gradio_app(mock_fastapi_app, gradio_app, path="/gradio")
            assert mounted_app is not None, "Gradio app mounting failed"
        except Exception as e:
            pytest.fail(f"Gradio app mounting failed with error: {str(e)}")

    def test_gradio_functions_exist(self):
        """Test if all required Gradio functions exist"""
        from app.ui.gradio_app import (
            get_models_list,
            get_providers_info,
            get_tools_info,
            test_query,
            update_settings,
        )

        assert callable(get_models_list), "get_models_list is not callable"
        assert callable(get_providers_info), "get_providers_info is not callable"
        assert callable(get_tools_info), "get_tools_info is not callable"
        assert callable(test_query), "test_query is not callable"
        assert callable(update_settings), "update_settings is not callable"
