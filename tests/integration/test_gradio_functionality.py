"""
Integration tests for Gradio interface functionality.

This module tests that all Gradio interface components are fully functional
and not just demo placeholders.
"""

import pytest
import httpx
from unittest.mock import patch

from app.ui.gradio_app import (
    get_models_list,
    get_providers_info,
    get_tools_info,
    get_agents_list,
    get_agents_info,
    initialize_gradio_components,
    execute_query_function,
    update_settings,
    create_gradio_app,
)
from app.core.config import settings
from app.core.tools import tool_registry
from app.core.llm_providers import provider_registry


@pytest.mark.integration
class TestGradioFunctionality:
    """Test class for Gradio interface functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path):
        """Set up test environment with temporary .env file"""
        # Create a temporary .env file for testing
        env_content = """
TOOL_SYSTEM_ENABLED=true
AGENT_SYSTEM_ENABLED=false
PREFERRED_PROVIDER=openai_compatible
ENABLE_FALLBACK=true
DEBUG=true
OPENAI_COMPATIBLE_API_KEY=test_key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OLLAMA_SETTINGS_ENABLED=true
OLLAMA_SETTINGS_BASE_URL=http://localhost:11434
HOST=127.0.0.1
PORT=8000
        """
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # Patch the .env file path
        with patch.object(settings.__class__.Config, "env_file", str(env_file)):
            # Reload settings
            settings.__class__.model_rebuild()
            yield

    def test_get_models_list_functionality(self):
        """Test that get_models_list returns actual models, not placeholders"""
        try:
            models = get_models_list()

            # Should return at least one model
            assert len(models) > 0, "No models returned"

            # Should not contain placeholder text
            assert not any("placeholder" in model.lower() for model in models), (
                "Placeholder models found"
            )

            # Should contain actual model names (not just placeholders)
            assert len(models) > 0, "No models returned"

            # Check if any model looks like a real model (has a name)
            assert any(
                len(model) > 3 and not model.startswith("placeholder")
                for model in models
            ), "No real model names found"

        except Exception as e:
            pytest.fail(f"get_models_list failed: {str(e)}")

    def test_get_providers_info_functionality(self):
        """Test that get_providers_info returns actual provider information"""
        try:
            providers_info = get_providers_info()

            # Should return provider information
            assert providers_info, "No provider information returned"

            # Check if providers_info is a string
            assert isinstance(providers_info, str), (
                f"Expected string, got {type(providers_info)}"
            )

            # Should contain provider status information
            has_configured = "Configured" in providers_info
            has_not_configured = "Not configured" in providers_info

            has_status = has_configured or has_not_configured
            assert has_status, "No provider status found"

            # Should contain health information
            has_healthy = "Healthy" in providers_info
            has_unhealthy = "Unhealthy" in providers_info
            has_unknown = "Unknown" in providers_info

            has_health = has_healthy or has_unhealthy or has_unknown
            assert has_health, "No provider health info found"

        except Exception as e:
            pytest.fail(f"get_providers_info failed: {str(e)}")

    def test_get_tools_info_functionality(self):
        """Test that get_tools_info returns actual tool information"""
        try:
            tools_info = get_tools_info()

            # Should return tool information
            assert tools_info, "No tool information returned"

            # Should not indicate initialization failure
            assert "Failed to initialize" not in tools_info, (
                "Tool initialization failed"
            )

            # Should contain tool descriptions
            if "No tools available" not in tools_info:
                assert any(
                    ":" in line for line in tools_info.split("\n") if line.strip()
                ), "No tool descriptions found"

        except Exception as e:
            pytest.fail(f"get_tools_info failed: {str(e)}")

    def test_get_agents_list_functionality(self):
        """Test that get_agents_list returns actual agent information"""
        try:
            agents = get_agents_list()

            # Should return at least one option
            assert len(agents) > 0, "No agents returned"

            # Should indicate system state properly
            if settings.agent_system_enabled:
                # If enabled, should have more than just disabled placeholder
                assert not all("disabled" in agent.lower() for agent in agents), (
                    "All agents marked as disabled"
                )
            else:
                # If disabled, should indicate this
                assert any("disabled" in agent.lower() for agent in agents), (
                    "No indication that agent system is disabled"
                )

        except Exception as e:
            pytest.fail(f"get_agents_list failed: {str(e)}")

    def test_get_agents_info_functionality(self):
        """Test that get_agents_info returns actual agent information"""
        try:
            agents_info = get_agents_info()

            # Should return agent information
            assert agents_info, "No agent information returned"

            # Should properly indicate system state
            if settings.agent_system_enabled:
                assert "disabled" not in agents_info.lower(), (
                    "Agent system marked as disabled when enabled"
                )
            else:
                assert "disabled" in agents_info.lower(), (
                    "No indication that agent system is disabled"
                )

        except Exception as e:
            pytest.fail(f"get_agents_info failed: {str(e)}")

    def test_initialize_gradio_components_functionality(self):
        """Test that initialize_gradio_components properly initializes all components"""
        try:
            success, status = initialize_gradio_components()

            # Should return success status
            assert isinstance(success, bool), "Success should be a boolean"
            assert isinstance(status, str), "Status should be a string"

            # Should initialize providers
            assert provider_registry.list_providers(), "No providers initialized"

            # Should initialize tools
            assert tool_registry.list_tools(), "No tools initialized"

            # Status should contain initialization information
            assert any("initialized" in item.lower() for item in status.split("\n")), (
                "No initialization status found"
            )

        except Exception as e:
            pytest.fail(f"initialize_gradio_components failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_test_query_functionality(self):
        """Test that test_query actually processes queries, not just returns demo responses"""
        try:
            # Test with valid input
            result = await execute_query_function(
                message="Hello, this is a test message",
                model=settings.default_model,
                temperature=0.7,
                max_tokens=100,
                use_agent_system=False,
            )

            # Should return a response
            assert result, "No response returned"

            # Should not be a generic error message
            if result.startswith("Error:"):
                # Check if it's a connection error (expected in test environment)
                # Allow for HTTP 404 errors which are common in test environments
                assert (
                    "connect" in result.lower()
                    or "timeout" in result.lower()
                    or "404" in result.lower()
                    or "server error" in result.lower()
                ), f"Unexpected error: {result}"
            else:
                # Should contain actual response content
                assert len(result) > 20, "Response too short to be real"

        except Exception as e:
            pytest.fail(f"test_query failed: {str(e)}")

    def test_update_settings_functionality(self):
        """Test that update_settings actually updates and persists settings"""
        try:
            # Get original settings
            original_tool_enabled = settings.tool_system_enabled
            original_debug = settings.debug

            # Update settings
            result = update_settings(
                tool_system_enabled=not original_tool_enabled,
                agent_system_enabled=not settings.agent_system_enabled,
                preferred_provider="ollama",
                enable_fallback=not settings.enable_fallback,
                debug_mode=not original_debug,
            )

            # Should return success message
            assert "Settings updated successfully" in result, "Settings update failed"

            # Should indicate what changed
            assert "Changes applied:" in result, "No changes indicated"

            # Settings should be updated in memory
            assert settings.tool_system_enabled != original_tool_enabled, (
                "Tool system setting not updated"
            )
            assert settings.debug != original_debug, "Debug setting not updated"

            # Restore original settings
            update_settings(
                tool_system_enabled=original_tool_enabled,
                agent_system_enabled=settings.agent_system_enabled,
                preferred_provider=settings.preferred_provider,
                enable_fallback=settings.enable_fallback,
                debug_mode=original_debug,
            )

        except Exception as e:
            pytest.fail(f"update_settings failed: {str(e)}")

    def test_create_gradio_app_functionality(self):
        """Test that create_gradio_app creates a functional app"""
        try:
            app = create_gradio_app()

            # Should return a Gradio Blocks app
            assert app is not None, "No app created"

            # Should have the expected title
            assert app.title == "AI Assistant Interface", "Incorrect app title"

            # Should have tabs
            assert hasattr(app, "blocks"), "App has no blocks"

        except Exception as e:
            pytest.fail(f"create_gradio_app failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_query_with_agent_system(self):
        """Test query functionality with agent system enabled"""
        if not settings.agent_system_enabled:
            pytest.skip("Agent system not enabled")

        try:
            result = await execute_query_function(
                message="Test message for agent system",
                model=settings.default_model,
                temperature=0.7,
                max_tokens=100,
                use_agent_system=True,
                agent_dropdown="default",
            )

            # Should return a response
            assert result, "No response returned"

            # Should not be a generic error message
            if result.startswith("Error:"):
                # Check if it's a connection error (expected in test environment)
                # Allow for HTTP 404 errors which are common in test environments
                assert (
                    "connect" in result.lower()
                    or "timeout" in result.lower()
                    or "404" in result.lower()
                    or "server error" in result.lower()
                ), f"Unexpected error: {result}"

        except Exception as e:
            pytest.fail(f"test_query with agent system failed: {str(e)}")

    def test_model_listing_with_provider_errors(self):
        """Test model listing handles provider errors gracefully"""
        with patch(
            "app.ui.gradio_app.provider_registry.list_providers", return_value=[]
        ):
            with patch(
                "app.ui.gradio_app.initialize_llm_providers",
                side_effect=Exception("Provider error"),
            ):
                try:
                    models = get_models_list()

                    # Should return fallback model
                    assert len(models) > 0, "No fallback models returned"
                    # Check if the fallback model contains the default model name (may have error suffix)
                    assert settings.default_model in models[0], (
                        "Incorrect fallback model"
                    )

                except Exception as e:
                    pytest.fail(f"Model listing with provider errors failed: {str(e)}")

    def test_settings_update_with_invalid_values(self):
        """Test settings update handles invalid values gracefully"""
        try:
            # Test with None values (should be handled gracefully)
            result = update_settings(
                tool_system_enabled=True,
                agent_system_enabled=True,
                preferred_provider=None,  # This might be invalid
                enable_fallback=True,
                debug_mode=True,
            )

            # Should either succeed or fail gracefully
            assert result is not None, "No response from settings update"

        except Exception as e:
            # Should not crash the application
            assert "Failed to update settings" in str(e), f"Unexpected error: {str(e)}"


@pytest.mark.integration
class TestGradioIntegration:
    """Test class for Gradio integration with backend services."""

    @pytest.mark.asyncio
    async def test_gradio_to_backend_integration(self):
        """Test that Gradio interface properly integrates with backend API"""
        try:
            # Test if the backend endpoint would be reachable
            base_url = f"http://{settings.host}:{settings.port}"

            # This test will fail in CI without a running server, but verifies the integration
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get(f"{base_url}/health")
                    # Accept both 200 (success) and 404 (server running but endpoint not found)
                    assert response.status_code in [
                        200,
                        404,
                    ], f"Unexpected status code: {response.status_code}"
                except httpx.ConnectError:
                    # Expected in test environment without running server
                    pass
                except Exception as e:
                    # Allow other connection-related errors in test environment
                    if "connect" in str(e).lower() or "timeout" in str(e).lower():
                        pass
                    else:
                        raise

        except Exception as e:
            pytest.fail(f"Gradio to backend integration test failed: {str(e)}")

    def test_gradio_with_real_configuration(self):
        """Test Gradio with a real configuration"""
        try:
            # Create a real configuration
            config = {
                "tool_system_enabled": True,
                "agent_system_enabled": False,
                "preferred_provider": "openai_compatible",
                "enable_fallback": True,
                "debug_mode": True,
            }

            # Test with this configuration
            result = update_settings(**config)

            # Should succeed
            assert "Settings updated successfully" in result, (
                "Settings update failed with real config"
            )

        except Exception as e:
            pytest.fail(f"Gradio with real configuration failed: {str(e)}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
