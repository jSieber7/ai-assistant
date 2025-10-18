"""
Unit tests for the Gradio settings interface integration.

Tests the integration of the settings page with the main Gradio app.
"""

import unittest
from unittest.mock import patch, MagicMock


from app.ui.gradio_app import create_gradio_app


class TestGradioSettingsIntegration(unittest.TestCase):
    """Test cases for Gradio settings integration."""

    @patch("app.ui.gradio_app.initialize_gradio_components")
    @patch("app.ui.gradio_app.create_settings_page")
    def test_settings_page_in_main_app(self, mock_create_settings, mock_init):
        """Test that the settings page is properly integrated into the main app."""
        # Mock the settings page creation
        mock_settings_app = MagicMock()
        mock_create_settings.return_value = mock_settings_app

        # Mock initialization
        mock_init.return_value = (True, "All components initialized")

        # Create the main Gradio app
        with patch("app.ui.gradio_app.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            # Mock all the other components
            with (
                patch("app.ui.gradio_app.gr.Markdown"),
                patch("app.ui.gradio_app.gr.Tabs"),
                patch("app.ui.gradio_app.gr.TabItem"),
                patch("app.ui.gradio_app.gr.Textbox"),
                patch("app.ui.gradio_app.gr.Button"),
                patch("app.ui.gradio_app.gr.Row"),
                patch("app.ui.gradio_app.gr.Column"),
                patch("app.ui.gradio_app.gr.Slider"),
                patch("app.ui.gradio_app.gr.Number"),
                patch("app.ui.gradio_app.gr.Dropdown"),
                patch("app.ui.gradio_app.gr.Checkbox"),
                patch("app.ui.gradio_app.gr.Code"),
            ):
                create_gradio_app()

                # Verify the settings page was created
                mock_create_settings.assert_called_once()

    @patch("app.ui.gradio_app.initialize_gradio_components")
    @patch("app.ui.gradio_app.create_settings_page")
    def test_settings_tab_order(self, mock_create_settings, mock_init):
        """Test that the settings tab appears in the correct position."""
        # Mock the settings page creation
        mock_settings_app = MagicMock()
        mock_create_settings.return_value = mock_settings_app

        # Mock initialization
        mock_init.return_value = (True, "All components initialized")

        # Track tab creation order
        tab_order = []

        def track_tab_creation(label, **kwargs):
            tab_order.append(label)
            mock_tab = MagicMock()
            return mock_tab

        with patch("app.ui.gradio_app.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            with patch("app.ui.gradio_app.gr.Tabs") as mock_tabs:
                mock_tabs_context = MagicMock()
                mock_tabs.return_value.__enter__.return_value = mock_tabs_context

                # Mock TabItem to track creation order
                with patch(
                    "app.ui.gradio_app.gr.TabItem", side_effect=track_tab_creation
                ):
                    # Mock all other components
                    with (
                        patch("app.ui.gradio_app.gr.Markdown"),
                        patch("app.ui.gradio_app.gr.Textbox"),
                        patch("app.ui.gradio_app.gr.Button"),
                        patch("app.ui.gradio_app.gr.Row"),
                        patch("app.ui.gradio_app.gr.Column"),
                        patch("app.ui.gradio_app.gr.Slider"),
                        patch("app.ui.gradio_app.gr.Number"),
                        patch("app.ui.gradio_app.gr.Dropdown"),
                        patch("app.ui.gradio_app.gr.Checkbox"),
                        patch("app.ui.gradio_app.gr.Code"),
                    ):
                        create_gradio_app()

                        # Verify settings tab is first
                        self.assertTrue(len(tab_order) > 0)
                        self.assertEqual(tab_order[0], "🔐 Settings")

    @patch("app.ui.gradio_app.initialize_gradio_components")
    @patch("app.ui.gradio_app.create_settings_page")
    def test_settings_page_error_handling(self, mock_create_settings, mock_init):
        """Test error handling when settings page creation fails."""
        # Mock settings page creation to raise an exception
        mock_create_settings.side_effect = Exception("Settings page creation failed")

        # Mock initialization
        mock_init.return_value = (True, "All components initialized")

        # The main app should still be created even if settings page fails
        with patch("app.ui.gradio_app.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            # Mock all the other components
            with (
                patch("app.ui.gradio_app.gr.Markdown"),
                patch("app.ui.gradio_app.gr.Tabs"),
                patch("app.ui.gradio_app.gr.TabItem"),
                patch("app.ui.gradio_app.gr.Textbox"),
                patch("app.ui.gradio_app.gr.Button"),
                patch("app.ui.gradio_app.gr.Row"),
                patch("app.ui.gradio_app.gr.Column"),
                patch("app.ui.gradio_app.gr.Slider"),
                patch("app.ui.gradio_app.gr.Number"),
                patch("app.ui.gradio_app.gr.Dropdown"),
                patch("app.ui.gradio_app.gr.Checkbox"),
                patch("app.ui.gradio_app.gr.Code"),
            ):
                # This should not raise an exception
                try:
                    create_gradio_app()
                    # If we get here, error handling worked
                    self.assertTrue(True)
                except Exception as e:
                    self.fail(
                        f"Main app creation failed when settings page failed: {e}"
                    )


class TestSettingsPageComponents(unittest.TestCase):
    """Test cases for individual settings page components."""

    @patch("app.ui.settings_page.secure_settings")
    def test_settings_page_initialization(self, mock_secure_settings):
        """Test that the settings page initializes with current settings."""
        # Mock current settings
        mock_secure_settings.get_all_settings.return_value = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "base_url": "https://openrouter.ai/api/v1",
                }
            },
            "system_config": {"tool_system_enabled": True},
        }

        # Create the settings page
        with patch("app.ui.settings_page.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            # Mock all Gradio components
            with (
                patch("app.ui.settings_page.gr.Markdown"),
                patch("app.ui.settings_page.gr.Tabs"),
                patch("app.ui.settings_page.gr.TabItem"),
                patch("app.ui.settings_page.gr.Checkbox"),
                patch("app.ui.settings_page.gr.Textbox"),
                patch("app.ui.settings_page.gr.Number"),
                patch("app.ui.settings_page.gr.Slider"),
                patch("app.ui.settings_page.gr.Dropdown"),
                patch("app.ui.settings_page.gr.Button"),
                patch("app.ui.settings_page.gr.Row"),
                patch("app.ui.settings_page.gr.Column"),
                patch("app.ui.settings_page.gr.Group"),
                patch("app.ui.settings_page.gr.Code"),
            ):
                from app.ui.settings_page import create_settings_page

                create_settings_page()

                # Verify current settings were retrieved
                mock_secure_settings.get_all_settings.assert_called_once()

    @patch("app.ui.settings_page.secure_settings")
    @patch("app.ui.settings_page.validate_api_key")
    def test_api_key_validation_component(self, mock_validate, mock_secure_settings):
        """Test the API key validation component."""
        # Mock validation response
        mock_validate.return_value = "✅ API key is valid"

        # Mock current settings
        mock_secure_settings.get_all_settings.return_value = {}

        # Create the settings page
        with patch("app.ui.settings_page.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            # Mock all Gradio components
            with (
                patch("app.ui.settings_page.gr.Markdown"),
                patch("app.ui.settings_page.gr.Tabs"),
                patch("app.ui.settings_page.gr.TabItem"),
                patch("app.ui.settings_page.gr.Checkbox"),
                patch("app.ui.settings_page.gr.Textbox"),
                patch("app.ui.settings_page.gr.Number"),
                patch("app.ui.settings_page.gr.Slider"),
                patch("app.ui.settings_page.gr.Dropdown"),
                patch("app.ui.settings_page.gr.Button") as mock_button,
                patch("app.ui.settings_page.gr.Row"),
                patch("app.ui.settings_page.gr.Column"),
                patch("app.ui.settings_page.gr.Group"),
                patch("app.ui.settings_page.gr.Code"),
            ):
                # Mock button click
                mock_button.return_value.click = MagicMock()

                from app.ui.settings_page import create_settings_page

                create_settings_page()

                # Verify validation function is connected to button
                self.assertTrue(mock_button.return_value.click.called)

    @patch("app.ui.settings_page.secure_settings")
    @patch("app.ui.settings_page.update_llm_provider_settings")
    def test_settings_update_component(self, mock_update, mock_secure_settings):
        """Test the settings update component."""
        # Mock update response
        mock_update.return_value = "✅ Settings updated successfully"

        # Mock current settings
        mock_secure_settings.get_all_settings.return_value = {}

        # Create the settings page
        with patch("app.ui.settings_page.gr.Blocks") as mock_blocks:
            mock_context = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_context

            # Mock all Gradio components
            with (
                patch("app.ui.settings_page.gr.Markdown"),
                patch("app.ui.settings_page.gr.Tabs"),
                patch("app.ui.settings_page.gr.TabItem"),
                patch("app.ui.settings_page.gr.Checkbox"),
                patch("app.ui.settings_page.gr.Textbox"),
                patch("app.ui.settings_page.gr.Number"),
                patch("app.ui.settings_page.gr.Slider"),
                patch("app.ui.settings_page.gr.Dropdown"),
                patch("app.ui.settings_page.gr.Button") as mock_button,
                patch("app.ui.settings_page.gr.Row"),
                patch("app.ui.settings_page.gr.Column"),
                patch("app.ui.settings_page.gr.Group"),
                patch("app.ui.settings_page.gr.Code"),
            ):
                # Mock button click
                mock_button.return_value.click = MagicMock()

                from app.ui.settings_page import create_settings_page

                create_settings_page()

                # Verify update function is connected to button
                self.assertTrue(mock_button.return_value.click.called)


if __name__ == "__main__":
    unittest.main()
