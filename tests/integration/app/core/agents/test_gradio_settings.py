"""
Integration tests for the Gradio settings interface.

Tests the complete workflow of configuring settings through the Gradio UI.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


from app.core.secure_settings import secure_settings
from app.ui.settings_page import (
    get_current_settings,
    validate_api_key,
    update_llm_provider_settings,
    update_system_settings,
    update_multi_writer_settings,
    export_settings,
    import_settings,
)


class TestGradioSettingsInterface(unittest.TestCase):
    """Test cases for the Gradio settings interface functions."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Patch the secure_settings module to use our test directory
        self.mock_manager = MagicMock()
        self.original_settings = secure_settings

        # Create a mock secure settings manager
        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            # Configure mock return values
            self.mock_manager.get_all_settings.return_value = {
                "llm_providers": {
                    "openai_compatible": {
                        "enabled": True,
                        "api_key": "",
                        "base_url": "https://openrouter.ai/api/v1",
                        "default_model": "anthropic/claude-3.5-sonnet",
                    },
                    "ollama": {
                        "enabled": True,
                        "base_url": "http://localhost:11434",
                        "default_model": "llama2",
                    },
                },
                "external_services": {
                    "firecrawl": {
                        "enabled": False,
                        "docker_url": "http://firecrawl-api:3002",
                    },
                    "jina_reranker": {
                        "enabled": False,
                        "url": "http://jina-reranker:8080",
                    },
                },
                "system_config": {
                    "tool_system_enabled": True,
                    "agent_system_enabled": True,
                    "preferred_provider": "openai_compatible",
                },
            }

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_current_settings(self):
        """Test getting current settings for the UI."""
        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            settings = get_current_settings()

            # Verify settings were retrieved
            self.mock_manager.get_all_settings.assert_called_once()
            self.assertIn("llm_providers", settings)
            self.assertIn("external_services", settings)
            self.assertIn("system_config", settings)

    def test_validate_api_key_function(self):
        """Test the API key validation function."""
        # Mock the validation method
        self.mock_manager.validate_api_key.return_value = True

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = validate_api_key("openai_compatible", "test_key")

            # Verify validation was called
            self.mock_manager.validate_api_key.assert_called_once_with(
                "openai_compatible", "test_key"
            )
            self.assertEqual(result, "✅ API key is valid")

    def test_validate_api_key_empty(self):
        """Test API key validation with empty key."""
        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = validate_api_key("openai_compatible", "")
            self.assertEqual(result, "⚠️ Please enter an API key")

    def test_update_llm_provider_settings(self):
        """Test updating LLM provider settings."""
        # Mock the set_category method
        self.mock_manager.set_category.return_value = None

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            with patch("app.ui.settings_page.initialize_llm_providers") as mock_init:
                result = update_llm_provider_settings(
                    openai_enabled=True,
                    openai_api_key="test_key",
                    openai_base_url="https://test.api.com",
                    openai_default_model="test-model",
                    openai_provider_name="TestProvider",
                    openai_timeout=30,
                    openai_max_retries=3,
                    ollama_enabled=True,
                    ollama_base_url="http://localhost:11434",
                    ollama_default_model="llama2",
                    ollama_timeout=30,
                    ollama_max_retries=3,
                    ollama_temperature=0.7,
                    ollama_max_tokens=1000,
                    ollama_streaming=True,
                )

                # Verify settings were saved
                self.mock_manager.set_category.assert_called_once()
                call_args = self.mock_manager.set_category.call_args[0]

                # Check the category and settings
                self.assertEqual(call_args[0], "llm_providers")

                settings = call_args[1]
                self.assertEqual(settings["openai_compatible"]["enabled"], True)
                self.assertEqual(settings["openai_compatible"]["api_key"], "test_key")
                self.assertEqual(settings["ollama"]["enabled"], True)
                self.assertEqual(settings["ollama"]["temperature"], 0.7)

                # Verify providers were reinitialized
                mock_init.assert_called_once()

                # Check result message
                self.assertIn("updated successfully", result)


    def test_update_system_settings(self):
        """Test updating system settings."""
        # Mock the set_category method
        self.mock_manager.set_category.return_value = None

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = update_system_settings(
                tool_system_enabled=False,
                agent_system_enabled=False,
                preferred_provider="ollama",
                enable_fallback=False,
                debug_mode=False,
                host="0.0.0.0",
                port=8001,
                environment="production",
                secret_key="new_secret_key",
            )

            # Verify settings were saved
            self.mock_manager.set_category.assert_called_once()
            call_args = self.mock_manager.set_category.call_args[0]

            # Check the category and settings
            self.assertEqual(call_args[0], "system_config")

            settings = call_args[1]
            self.assertEqual(settings["tool_system_enabled"], False)
            self.assertEqual(settings["agent_system_enabled"], False)
            self.assertEqual(settings["preferred_provider"], "ollama")
            self.assertEqual(settings["debug"], False)
            self.assertEqual(settings["host"], "0.0.0.0")
            self.assertEqual(settings["port"], 8001)
            self.assertEqual(settings["environment"], "production")
            self.assertEqual(settings["secret_key"], "new_secret_key")

            # Check result message
            self.assertIn("updated successfully", result)

    def test_update_multi_writer_settings(self):
        """Test updating multi-writer settings."""
        # Mock the set_category method
        self.mock_manager.set_category.return_value = None

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = update_multi_writer_settings(
                multi_writer_enabled=True,
                mongodb_connection_string="mongodb://test:27017",
                mongodb_database_name="test_db",
            )

            # Verify settings were saved
            self.mock_manager.set_category.assert_called_once()
            call_args = self.mock_manager.set_category.call_args[0]

            # Check the category and settings
            self.assertEqual(call_args[0], "multi_writer")

            settings = call_args[1]
            self.assertEqual(settings["enabled"], True)
            self.assertEqual(
                settings["mongodb_connection_string"], "mongodb://test:27017"
            )
            self.assertEqual(settings["mongodb_database_name"], "test_db")

            # Check result message
            self.assertIn("updated successfully", result)

    def test_export_settings(self):
        """Test exporting settings."""
        # Mock export method
        test_settings = {
            "llm_providers": {
                "openai_compatible": {"enabled": True, "api_key": "masked_key"}
            }
        }
        self.mock_manager.export_settings.return_value = json.dumps(test_settings)

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = export_settings(include_secrets=False)

            # Verify export was called
            self.mock_manager.export_settings.assert_called_once_with(
                include_secrets=False
            )

            # Check result
            exported_data = json.loads(result)
            self.assertIn("llm_providers", exported_data)
            self.assertEqual(
                exported_data["llm_providers"]["openai_compatible"]["enabled"], True
            )

    def test_import_settings(self):
        """Test importing settings."""
        # Mock import method
        self.mock_manager.import_settings.return_value = None

        test_settings = {"llm_providers": {"openai_compatible": {"enabled": False}}}
        settings_json = json.dumps(test_settings)

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = import_settings(settings_json)

            # Verify import was called
            self.mock_manager.import_settings.assert_called_once_with(
                settings_json, merge=True
            )

            # Check result
            self.assertIn("imported successfully", result)

    def test_import_settings_empty(self):
        """Test importing settings with empty JSON."""
        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = import_settings("")
            self.assertIn("Please provide valid JSON", result)

    def test_update_settings_with_validation_error(self):
        """Test updating settings when validation fails."""
        # Mock the set_category method to raise an exception
        self.mock_manager.set_category.side_effect = Exception("Test error")

        with patch("app.ui.settings_page.secure_settings", self.mock_manager):
            result = update_llm_provider_settings(
                openai_enabled=True,
                openai_api_key="test_key",
                openai_base_url="https://test.api.com",
                openai_default_model="test-model",
                openai_provider_name="TestProvider",
                openai_timeout=30,
                openai_max_retries=3,
                ollama_enabled=True,
                ollama_base_url="http://localhost:11434",
                ollama_default_model="llama2",
                ollama_timeout=30,
                ollama_max_retries=3,
                ollama_temperature=0.7,
                ollama_max_tokens=1000,
                ollama_streaming=True,
            )

            # Check error message
            self.assertIn("Failed to update", result)

    @patch("app.ui.settings_page.gr.Blocks")
    def test_create_settings_page(self, mock_blocks):
        """Test creating the settings page."""
        # Mock the Gradio Blocks context manager
        mock_context = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_context
        mock_context.tab = MagicMock()
        mock_context.tab.return_value.__enter__.return_value = mock_context
        mock_context.row = MagicMock()
        mock_context.row.return_value.__enter__.return_value = mock_context
        mock_context.column = MagicMock()
        mock_context.column.return_value.__enter__.return_value = mock_context
        mock_context.group = MagicMock()
        mock_context.group.return_value.__enter__.return_value = mock_context

        # Mock all Gradio components
        with (
            patch("app.ui.settings_page.gr.Markdown"),
            patch("app.ui.settings_page.gr.Checkbox"),
            patch("app.ui.settings_page.gr.Textbox"),
            patch("app.ui.settings_page.gr.Number"),
            patch("app.ui.settings_page.gr.Slider"),
            patch("app.ui.settings_page.gr.Dropdown"),
            patch("app.ui.settings_page.gr.Button"),
            patch("app.ui.settings_page.gr.Code"),
            patch("app.ui.settings_page.gr.Tabs"),
        ):
            with patch("app.ui.settings_page.get_current_settings", return_value={}):
                from app.ui.settings_page import create_settings_page

                app = create_settings_page()

                # Verify the app was created
                self.assertIsNotNone(app)


class TestGradioSettingsWorkflow(unittest.TestCase):
    """Test the complete workflow of configuring settings through Gradio."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_settings_workflow(self):
        """Test the complete workflow of configuring and using settings."""
        # This is an integration test that simulates the complete user workflow

        # 1. Create a secure settings manager
        from app.core.secure_settings import SecureSettingsManager

        settings_manager = SecureSettingsManager(settings_dir=self.temp_dir)

        # 2. Configure OpenAI-compatible provider
        settings_manager.set_category(
            "llm_providers",
            {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test_openai_key",
                    "base_url": "https://test.openai.com",
                    "default_model": "test-model",
                },
                "ollama": {"enabled": False},
            },
        )

        # 3. Configure external services

        # 4. Configure system settings
        settings_manager.set_category(
            "system_config",
            {
                "tool_system_enabled": False,
                "agent_system_enabled": True,
                "preferred_provider": "openai_compatible",
            },
        )

        # 5. Verify settings were saved
        self.assertEqual(
            settings_manager.get_setting("llm_providers", "openai_compatible", {}).get(
                "api_key"
            ),
            "test_openai_key",
        )
        self.assertFalse(
            settings_manager.get_setting("system_config", "tool_system_enabled")
        )

        # 6. Test export functionality
        exported = settings_manager.export_settings(include_secrets=True)
        exported_data = json.loads(exported)

        self.assertEqual(
            exported_data["llm_providers"]["openai_compatible"]["api_key"],
            "test_openai_key",
        )

        # 7. Test import functionality
        new_settings = {
            "llm_providers": {"openai_compatible": {"api_key": "updated_key"}}
        }
        settings_manager.import_settings(json.dumps(new_settings), merge=True)

        # 8. Verify imported settings
        self.assertEqual(
            settings_manager.get_setting("llm_providers", "openai_compatible", {}).get(
                "api_key"
            ),
            "updated_key",
        )


    @patch("app.core.secure_settings.httpx.Client")
    def test_api_key_validation_workflow(self, mock_client):
        """Test the API key validation workflow."""
        # Create a secure settings manager
        from app.core.secure_settings import SecureSettingsManager

        settings_manager = SecureSettingsManager(settings_dir=self.temp_dir)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # 1. Validate API key before saving
        is_valid = settings_manager.validate_api_key("openai_compatible", "test_key")
        self.assertTrue(is_valid)

        # 2. Save the API key
        settings_manager.set_setting(
            "llm_providers", "openai_compatible", "api_key", "test_key"
        )

        # 3. Verify the key was saved
        saved_key = settings_manager.get_setting(
            "llm_providers", "openai_compatible", "api_key"
        )
        self.assertEqual(saved_key, "test_key")

        # 4. Verify the key is encrypted
        settings_file = Path(self.temp_dir) / "secure_settings.enc"
        with open(settings_file, "rb") as f:
            encrypted_content = f.read()
        self.assertNotIn(b"test_key", encrypted_content)


if __name__ == "__main__":
    unittest.main()
