"""
Unit tests for the secure settings system.

Tests the encryption, storage, and retrieval of sensitive configuration data.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


from app.core.secure_settings import SecureSettingsManager


class TestSecureSettingsManager(unittest.TestCase):
    """Test cases for the SecureSettingsManager class."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.settings_manager = SecureSettingsManager(settings_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_creates_directory(self):
        """Test that initialization creates the settings directory."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(Path(self.temp_dir).is_dir())

    def test_default_settings_structure(self):
        """Test that default settings have the expected structure."""
        defaults = self.settings_manager._get_default_settings()

        # Check main categories exist
        self.assertIn("llm_providers", defaults)
        self.assertIn("external_services", defaults)
        self.assertIn("system_config", defaults)
        self.assertIn("multi_writer", defaults)

        # Check LLM providers
        self.assertIn("openai_compatible", defaults["llm_providers"])
        self.assertIn("ollama", defaults["llm_providers"])

        # Check external services
        self.assertIn("firecrawl", defaults["external_services"])
        self.assertIn("jina_reranker", defaults["external_services"])
        self.assertIn("searxng", defaults["external_services"])

    def test_setting_and_getting_values(self):
        """Test setting and getting individual settings."""
        # Test setting a new value
        self.settings_manager.set_setting("test_category", "test_key", "test_value")

        # Test getting the value
        result = self.settings_manager.get_setting("test_category", "test_key")
        self.assertEqual(result, "test_value")

        # Test getting with default
        result = self.settings_manager.get_setting(
            "test_category", "nonexistent", "default"
        )
        self.assertEqual(result, "default")

    def test_setting_and_getting_categories(self):
        """Test setting and getting entire categories."""
        test_category = {
            "key1": "value1",
            "key2": "value2",
            "nested": {"subkey": "subvalue"},
        }

        # Set entire category
        self.settings_manager.set_category("test_category", test_category)

        # Get entire category
        result = self.settings_manager.get_category("test_category")
        self.assertEqual(result, test_category)

    def test_persistence_across_instances(self):
        """Test that settings persist across different instances."""
        # Set a value in first instance
        self.settings_manager.set_setting(
            "test_category", "persistent_key", "persistent_value"
        )

        # Create new instance with same directory
        new_manager = SecureSettingsManager(settings_dir=self.temp_dir)

        # Check value persists
        result = new_manager.get_setting("test_category", "persistent_key")
        self.assertEqual(result, "persistent_value")

    def test_encryption_of_sensitive_data(self):
        """Test that sensitive data is actually encrypted."""
        # Set a sensitive value
        self.settings_manager.set_setting(
            "test_category", "api_key", "secret_key_value"
        )

        # Check that the encrypted file doesn't contain the plain text
        settings_file = Path(self.temp_dir) / "secure_settings.enc"
        self.assertTrue(settings_file.exists())

        with open(settings_file, "rb") as f:
            encrypted_content = f.read()

        # The plain text should not be in the encrypted content
        self.assertNotIn(b"secret_key_value", encrypted_content)
        self.assertNotIn(b"api_key", encrypted_content)

    def test_masking_of_sensitive_values(self):
        """Test that sensitive values are masked in get_all_settings."""
        # Set sensitive values
        self.settings_manager.set_setting("test_category", "api_key", "secret123")
        self.settings_manager.set_setting("test_category", "password", "pass456")
        self.settings_manager.set_setting("test_category", "normal_value", "visible")

        # Get all settings (should be masked)
        all_settings = self.settings_manager.get_all_settings()

        # Check sensitive values are masked
        self.assertEqual(all_settings["test_category"]["api_key"], "secr***")
        self.assertEqual(all_settings["test_category"]["password"], "pass***")

        # Check normal values are not masked
        self.assertEqual(all_settings["test_category"]["normal_value"], "visible")

    def test_export_settings_without_secrets(self):
        """Test exporting settings without including secrets."""
        # Set both sensitive and non-sensitive values
        self.settings_manager.set_setting("test_category", "api_key", "secret123")
        self.settings_manager.set_setting("test_category", "normal_value", "visible")

        # Export without secrets
        exported = self.settings_manager.export_settings(include_secrets=False)
        exported_data = json.loads(exported)

        # Check sensitive values are masked
        self.assertEqual(exported_data["test_category"]["api_key"], "secr***")

        # Check normal values are not masked
        self.assertEqual(exported_data["test_category"]["normal_value"], "visible")

    def test_export_settings_with_secrets(self):
        """Test exporting settings with secrets included."""
        # Set a sensitive value
        self.settings_manager.set_setting("test_category", "api_key", "secret123")

        # Export with secrets
        exported = self.settings_manager.export_settings(include_secrets=True)
        exported_data = json.loads(exported)

        # Check sensitive values are included
        self.assertEqual(exported_data["test_category"]["api_key"], "secret123")

    def test_import_settings(self):
        """Test importing settings from JSON."""
        # Create test settings JSON
        test_settings = {
            "imported_category": {"key1": "value1", "key2": "value2"},
            "existing_category": {"new_key": "new_value"},
        }
        settings_json = json.dumps(test_settings)

        # Import settings
        self.settings_manager.import_settings(settings_json, merge=True)

        # Check imported values exist
        self.assertEqual(
            self.settings_manager.get_setting("imported_category", "key1"), "value1"
        )
        self.assertEqual(
            self.settings_manager.get_setting("imported_category", "key2"), "value2"
        )
        self.assertEqual(
            self.settings_manager.get_setting("existing_category", "new_key"),
            "new_value",
        )

    def test_import_settings_replace(self):
        """Test importing settings with replace mode."""
        # Set an initial value
        self.settings_manager.set_setting(
            "test_category", "existing_key", "existing_value"
        )

        # Create test settings JSON that replaces the category
        test_settings = {"test_category": {"new_key": "new_value"}}
        settings_json = json.dumps(test_settings)

        # Import with replace
        self.settings_manager.import_settings(settings_json, merge=False)

        # Check old value is gone and new value exists
        self.assertIsNone(
            self.settings_manager.get_setting("test_category", "existing_key")
        )
        self.assertEqual(
            self.settings_manager.get_setting("test_category", "new_key"), "new_value"
        )

    @patch("app.core.secure_settings.httpx.Client")
    def test_validate_api_key_openai_compatible(self, mock_client):
        """Test API key validation for OpenAI-compatible provider."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Test validation
        result = self.settings_manager.validate_api_key("openai_compatible", "test_key")
        self.assertTrue(result)

        # Verify correct endpoint was called
        mock_client.return_value.__enter__.return_value.get.assert_called_once()
        call_args = mock_client.return_value.__enter__.return_value.get.call_args
        self.assertIn("models", call_args[0][0])

    @patch("app.core.secure_settings.httpx.Client")
    def test_validate_api_key_jina(self, mock_client):
        """Test API key validation for Jina provider."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 422  # Expected for empty request
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Test validation
        result = self.settings_manager.validate_api_key("jina_reranker", "test_key")
        self.assertTrue(result)

        # Verify correct endpoint was called
        mock_client.return_value.__enter__.return_value.get.assert_called_once()
        call_args = mock_client.return_value.__enter__.return_value.get.call_args
        self.assertIn("rerank", call_args[0][0])

    def test_validate_api_key_empty(self):
        """Test API key validation with empty key."""
        result = self.settings_manager.validate_api_key("openai_compatible", "")
        self.assertFalse(result)

        result = self.settings_manager.validate_api_key("openai_compatible", None)
        self.assertFalse(result)

    @patch("app.core.secure_settings.httpx.Client")
    def test_validate_api_key_failure(self, mock_client):
        """Test API key validation with failed request."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Test validation
        result = self.settings_manager.validate_api_key(
            "openai_compatible", "invalid_key"
        )
        self.assertFalse(result)

    @patch("app.core.secure_settings.httpx.Client")
    def test_validate_api_key_exception(self, mock_client):
        """Test API key validation with request exception."""
        # Mock exception
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception(
            "Network error"
        )

        # Test validation
        result = self.settings_manager.validate_api_key("openai_compatible", "test_key")
        self.assertFalse(result)


class TestSecureSettingsIntegration(unittest.TestCase):
    """Integration tests for secure settings with the main configuration system."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("app.core.secure_settings.SecureSettingsManager")
    def test_config_loads_from_secure_settings(self, mock_manager):
        """Test that the main config system loads from secure settings."""
        # Mock secure settings
        mock_secure_settings = {
            "system_config": {
                "tool_system_enabled": False,
                "agent_system_enabled": False,
                "preferred_provider": "ollama",
            },
            "llm_providers": {
                "openai_compatible": {
                    "enabled": False,
                    "api_key": "test_key",
                    "base_url": "https://test.api.com",
                }
            },
        }

        mock_instance = MagicMock()
        mock_instance.get_category.return_value = mock_secure_settings.get(
            "system_config", {}
        )
        mock_instance.get_category.side_effect = (
            lambda category, default={}: mock_secure_settings.get(category, default)
        )
        mock_manager.return_value = mock_instance

        # Test that settings load from secure settings
        with patch("app.core.config.secure_settings", mock_instance):
            from app.core.config import Settings

            settings = Settings()

            # Check that values were loaded from secure settings
            self.assertEqual(settings.preferred_provider, "ollama")
            self.assertEqual(settings.tool_system_enabled, False)
            self.assertEqual(settings.agent_system_enabled, False)

    def test_gradio_settings_page_creation(self):
        """Test that the Gradio settings page can be created without errors."""
        # Gradio settings page has been removed - this test is now deprecated
        self.skipTest("Gradio settings page has been removed from the codebase")


if __name__ == "__main__":
    unittest.main()
