"""
Unit tests for secure settings management system.

Tests SecureSettingsManager and related functions.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, Mock, mock_open
from pathlib import Path

from app.core.secure_settings import SecureSettingsManager


class TestSecureSettingsManager:
    """Test SecureSettingsManager class"""
    
    def test_init_with_default_settings_dir(self):
        """Test initializing with default settings directory"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.dict(os.environ, {}, clear=True):
            
            manager = SecureSettingsManager()
            
            assert manager.settings_dir == Path.home() / ".ai_assistant"
            assert manager.settings_file == manager.settings_dir / "secure_settings.enc"
            assert manager.key_file == manager.settings_dir / "key.derived"
    
    def test_init_with_custom_settings_dir(self):
        """Test initializing with custom settings directory"""
        custom_dir = "/custom/settings/dir"
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            manager = SecureSettingsManager(custom_dir)
            
            assert manager.settings_dir == Path(custom_dir)
            assert manager.settings_file == manager.settings_dir / "secure_settings.enc"
            assert manager.key_file == manager.settings_dir / "key.derived"
    
    def test_init_with_env_var_settings_dir(self):
        """Test initializing with environment variable settings directory"""
        env_dir = "/env/settings/dir"
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.dict(os.environ, {"AI_ASSISTANT_SETTINGS_DIR": env_dir}):
            
            manager = SecureSettingsManager()
            
            assert manager.settings_dir == Path(env_dir)
    
    def test_init_creates_settings_directory(self):
        """Test that initialization creates settings directory"""
        with patch('os.makedirs') as mock_makedirs, \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_load_settings'), \
             patch.object(SecureSettingsManager, '_init_encryption'):
            
            SecureSettingsManager()
            
            mock_makedirs.assert_called_once_with(exist_ok=True)
    
    def test_get_default_settings(self):
        """Test getting default settings structure"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            settings = manager.settings
            
            # Check top-level categories
            assert "llm_providers" in settings
            assert "external_services" in settings
            assert "system_config" in settings
            assert "multi_writer" in settings
            assert "langchain_integration" in settings
            
            # Check LLM providers
            assert "openai_compatible" in settings["llm_providers"]
            assert "ollama" in settings["llm_providers"]
            
            # Check external services
            assert "firecrawl" in settings["external_services"]
            assert "jina_reranker" in settings["external_services"]
            assert "searxng" in settings["external_services"]
            
            # Check system config
            assert "tool_system_enabled" in settings["system_config"]
            assert "agent_system_enabled" in settings["system_config"]
            assert "deep_agents_enabled" in settings["system_config"]
            
            # Check multi-writer
            assert "enabled" in settings["multi_writer"]
            assert "mongodb_connection_string" in settings["multi_writer"]
            
            # Check LangChain integration
            assert "use_langchain_llm" in settings["langchain_integration"]
            assert "use_langchain_tools" in settings["langchain_integration"]
    
    def test_load_existing_settings(self):
        """Test loading existing settings from file"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test-key"
                }
            },
            "system_config": {
                "debug": False
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=True), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch('builtins.open', mock_open(read_data=b'encrypted_data')), \
             patch.object(SecureSettingsManager, 'cipher') as mock_cipher:
            
            mock_cipher.decrypt.return_value = json.dumps(test_settings).encode()
            
            manager = SecureSettingsManager()
            
            assert manager.settings["llm_providers"]["openai_compatible"]["api_key"] == "test-key"
            assert manager.settings["system_config"]["debug"] is False
    
    def test_load_settings_file_not_exists(self):
        """Test loading settings when file doesn't exist"""
        with patch('os.makedirs'), \
             patch('os.path.exists', side_effect=lambda path: not path.endswith('.enc')), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            
            # Should create default settings and save them
            mock_save.assert_called_once()
            assert "llm_providers" in manager.settings
    
    def test_load_settings_corrupted_file(self):
        """Test loading settings when file is corrupted"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=True), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch('builtins.open', mock_open(read_data=b'corrupted_data')), \
             patch.object(SecureSettingsManager, 'cipher') as mock_cipher, \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            mock_cipher.decrypt.side_effect = Exception("Decryption failed")
            
            manager = SecureSettingsManager()
            
            # Should fall back to default settings and save them
            mock_save.assert_called_once()
            assert "llm_providers" in manager.settings
    
    def test_save_settings(self):
        """Test saving settings to file"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test-key"
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            manager._save_settings()
            
            mock_save.assert_called_once()
    
    def test_save_settings_encryption(self):
        """Test that settings are encrypted when saved"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test-key"
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch.object(SecureSettingsManager, 'cipher') as mock_cipher:
            
            mock_cipher.encrypt.return_value = b'encrypted_data'
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            manager._save_settings()
            
            mock_cipher.encrypt.assert_called_once()
            mock_file.return_value.write.assert_called_once_with(b'encrypted_data')
    
    def test_get_setting(self):
        """Test getting a specific setting value"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test-key"
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            
            # Test existing setting
            assert manager.get_setting("llm_providers", "openai_compatible") == {
                "enabled": True,
                "api_key": "test-key"
            }
            
            # Test nested setting
            assert manager.get_setting("llm_providers", "enabled", False) is None  # Wrong level
            
            # Test with default
            assert manager.get_setting("nonexistent", "key", "default") == "default"
    
    def test_get_setting_error_handling(self):
        """Test error handling in get_setting"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            # Set settings to None to trigger an error
            manager.settings = None
            
            # Should return default value
            assert manager.get_setting("category", "key", "default") == "default"
    
    def test_set_setting(self):
        """Test setting a specific setting value"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            
            # Set new setting
            manager.set_setting("test_category", "test_key", "test_value")
            
            assert manager.settings["test_category"]["test_key"] == "test_value"
            mock_save.assert_called_once()
            
            # Update existing setting
            manager.set_setting("test_category", "test_key", "updated_value")
            
            assert manager.settings["test_category"]["test_key"] == "updated_value"
    
    def test_set_setting_error_handling(self):
        """Test error handling in set_setting"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'), \
             patch.object(SecureSettingsManager, '_save_settings', side_effect=Exception("Save failed")):
            
            manager = SecureSettingsManager()
            
            # Should raise exception
            with pytest.raises(Exception):
                manager.set_setting("test_category", "test_key", "test_value")
    
    def test_get_category(self):
        """Test getting all settings in a category"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "test-key"
                },
                "ollama": {
                    "enabled": False
                }
            },
            "other_category": {
                "other_key": "other_value"
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            
            # Test existing category
            category = manager.get_category("llm_providers")
            assert category["openai_compatible"]["enabled"] is True
            assert category["ollama"]["enabled"] is False
            
            # Test nonexistent category with default
            category = manager.get_category("nonexistent", {"default": True})
            assert category["default"] is True
            
            # Test nonexistent category without default
            category = manager.get_category("nonexistent")
            assert category == {}
    
    def test_set_category(self):
        """Test setting multiple values in a category"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            
            # Set new category
            new_values = {
                "key1": "value1",
                "key2": "value2"
            }
            manager.set_category("test_category", new_values)
            
            assert manager.settings["test_category"]["key1"] == "value1"
            assert manager.settings["test_category"]["key2"] == "value2"
            mock_save.assert_called_once()
            
            # Update existing category
            updated_values = {
                "key2": "updated_value2",
                "key3": "value3"
            }
            manager.set_category("test_category", updated_values)
            
            assert manager.settings["test_category"]["key1"] == "value1"  # Unchanged
            assert manager.settings["test_category"]["key2"] == "updated_value2"  # Updated
            assert manager.settings["test_category"]["key3"] == "value3"  # New
    
    def test_get_all_settings_masked(self):
        """Test getting all settings with sensitive values masked"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "secret-api-key-123456",
                    "timeout": 30
                },
                "ollama": {
                    "enabled": False,
                    "password": "secret-password"
                }
            },
            "safe_category": {
                "safe_value": "not-secret"
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            
            masked_settings = manager.get_all_settings()
            
            # Check that sensitive values are masked
            assert masked_settings["llm_providers"]["openai_compatible"]["api_key"] == "sec********3456"
            assert masked_settings["llm_providers"]["ollama"]["password"] == "***"
            
            # Check that non-sensitive values are not masked
            assert masked_settings["llm_providers"]["openai_compatible"]["enabled"] is True
            assert masked_settings["llm_providers"]["openai_compatible"]["timeout"] == 30
            assert masked_settings["safe_category"]["safe_value"] == "not-secret"
    
    def test_validate_api_key_openai_compatible(self):
        """Test validating OpenAI-compatible API key"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'), \
             patch('httpx.Client') as mock_client:
            
            # Mock successful response
            mock_client.return_value.__enter__.return_value.get.return_value.status_code = 200
            
            manager = SecureSettingsManager()
            
            # Test valid key
            result = manager.validate_api_key("openai_compatible", "valid-key")
            assert result is True
            
            # Test invalid key
            mock_client.return_value.__enter__.return_value.get.return_value.status_code = 401
            result = manager.validate_api_key("openai_compatible", "invalid-key")
            assert result is False
            
            # Test empty key
            result = manager.validate_api_key("openai_compatible", "")
            assert result is False
    
    def test_validate_api_key_jina_reranker(self):
        """Test validating Jina reranker API key"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'), \
             patch('httpx.Client') as mock_client:
            
            # Mock successful response
            mock_client.return_value.__enter__.return_value.get.return_value.status_code = 200
            
            manager = SecureSettingsManager()
            
            # Test valid key
            result = manager.validate_api_key("jina_reranker", "valid-key")
            assert result is True
            
            # Test invalid key (401)
            mock_client.return_value.__enter__.return_value.get.return_value.status_code = 401
            result = manager.validate_api_key("jina_reranker", "invalid-key")
            assert result is False
            
            # Test empty request (422 is acceptable)
            mock_client.return_value.__enter__.return_value.get.return_value.status_code = 422
            result = manager.validate_api_key("jina_reranker", "test-key")
            assert result is True
            
            # Test empty key
            result = manager.validate_api_key("jina_reranker", "")
            assert result is False
    
    def test_validate_api_key_unknown_provider(self):
        """Test validating API key for unknown provider"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            
            # Test unknown provider (should return True for non-empty key)
            result = manager.validate_api_key("unknown_provider", "some-key")
            assert result is True
            
            # Test empty key
            result = manager.validate_api_key("unknown_provider", "")
            assert result is False
    
    def test_validate_api_key_exception_handling(self):
        """Test exception handling in API key validation"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'), \
             patch('httpx.Client', side_effect=Exception("Network error")):
            
            manager = SecureSettingsManager()
            
            # Should return False on exception
            result = manager.validate_api_key("openai_compatible", "test-key")
            assert result is False
    
    def test_export_settings_without_secrets(self):
        """Test exporting settings without secrets"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "secret-api-key-123456"
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            
            exported = manager.export_settings(include_secrets=False)
            exported_data = json.loads(exported)
            
            # Check that API key is masked
            assert "secret" not in exported_data["llm_providers"]["openai_compatible"]["api_key"]
            assert exported_data["llm_providers"]["openai_compatible"]["api_key"].startswith("sec")
    
    def test_export_settings_with_secrets(self):
        """Test exporting settings with secrets"""
        test_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "secret-api-key-123456"
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            manager.settings = test_settings
            
            exported = manager.export_settings(include_secrets=True)
            exported_data = json.loads(exported)
            
            # Check that API key is included
            assert exported_data["llm_providers"]["openai_compatible"]["api_key"] == "secret-api-key-123456"
    
    def test_import_settings_merge(self):
        """Test importing settings with merge"""
        existing_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "existing-key"
                }
            },
            "existing_category": {
                "existing_key": "existing_value"
            }
        }
        
        new_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "timeout": 60  # New field
                }
            },
            "new_category": {
                "new_key": "new_value"
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            manager.settings = existing_settings
            
            manager.import_settings(json.dumps(new_settings), merge=True)
            
            # Check that existing settings are preserved
            assert manager.settings["llm_providers"]["openai_compatible"]["enabled"] is True
            assert manager.settings["llm_providers"]["openai_compatible"]["api_key"] == "existing-key"
            assert manager.settings["existing_category"]["existing_key"] == "existing_value"
            
            # Check that new settings are added
            assert manager.settings["llm_providers"]["openai_compatible"]["timeout"] == 60
            assert manager.settings["new_category"]["new_key"] == "new_value"
            
            mock_save.assert_called_once()
    
    def test_import_settings_replace(self):
        """Test importing settings with replace"""
        existing_settings = {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "existing-key"
                }
            }
        }
        
        new_settings = {
            "llm_providers": {
                "ollama": {
                    "enabled": False
                }
            }
        }
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings') as mock_save:
            
            manager = SecureSettingsManager()
            manager.settings = existing_settings
            
            manager.import_settings(json.dumps(new_settings), merge=False)
            
            # Check that settings are replaced
            assert "openai_compatible" not in manager.settings["llm_providers"]
            assert manager.settings["llm_providers"]["ollama"]["enabled"] is False
            
            mock_save.assert_called_once()
    
    def test_import_settings_invalid_json(self):
        """Test importing settings with invalid JSON"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.object(SecureSettingsManager, '_init_encryption'), \
             patch.object(SecureSettingsManager, '_save_settings'):
            
            manager = SecureSettingsManager()
            
            # Should raise exception for invalid JSON
            with pytest.raises(Exception):
                manager.import_settings("invalid json", merge=True)


class TestSecureSettingsManagerEncryption:
    """Test encryption functionality of SecureSettingsManager"""
    
    def test_init_encryption(self):
        """Test encryption initialization"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.dict(os.environ, {
                 "COMPUTERNAME": "test-computer",
                 "HOME": "/home/testuser"
             }):
            
            manager = SecureSettingsManager()
            manager._init_encryption()
            
            # Check that cipher is initialized
            assert manager.cipher is not None
    
    def test_encryption_key_derivation(self):
        """Test that encryption key is derived consistently"""
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch.dict(os.environ, {
                 "COMPUTERNAME": "test-computer",
                 "HOME": "/home/testuser"
             }):
            
            manager1 = SecureSettingsManager()
            manager1._init_encryption()
            
            manager2 = SecureSettingsManager()
            manager2._init_encryption()
            
            # Both managers should have the same cipher (same key derivation)
            assert type(manager1.cipher) == type(manager2.cipher)


if __name__ == "__main__":
    pytest.main([__file__])