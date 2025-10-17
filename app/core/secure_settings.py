"""
Secure settings management system for the AI Assistant.

This module provides a secure way to store and manage sensitive configuration
like API keys without exposing them in .env files.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class SecureSettingsManager:
    """Manages secure storage of sensitive configuration data."""
    
    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize the secure settings manager.
        
        Args:
            settings_dir: Directory to store encrypted settings. Defaults to ~/.ai_assistant
        """
        if settings_dir is None:
            settings_dir = os.path.expanduser("~/.ai_assistant")
        
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(exist_ok=True)
        self.settings_file = self.settings_dir / "secure_settings.enc"
        self.key_file = self.settings_dir / "key.derived"
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing settings or create defaults
        self._load_settings()
    
    def _init_encryption(self):
        """Initialize encryption using system-based key derivation."""
        # Use a combination of user-specific and machine-specific data for key derivation
        machine_id = os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "default"))
        user_home = os.path.expanduser("~")
        seed_data = f"{user_home}:{machine_id}:ai_assistant_v1".encode()
        
        # Generate a salt (in production, this should be stored securely)
        salt = b"ai_assistant_salt_v1"  # Fixed salt for consistency
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(seed_data))
        
        self.cipher = Fernet(key)
        logger.debug("Encryption initialized successfully")
    
    def _load_settings(self):
        """Load settings from encrypted storage."""
        if not self.settings_file.exists():
            self.settings = self._get_default_settings()
            self._save_settings()
            return
        
        try:
            with open(self.settings_file, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            self.settings = json.loads(decrypted_data.decode())
            logger.info("Secure settings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load secure settings: {e}")
            self.settings = self._get_default_settings()
            self._save_settings()
    
    def _save_settings(self):
        """Save settings to encrypted storage."""
        try:
            settings_json = json.dumps(self.settings, indent=2)
            encrypted_data = self.cipher.encrypt(settings_json.encode())
            
            with open(self.settings_file, "wb") as f:
                f.write(encrypted_data)
            
            logger.info("Secure settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save secure settings: {e}")
            raise
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings structure."""
        return {
            "llm_providers": {
                "openai_compatible": {
                    "enabled": True,
                    "api_key": "",
                    "base_url": "https://openrouter.ai/api/v1",
                    "default_model": "anthropic/claude-3.5-sonnet",
                    "provider_name": "",
                    "timeout": 30,
                    "max_retries": 3
                },
                "ollama": {
                    "enabled": True,
                    "base_url": "http://localhost:11434",
                    "default_model": "llama2",
                    "timeout": 30,
                    "max_retries": 3,
                    "temperature": 0.7,
                    "max_tokens": None,
                    "streaming": True
                }
            },
            "external_services": {
                "firecrawl": {
                    "enabled": False,
                    "deployment_mode": "docker",
                    "docker_url": "http://firecrawl-api:3002",
                    "bull_auth_key": "",
                    "scraping_enabled": True,
                    "max_concurrent_scrapes": 5,
                    "scrape_timeout": 60
                },
                "jina_reranker": {
                    "enabled": False,
                    "api_key": "",
                    "url": "http://jina-reranker:8080",
                    "model": "jina-reranker-v2-base-multilingual",
                    "timeout": 30,
                    "cache_ttl": 3600,
                    "max_retries": 3
                },
                "searxng": {
                    "secret_key": "",
                    "url": "http://searxng:8080"
                }
            },
            "system_config": {
                "tool_system_enabled": True,
                "agent_system_enabled": True,
                "preferred_provider": "openai_compatible",
                "enable_fallback": True,
                "debug": True,
                "host": "127.0.0.1",
                "port": 8000,
                "environment": "development",
                "secret_key": ""
            },
            "multi_writer": {
                "enabled": False,
                "mongodb_connection_string": "mongodb://localhost:27017",
                "mongodb_database_name": "multi_writer_system"
            }
        }
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            category: The category of the setting (e.g., "llm_providers")
            key: The key within the category
            default: Default value if not found
            
        Returns:
            The setting value or default
        """
        try:
            return self.settings.get(category, {}).get(key, default)
        except Exception as e:
            logger.error(f"Error getting setting {category}.{key}: {e}")
            return default
    
    def set_setting(self, category: str, key: str, value: Any):
        """
        Set a specific setting value.
        
        Args:
            category: The category of the setting
            key: The key within the category
            value: The value to set
        """
        try:
            if category not in self.settings:
                self.settings[category] = {}
            
            self.settings[category][key] = value
            self._save_settings()
            logger.info(f"Setting {category}.{key} updated successfully")
        except Exception as e:
            logger.error(f"Error setting {category}.{key}: {e}")
            raise
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """
        Get all settings in a category.
        
        Args:
            category: The category name
            
        Returns:
            Dictionary of all settings in the category
        """
        return self.settings.get(category, {})
    
    def set_category(self, category: str, values: Dict[str, Any]):
        """
        Set multiple values in a category.
        
        Args:
            category: The category name
            values: Dictionary of values to set
        """
        try:
            if category not in self.settings:
                self.settings[category] = {}
            
            self.settings[category].update(values)
            self._save_settings()
            logger.info(f"Category {category} updated successfully")
        except Exception as e:
            logger.error(f"Error updating category {category}: {e}")
            raise
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings (with sensitive values masked)."""
        masked_settings = {}
        
        for category, values in self.settings.items():
            masked_settings[category] = {}
            for key, value in values.items():
                if any(sensitive in key.lower() for sensitive in ["key", "password", "secret", "auth"]):
                    if value and len(str(value)) > 4:
                        masked_settings[category][key] = f"{str(value)[:4]}{'*' * (len(str(value)) - 4)}"
                    else:
                        masked_settings[category][key] = "***" if value else ""
                else:
                    masked_settings[category][key] = value
        
        return masked_settings
    
    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key by testing a simple request.
        
        Args:
            provider: The provider name (e.g., "openai_compatible")
            api_key: The API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False
        
        try:
            import httpx
            
            if provider == "openai_compatible":
                # Test with OpenAI-compatible API
                base_url = self.get_setting("llm_providers", "openai_compatible", {}).get("base_url", "https://openrouter.ai/api/v1")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                with httpx.Client(timeout=10) as client:
                    response = client.get(f"{base_url}/models", headers=headers)
                    return response.status_code == 200
            
            elif provider == "jina_reranker":
                # Test with Jina API
                with httpx.Client(timeout=10) as client:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    response = client.get("https://api.jina.ai/v1/rerank", headers=headers)
                    return response.status_code in [200, 400, 422]  # 422 is OK for empty request
            
            return True  # For other providers, just check if key is not empty
            
        except Exception as e:
            logger.error(f"API key validation failed for {provider}: {e}")
            return False
    
    def export_settings(self, include_secrets: bool = False) -> str:
        """
        Export settings to JSON string.
        
        Args:
            include_secrets: Whether to include sensitive values
            
        Returns:
            JSON string of settings
        """
        if include_secrets:
            return json.dumps(self.settings, indent=2)
        else:
            return json.dumps(self.get_all_settings(), indent=2)
    
    def import_settings(self, settings_json: str, merge: bool = True):
        """
        Import settings from JSON string.
        
        Args:
            settings_json: JSON string of settings
            merge: Whether to merge with existing settings or replace
        """
        try:
            new_settings = json.loads(settings_json)
            
            if merge:
                for category, values in new_settings.items():
                    if category not in self.settings:
                        self.settings[category] = {}
                    self.settings[category].update(values)
            else:
                self.settings = new_settings
            
            self._save_settings()
            logger.info("Settings imported successfully")
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            raise


# Global instance
secure_settings = SecureSettingsManager()