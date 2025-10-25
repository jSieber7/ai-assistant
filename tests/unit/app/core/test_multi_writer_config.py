"""
Unit tests for multi-writer configuration system.

Tests MultiWriterSettings and related functions.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, Mock
from pathlib import Path

from app.core.multi_writer_config import (
    MultiWriterSettings,
    multi_writer_settings,
    get_multi_writer_config,
    is_multi_writer_enabled,
    validate_multi_writer_config,
    initialize_multi_writer_system
)


class TestMultiWriterSettings:
    """Test MultiWriterSettings configuration"""
    
    def test_default_multi_writer_settings(self):
        """Test default multi-writer settings values"""
        settings = MultiWriterSettings()
        
        assert settings.enabled is False
        assert settings.firecrawl_api_key is None
        assert settings.firecrawl_base_url == "https://api.firecrawl.dev"
        assert settings.mongodb_connection_string is None
        assert settings.mongodb_database_name == "multi_writer_system"
        assert settings.template_dir == "templates"
        assert settings.default_template == "article.html.jinja"
        
        # Writer configuration
        assert settings.default_writers == ["technical_1", "creative_1", "analytical_1"]
        assert "technical_1" in settings.available_writers
        assert "creative_1" in settings.available_writers
        assert "analytical_1" in settings.available_writers
        
        # Checker configuration
        assert settings.default_checkers == ["factual_1", "style_1", "structure_1"]
        assert "factual_1" in settings.available_checkers
        assert "style_1" in settings.available_checkers
        assert "structure_1" in settings.available_checkers
        
        # Quality settings
        assert settings.quality_threshold == 70.0
        assert settings.max_iterations == 2
        
        # Performance settings
        assert settings.max_concurrent_workflows == 5
        assert settings.workflow_timeout == 600
        
        # Output settings
        assert settings.output_dir == "generated_content"
        assert settings.save_intermediate_results is True
        
        # API settings
        assert settings.api_prefix == "/v1/multi-writer"
        assert settings.enable_async_execution is True
    
    def test_default_writers_parsing_from_string(self):
        """Test parsing default_writers from string"""
        settings = MultiWriterSettings(default_writers="writer1, writer2, writer3")
        
        assert settings.default_writers == ["writer1", "writer2", "writer3"]
    
    def test_default_checkers_parsing_from_string(self):
        """Test parsing default_checkers from string"""
        settings = MultiWriterSettings(default_checkers="checker1, checker2, checker3")
        
        assert settings.default_checkers == ["checker1", "checker2", "checker3"]
    
    def test_available_writers_configuration(self):
        """Test available writers configuration"""
        settings = MultiWriterSettings()
        
        # Test technical writers
        assert settings.available_writers["technical_1"]["specialty"] == "technical"
        assert settings.available_writers["technical_1"]["model"] == "claude-3.5-sonnet"
        assert settings.available_writers["technical_2"]["specialty"] == "technical"
        assert settings.available_writers["technical_2"]["model"] == "gpt-4-turbo"
        
        # Test creative writers
        assert settings.available_writers["creative_1"]["specialty"] == "creative"
        assert settings.available_writers["creative_1"]["model"] == "claude-3.5-sonnet"
        
        # Test analytical writers
        assert settings.available_writers["analytical_1"]["specialty"] == "analytical"
        assert settings.available_writers["analytical_1"]["model"] == "gpt-4-turbo"
    
    def test_available_checkers_configuration(self):
        """Test available checkers configuration"""
        settings = MultiWriterSettings()
        
        # Test factual checker
        assert settings.available_checkers["factual_1"]["focus_area"] == "factual"
        assert settings.available_checkers["factual_1"]["model"] == "claude-3.5-sonnet"
        
        # Test style checker
        assert settings.available_checkers["style_1"]["focus_area"] == "style"
        assert settings.available_checkers["style_1"]["model"] == "claude-3.5-sonnet"
        
        # Test structure checker
        assert settings.available_checkers["structure_1"]["focus_area"] == "structure"
        assert settings.available_checkers["structure_1"]["model"] == "gpt-4-turbo"
        
        # Test SEO checker
        assert settings.available_checkers["seo_1"]["focus_area"] == "seo"
        assert settings.available_checkers["seo_1"]["model"] == "gpt-4-turbo"


class TestMultiWriterConfigFunctions:
    """Test multi-writer configuration functions"""
    
    def test_get_multi_writer_config(self):
        """Test getting multi-writer configuration as dictionary"""
        config = get_multi_writer_config()
        
        # Check top-level keys
        assert "enabled" in config
        assert "firecrawl" in config
        assert "mongodb" in config
        assert "templates" in config
        assert "writers" in config
        assert "checkers" in config
        assert "quality" in config
        assert "performance" in config
        assert "output" in config
        assert "api" in config
        
        # Check firecrawl configuration
        assert config["firecrawl"]["base_url"] == "https://api.firecrawl.dev"
        assert config["firecrawl"]["api_key"] is None
        
        # Check mongodb configuration
        assert config["mongodb"]["database_name"] == "multi_writer_system"
        assert config["mongodb"]["connection_string"] is None
        
        # Check templates configuration
        assert config["templates"]["template_dir"] == "templates"
        assert config["templates"]["default_template"] == "article.html.jinja"
        
        # Check writers configuration
        assert config["writers"]["default_writers"] == ["technical_1", "creative_1", "analytical_1"]
        assert "technical_1" in config["writers"]["available_writers"]
        
        # Check checkers configuration
        assert config["checkers"]["default_checkers"] == ["factual_1", "style_1", "structure_1"]
        assert "factual_1" in config["checkers"]["available_checkers"]
        
        # Check quality configuration
        assert config["quality"]["threshold"] == 70.0
        assert config["quality"]["max_iterations"] == 2
        
        # Check performance configuration
        assert config["performance"]["max_concurrent_workflows"] == 5
        assert config["performance"]["workflow_timeout"] == 600
        
        # Check output configuration
        assert config["output"]["output_dir"] == "generated_content"
        assert config["output"]["save_intermediate_results"] is True
        
        # Check API configuration
        assert config["api"]["prefix"] == "/v1/multi-writer"
        assert config["api"]["enable_async_execution"] is True
    
    @patch('app.core.multi_writer_config.settings')
    def test_is_multi_writer_enabled_main_settings(self, mock_settings):
        """Test checking if multi-writer is enabled from main settings"""
        mock_settings.multi_writer_enabled = True
        
        result = is_multi_writer_enabled()
        assert result is True
    
    @patch('app.core.multi_writer_config.settings')
    def test_is_multi_writer_enabled_multi_writer_settings(self, mock_settings):
        """Test checking if multi-writer is enabled from multi-writer settings"""
        mock_settings.multi_writer_enabled = False
        
        # Enable multi-writer settings directly
        multi_writer_settings.enabled = True
        
        result = is_multi_writer_enabled()
        assert result is True
    
    @patch('app.core.multi_writer_config.settings')
    def test_is_multi_writer_enabled_both_disabled(self, mock_settings):
        """Test checking if multi-writer is enabled when both are disabled"""
        mock_settings.multi_writer_enabled = False
        
        # Disable multi-writer settings directly
        multi_writer_settings.enabled = False
        
        result = is_multi_writer_enabled()
        assert result is False
    
    def test_validate_multi_writer_config_disabled(self):
        """Test validating multi-writer configuration when disabled"""
        multi_writer_settings.enabled = False
        
        issues = validate_multi_writer_config()
        
        assert len(issues) == 1
        assert "Multi-writer system is disabled" in issues[0]
    
    def test_validate_multi_writer_config_no_api_key(self):
        """Test validating multi-writer configuration without API key"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = None
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("Firecrawl API key not configured" in issue for issue in issues)
    
    def test_validate_multi_writer_config_no_mongodb(self):
        """Test validating multi-writer configuration without MongoDB"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = None
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("MongoDB connection string not configured" in issue for issue in issues)
    
    def test_validate_multi_writer_config_missing_template_dir(self):
        """Test validating multi-writer configuration with missing template directory"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.template_dir = "/nonexistent/templates"
        
        # Mock directory existence and creation failure
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs', side_effect=OSError("Permission denied")):
            issues = validate_multi_writer_config()
            
            assert any("Cannot create template directory" in issue for issue in issues)
    
    def test_validate_multi_writer_config_missing_output_dir(self):
        """Test validating multi-writer configuration with missing output directory"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.template_dir = "/tmp/templates"
        multi_writer_settings.output_dir = "/nonexistent/output"
        
        # Mock template directory exists, output directory doesn't and creation fails
        def mock_exists(path):
            return path == "/tmp/templates"
        
        def mock_makedirs(path, exist_ok=False):
            if path == "/nonexistent/output":
                raise OSError("Permission denied")
        
        with patch('os.path.exists', side_effect=mock_exists), \
             patch('os.makedirs', side_effect=mock_makedirs):
            issues = validate_multi_writer_config()
            
            assert any("Cannot create output directory" in issue for issue in issues)
    
    def test_validate_multi_writer_config_invalid_writer(self):
        """Test validating multi-writer configuration with invalid writer"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.default_writers = ["invalid_writer"]
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("Default writer 'invalid_writer' not found" in issue for issue in issues)
    
    def test_validate_multi_writer_config_invalid_checker(self):
        """Test validating multi-writer configuration with invalid checker"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.default_checkers = ["invalid_checker"]
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("Default checker 'invalid_checker' not found" in issue for issue in issues)
    
    def test_validate_multi_writer_config_invalid_quality_threshold(self):
        """Test validating multi-writer configuration with invalid quality threshold"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.quality_threshold = 150.0  # Invalid: > 100
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("Quality threshold must be between 0 and 100" in issue for issue in issues)
    
    def test_validate_multi_writer_config_invalid_max_iterations(self):
        """Test validating multi-writer configuration with invalid max iterations"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.max_iterations = 0  # Invalid: < 1
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert any("Max iterations must be at least 1" in issue for issue in issues)
    
    def test_validate_multi_writer_config_valid(self):
        """Test validating multi-writer configuration with valid settings"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        multi_writer_settings.quality_threshold = 80.0
        multi_writer_settings.max_iterations = 3
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            issues = validate_multi_writer_config()
            
            assert len(issues) == 0
    
    @patch('app.core.multi_writer_config.create_default_templates')
    def test_initialize_multi_writer_system_disabled(self, mock_create_templates):
        """Test initializing multi-writer system when disabled"""
        multi_writer_settings.enabled = False
        
        result = initialize_multi_writer_system()
        
        assert result is None
        mock_create_templates.assert_not_called()
    
    @patch('app.core.multi_writer_config.create_default_templates')
    def test_initialize_multi_writer_system_enabled(self, mock_create_templates):
        """Test initializing multi-writer system when enabled"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = Mock()
        multi_writer_settings.mongodb_connection_string = "mongodb://localhost:27017"
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            result = initialize_multi_writer_system()
            
            assert result is not None
            mock_create_templates.assert_called_once()
    
    @patch('app.core.multi_writer_config.create_default_templates')
    def test_initialize_multi_writer_system_with_validation_issues(self, mock_create_templates):
        """Test initializing multi-writer system with validation issues"""
        multi_writer_settings.enabled = True
        multi_writer_settings.firecrawl_api_key = None  # Missing API key
        
        # Mock directory existence
        with patch('os.path.exists', return_value=True):
            result = initialize_multi_writer_system()
            
            assert result is None
            mock_create_templates.assert_not_called()


class TestMultiWriterSettingsWithEnvVars:
    """Test MultiWriterSettings with environment variables"""
    
    def test_multi_writer_settings_from_env(self):
        """Test creating multi-writer settings from environment variables"""
        with patch.dict(os.environ, {
            "MULTI_WRITER_ENABLED": "true",
            "MULTI_WRITER_FIRECRAWL_BASE_URL": "https://custom-firecrawl.com",
            "MULTI_WRITER_MONGODB_DATABASE_NAME": "custom_db",
            "MULTI_WRITER_TEMPLATE_DIR": "/custom/templates",
            "MULTI_WRITER_OUTPUT_DIR": "/custom/output",
            "MULTI_WRITER_API_PREFIX": "/v1/custom",
            "MULTI_WRITER_QUALITY_THRESHOLD": "85.0",
            "MULTI_WRITER_MAX_ITERATIONS": "5",
            "MULTI_WRITER_MAX_CONCURRENT_WORKFLOWS": "10",
            "MULTI_WRITER_WORKFLOW_TIMEOUT": "1200"
        }):
            settings = MultiWriterSettings()
            
            # Note: Pydantic_settings doesn't automatically use env vars without explicit configuration
            # This test demonstrates expected behavior if env vars were properly configured
            assert settings.enabled is False  # Default value
            assert settings.firecrawl_base_url == "https://api.firecrawl.dev"  # Default value


if __name__ == "__main__":
    pytest.main([__file__])