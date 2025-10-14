"""
Configuration for multi-writer/checker system
"""
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
from pydantic import SecretStr
import logging

logger = logging.getLogger(__name__)


class MultiWriterSettings(BaseSettings):
    """Multi-writer/checker system configuration"""
    
    # System enable/disable
    enabled: bool = False  # Disabled by default as requested
    
    # Firecrawl API settings
    firecrawl_api_key: Optional[SecretStr] = None
    firecrawl_base_url: str = "https://api.firecrawl.dev"
    
    # MongoDB settings
    mongodb_connection_string: Optional[str] = None
    mongodb_database_name: str = "multi_writer_system"
    
    # Template settings
    template_dir: str = "templates"
    default_template: str = "article.html.jinja"
    
    # Writer configuration
    default_writers: List[str] = ["technical_1", "creative_1", "analytical_1"]
    available_writers: Dict[str, Dict[str, Any]] = {
        "technical_1": {
            "specialty": "technical",
            "model": "claude-3.5-sonnet"
        },
        "technical_2": {
            "specialty": "technical", 
            "model": "gpt-4-turbo"
        },
        "creative_1": {
            "specialty": "creative",
            "model": "claude-3.5-sonnet"
        },
        "analytical_1": {
            "specialty": "analytical",
            "model": "gpt-4-turbo"
        }
    }
    
    # Checker configuration
    default_checkers: List[str] = ["factual_1", "style_1", "structure_1"]
    available_checkers: Dict[str, Dict[str, Any]] = {
        "factual_1": {
            "focus_area": "factual",
            "model": "claude-3.5-sonnet"
        },
        "style_1": {
            "focus_area": "style",
            "model": "claude-3.5-sonnet"
        },
        "structure_1": {
            "focus_area": "structure",
            "model": "gpt-4-turbo"
        },
        "seo_1": {
            "focus_area": "seo",
            "model": "gpt-4-turbo"
        }
    }
    
    # Quality settings
    quality_threshold: float = 70.0
    max_iterations: int = 2
    
    # Performance settings
    max_concurrent_workflows: int = 5
    workflow_timeout: int = 600  # 10 minutes
    
    # Output settings
    output_dir: str = "generated_content"
    save_intermediate_results: bool = True
    
    # API settings
    api_prefix: str = "/v1/multi-writer"
    enable_async_execution: bool = True
    
    class Config:
        env_prefix = "MULTI_WRITER_"
        env_file = ".env"


# Global settings instance
multi_writer_settings = MultiWriterSettings()


def get_multi_writer_config() -> Dict[str, Any]:
    """Get multi-writer system configuration as dictionary"""
    return {
        "enabled": multi_writer_settings.enabled,
        "firecrawl": {
            "api_key": multi_writer_settings.firecrawl_api_key.get_secret_value() if multi_writer_settings.firecrawl_api_key else None,
            "base_url": multi_writer_settings.firecrawl_base_url
        },
        "mongodb": {
            "connection_string": multi_writer_settings.mongodb_connection_string,
            "database_name": multi_writer_settings.mongodb_database_name
        },
        "templates": {
            "template_dir": multi_writer_settings.template_dir,
            "default_template": multi_writer_settings.default_template
        },
        "writers": {
            "default_writers": multi_writer_settings.default_writers,
            "available_writers": multi_writer_settings.available_writers
        },
        "checkers": {
            "default_checkers": multi_writer_settings.default_checkers,
            "available_checkers": multi_writer_settings.available_checkers
        },
        "quality": {
            "threshold": multi_writer_settings.quality_threshold,
            "max_iterations": multi_writer_settings.max_iterations
        },
        "performance": {
            "max_concurrent_workflows": multi_writer_settings.max_concurrent_workflows,
            "workflow_timeout": multi_writer_settings.workflow_timeout
        },
        "output": {
            "output_dir": multi_writer_settings.output_dir,
            "save_intermediate_results": multi_writer_settings.save_intermediate_results
        },
        "api": {
            "prefix": multi_writer_settings.api_prefix,
            "enable_async_execution": multi_writer_settings.enable_async_execution
        }
    }


def is_multi_writer_enabled() -> bool:
    """Check if multi-writer system is enabled"""
    return multi_writer_settings.enabled


def validate_multi_writer_config() -> List[str]:
    """Validate multi-writer configuration and return list of issues"""
    issues = []
    
    if not multi_writer_settings.enabled:
        return ["Multi-writer system is disabled"]
    
    # Check Firecrawl API key
    if not multi_writer_settings.firecrawl_api_key:
        issues.append("Firecrawl API key not configured")
    
    # Check MongoDB connection
    if not multi_writer_settings.mongodb_connection_string:
        issues.append("MongoDB connection string not configured")
    
    # Check template directory
    import os
    if not os.path.exists(multi_writer_settings.template_dir):
        try:
            os.makedirs(multi_writer_settings.template_dir, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create template directory: {str(e)}")
    
    # Check output directory
    if not os.path.exists(multi_writer_settings.output_dir):
        try:
            os.makedirs(multi_writer_settings.output_dir, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {str(e)}")
    
    # Validate writers and checkers
    for writer_id in multi_writer_settings.default_writers:
        if writer_id not in multi_writer_settings.available_writers:
            issues.append(f"Default writer '{writer_id}' not found in available writers")
    
    for checker_id in multi_writer_settings.default_checkers:
        if checker_id not in multi_writer_settings.available_checkers:
            issues.append(f"Default checker '{checker_id}' not found in available checkers")
    
    # Validate quality threshold
    if not 0 <= multi_writer_settings.quality_threshold <= 100:
        issues.append("Quality threshold must be between 0 and 100")
    
    # Validate max iterations
    if multi_writer_settings.max_iterations < 1:
        issues.append("Max iterations must be at least 1")
    
    return issues


def initialize_multi_writer_system():
    """Initialize the multi-writer system if enabled"""
    if not multi_writer_settings.enabled:
        logger.info("Multi-writer system is disabled")
        return None
    
    # Validate configuration
    issues = validate_multi_writer_config()
    if issues:
        logger.error(f"Multi-writer configuration issues: {'; '.join(issues)}")
        return None
    
    # Create default templates if they don't exist
    from app.core.templating.jinja_processor import create_default_templates
    from pathlib import Path
    
    template_dir = Path(multi_writer_settings.template_dir)
    create_default_templates(template_dir)
    
    logger.info("Multi-writer system initialized successfully")
    return get_multi_writer_config()