"""
Configuration management for the AI Assistant tool system.

This module provides configuration settings for the tool system, including
global settings and tool-specific configurations.
"""

from pydantic_settings import BaseSettings


class ToolSystemSettings(BaseSettings):
    """Configuration for the tool system"""

    # Global tool system settings
    tool_calling_enabled: bool = True
    max_tools_per_query: int = 3
    tool_timeout_seconds: int = 30
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 300

    # Individual tool settings
    calculator_tool_enabled: bool = True
    time_tool_enabled: bool = True
    echo_tool_enabled: bool = True

    # Tool-specific configurations
    calculator_precision: int = 10
    default_timezone: str = "UTC"

    class Config:
        env_prefix = "TOOL_"
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


tool_settings = ToolSystemSettings()
