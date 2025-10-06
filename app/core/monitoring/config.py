"""
Monitoring configuration for the AI Assistant Tool System.

This module provides configuration settings for the monitoring system,
including metrics collection, health checks, and performance tracking.
"""

from enum import Enum
from typing import Dict, Any
from pydantic import BaseSettings, Field


class MonitoringLevel(str, Enum):
    """Monitoring detail levels"""

    BASIC = "basic"  # Minimal monitoring - essential metrics only
    STANDARD = "standard"  # Standard monitoring - most common metrics
    DETAILED = "detailed"  # Detailed monitoring - all metrics including performance


class MetricsBackend(str, Enum):
    """Supported metrics backends"""

    PROMETHEUS = "prometheus"  # Prometheus metrics format
    INFLUXDB = "influxdb"  # InfluxDB line protocol
    CUSTOM = "custom"  # Custom JSON format


class MonitoringConfig(BaseSettings):
    """
    Configuration for the monitoring system.

    These settings control what is monitored, how often, and where the data is stored.
    """

    # Global monitoring settings
    monitoring_enabled: bool = Field(
        True, description="Enable/disable the entire monitoring system"
    )
    monitoring_level: MonitoringLevel = Field(
        MonitoringLevel.STANDARD, description="Level of monitoring detail"
    )

    # Metrics collection settings
    metrics_backend: MetricsBackend = Field(
        MetricsBackend.PROMETHEUS, description="Metrics storage backend"
    )
    metrics_collection_interval: int = Field(
        30, description="Metrics collection interval in seconds"
    )
    metrics_retention_days: int = Field(7, description="How long to keep metrics data")

    # Health check settings
    health_check_interval: int = Field(
        60, description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(10, description="Health check timeout in seconds")

    # Performance tracking settings
    performance_tracking_enabled: bool = Field(
        True, description="Enable performance tracking"
    )
    track_tool_performance: bool = Field(
        True, description="Track tool execution performance"
    )
    track_agent_performance: bool = Field(
        True, description="Track agent execution performance"
    )
    track_api_performance: bool = Field(
        True, description="Track API request/response performance"
    )

    # Alerting settings
    alerting_enabled: bool = Field(False, description="Enable alerting system")
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "api_error_rate": 0.1,  # 10% error rate threshold
            "tool_timeout_rate": 0.05,  # 5% timeout rate threshold
            "agent_failure_rate": 0.1,  # 10% failure rate threshold
            "response_time_p95": 5.0,  # 5 seconds P95 response time
        },
        description="Alert thresholds for various metrics",
    )

    # Prometheus-specific settings
    prometheus_port: int = Field(9090, description="Prometheus metrics port")
    prometheus_path: str = Field(
        "/metrics", description="Prometheus metrics endpoint path"
    )

    # Logging settings
    log_metrics: bool = Field(False, description="Log metrics to application logs")
    log_level: str = Field("INFO", description="Log level for monitoring system")

    # External service monitoring
    external_services: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "openrouter": {
                "enabled": True,
                "health_check_url": "https://openrouter.ai/api/v1/models",
                "timeout": 5,
            },
            "github": {
                "enabled": True,
                "health_check_url": "https://api.github.com",
                "timeout": 5,
            },
        },
        description="External services to monitor for health checks",
    )

    # Custom metrics configuration
    custom_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metrics configuration"
    )

    class Config:
        env_prefix = "monitoring_"
        case_sensitive = False


# Global monitoring configuration instance
monitoring_config = MonitoringConfig()


def get_monitoring_config() -> MonitoringConfig:
    """Get the current monitoring configuration"""
    return monitoring_config


def update_monitoring_config(**kwargs) -> MonitoringConfig:
    """Update monitoring configuration with new values"""
    global monitoring_config

    # Create a new config with updated values
    updated_config = monitoring_config.copy(update=kwargs)
    monitoring_config = updated_config

    return monitoring_config


def disable_monitoring() -> None:
    """Disable the entire monitoring system"""
    global monitoring_config
    monitoring_config = monitoring_config.copy(update={"monitoring_enabled": False})


def enable_monitoring() -> None:
    """Enable the entire monitoring system"""
    global monitoring_config
    monitoring_config = monitoring_config.copy(update={"monitoring_enabled": True})


def set_monitoring_level(level: MonitoringLevel) -> None:
    """Set the monitoring detail level"""
    global monitoring_config
    monitoring_config = monitoring_config.copy(update={"monitoring_level": level})


def configure_for_environment(environment: str) -> MonitoringConfig:
    """
    Configure monitoring settings based on environment.

    Args:
        environment: Deployment environment (development, staging, production)

    Returns:
        Configured monitoring settings
    """
    global monitoring_config

    if environment == "production":
        config_updates = {
            "monitoring_enabled": True,
            "monitoring_level": MonitoringLevel.DETAILED,
            "alerting_enabled": True,
            "log_metrics": False,  # Don't log metrics in production to avoid noise
            "metrics_retention_days": 30,  # Keep metrics longer in production
        }
    elif environment == "staging":
        config_updates = {
            "monitoring_enabled": True,
            "monitoring_level": MonitoringLevel.STANDARD,
            "alerting_enabled": True,
            "log_metrics": True,
            "metrics_retention_days": 7,
        }
    else:  # development
        config_updates = {
            "monitoring_enabled": True,
            "monitoring_level": MonitoringLevel.BASIC,
            "alerting_enabled": False,
            "log_metrics": True,
            "metrics_retention_days": 1,
        }

    monitoring_config = monitoring_config.copy(update=config_updates)
    return monitoring_config
