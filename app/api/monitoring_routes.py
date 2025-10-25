"""
Monitoring API routes for the AI Assistant Tool System.

This module provides endpoints for:
- Health checks and system status
- Metrics collection and Prometheus metrics
- Performance statistics
- System monitoring data
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, PlainTextResponse
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.monitoring.health import health_monitor, HealthStatus
from app.core.monitoring.metrics import metrics_collector
from app.core.monitoring.config import monitoring_config
from app.core.langchain.monitoring import LangChainMonitoring

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health", summary="System Health Check")
async def health_check(
    detailed: bool = Query(False, description="Return detailed health information"),
) -> Dict[str, Any]:
    """
    Perform a comprehensive health check of the system.

    Returns overall system health status along with individual component checks.
    """
    if not monitoring_config.monitoring_enabled:
        return {
            "status": HealthStatus.UNKNOWN.value,
            "message": "Monitoring system is disabled",
            "monitoring_enabled": False,
        }

    health_status = await health_monitor.perform_health_check()

    if detailed:
        return {
            "status": health_status.status.value,
            "timestamp": health_status.timestamp.isoformat(),
            "uptime_seconds": round(health_status.uptime_seconds, 2),
            "checks": [
                {
                    "name": check.check_name,
                    "type": check.check_type.value,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": round(check.response_time, 3),
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details,
                }
                for check in health_status.checks
            ],
            "system_info": health_status.system_info,
        }
    else:
        return {
            "status": health_status.status.value,
            "message": f"System is {health_status.status.value}",
            "timestamp": health_status.timestamp.isoformat(),
            "uptime_seconds": round(health_status.uptime_seconds, 2),
            "checks_performed": len(health_status.checks),
            "healthy_checks": len(
                [c for c in health_status.checks if c.status == HealthStatus.HEALTHY]
            ),
            "degraded_checks": len(
                [c for c in health_status.checks if c.status == HealthStatus.DEGRADED]
            ),
            "unhealthy_checks": len(
                [c for c in health_status.checks if c.status == HealthStatus.UNHEALTHY]
            ),
        }


@router.get("/health/summary", summary="Health Summary")
async def health_summary() -> Dict[str, Any]:
    """
    Get a summary of the current system health status.

    This endpoint provides a lightweight health check without performing
    new health checks (uses cached results).
    """
    return health_monitor.get_health_summary()


@router.get("/metrics", summary="Prometheus Metrics")
async def get_metrics() -> Response:
    """
    Get Prometheus metrics in text format.

    This endpoint exposes metrics in Prometheus format for scraping by
    monitoring systems like Prometheus or Grafana.
    """
    if not monitoring_config.monitoring_enabled:
        return PlainTextResponse("# Monitoring system is disabled\n", status_code=503)

    if monitoring_config.metrics_backend != "prometheus":
        return PlainTextResponse(
            f"# Prometheus metrics backend is not enabled (current: {monitoring_config.metrics_backend})\n",
            status_code=503,
        )

    try:
        metrics_data = metrics_collector.get_prometheus_metrics()
        return Response(content=metrics_data, media_type="text/plain; version=0.0.4")
    except Exception as e:
        return PlainTextResponse(
            f"# Error retrieving metrics: {str(e)}\n", status_code=500
        )


@router.get("/metrics/summary", summary="Metrics Summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of collected metrics.

    This endpoint provides a JSON summary of key metrics for easy consumption
    by monitoring dashboards and alerting systems.
    """
    if not monitoring_config.monitoring_enabled:
        return {"monitoring_enabled": False, "message": "Monitoring system is disabled"}

    system_summary = metrics_collector.get_system_summary()
    tool_metrics = metrics_collector.get_tool_metrics()
    agent_metrics = metrics_collector.get_agent_metrics()

    return {
        "monitoring_enabled": True,
        "system_summary": system_summary,
        "tools_registered": len(tool_metrics),
        "agents_registered": len(agent_metrics),
        "collection_timestamp": system_summary.get("timestamp", "unknown"),
        "metrics_backend": monitoring_config.metrics_backend.value,
    }


@router.get("/metrics/tools", summary="Tool Metrics")
async def get_tool_metrics(
    tool_name: Optional[str] = Query(
        None, description="Specific tool name to filter by"
    ),
) -> Dict[str, Any]:
    """
    Get detailed metrics for tools.

    Returns performance metrics for all tools or a specific tool if provided.
    """
    if not monitoring_config.monitoring_enabled:
        return {"monitoring_enabled": False, "message": "Monitoring system is disabled"}

    metrics = metrics_collector.get_tool_metrics(tool_name)

    if tool_name and not metrics:
        raise HTTPException(
            status_code=404, detail=f"No metrics found for tool: {tool_name}"
        )

    return {
        "monitoring_enabled": True,
        "tool_name": tool_name if tool_name else "all",
        "metrics": metrics,
        "total_tools_tracked": len(metrics) if not tool_name else 1,
    }


@router.get("/metrics/agents", summary="Agent Metrics")
async def get_agent_metrics(
    agent_name: Optional[str] = Query(
        None, description="Specific agent name to filter by"
    ),
) -> Dict[str, Any]:
    """
    Get detailed metrics for agents.

    Returns performance metrics for all agents or a specific agent if provided.
    """
    if not monitoring_config.monitoring_enabled:
        return {"monitoring_enabled": False, "message": "Monitoring system is disabled"}

    metrics = metrics_collector.get_agent_metrics(agent_name)

    if agent_name and not metrics:
        raise HTTPException(
            status_code=404, detail=f"No metrics found for agent: {agent_name}"
        )

    return {
        "monitoring_enabled": True,
        "agent_name": agent_name if agent_name else "all",
        "metrics": metrics,
        "total_agents_tracked": len(metrics) if not agent_name else 1,
    }


@router.get("/status", summary="System Status")
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status information.

    This endpoint combines health, metrics, and configuration information
    to provide a complete overview of the system status.
    """
    from app.core.tools.execution.registry import tool_registry
    from app.core.agents.management.registry import agent_registry
    from app.core.config import settings

    # Get health summary
    health_summary = health_monitor.get_health_summary()

    # Get metrics summary
    metrics_summary = {}
    if monitoring_config.monitoring_enabled:
        metrics_summary = metrics_collector.get_system_summary()

    # Get registry information
    tool_stats = tool_registry.get_registry_stats()
    agent_stats = (
        agent_registry.get_registry_stats() if settings.agent_system_enabled else {}
    )

    return {
        "system": {
            "version": "1.0.0",  # This should come from app/__init__.py
            "environment": settings.environment,
            "monitoring_enabled": monitoring_config.monitoring_enabled,
            "tool_system_enabled": settings.tool_system_enabled,
            "agent_system_enabled": settings.agent_system_enabled,
        },
        "health": health_summary,
        "metrics": metrics_summary,
        "registries": {"tools": tool_stats, "agents": agent_stats},
        "monitoring_config": {
            "level": monitoring_config.monitoring_level.value,
            "metrics_backend": monitoring_config.metrics_backend.value,
            "performance_tracking": monitoring_config.performance_tracking_enabled,
        },
    }


@router.get("/config", summary="Monitoring Configuration")
async def get_monitoring_config() -> Dict[str, Any]:
    """
    Get the current monitoring configuration.

    This endpoint returns the monitoring configuration settings for
    debugging and configuration verification purposes.
    """
    return {
        "monitoring_enabled": monitoring_config.monitoring_enabled,
        "monitoring_level": monitoring_config.monitoring_level.value,
        "metrics_backend": monitoring_config.metrics_backend.value,
        "metrics_collection_interval": monitoring_config.metrics_collection_interval,
        "metrics_retention_days": monitoring_config.metrics_retention_days,
        "health_check_interval": monitoring_config.health_check_interval,
        "health_check_timeout": monitoring_config.health_check_timeout,
        "performance_tracking_enabled": monitoring_config.performance_tracking_enabled,
        "track_tool_performance": monitoring_config.track_tool_performance,
        "track_agent_performance": monitoring_config.track_agent_performance,
        "track_api_performance": monitoring_config.track_api_performance,
        "alerting_enabled": monitoring_config.alerting_enabled,
        "prometheus_port": monitoring_config.prometheus_port,
        "prometheus_path": monitoring_config.prometheus_path,
        "log_metrics": monitoring_config.log_metrics,
        "log_level": monitoring_config.log_level,
    }


@router.post("/monitoring/reset", summary="Reset Monitoring Data")
async def reset_monitoring_data() -> Dict[str, Any]:
    """
    Reset all monitoring data (metrics and health checks).

    WARNING: This will clear all collected metrics and health check history.
    Use with caution in production environments.
    """
    # This is a placeholder - in a real implementation, we would reset metrics
    # For now, we'll just return a message since our metrics are in-memory
    return {
        "message": "Monitoring reset functionality not yet implemented",
        "warning": "This would clear all collected metrics and health data",
        "status": "not_implemented",
    }


# LangChain Monitoring Endpoints

@router.get("/langchain/status", summary="LangChain Component Status")
async def get_langchain_status() -> Dict[str, Any]:
    """
    Get status of all LangChain components.
    
    This endpoint provides information about the status of LangChain LLM Manager,
    Tool Registry, Agent Manager, and Memory Manager.
    """
    try:
        langchain_monitoring = LangChainMonitoring()
        await langchain_monitoring.initialize()
        
        # Get component status
        llm_status = await langchain_monitoring.get_component_status("llm")
        tool_status = await langchain_monitoring.get_component_status("tool")
        agent_status = await langchain_monitoring.get_component_status("agent")
        memory_status = await langchain_monitoring.get_component_status("memory")
        workflow_status = await langchain_monitoring.get_component_status("workflow")
        
        return {
            "monitoring_enabled": True,
            "components": {
                "llm_manager": llm_status,
                "tool_registry": tool_status,
                "agent_manager": agent_status,
                "memory_manager": memory_status,
                "workflow_manager": workflow_status
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "monitoring_enabled": False,
            "error": str(e),
            "message": "Failed to get LangChain component status"
        }


@router.get("/langchain/metrics", summary="LangChain Metrics")
async def get_langchain_metrics(
    component_type: Optional[str] = Query(
        None, description="Filter by component type (llm, tool, agent, memory, workflow)"
    ),
    component_name: Optional[str] = Query(
        None, description="Filter by specific component name"
    ),
    metric_name: Optional[str] = Query(
        None, description="Filter by specific metric name"
    ),
    limit: int = Query(100, description="Maximum number of metrics to return")
) -> Dict[str, Any]:
    """
    Get LangChain component metrics.
    
    This endpoint provides detailed metrics for LangChain components, including
    performance data, error rates, and usage statistics.
    """
    try:
        langchain_monitoring = LangChainMonitoring()
        await langchain_monitoring.initialize()
        
        # Get metrics with filters
        metrics = await langchain_monitoring.get_metrics(
            component_type=component_type,
            component_name=component_name,
            metric_name=metric_name,
            limit=limit
        )
        
        # Get summary statistics
        summary = await langchain_monitoring.get_metrics_summary(
            component_type=component_type,
            component_name=component_name
        )
        
        return {
            "monitoring_enabled": True,
            "filters": {
                "component_type": component_type,
                "component_name": component_name,
                "metric_name": metric_name,
                "limit": limit
            },
            "metrics": metrics,
            "summary": summary,
            "total_metrics": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "monitoring_enabled": False,
            "error": str(e),
            "message": "Failed to get LangChain metrics"
        }


@router.get("/langchain/performance", summary="LangChain Performance Data")
async def get_langchain_performance(
    component_type: Optional[str] = Query(
        None, description="Filter by component type"
    ),
    component_name: Optional[str] = Query(
        None, description="Filter by specific component name"
    ),
    time_range: str = Query("1h", description="Time range (1h, 6h, 24h, 7d)")
) -> Dict[str, Any]:
    """
    Get LangChain performance data.
    
    This endpoint provides performance metrics including response times,
    throughput, and error rates over the specified time range.
    """
    try:
        langchain_monitoring = LangChainMonitoring()
        await langchain_monitoring.initialize()
        
        # Get performance data
        performance_data = await langchain_monitoring.get_performance_data(
            component_type=component_type,
            component_name=component_name,
            time_range=time_range
        )
        
        return {
            "monitoring_enabled": True,
            "filters": {
                "component_type": component_type,
                "component_name": component_name,
                "time_range": time_range
            },
            "performance_data": performance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "monitoring_enabled": False,
            "error": str(e),
            "message": "Failed to get LangChain performance data"
        }


@router.get("/langchain/health", summary="LangChain Health Check")
async def get_langchain_health() -> Dict[str, Any]:
    """
    Perform health check on LangChain components.
    
    This endpoint checks the health and availability of all LangChain components
    and returns detailed health information.
    """
    try:
        langchain_monitoring = LangChainMonitoring()
        await langchain_monitoring.initialize()
        
        # Perform health check
        health_data = await langchain_monitoring.perform_health_check()
        
        return {
            "monitoring_enabled": True,
            "health": health_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "monitoring_enabled": False,
            "error": str(e),
            "message": "Failed to perform LangChain health check"
        }


@router.post("/langchain/metrics/clear", summary="Clear LangChain Metrics")
async def clear_langchain_metrics(
    component_type: Optional[str] = Query(
        None, description="Clear metrics for specific component type"
    ),
    component_name: Optional[str] = Query(
        None, description="Clear metrics for specific component name"
    )
) -> Dict[str, Any]:
    """
    Clear LangChain metrics.
    
    WARNING: This will permanently delete the specified metrics.
    Use with caution in production environments.
    """
    try:
        langchain_monitoring = LangChainMonitoring()
        await langchain_monitoring.initialize()
        
        # Clear metrics
        cleared_count = await langchain_monitoring.clear_metrics(
            component_type=component_type,
            component_name=component_name
        )
        
        return {
            "monitoring_enabled": True,
            "cleared_metrics_count": cleared_count,
            "filters": {
                "component_type": component_type,
                "component_name": component_name
            },
            "message": f"Successfully cleared {cleared_count} metrics",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "monitoring_enabled": False,
            "error": str(e),
            "message": "Failed to clear LangChain metrics"
        }
