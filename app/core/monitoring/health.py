"""
Health monitoring system for the AI Assistant Tool System.

This module provides comprehensive health checks for:
- Tool registry health
- Agent registry health
- External service dependencies
- System resource monitoring
- Health status aggregation
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import psutil
import requests
from enum import Enum

from .config import monitoring_config

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(str, Enum):
    """Health check type enumeration"""

    TOOL_REGISTRY = "tool_registry"
    AGENT_REGISTRY = "agent_registry"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCES = "system_resources"
    DATABASE = "database"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    check_type: HealthCheckType
    check_name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time: float


@dataclass
class SystemHealth:
    """Overall system health status"""

    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime
    uptime_seconds: float
    system_info: Dict[str, Any]


class HealthMonitor:
    """
    Health monitoring system for the tool system.

    This class performs health checks on various system components and
    provides an overall health status for the system.
    """

    def __init__(self):
        self._health_checks: Dict[str, Callable] = {}
        self._last_health_check: Optional[SystemHealth] = None
        self._start_time = time.time()
        self._health_check_interval = monitoring_config.health_check_interval
        self._health_check_timeout = monitoring_config.health_check_timeout

        # Register default health checks
        self._register_default_checks()

        logger.info("HealthMonitor initialized")

    def _register_default_checks(self):
        """Register default health checks"""
        self.register_health_check(
            "tool_registry", HealthCheckType.TOOL_REGISTRY, self._check_tool_registry
        )

        self.register_health_check(
            "agent_registry", HealthCheckType.AGENT_REGISTRY, self._check_agent_registry
        )

        self.register_health_check(
            "system_resources",
            HealthCheckType.SYSTEM_RESOURCES,
            self._check_system_resources,
        )

        self.register_health_check(
            "openrouter_api",
            HealthCheckType.EXTERNAL_SERVICE,
            self._check_openrouter_api,
        )

    def register_health_check(
        self, name: str, check_type: HealthCheckType, check_func: Callable
    ):
        """Register a custom health check"""
        self._health_checks[name] = {"type": check_type, "function": check_func}
        logger.info(f"Registered health check: {name} ({check_type.value})")

    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        if name in self._health_checks:
            del self._health_checks[name]
            logger.info(f"Unregistered health check: {name}")

    async def perform_health_check(self) -> SystemHealth:
        """Perform all registered health checks and return overall health status"""
        check_results = []

        # Run health checks concurrently
        tasks = []
        for name, check_info in self._health_checks.items():
            task = self._run_single_check(
                name, check_info["type"], check_info["function"]
            )
            tasks.append(task)

        # Wait for all checks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed with exception: {result}")
                continue
            if result:
                check_results.append(result)

        # Determine overall health status
        overall_status = self._determine_overall_health(check_results)

        # Get system info
        system_info = self._get_system_info()

        health_status = SystemHealth(
            status=overall_status,
            checks=check_results,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self._start_time,
            system_info=system_info,
        )

        self._last_health_check = health_status
        return health_status

    async def _run_single_check(
        self, name: str, check_type: HealthCheckType, check_func: Callable
    ) -> Optional[HealthCheckResult]:
        """Run a single health check with timeout"""
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(check_func), timeout=self._health_check_timeout
            )

            response_time = time.time() - start_time

            return HealthCheckResult(
                check_type=check_type,
                check_name=name,
                status=result["status"],
                message=result["message"],
                details=result.get("details", {}),
                timestamp=datetime.now(),
                response_time=response_time,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Health check '{name}' timed out")
            return HealthCheckResult(
                check_type=check_type,
                check_name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                details={"timeout_seconds": self._health_check_timeout},
                timestamp=datetime.now(),
                response_time=self._health_check_timeout,
            )
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheckResult(
                check_type=check_type,
                check_name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
            )

    def _determine_overall_health(
        self, check_results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Determine overall health status based on individual check results"""
        if not check_results:
            return HealthStatus.UNKNOWN

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0,
        }

        for result in check_results:
            status_counts[result.status] += 1

        # Decision logic for overall health
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] == len(check_results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_used_mb": round(memory_info.rss / 1024 / 1024, 2),
                "memory_percent": process.memory_percent(),
                "thread_count": process.num_threads(),
                "open_files": (
                    len(process.open_files()) if hasattr(process, "open_files") else 0
                ),
                "disk_usage": {
                    "total_gb": round(
                        psutil.disk_usage("/").total / 1024 / 1024 / 1024, 2
                    ),
                    "used_gb": round(
                        psutil.disk_usage("/").used / 1024 / 1024 / 1024, 2
                    ),
                    "free_gb": round(
                        psutil.disk_usage("/").free / 1024 / 1024 / 1024, 2
                    ),
                    "percent": psutil.disk_usage("/").percent,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def _check_tool_registry(self) -> Dict[str, Any]:
        """Check tool registry health"""
        try:
            from ..tools.execution.registry import tool_registry

            stats = tool_registry.get_registry_stats()
            total_tools = stats["total_tools"]
            enabled_tools = stats["enabled_tools"]

            if total_tools == 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No tools registered in the system",
                    "details": stats,
                }

            if enabled_tools == 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No tools are currently enabled",
                    "details": stats,
                }

            return {
                "status": HealthStatus.HEALTHY,
                "message": f"Tool registry healthy ({enabled_tools}/{total_tools} tools enabled)",
                "details": stats,
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Tool registry check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_agent_registry(self) -> Dict[str, Any]:
        """Check agent registry health"""
        try:
            from ..agents.management.registry import agent_registry

            stats = agent_registry.get_registry_stats()
            total_agents = stats["total_agents"]
            active_agents = stats["active_agents"]

            if total_agents == 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No agents registered in the system",
                    "details": stats,
                }

            if active_agents == 0:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No agents are currently active",
                    "details": stats,
                }

            return {
                "status": HealthStatus.HEALTHY,
                "message": f"Agent registry healthy ({active_agents}/{total_agents} agents active)",
                "details": stats,
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Agent registry check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage("/").percent

            # Define thresholds
            cpu_threshold = 90.0
            memory_threshold = 85.0
            disk_threshold = 90.0

            issues = []

            if cpu_percent > cpu_threshold:
                issues.append(f"CPU usage high: {cpu_percent}%")

            if memory_percent > memory_threshold:
                issues.append(f"Memory usage high: {memory_percent}%")

            if disk_percent > disk_threshold:
                issues.append(f"Disk usage high: {disk_percent}%")

            if issues:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "System resources under pressure",
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent,
                        "issues": issues,
                    },
                }

            return {
                "status": HealthStatus.HEALTHY,
                "message": "System resources within normal limits",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                },
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"System resource check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_openrouter_api(self) -> Dict[str, Any]:
        """Check OpenRouter API connectivity"""
        try:
            from ..config import settings

            if not settings.openrouter_api_key:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "OpenRouter API key not configured",
                    "details": {"configured": False},
                }

            # Simple connectivity check - try to list models
            response = requests.get(
                f"{settings.openrouter_base_url}/models",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=5,
            )

            if response.status_code == 200:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "OpenRouter API connectivity verified",
                    "details": {
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                    },
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"OpenRouter API returned status {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                    },
                }

        except requests.exceptions.Timeout:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": "OpenRouter API request timed out",
                "details": {"timeout_seconds": 5},
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"OpenRouter API check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def get_last_health_check(self) -> Optional[SystemHealth]:
        """Get the last health check result"""
        return self._last_health_check

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of the current health status"""
        if not self._last_health_check:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(time.time() - self._start_time, 2),
            }

        return {
            "status": self._last_health_check.status.value,
            "message": f"System is {self._last_health_check.status.value}",
            "timestamp": self._last_health_check.timestamp.isoformat(),
            "uptime_seconds": round(self._last_health_check.uptime_seconds, 2),
            "checks_performed": len(self._last_health_check.checks),
            "system_info": self._last_health_check.system_info,
        }


# Global health monitor instance
health_monitor = HealthMonitor()
