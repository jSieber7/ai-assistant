"""
Tests for the monitoring and metrics system.

This module contains tests for:
- Metrics collection functionality
- Health monitoring system
- Monitoring API endpoints
- Integration with tool and agent systems
"""

import pytest
import asyncio
import time
from fastapi.testclient import TestClient

from app.main import app
from app.core.monitoring.metrics import MetricsCollector
from app.core.monitoring.health import HealthMonitor, HealthStatus, HealthCheckType
from app.core.monitoring.config import monitoring_config, MonitoringLevel
from app.core.tools.base.base import BaseTool
from app.core.agents.base.base import BaseAgent, AgentResult


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collection functionality"""

    def setup_method(self):
        """Set up a fresh metrics collector for each test"""
        self.metrics_collector = MetricsCollector()

    def test_record_tool_execution_success(self):
        """Test recording successful tool execution"""
        tool_name = "test_tool"
        execution_time = 0.5

        self.metrics_collector.record_tool_execution(
            tool_name=tool_name, success=True, execution_time=execution_time
        )

        metrics = self.metrics_collector.get_tool_metrics(tool_name)
        assert metrics["tool_name"] == tool_name
        assert metrics["execution_count"] == 1
        assert metrics["success_count"] == 1
        assert metrics["failure_count"] == 0
        assert metrics["total_execution_time"] == execution_time

    def test_record_tool_execution_failure(self):
        """Test recording failed tool execution"""
        tool_name = "test_tool"
        execution_time = 0.3
        error_type = "TimeoutError"

        self.metrics_collector.record_tool_execution(
            tool_name=tool_name,
            success=False,
            execution_time=execution_time,
            error_type=error_type,
        )

        metrics = self.metrics_collector.get_tool_metrics(tool_name)
        assert metrics["execution_count"] == 1
        assert metrics["success_count"] == 0
        assert metrics["failure_count"] == 1
        assert metrics["error_types"][error_type] == 1

    def test_record_agent_execution(self):
        """Test recording agent execution metrics"""
        agent_name = "test_agent"
        execution_time = 1.2
        tools_used = ["tool1", "tool2"]

        self.metrics_collector.record_agent_execution(
            agent_name=agent_name,
            success=True,
            execution_time=execution_time,
            tools_used=tools_used,
        )

        metrics = self.metrics_collector.get_agent_metrics(agent_name)
        assert metrics["agent_name"] == agent_name
        assert metrics["execution_count"] == 1
        assert metrics["success_count"] == 1
        assert metrics["tools_used"]["tool1"] == 1
        assert metrics["tools_used"]["tool2"] == 1

    def test_measure_tool_execution_context_manager(self):
        """Test the context manager for tool execution measurement"""
        tool_name = "test_tool"

        with self.metrics_collector.measure_tool_execution(tool_name):
            time.sleep(0.1)  # Simulate some work

        metrics = self.metrics_collector.get_tool_metrics(tool_name)
        assert metrics["execution_count"] == 1
        assert metrics["success_count"] == 1
        assert metrics["total_execution_time"] > 0

    def test_measure_tool_execution_context_manager_exception(self):
        """Test context manager with exception handling"""
        tool_name = "test_tool"

        with pytest.raises(ValueError):
            with self.metrics_collector.measure_tool_execution(tool_name):
                raise ValueError("Test exception")

        metrics = self.metrics_collector.get_tool_metrics(tool_name)
        assert metrics["execution_count"] == 1
        assert metrics["success_count"] == 0
        assert metrics["failure_count"] == 1

    def test_get_system_summary(self):
        """Test system summary generation"""
        # Add some metrics
        self.metrics_collector.record_tool_execution("tool1", True, 0.5)
        self.metrics_collector.record_tool_execution("tool2", False, 0.3)
        self.metrics_collector.record_agent_execution("agent1", True, 1.0)

        summary = self.metrics_collector.get_system_summary()

        assert summary["total_tool_executions"] == 2
        assert summary["total_agent_executions"] == 1
        assert 0 <= summary["tool_success_rate"] <= 100
        assert 0 <= summary["agent_success_rate"] <= 100

    def test_prometheus_metrics_generation(self):
        """Test Prometheus metrics format generation"""
        # Add some metrics
        self.metrics_collector.record_tool_execution("test_tool", True, 0.5)

        metrics_data = self.metrics_collector.get_prometheus_metrics()

        assert isinstance(metrics_data, bytes)
        assert len(metrics_data) > 0
        # Check for expected metric names in the output
        metrics_text = metrics_data.decode("utf-8")
        assert "tool_execution_total" in metrics_text
        assert "tool_execution_duration_seconds" in metrics_text


@pytest.mark.unit
class TestHealthMonitor:
    """Test health monitoring functionality"""

    def setup_method(self):
        """Set up a fresh health monitor for each test"""
        self.health_monitor = HealthMonitor()

    @pytest.mark.asyncio
    async def test_perform_health_check(self):
        """Test performing a complete health check"""
        health_status = await self.health_monitor.perform_health_check()

        assert health_status.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]
        assert len(health_status.checks) > 0
        assert health_status.uptime_seconds > 0
        assert isinstance(health_status.timestamp, type(health_status.timestamp))

    def test_register_custom_health_check(self):
        """Test registering a custom health check"""

        def custom_check():
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Custom check passed",
                "details": {"custom": "data"},
            }

        self.health_monitor.register_health_check(
            "custom_check", HealthCheckType.EXTERNAL_SERVICE, custom_check
        )

        assert "custom_check" in self.health_monitor._health_checks

    def test_unregister_health_check(self):
        """Test unregistering a health check"""

        # First register a check
        def custom_check():
            return {"status": HealthStatus.HEALTHY, "message": "test"}

        self.health_monitor.register_health_check(
            "test_check", HealthCheckType.EXTERNAL_SERVICE, custom_check
        )

        # Then unregister it
        self.health_monitor.unregister_health_check("test_check")
        assert "test_check" not in self.health_monitor._health_checks

    def test_determine_overall_health(self):
        """Test overall health status determination logic"""
        from app.core.monitoring.health import HealthCheckResult

        # Test with all healthy checks
        healthy_checks = [
            HealthCheckResult(
                check_type=HealthCheckType.TOOL_REGISTRY,
                check_name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=type(
                    "MockDateTime", (), {"isoformat": lambda: "2023-01-01T00:00:00"}
                )(),
                response_time=0.1,
            )
        ]

        status = self.health_monitor._determine_overall_health(healthy_checks)
        assert status == HealthStatus.HEALTHY

        # Test with unhealthy checks
        unhealthy_checks = [
            HealthCheckResult(
                check_type=HealthCheckType.TOOL_REGISTRY,
                check_name="test",
                status=HealthStatus.UNHEALTHY,
                message="Failed",
                details={},
                timestamp=type(
                    "MockDateTime", (), {"isoformat": lambda: "2023-01-01T00:00:00"}
                )(),
                response_time=0.1,
            )
        ]

        status = self.health_monitor._determine_overall_health(unhealthy_checks)
        assert status == HealthStatus.UNHEALTHY

    def test_get_system_info(self):
        """Test system information collection"""
        system_info = self.health_monitor._get_system_info()

        # Check that we get some system metrics
        assert isinstance(system_info, dict)
        # Should have at least some of these keys
        expected_keys = ["cpu_percent", "memory_used_mb", "memory_percent"]
        assert any(key in system_info for key in expected_keys)


@pytest.mark.unit
class TestMonitoringAPI:
    """Test monitoring API endpoints"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/monitoring/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data

    def test_health_check_detailed(self):
        """Test detailed health check endpoint"""
        response = self.client.get("/monitoring/health?detailed=true")

        assert response.status_code == 200
        data = response.json()
        assert "checks" in data
        assert "system_info" in data

    def test_health_summary_endpoint(self):
        """Test health summary endpoint"""
        response = self.client.get("/monitoring/health/summary")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = self.client.get("/monitoring/metrics")

        # Should return either metrics or a service unavailable status
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            assert "text/plain" in response.headers["content-type"]

    def test_metrics_summary_endpoint(self):
        """Test metrics summary endpoint"""
        response = self.client.get("/monitoring/metrics/summary")

        assert response.status_code == 200
        data = response.json()
        assert "monitoring_enabled" in data
        assert "system_summary" in data

    def test_tool_metrics_endpoint(self):
        """Test tool metrics endpoint"""
        response = self.client.get("/monitoring/metrics/tools")

        assert response.status_code == 200
        data = response.json()
        assert "monitoring_enabled" in data
        assert "metrics" in data

    def test_agent_metrics_endpoint(self):
        """Test agent metrics endpoint"""
        response = self.client.get("/monitoring/metrics/agents")

        assert response.status_code == 200
        data = response.json()
        assert "monitoring_enabled" in data
        assert "metrics" in data

    def test_system_status_endpoint(self):
        """Test system status endpoint"""
        response = self.client.get("/monitoring/status")

        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "health" in data
        assert "metrics" in data
        assert "registries" in data

    def test_monitoring_config_endpoint(self):
        """Test monitoring configuration endpoint"""
        response = self.client.get("/monitoring/config")

        assert response.status_code == 200
        data = response.json()
        assert "monitoring_enabled" in data
        assert "monitoring_level" in data
        assert "metrics_backend" in data


@pytest.mark.unit
class TestMonitoringIntegration:
    """Test monitoring system integration with tools and agents"""

    def test_tool_integration(self):
        """Test that tools integrate with monitoring system"""

        # Mock tool that uses monitoring
        class MockTool(BaseTool):
            @property
            def name(self):
                return "mock_tool"

            @property
            def description(self):
                return "A mock tool for testing"

            async def execute(self, **kwargs):
                return {"result": "success"}

        tool = MockTool()

        # Test that the tool can be executed with monitoring
        # This tests the integration in BaseTool.execute_with_timeout
        async def test_tool_execution():
            result = await tool.execute_with_timeout(query="test")
            return result

        # Run the async test
        result = asyncio.run(test_tool_execution())
        assert result.success is True
        assert result.tool_name == "mock_tool"
        assert result.execution_time > 0

    def test_agent_integration(self):
        """Test that agents integrate with monitoring system"""

        # Mock agent that uses monitoring
        class MockAgent(BaseAgent):
            @property
            def name(self):
                return "mock_agent"

            @property
            def description(self):
                return "A mock agent for testing"

            async def _process_message_impl(
                self, message, conversation_id=None, context=None
            ):
                return AgentResult(
                    success=True,
                    response="Mock response",
                    agent_name=self.name,
                    execution_time=0.1,
                    conversation_id=conversation_id,
                )

        from app.core.tools.execution.registry import ToolRegistry

        agent = MockAgent(ToolRegistry())

        # Test agent message processing with monitoring
        async def test_agent_execution():
            result = await agent.process_message("test message")
            return result

        result = asyncio.run(test_agent_execution())
        assert result.success is True
        assert result.agent_name == "mock_agent"


@pytest.mark.unit
class TestMonitoringConfiguration:
    """Test monitoring configuration functionality"""

    def test_configuration_loading(self):
        """Test that monitoring configuration loads correctly"""
        config = monitoring_config

        assert hasattr(config, "monitoring_enabled")
        assert hasattr(config, "monitoring_level")
        assert hasattr(config, "metrics_backend")
        assert hasattr(config, "performance_tracking_enabled")

    def test_configuration_environment_variables(self):
        """Test that configuration respects environment variables"""
        # This test would require setting environment variables
        # For now, we'll just test that the configuration object is properly structured
        config = monitoring_config

        # Check that the configuration has the expected attributes
        expected_attributes = [
            "monitoring_enabled",
            "monitoring_level",
            "metrics_backend",
            "metrics_collection_interval",
            "health_check_interval",
            "performance_tracking_enabled",
            "track_tool_performance",
            "track_agent_performance",
            "track_api_performance",
        ]

        for attr in expected_attributes:
            assert hasattr(config, attr)

    def test_configuration_environment_specific(self):
        """Test environment-specific configuration"""
        from app.core.monitoring.config import configure_for_environment

        # Test production configuration
        prod_config = configure_for_environment("production")
        assert prod_config.monitoring_enabled is True
        assert prod_config.monitoring_level == MonitoringLevel.DETAILED
        assert prod_config.alerting_enabled is True

        # Test development configuration
        dev_config = configure_for_environment("development")
        assert dev_config.monitoring_enabled is True
        assert dev_config.monitoring_level == MonitoringLevel.BASIC
        assert dev_config.alerting_enabled is False


@pytest.mark.skip(reason="Requires specific environment setup")
class TestMonitoringInProduction:
    """Tests that require production-like environment setup"""

    def test_prometheus_integration(self):
        """Test full Prometheus integration (requires Prometheus server)"""
        # This test would require a running Prometheus instance
        # For now, we'll skip it but document what it would test
        pass

    def test_alerting_system(self):
        """Test alerting system functionality (requires alerting setup)"""
        # This test would require alerting infrastructure
        pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
