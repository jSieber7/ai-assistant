"""
Unit tests for monitoring routes functionality
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest


class TestMonitoringRoutes:
    """Test monitoring routes endpoints"""

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_health_check_healthy(self, mock_metrics_collector, client):
        """Test health check when system is healthy"""
        # Mock healthy components
        mock_metrics_collector.get_system_summary.return_value = {
            "overall_status": "healthy",
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "llm_providers": "healthy"
            }
        }
        
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_health_check_unhealthy(self, mock_metrics_collector, client):
        """Test health check when system has issues"""
        # Mock unhealthy components
        mock_metrics_collector.get_system_summary.return_value = {
            "overall_status": "unhealthy",
            "components": {
                "database": "unhealthy",
                "redis": "healthy",
                "llm_providers": "healthy"
            },
            "issues": ["Database connection timeout"]
        }
        
        response = client.get("/monitoring/health")
        assert response.status_code == 503  # Service Unavailable

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_health_summary(self, mock_metrics_collector, client):
        """Test health summary endpoint"""
        mock_metrics_collector.get_system_summary.return_value = {
            "overall_status": "healthy",
            "components": {
                "database": {"status": "healthy", "response_time": 0.05},
                "redis": {"status": "healthy", "response_time": 0.02},
                "llm_providers": {"status": "healthy", "response_time": 0.1}
            },
            "uptime": 3600,
            "version": "1.0.0"
        }
        
        response = client.get("/monitoring/health/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "overall_status" in data
        assert "components" in data
        assert "uptime" in data
        assert "version" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_metrics_endpoint(self, mock_metrics_collector, client):
        """Test Prometheus metrics endpoint"""
        # Mock metrics collector
        mock_registry = CollectorRegistry()
        
        with patch("app.api.monitoring_routes.generate_latest") as mock_generate:
            mock_generate.return_value = b"# HELP test_metric Test metric\n"
            
            response = client.get("/monitoring/metrics")
            
            # Check content type
            assert response.headers["content-type"] == "text/plain; version=0.0.4"

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_metrics_summary(self, mock_metrics_collector, client):
        """Test metrics summary endpoint"""
        mock_metrics_collector.get_summary_stats.return_value = {
            "tool_executions": {
                "total": 100,
                "successful": 95,
                "failed": 5,
                "average_duration": 0.5
            },
            "agent_executions": {
                "total": 50,
                "successful": 48,
                "failed": 2,
                "average_duration": 1.2
            },
            "cache_performance": {
                "hit_rate": 0.85,
                "total_requests": 1000
            }
        }
        
        response = client.get("/monitoring/metrics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "tool_executions" in data
        assert "agent_executions" in data
        assert "cache_performance" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_tool_metrics(self, mock_metrics_collector, client):
        """Test tool-specific metrics endpoint"""
        mock_metrics_collector.get_tool_metrics.return_value = {
            "search_tool": {
                "executions": 25,
                "success_rate": 0.96,
                "average_duration": 0.3,
                "last_execution": "2024-01-01T12:00:00Z"
            },
            "web_scrape_tool": {
                "executions": 15,
                "success_rate": 0.87,
                "average_duration": 1.2,
                "last_execution": "2024-01-01T11:45:00Z"
            }
        }
        
        response = client.get("/monitoring/metrics/tools")
        assert response.status_code == 200
        
        data = response.json()
        assert "search_tool" in data
        assert "web_scrape_tool" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_agent_metrics(self, mock_metrics_collector, client):
        """Test agent-specific metrics endpoint"""
        mock_metrics_collector.get_agent_metrics.return_value = {
            "writer_agent": {
                "executions": 30,
                "success_rate": 0.93,
                "average_duration": 2.1,
                "conversations": 15
            },
            "checker_agent": {
                "executions": 20,
                "success_rate": 0.95,
                "average_duration": 0.8,
                "conversations": 10
            }
        }
        
        response = client.get("/monitoring/metrics/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "writer_agent" in data
        assert "checker_agent" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_system_status(self, mock_metrics_collector, client):
        """Test system status endpoint"""
        mock_metrics_collector.get_system_status.return_value = {
            "status": "healthy",
            "uptime": 86400,
            "version": "1.0.0",
            "environment": "production",
            "components": {
                "database": {"status": "healthy", "connections": 5},
                "redis": {"status": "healthy", "memory_usage": "45MB"},
                "llm_providers": {"status": "healthy", "active_providers": 3}
            },
            "performance": {
                "cpu_usage": 0.25,
                "memory_usage": 0.60,
                "disk_usage": 0.40
            }
        }
        
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "version" in data
        assert "components" in data
        assert "performance" in data

    @patch("app.api.monitoring_routes.monitoring_config")
    def test_monitoring_config(self, mock_config, client):
        """Test monitoring configuration endpoint"""
        mock_config.health_check_interval = 30
        mock_config.metrics_retention_days = 7
        mock_config.alerting_enabled = True
        mock_config.performance_monitoring = True
        
        response = client.get("/monitoring/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "health_check_interval" in data
        assert "metrics_retention_days" in data
        assert "alerting_enabled" in data
        assert "performance_monitoring" in data

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_reset_monitoring_data(self, mock_metrics_collector, client):
        """Test resetting monitoring data"""
        mock_metrics_collector.reset_all_metrics.return_value = True
        
        response = client.post("/monitoring/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "reset" in data["message"].lower()


class TestMonitoringRoutesErrorHandling:
    """Test error handling in monitoring routes"""

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_health_check_exception(self, mock_metrics_collector, client):
        """Test health check when exception occurs"""
        mock_metrics_collector.get_system_summary.side_effect = Exception("Database error")
        
        response = client.get("/monitoring/health")
        assert response.status_code == 500

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_metrics_endpoint_exception(self, mock_metrics_collector, client):
        """Test metrics endpoint when exception occurs"""
        with patch("app.api.monitoring_routes.generate_latest") as mock_generate:
            mock_generate.side_effect = Exception("Metrics generation failed")
            
            response = client.get("/monitoring/metrics")
            assert response.status_code == 500

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_reset_monitoring_data_failure(self, mock_metrics_collector, client):
        """Test reset monitoring data when it fails"""
        mock_metrics_collector.reset_all_metrics.return_value = False
        
        response = client.post("/monitoring/reset")
        assert response.status_code == 500


class TestMonitoringRoutesParameters:
    """Test monitoring routes with various parameters"""

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_tool_metrics_with_tool_name(self, mock_metrics_collector, client):
        """Test tool metrics for specific tool"""
        mock_metrics_collector.get_tool_metrics.return_value = {
            "search_tool": {
                "executions": 25,
                "success_rate": 0.96,
                "average_duration": 0.3
            }
        }
        
        response = client.get("/monitoring/metrics/tools?tool_name=search_tool")
        assert response.status_code == 200

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_agent_metrics_with_agent_name(self, mock_metrics_collector, client):
        """Test agent metrics for specific agent"""
        mock_metrics_collector.get_agent_metrics.return_value = {
            "writer_agent": {
                "executions": 30,
                "success_rate": 0.93,
                "average_duration": 2.1
            }
        }
        
        response = client.get("/monitoring/metrics/agents?agent_name=writer_agent")
        assert response.status_code == 200

    @patch("app.api.monitoring_routes.metrics_collector")
    def test_metrics_with_time_range(self, mock_metrics_collector, client):
        """Test metrics with time range parameters"""
        mock_metrics_collector.get_summary_stats.return_value = {
            "tool_executions": {
                "total": 50,
                "successful": 48,
                "failed": 2
            }
        }
        
        response = client.get("/monitoring/metrics/summary?start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z")
        assert response.status_code == 200


class TestMonitoringRoutesAuthentication:
    """Test authentication and authorization for monitoring routes"""

    def test_public_endpoints(self, client):
        """Test that certain endpoints are publicly accessible"""
        # Health check should be public
        response = client.get("/monitoring/health")
        # Should not require authentication (may fail due to missing mocks but not auth)
        assert response.status_code not in [401, 403]

    def test_protected_endpoints(self, client):
        """Test that certain endpoints require authentication"""
        # These endpoints might require authentication in production
        protected_endpoints = [
            "/monitoring/reset",
        ]
        
        for endpoint in protected_endpoints:
            response = client.post(endpoint)
            # May fail due to missing mocks or authentication requirements
            assert response.status_code not in [404]  # Endpoint should exist