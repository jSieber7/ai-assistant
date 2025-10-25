"""
Unit tests for LangChain Monitoring System.

This module tests metric collection, performance tracking,
and integration with database persistence.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.core.langchain.monitoring import LangChainMonitoring
from app.core.secure_settings import secure_settings


class TestLangChainMonitoring:
    """Test cases for LangChain Monitoring System"""
    
    @pytest.fixture
    async def monitoring(self):
        """Create a LangChain Monitoring instance for testing"""
        monitor = LangChainMonitoring()
        await monitor.initialize()
        return monitor
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'monitoring_enabled'): True,
            ('langchain', 'metrics_retention_days'): 30,
            ('langchain', 'performance_tracking'): True,
        }.get((section, key), default)
        return mock_settings
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock database client for testing"""
        db_client = Mock()
        db_client.save_metric = AsyncMock()
        db_client.get_metrics = AsyncMock(return_value=[])
        db_client.delete_old_metrics = AsyncMock(return_value=10)
        return db_client
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of monitoring system"""
        with patch('app.core.langchain.monitoring.secure_settings', mock_settings):
            with patch('app.core.langchain.monitoring.get_langchain_client') as mock_get_client:
                mock_get_client.return_value = Mock()
                
                monitor = LangChainMonitoring()
                
                # Test initialization
                await monitor.initialize()
                
                # Verify initialization
                assert monitor._initialized is True
                assert monitor._db_client is not None
                assert isinstance(monitor._metrics, dict)
    
    async def test_record_metric_success(self, monitoring, mock_db_client):
        """Test successful metric recording"""
        monitoring._db_client = mock_db_client
        
        # Record a metric
        await monitoring.record_metric(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            metric_name="request_count",
            metric_value=1,
            metadata={"model": "gpt-3.5-turbo"}
        )
        
        # Verify metric was saved
        mock_db_client.save_metric.assert_called_once()
        call_args = mock_db_client.save_metric.call_args[1]
        
        assert call_args["component_type"] == "llm"
        assert call_args["component_name"] == "gpt-3.5-turbo"
        assert call_args["metric_name"] == "request_count"
        assert call_args["metric_value"] == 1
        assert call_args["metadata"]["model"] == "gpt-3.5-turbo"
    
    async def test_record_metric_with_validation(self, monitoring, mock_db_client):
        """Test metric recording with validation"""
        monitoring._db_client = mock_db_client
        
        # Test invalid component type
        with pytest.raises(ValueError, match="Invalid component type"):
            await monitoring.record_metric(
                component_type="invalid_type",
                component_name="test",
                metric_name="test_metric",
                metric_value=1
            )
        
        # Test invalid metric value
        with pytest.raises(ValueError, match="Metric value must be numeric"):
            await monitoring.record_metric(
                component_type="llm",
                component_name="test",
                metric_name="test_metric",
                metric_value="invalid"
            )
    
    async def test_get_metrics(self, monitoring, mock_db_client):
        """Test getting metrics with filters"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_metrics = [
            {
                "id": 1,
                "component_type": "llm",
                "component_name": "gpt-3.5-turbo",
                "metric_name": "request_count",
                "metric_value": 10,
                "metadata": {"model": "gpt-3.5-turbo"},
                "created_at": datetime.now()
            },
            {
                "id": 2,
                "component_type": "tool",
                "component_name": "web_search",
                "metric_name": "execution_count",
                "metric_value": 5,
                "metadata": {"tool": "web_search"},
                "created_at": datetime.now()
            }
        ]
        mock_db_client.get_metrics.return_value = mock_metrics
        
        # Get metrics with filters
        metrics = await monitoring.get_metrics(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            limit=10
        )
        
        # Verify filters were applied
        mock_db_client.get_metrics.assert_called_once_with(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            metric_name=None,
            limit=10
        )
        
        # Verify returned metrics
        assert len(metrics) == 2
    
    async def test_get_metrics_summary(self, monitoring, mock_db_client):
        """Test getting metrics summary"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_summary = {
            "total_metrics": 100,
            "metrics_by_type": {
                "llm": 50,
                "tool": 30,
                "agent": 20
            },
            "metrics_by_name": {
                "request_count": 50,
                "execution_count": 30,
                "success_count": 20
            }
        }
        mock_db_client.get_metrics_summary.return_value = mock_summary
        
        # Get summary
        summary = await monitoring.get_metrics_summary(
            component_type="llm",
            component_name="gpt-3.5-turbo"
        )
        
        # Verify filters were applied
        mock_db_client.get_metrics_summary.assert_called_once_with(
            component_type="llm",
            component_name="gpt-3.5-turbo"
        )
        
        # Verify summary
        assert summary["total_metrics"] == 100
        assert summary["metrics_by_type"]["llm"] == 50
        assert summary["metrics_by_name"]["request_count"] == 50
    
    async def test_get_component_status(self, monitoring, mock_db_client):
        """Test getting component status"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_status = {
            "component_type": "llm",
            "component_name": "gpt-3.5-turbo",
            "status": "healthy",
            "last_activity": datetime.now(),
            "error_count": 0,
            "success_count": 100
        }
        mock_db_client.get_component_status.return_value = mock_status
        
        # Get component status
        status = await monitoring.get_component_status("llm")
        
        # Verify call was made
        mock_db_client.get_component_status.assert_called_once_with(
            component_type="llm",
            component_name=None
        )
        
        # Verify status
        assert status["component_type"] == "llm"
        assert status["status"] == "healthy"
        assert status["error_count"] == 0
        assert status["success_count"] == 100
    
    async def test_get_performance_data(self, monitoring, mock_db_client):
        """Test getting performance data"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_performance = [
            {
                "timestamp": datetime.now() - timedelta(hours=1),
                "avg_response_time": 0.5,
                "request_count": 10,
                "error_rate": 0.0
            },
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "avg_response_time": 0.7,
                "request_count": 15,
                "error_rate": 0.1
            }
        ]
        mock_db_client.get_performance_data.return_value = mock_performance
        
        # Get performance data
        performance = await monitoring.get_performance_data(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            time_range="1h"
        )
        
        # Verify call was made
        mock_db_client.get_performance_data.assert_called_once_with(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            time_range="1h"
        )
        
        # Verify performance data
        assert len(performance) == 2
        assert performance[0]["avg_response_time"] == 0.5
        assert performance[1]["error_rate"] == 0.1
    
    async def test_perform_health_check(self, monitoring, mock_db_client):
        """Test performing health check"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_health = {
            "overall_status": "healthy",
            "components": {
                "llm": {"status": "healthy", "last_check": datetime.now()},
                "tool": {"status": "healthy", "last_check": datetime.now()},
                "agent": {"status": "degraded", "last_check": datetime.now()}
            },
            "timestamp": datetime.now()
        }
        mock_db_client.get_health_status.return_value = mock_health
        
        # Perform health check
        health = await monitoring.perform_health_check()
        
        # Verify call was made
        mock_db_client.get_health_status.assert_called_once()
        
        # Verify health data
        assert health["overall_status"] == "healthy"
        assert len(health["components"]) == 3
        assert health["components"]["llm"]["status"] == "healthy"
        assert health["components"]["agent"]["status"] == "degraded"
    
    async def test_clear_metrics(self, monitoring, mock_db_client):
        """Test clearing metrics"""
        monitoring._db_client = mock_db_client
        
        # Clear metrics with filters
        cleared_count = await monitoring.clear_metrics(
            component_type="llm",
            component_name="gpt-3.5-turbo"
        )
        
        # Verify call was made
        mock_db_client.delete_old_metrics.assert_called_once_with(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            older_than_days=0
        )
        
        # Verify cleared count
        assert cleared_count == 10
    
    async def test_cleanup_old_metrics(self, monitoring, mock_db_client):
        """Test cleaning up old metrics"""
        monitoring._db_client = mock_db_client
        
        # Clean up metrics older than 7 days
        cleaned_count = await monitoring.cleanup_old_metrics(days=7)
        
        # Verify call was made
        mock_db_client.delete_old_metrics.assert_called_once_with(
            component_type=None,
            component_name=None,
            older_than_days=7
        )
        
        # Verify cleaned count
        assert cleaned_count == 10
    
    async def test_track_request_context_manager(self, monitoring, mock_db_client):
        """Test request tracking context manager"""
        monitoring._db_client = mock_db_client
        
        # Use context manager
        with monitoring.track_request(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            metadata={"model": "gpt-3.5-turbo"}
        ):
            # Simulate some work
            await asyncio.sleep(0.1)
        
        # Verify metrics were recorded
        assert mock_db_client.save_metric.call_count >= 2  # start and end metrics
    
    async def test_track_request_context_manager_with_error(self, monitoring, mock_db_client):
        """Test request tracking context manager with error"""
        monitoring._db_client = mock_db_client
        
        # Use context manager with error
        try:
            with monitoring.track_request(
                component_type="llm",
                component_name="gpt-3.5-turbo"
            ):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected error
        
        # Verify error metric was recorded
        call_args_list = [call[1] for call in mock_db_client.save_metric.call_args_list]
        error_metrics = [args for args in call_args_list if args.get("metric_name") == "error"]
        
        assert len(error_metrics) > 0
        assert error_metrics[0]["component_type"] == "llm"
        assert error_metrics[0]["component_name"] == "gpt-3.5-turbo"
        assert "Test error" in error_metrics[0]["metadata"]["error"]
    
    async def test_shutdown(self, monitoring):
        """Test shutdown functionality"""
        # Mock database client shutdown
        monitoring._db_client = Mock()
        monitoring._db_client.close = AsyncMock()
        
        await monitoring.shutdown()
        
        # Verify shutdown was called
        monitoring._db_client.close.assert_called_once()
        
        # Verify monitoring is marked as not initialized
        assert monitoring._initialized is False
    
    async def test_concurrent_metric_recording(self, monitoring, mock_db_client):
        """Test concurrent metric recording"""
        monitoring._db_client = mock_db_client
        
        # Record metrics concurrently
        tasks = [
            monitoring.record_metric(
                component_type="llm",
                component_name="gpt-3.5-turbo",
                metric_name="request_count",
                metric_value=1
            )
            for i in range(10)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all metrics were recorded
        assert mock_db_client.save_metric.call_count == 10
    
    async def test_metric_aggregation(self, monitoring, mock_db_client):
        """Test metric aggregation functionality"""
        monitoring._db_client = mock_db_client
        
        # Record multiple metrics for same component
        for i in range(5):
            await monitoring.record_metric(
                component_type="llm",
                component_name="gpt-3.5-turbo",
                metric_name="response_time",
                metric_value=0.1 + (i * 0.1)
            )
        
        # Get aggregated metrics
        metrics = await monitoring.get_metrics(
            component_type="llm",
            component_name="gpt-3.5-turbo",
            metric_name="response_time"
        )
        
        # Verify metrics were recorded
        assert mock_db_client.save_metric.call_count == 5
    
    async def test_time_range_validation(self, monitoring, mock_db_client):
        """Test time range validation"""
        monitoring._db_client = mock_db_client
        
        # Test invalid time range
        with pytest.raises(ValueError, match="Invalid time range"):
            await monitoring.get_performance_data(
                component_type="llm",
                time_range="invalid_range"
            )
        
        # Test valid time ranges
        for time_range in ["1h", "6h", "24h", "7d"]:
            # Should not raise exception
            await monitoring.get_performance_data(
                component_type="llm",
                time_range=time_range
            )
    
    async def test_get_statistics(self, monitoring, mock_db_client):
        """Test getting monitoring statistics"""
        monitoring._db_client = mock_db_client
        
        # Mock database response
        mock_stats = {
            "total_metrics": 1000,
            "metrics_by_component_type": {
                "llm": 400,
                "tool": 300,
                "agent": 200,
                "memory": 100
            },
            "oldest_metric": datetime.now() - timedelta(days=30),
            "newest_metric": datetime.now(),
            "retention_days": 30
        }
        mock_db_client.get_monitoring_statistics.return_value = mock_stats
        
        # Get statistics
        stats = await monitoring.get_statistics()
        
        # Verify call was made
        mock_db_client.get_monitoring_statistics.assert_called_once()
        
        # Verify statistics
        assert stats["total_metrics"] == 1000
        assert stats["metrics_by_component_type"]["llm"] == 400
        assert "oldest_metric" in stats
        assert "newest_metric" in stats
    
    async def test_error_handling_during_recording(self, monitoring, mock_db_client):
        """Test error handling during metric recording"""
        # Mock database to raise error
        mock_db_client.save_metric.side_effect = Exception("Database error")
        monitoring._db_client = mock_db_client
        
        # Record metric should handle error gracefully
        with pytest.raises(Exception, match="Database error"):
            await monitoring.record_metric(
                component_type="llm",
                component_name="gpt-3.5-turbo",
                metric_name="request_count",
                metric_value=1
            )
    
    async def test_disabled_monitoring(self, mock_settings):
        """Test behavior when monitoring is disabled"""
        # Mock settings to disable monitoring
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'monitoring_enabled'): False,
        }.get((section, key), default)
        
        with patch('app.core.langchain.monitoring.secure_settings', mock_settings):
            monitor = LangChainMonitoring()
            await monitor.initialize()
            
            # Try to record metric
            await monitor.record_metric(
                component_type="llm",
                component_name="gpt-3.5-turbo",
                metric_name="request_count",
                metric_value=1
            )
            
            # Should not record anything when disabled
            assert monitor._db_client is None


if __name__ == "__main__":
    pytest.main([__file__])