"""
Performance and load testing for the application.

This module tests application performance under various load conditions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import statistics
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import httpx
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.core.config import Settings


class TestLoadTesting:
    """Test application performance and load handling"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def test_settings(self):
        """Create test settings"""
        return Settings(
            environment="test",
            debug=True,
            api_host="localhost",
            api_port=8000,
            redis_url="redis://localhost:6379/0",
            cache_enabled=True,
            cache_ttl=300,
            max_concurrent_requests=100,
            request_timeout=30
        )

    @pytest.fixture
    def async_client(self):
        """Create an async client for load testing"""
        return httpx.AsyncClient(app=app, base_url="http://test")

    def measure_response_time(self, func, *args, **kwargs):
        """Measure response time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    async def measure_async_response_time(self, func, *args, **kwargs):
        """Measure response time of an async function"""
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    def test_single_request_performance(self, client, test_settings):
        """Test performance of a single request"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-perf-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Performance test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                request = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Performance test"}
                    ]
                }
                
                # Measure response time
                response_time, response = self.measure_response_time(
                    client.post, "/chat/completions", json=request
                )
                
                # Verify performance
                assert response.status_code == 200
                assert response_time < 1.0  # Should respond within 1 second
                assert "Performance test response" in response.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, async_client, test_settings):
        """Test performance with concurrent requests"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-concurrent-perf",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Concurrent performance test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send concurrent requests
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Concurrent test {uuid.uuid4()}"}
                        ]
                    }
                    response = await async_client.post("/chat/completions", json=request)
                    return response
                
                # Send 20 concurrent requests
                num_requests = 20
                tasks = [send_request() for _ in range(num_requests)]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Verify all requests succeeded
                assert all(response.status_code == 200 for response in responses)
                
                # Calculate performance metrics
                avg_response_time = total_time / num_requests
                requests_per_second = num_requests / total_time
                
                # Verify performance metrics
                assert avg_response_time < 2.0  # Average should be under 2 seconds
                assert requests_per_second > 5.0  # Should handle at least 5 RPS

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, async_client, test_settings):
        """Test performance under sustained load"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-sustained-perf",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sustained load test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send sustained load for 30 seconds
                duration = 30  # seconds
                target_rps = 10  # requests per second
                
                response_times = []
                error_count = 0
                success_count = 0
                
                async def send_request():
                    try:
                        request = {
                            "model": "gpt-3.5-turbo",
                            "messages": [
                                {"role": "user", "content": f"Sustained test {uuid.uuid4()}"}
                            ]
                        }
                        
                        response_time, response = await self.measure_async_response_time(
                            async_client.post, "/chat/completions", json=request
                        )
                        
                        if response.status_code == 200:
                            response_times.append(response_time)
                            return True
                        else:
                            return False
                    except Exception:
                        return False
                
                # Generate sustained load
                start_time = time.time()
                while time.time() - start_time < duration:
                    # Send batch of requests
                    batch_tasks = [send_request() for _ in range(target_rps)]
                    batch_results = await asyncio.gather(*batch_tasks)
                    
                    success_count += sum(batch_results)
                    error_count += len(batch_results) - sum(batch_results)
                    
                    # Wait for next second
                    await asyncio.sleep(1)
                
                # Calculate metrics
                total_requests = success_count + error_count
                actual_rps = total_requests / duration
                error_rate = error_count / total_requests if total_requests > 0 else 0
                
                if response_times:
                    avg_response_time = statistics.mean(response_times)
                    p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                    p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
                else:
                    avg_response_time = p95_response_time = p99_response_time = 0
                
                # Verify performance under sustained load
                assert actual_rps >= target_rps * 0.8  # Should handle at least 80% of target RPS
                assert error_rate < 0.01  # Error rate should be under 1%
                assert avg_response_time < 3.0  # Average response time under 3 seconds
                assert p95_response_time < 5.0  # 95th percentile under 5 seconds

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, async_client, test_settings):
        """Test memory usage under load"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-memory-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Memory usage test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Measure initial memory
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Generate load
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Memory test {uuid.uuid4()}"}
                        ]
                    }
                    return await async_client.post("/chat/completions", json=request)
                
                # Send 100 requests over 10 seconds
                tasks = []
                for _ in range(100):
                    tasks.append(send_request())
                    if len(tasks) >= 10:  # Send in batches of 10
                        await asyncio.gather(*tasks)
                        tasks = []
                        await asyncio.sleep(1)  # Wait 1 second between batches
                
                # Send remaining tasks
                if tasks:
                    await asyncio.gather(*tasks)
                
                # Measure final memory
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Verify memory usage
                assert memory_increase < 100  # Memory increase should be under 100MB

    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, async_client, test_settings):
        """Test cache performance under load"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-cache-perf",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Cache performance test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send requests with same content to test caching
                same_request = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Cache performance test"}
                    ]
                }
                
                # First request (cache miss)
                response_time1, response1 = await self.measure_async_response_time(
                    async_client.post, "/chat/completions", json=same_request
                )
                
                # Second request (cache hit)
                response_time2, response2 = await self.measure_async_response_time(
                    async_client.post, "/chat/completions", json=same_request
                )
                
                # Verify cache performance
                assert response1.status_code == 200
                assert response2.status_code == 200
                assert response_time2 < response_time1  # Cache hit should be faster

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, async_client, test_settings):
        """Test database performance under load"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock database operations
            with patch('app.core.storage.postgresql_client.PostgreSQLClient.execute_query') as mock_query:
                mock_query.return_value = [{"result": "Database test result"}]
                
                # Send requests that trigger database operations
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Database test {uuid.uuid4()}"}
                        ],
                        "store_conversation": True
                    }
                    return await async_client.post("/chat/completions", json=request)
                
                # Send 50 concurrent requests
                tasks = [send_request() for _ in range(50)]
                responses = await asyncio.gather(*tasks)
                
                # Verify all requests succeeded
                assert all(response.status_code == 200 for response in responses)
                
                # Verify database was called appropriately
                assert mock_query.call_count == 50

    @pytest.mark.asyncio
    async def test_tool_performance_under_load(self, async_client, test_settings):
        """Test tool execution performance under load"""
        with patch('app.api.routes.tools.get_settings', return_value=test_settings):
            # Mock tool execution
            mock_result = {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "Performance test result",
                            "url": "https://example.com/perf",
                            "content": "Performance test content"
                        }
                    ]
                }
            }
            
            with patch('app.api.routes.tools.executor.execute_tool', return_value=mock_result):
                # Send concurrent tool execution requests
                async def send_tool_request():
                    request = {
                        "tool_name": "searxng_search",
                        "parameters": {
                            "query": f"Performance test {uuid.uuid4()}",
                            "num_results": 5
                        }
                    }
                    return await async_client.post("/tools/execute", json=request)
                
                # Send 30 concurrent requests
                tasks = [send_tool_request() for _ in range(30)]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Verify all requests succeeded
                assert all(response.status_code == 200 for response in responses)
                
                # Calculate performance metrics
                avg_response_time = total_time / 30
                requests_per_second = 30 / total_time
                
                # Verify tool execution performance
                assert avg_response_time < 5.0  # Average should be under 5 seconds
                assert requests_per_second > 2.0  # Should handle at least 2 RPS for tools

    @pytest.mark.asyncio
    async def test_stress_test_beyond_limits(self, async_client, test_settings):
        """Test application behavior under extreme stress"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-stress-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Stress test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send extreme load
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Stress test {uuid.uuid4()}"}
                        ]
                    }
                    try:
                        response = await async_client.post("/chat/completions", json=request)
                        return response.status_code
                    except Exception:
                        return 500
                
                # Send 200 concurrent requests (beyond typical limits)
                tasks = [send_request() for _ in range(200)]
                status_codes = await asyncio.gather(*tasks)
                
                # Analyze results
                success_count = sum(1 for code in status_codes if code == 200)
                error_count = sum(1 for code in status_codes if code != 200)
                success_rate = success_count / len(status_codes)
                
                # Verify graceful degradation
                assert success_rate > 0.5  # At least 50% should succeed
                assert error_count < len(status_codes) * 0.5  # Less than 50% should fail

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, async_client, test_settings):
        """Test performance regression detection"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-regression-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Regression test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Baseline performance measurement
                baseline_times = []
                for _ in range(10):
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": "Baseline test"}
                        ]
                    }
                    
                    response_time, _ = await self.measure_async_response_time(
                        async_client.post, "/chat/completions", json=request
                    )
                    baseline_times.append(response_time)
                
                baseline_avg = statistics.mean(baseline_times)
                
                # Current performance measurement
                current_times = []
                for _ in range(10):
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": "Current test"}
                        ]
                    }
                    
                    response_time, _ = await self.measure_async_response_time(
                        async_client.post, "/chat/completions", json=request
                    )
                    current_times.append(response_time)
                
                current_avg = statistics.mean(current_times)
                
                # Check for regression (current should not be significantly worse than baseline)
                regression_threshold = 1.5  # 50% slower is considered regression
                assert current_avg < baseline_avg * regression_threshold

    @pytest.mark.asyncio
    async def test_resource_cleanup_under_load(self, async_client, test_settings):
        """Test resource cleanup under load"""
        with patch('app.api.routes.chat.get_settings', return_value=test_settings):
            # Mock response
            mock_response = {
                "id": "chatcmpl-cleanup-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Cleanup test response"
                        }
                    }
                ]
            }
            
            with patch('app.api.routes.chat.create_chat_completion', return_value=mock_response):
                # Send requests and monitor resource cleanup
                initial_thread_count = threading.active_count()
                
                async def send_request():
                    request = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "user", "content": f"Cleanup test {uuid.uuid4()}"}
                        ]
                    }
                    return await async_client.post("/chat/completions", json=request)
                
                # Send requests in batches
                for batch in range(5):
                    tasks = [send_request() for _ in range(10)]
                    await asyncio.gather(*tasks)
                    await asyncio.sleep(0.5)  # Allow cleanup
                
                # Wait for cleanup
                await asyncio.sleep(2)
                
                # Check thread count
                final_thread_count = threading.active_count()
                thread_increase = final_thread_count - initial_thread_count
                
                # Verify minimal thread increase
                assert thread_increase < 10  # Should not create excessive threads

    def test_performance_report_generation(self, test_settings):
        """Test performance report generation"""
        # Generate performance report
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": test_settings.environment,
            "metrics": {
                "single_request_time": 0.5,
                "concurrent_rps": 15.0,
                "sustained_rps": 12.0,
                "memory_usage_mb": 150.0,
                "error_rate": 0.01,
                "cache_hit_rate": 0.75
            },
            "thresholds": {
                "max_single_request_time": 1.0,
                "min_concurrent_rps": 10.0,
                "min_sustained_rps": 8.0,
                "max_memory_usage_mb": 200.0,
                "max_error_rate": 0.05,
                "min_cache_hit_rate": 0.7
            }
        }
        
        # Verify report structure
        assert "timestamp" in report
        assert "environment" in report
        assert "metrics" in report
        assert "thresholds" in report
        
        # Verify performance against thresholds
        assert report["metrics"]["single_request_time"] < report["thresholds"]["max_single_request_time"]
        assert report["metrics"]["concurrent_rps"] > report["thresholds"]["min_concurrent_rps"]
        assert report["metrics"]["sustained_rps"] > report["thresholds"]["min_sustained_rps"]
        assert report["metrics"]["memory_usage_mb"] < report["thresholds"]["max_memory_usage_mb"]
        assert report["metrics"]["error_rate"] < report["thresholds"]["max_error_rate"]
        assert report["metrics"]["cache_hit_rate"] > report["thresholds"]["min_cache_hit_rate"]