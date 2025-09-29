import pytest
import subprocess
import time
import requests
import signal
import os
from pathlib import Path


@pytest.fixture
def test_server_process():
    """Start the FastAPI server in a subprocess for integration testing."""
    # Set test environment variables
    env = os.environ.copy()
    env['OPENROUTER_API_KEY'] = 'test-key-integration'
    env['ENVIRONMENT'] = 'testing'
    
    # Start the server
    process = subprocess.Popen(
        ['uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8001'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent  # Project root directory
    )
    
    # Wait for server to start
    time.sleep(3)
    
    yield process
    
    # Cleanup - terminate the server
    process.terminate()
    process.wait(timeout=5)


class TestApplicationIntegration:
    """Integration tests for the FastAPI application."""
    
    def test_server_starts_successfully(self, test_server_process):
        """Test that the server starts without errors."""
        # Check if process is still running
        assert test_server_process.poll() is None, "Server process terminated unexpectedly"
        
        # Try to connect to the server
        try:
            response = requests.get('http://127.0.0.1:8001/', timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.fail("Server failed to start or is not accessible")
    
    def test_health_endpoint_integration(self, test_server_process):
        """Test health endpoint in integrated environment."""
        response = requests.get('http://127.0.0.1:8001/health', timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
    
    def test_models_endpoint_integration(self, test_server_process):
        """Test models endpoint in integrated environment."""
        response = requests.get('http://127.0.0.1:8001/v1/models', timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data['object'] == 'list'
        assert len(data['data']) > 0


class TestConfigurationIntegration:
    """Test configuration and environment setup."""
    
    def test_environment_variables_loading(self):
        """Test that environment variables are properly loaded."""
        from app.core.config import Settings
        
        # Test with mock environment
        with pytest.MonkeyPatch().context() as m:
            m.setenv('OPENROUTER_API_KEY', 'test-key-123')
            m.setenv('DEFAULT_MODEL', 'test-model')
            
            settings = Settings()
            assert settings.openrouter_api_key.get_secret_value() == 'test-key-123'
            assert settings.default_model == 'test-model'
    
    def test_missing_api_key_handling(self):
        """Test error handling when API key is missing."""
        # Test that get_llm raises ValueError when API key is missing
        from unittest.mock import patch
        from app.core.config import get_llm
        
        # Mock the settings to have no API key
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.openrouter_api_key = None
            
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
                get_llm()


class TestDependencyInjection:
    """Test dependency injection and module imports."""
    
    def test_module_imports(self):
        """Test that all modules can be imported without errors."""
        # Test main application imports
        from app.main import app
        assert app is not None
        
        # Test API routes imports
        from app.api.routes import router
        assert router is not None
        
        # Test config imports
        from app.core.config import settings, get_llm
        assert settings is not None
        assert callable(get_llm)
    
    def test_fastapi_app_structure(self):
        """Test FastAPI app structure and routes."""
        from app.main import app
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/health', '/v1/models', '/v1/chat/completions']
        
        for expected_route in expected_routes:
            assert expected_route in routes, f"Route {expected_route} not found in app"
        
        # Check CORS middleware
        assert any('CORSMiddleware' in str(middleware) for middleware in app.user_middleware)


@pytest.mark.slow
class TestPerformance:
    """Performance-related integration tests."""
    
    def test_response_time(self, client):
        """Test that endpoints respond within acceptable time limits."""
        import time
        
        # Test root endpoint response time
        start_time = time.time()
        response = client.get('/')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_chat_completion_performance(self, client, chat_request_data, mock_llm):
        """Test chat completion response time."""
        import time
        
        start_time = time.time()
        response = client.post('/v1/chat/completions', json=chat_request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_server_recovery_after_error(self, test_server_process):
        """Test that server recovers after an error condition."""
        # Send a malformed request that might cause an error
        malformed_data = '{"invalid": "json"'
        try:
            response = requests.post(
                'http://127.0.0.1:8001/v1/chat/completions',
                data=malformed_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            # Server should handle the error gracefully
            assert response.status_code in [400, 422, 500]
        except requests.exceptions.RequestException:
            # Server might close connection, but should restart cleanly
            pass
        
        # Verify server is still responsive after error
        time.sleep(1)  # Give server time to recover
        response = requests.get('http://127.0.0.1:8001/health', timeout=5)
        assert response.status_code == 200


@pytest.mark.skip(reason="Requires real OpenRouter API key")
class TestRealAPIIntegration:
    """Integration tests with real OpenRouter API (requires API key)."""
    
    def test_real_openrouter_connection(self):
        """Test actual connection to OpenRouter API."""
        # This test requires a real API key and should be run separately
        from app.core.config import get_llm
        from langchain.schema import HumanMessage
        
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Say hello")])
        
        assert response is not None
        assert hasattr(response, 'content')
        assert isinstance(response.content, str)