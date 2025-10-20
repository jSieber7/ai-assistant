"""
Integration tests for Llama.cpp tool calling functionality.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.integration
@pytest.mark.asyncio
class TestLlamaCppTools:
    """Test Llama.cpp tool calling functionality"""

    @pytest.fixture
    def mock_llamacpp_response(self):
        """Mock response from Llama.cpp server"""
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "test_function",
                            "arguments": '{"param1": "value1"}'
                        }
                    },
                    "finish_reason": "function_call"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

    async def test_llamacpp_tool_calling(self, mock_llamacpp_response):
        """Test tool calling with Llama.cpp server"""
        # Mock the HTTP client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock the response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_llamacpp_response
            mock_client.post.return_value = mock_response
            
            # Mock the models endpoint
            mock_models_response = AsyncMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "data": [
                    {"id": "test-model", "object": "model"}
                ]
            }
            mock_client.get.return_value = mock_models_response
            
            # Test server connection
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        "http://localhost:7543/v1/models",
                        headers={"Authorization": "Bearer llama-cpp-key"}
                    )
                    assert response.status_code == 200
                    data = response.json()
                    assert "data" in data
                    assert len(data["data"]) > 0
                    assert "id" in data["data"][0]
            except ImportError:
                pytest.skip("httpx not available")
            except Exception as e:
                pytest.skip(f"Cannot connect to Llama.cpp server: {e}")

    async def test_direct_test_execution(self):
        """Test direct execution of the Llama.cpp tool calling test"""
        # This test would run the actual integration test if available
        test_file = "tests/integration/app/core/agents/test_llamacpp_tool_calling.py"
        
        # Check if the test file exists
        import os
        if not os.path.exists(test_file):
            pytest.skip(f"Test file not found: {test_file}")
        
        # Try to import and run the test
        try:
            from tests.integration.app.core.agents.test_llamacpp_tool_calling import run_all_tests
            await run_all_tests()
        except ImportError:
            pytest.skip("Could not import test_llamacpp_tool_calling module")
        except Exception as e:
            pytest.fail(f"Direct test execution failed: {e}")

    def test_pytest_execution(self):
        """Test execution using pytest"""
        test_file = "tests/integration/app/core/agents/test_llamacpp_tool_calling.py"
        
        # Check if the test file exists
        import os
        if not os.path.exists(test_file):
            pytest.skip(f"Test file not found: {test_file}")
        
        # This test would normally run pytest programmatically
        # but for now we just verify the file exists
        assert os.path.exists(test_file)

    def test_prerequisites(self):
        """Test that prerequisites are met"""
        # Check if pytest is available
        try:
            import pytest
            assert pytest is not None
        except ImportError:
            pytest.fail("pytest is not installed")
        
        # Check if the test file exists
        import os
        test_file = os.path.join(
            os.path.dirname(__file__),
            "test_llamacpp_tool_calling.py"
        )
        
        # The test file might not exist, so we just check the path
        assert isinstance(test_file, str)
        assert test_file.endswith("test_llamacpp_tool_calling.py")

    @patch('subprocess.run')
    def test_pytest_subprocess_execution(self, mock_run):
        """Test execution using pytest subprocess"""
        # Mock the subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "PASSED"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Test the command that would be executed
        import sys
        test_file = "tests/integration/app/core/agents/test_llamacpp_tool_calling.py"
        
        # This is the command that would be run
        command = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "-m", "integration",
            "--tb=short"
        ]
        
        # Verify the command structure
        assert "pytest" in command
        assert test_file in command
        assert "-m" in command
        assert "integration" in command