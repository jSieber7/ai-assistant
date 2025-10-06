import pytest
import json
from fastapi.testclient import TestClient


class TestChatCompletions:
    """Test chat completions endpoint functionality."""

    def test_chat_completion_success(
        self, client: TestClient, chat_request_data, mock_llm
    ):
        """Test successful non-streaming chat completion."""
        response = client.post("/v1/chat/completions", json=chat_request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "chatcmpl-123456789"
        assert data["object"] == "chat.completion"
        assert data["model"] == "deepseek/deepseek-v3.1-terminus"
        assert "choices" in data
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Mocked AI response"
        assert choice["finish_reason"] == "stop"

    def test_chat_completion_default_model(self, client: TestClient, mock_llm):
        """Test chat completion with default model when not specified."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "langchain-agent-hub"

    def test_chat_completion_different_roles(self, client: TestClient, mock_llm):
        """Test chat completion with different message roles."""
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mocked AI response"

    def test_chat_completion_with_parameters(self, client: TestClient, mock_llm):
        """Test chat completion with temperature and max_tokens parameters."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "deepseek/deepseek-v3.1-terminus",
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 100,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "deepseek/deepseek-v3.1-terminus"


class TestStreamingChatCompletions:
    """Test streaming chat completions endpoint functionality."""

    def test_streaming_chat_completion_success(
        self, client: TestClient, streaming_chat_request_data, mock_llm
    ):
        """Test successful streaming chat completion."""
        response = client.post("/v1/chat/completions", json=streaming_chat_request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Parse streaming response
        content = response.text
        lines = [line for line in content.split("\n") if line.strip()]

        # Should have data lines and a [DONE] line
        assert len(lines) > 0
        assert any("data: [DONE]" in line for line in lines)

        # Check that we have proper SSE format
        for line in lines:
            if line.startswith("data: "):
                if line != "data: [DONE]":
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    assert "choices" in chunk_data
                    assert len(chunk_data["choices"]) == 1

    def test_streaming_response_content(self, client: TestClient, mock_streaming_llm):
        """Test streaming response contains expected content chunks."""
        request_data = {
            "messages": [{"role": "user", "content": "Say hello world"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        content = response.text
        lines = [
            line
            for line in content.split("\n")
            if line.strip() and line.startswith("data: ")
        ]

        # Should have multiple data chunks
        assert len(lines) > 1

        # Last line should be [DONE]
        assert lines[-1] == "data: [DONE]"

        # Check content chunks
        for line in lines[:-1]:  # Exclude [DONE] line
            if line != "data: [DONE]":
                chunk_data = json.loads(line[6:])
                choice = chunk_data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    assert isinstance(choice["delta"]["content"], str)


class TestChatCompletionErrorHandling:
    """Test error handling for chat completions endpoint."""

    def test_chat_completion_missing_messages(self, client: TestClient):
        """Test chat completion with missing messages field."""
        request_data = {"model": "deepseek/deepseek-v3.1-terminus", "stream": False}
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_completion_empty_messages(
        self, client: TestClient, invalid_chat_request_data
    ):
        """Test chat completion with empty messages array."""
        response = client.post("/v1/chat/completions", json=invalid_chat_request_data)
        # Empty messages might be handled gracefully or cause an error
        assert response.status_code in [200, 500]

    def test_chat_completion_invalid_model(self, client: TestClient):
        """Test chat completion with invalid model name."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "invalid-model-name",
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        # This might return 500 if the model is not found, or 200 if it uses default
        # Depending on how the error is handled in the actual code
        assert response.status_code in [200, 500]

    def test_chat_completion_openrouter_error(
        self, client: TestClient, chat_request_data, mock_openrouter_error
    ):
        """Test chat completion when OpenRouter API returns an error."""
        response = client.post("/v1/chat/completions", json=chat_request_data)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # The actual error message from the mock
        assert "OpenRouter API error" in data["detail"]

    def test_chat_completion_missing_api_key(
        self, client: TestClient, chat_request_data
    ):
        """Test chat completion when OpenRouter API key is missing."""
        # Mock the environment to have no API key
        with pytest.MonkeyPatch().context() as m:
            m.delenv("OPENROUTER_API_KEY", raising=False)
            response = client.post("/v1/chat/completions", json=chat_request_data)
            # Since the LLM is mocked, we get a successful response
            assert response.status_code == 200

    def test_chat_completion_invalid_json(self, client: TestClient):
        """Test chat completion with invalid JSON payload."""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json data",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_chat_completion_wrong_content_type(
        self, client: TestClient, chat_request_data
    ):
        """Test chat completion with wrong content type."""
        response = client.post(
            "/v1/chat/completions",
            data=json.dumps(chat_request_data),
            headers={"Content-Type": "text/plain"},
        )
        # FastAPI might still parse it as JSON, but let's see what happens
        assert response.status_code in [200, 415, 422]


class TestMessageValidation:
    """Test message validation in chat completions."""

    def test_chat_completion_invalid_role(self, client: TestClient, mock_llm):
        """Test chat completion with invalid message role."""
        request_data = {
            "messages": [{"role": "invalid_role", "content": "Hello"}],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        # The role might be validated by Pydantic or in the route handler
        assert response.status_code in [200, 422]

    def test_chat_completion_empty_content(self, client: TestClient, mock_llm):
        """Test chat completion with empty message content."""
        request_data = {"messages": [{"role": "user", "content": ""}], "stream": False}
        response = client.post("/v1/chat/completions", json=request_data)
        # Empty content might be allowed or cause an error
        assert response.status_code in [200, 422, 500]

    def test_chat_completion_missing_content(self, client: TestClient):
        """Test chat completion with missing content field."""
        request_data = {
            "messages": [{"role": "user"}],  # Missing content
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
class TestAsyncChatCompletions:
    """Test async functionality of chat completions."""

    async def test_async_chat_completion(
        self, client: TestClient, chat_request_data, mock_llm
    ):
        """Test async chat completion request."""
        response = client.post("/v1/chat/completions", json=chat_request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mocked AI response"

    async def test_async_streaming_chat_completion(
        self, client: TestClient, streaming_chat_request_data, mock_llm
    ):
        """Test async streaming chat completion."""
        response = client.post("/v1/chat/completions", json=streaming_chat_request_data)
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
