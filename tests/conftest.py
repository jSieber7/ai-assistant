import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from langchain.schema import AIMessage


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm():
    """Mock the LLM factory function to return a controlled response."""
    with patch("app.api.routes.get_llm") as mock:
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = AIMessage(content="Mocked AI response")
        mock.return_value = mock_llm_instance
        yield mock


@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    with patch.dict(
        "os.environ",
        {
            "OPENROUTER_API_KEY": "test-key-123",
            "DEFAULT_MODEL": "deepseek/deepseek-v3.1-terminus",
        },
    ):
        yield


@pytest.fixture
def client(mock_llm, mock_env):
    """Create a test client with mocked dependencies."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def chat_request_data():
    """Sample chat request data for testing."""
    return {
        "messages": [
            {
                "role": "user",
                "content": "Hello, can you respond with just the word SUCCESS?",
            }
        ],
        "model": "deepseek/deepseek-v3.1-terminus",
        "stream": False,
    }


@pytest.fixture
def streaming_chat_request_data():
    """Sample streaming chat request data for testing."""
    return {
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
    }


@pytest.fixture
def invalid_chat_request_data():
    """Invalid chat request data for error testing."""
    return {"messages": [], "model": "invalid-model-name"}


@pytest.fixture
def mock_openrouter_error():
    """Mock an OpenRouter API error."""
    with patch("app.api.routes.get_llm") as mock:
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.side_effect = Exception("OpenRouter API error")
        mock.return_value = mock_llm_instance
        yield mock


class MockStreamingResponse:
    """Mock streaming response generator."""

    def __init__(self, content="Mocked streaming response"):
        self.content = content
        self.words = content.split()

    async def generate(self):
        """Generate streaming response chunks."""
        for i, word in enumerate(self.words):
            chunk = {
                "id": "chatcmpl-test-123",
                "object": "chat.completion.chunk",
                "model": "deepseek/deepseek-v3.1-terminus",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)

        # Final chunk
        final_chunk = {
            "id": "chatcmpl-test-123",
            "object": "chat.completion.chunk",
            "model": "deepseek/deepseek-v3.1-terminus",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {final_chunk}\n\n"
        yield "data: [DONE]\n\n"


@pytest.fixture
def mock_streaming_llm():
    """Mock LLM for streaming responses."""
    with patch("app.api.routes.get_llm") as mock:
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = AIMessage(
            content="Mocked streaming response"
        )
        mock.return_value = mock_llm_instance
        yield mock
