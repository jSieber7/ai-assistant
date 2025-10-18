"""
Unit tests for the Deep Search agent.

This module contains comprehensive tests for the Deep Search agent,
including mock testing of the search, scraping, and RAG pipeline.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from app.core.agents.deep_search_agent import DeepSearchAgent
from app.core.tools.base import ToolResult
from app.core.tools.registry import ToolRegistry


class MockEmbeddings(Embeddings):
    """Mock embeddings class for testing."""

    def __init__(self, dimension=1536):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for documents."""
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return mock embedding for query."""
        return [0.1] * self.dimension

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


from langchain_core.runnables import Runnable


class MockLLM(Runnable):
    """Mock LLM class for testing."""

    def __init__(self, responses: Dict[str, str] = None):
        super().__init__()
        self.responses = responses or {}
        self.call_count = 0

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        """Mock LLM call."""
        self.call_count += 1

        # Return predefined response if available
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default responses
        if "search query" in prompt.lower():
            return "artificial intelligence latest developments"
        elif "user question" in prompt.lower():
            return "Based on the provided documents, artificial intelligence has made significant advances in recent years."
        else:
            return "Mock response"

    async def _acall(self, prompt: str, stop=None, run_manager=None) -> str:
        """Async version of _call."""
        return self._call(prompt, stop, run_manager)

    async def ainvoke(self, input_data, config=None, **kwargs):
        """Mock async invoke method."""
        # Handle different input types
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict):
            # Extract prompt from dictionary
            prompt = (
                input_data.get("input", "")
                or input_data.get("prompt", "")
                or input_data.get("user_query", "")
            )
        else:
            prompt = str(input_data)

        response = self._call(prompt)
        return response

    def invoke(self, input_data, config=None, **kwargs):
        """Mock invoke method."""
        # Handle different input types
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict):
            # Extract prompt from dictionary
            prompt = (
                input_data.get("input", "")
                or input_data.get("prompt", "")
                or input_data.get("user_query", "")
            )
        else:
            prompt = str(input_data)

        response = self._call(prompt)
        return response

    def transform(self, func):
        """Mock transform method."""
        return MockTransformedLLM(self, func)

    @property
    def _llm_type(self) -> str:
        return "mock"


class MockResponse:
    """Mock response object."""

    def __init__(self, content: str):
        self.content = content


class MockTransformedLLM(Runnable):
    """Mock transformed LLM."""

    def __init__(self, llm, transform_func):
        super().__init__()
        self.llm = llm
        self.transform_func = transform_func

    async def ainvoke(self, input_data, config=None, **kwargs):
        """Mock async invoke."""
        transformed = self.transform_func(input_data)
        return await self.llm.ainvoke(transformed, config, **kwargs)

    def invoke(self, input_data, config=None, **kwargs):
        """Mock invoke."""
        transformed = self.transform_func(input_data)
        return self.llm.invoke(transformed, config, **kwargs)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, response_data: Any = None, success: bool = True):
        self.name = name
        self.response_data = response_data or {}
        self.success = success
        self.call_count = 0

    async def execute(self, **kwargs) -> ToolResult:
        """Mock tool execution."""
        self.call_count += 1
        return ToolResult(
            success=self.success,
            data=self.response_data,
            tool_name=self.name,
            execution_time=0.1,
        )


@pytest.fixture
def mock_embeddings():
    """Fixture for mock embeddings."""
    return MockEmbeddings()


@pytest.fixture
def mock_llm():
    """Fixture for mock LLM."""
    responses = {
        "search query": "artificial intelligence latest developments",
        "user question": "Based on the provided documents, artificial intelligence has made significant advances in recent years.",
    }
    return MockLLM(responses)


@pytest.fixture
def mock_tool_registry():
    """Fixture for mock tool registry."""
    registry = Mock(spec=ToolRegistry)

    # Mock SearXNG tool
    searxng_response = {
        "results": [
            {
                "title": "AI Advances in 2024",
                "url": "https://example.com/ai-2024",
                "content": "Artificial intelligence has seen remarkable advances in 2024.",
            },
            {
                "title": "Machine Learning Breakthroughs",
                "url": "https://example.com/ml-breakthroughs",
                "content": "Machine learning algorithms continue to improve rapidly.",
            },
        ]
    }

    # Mock Firecrawl tool
    firecrawl_response = {
        "content": "# AI Advances in 2024\n\nArtificial intelligence has seen remarkable advances in 2024. New models are more efficient and capable.",
        "title": "AI Advances in 2024",
        "source": "https://example.com/ai-2024",
    }

    # Mock Jina Reranker tool
    jina_response = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
        ]
    }

    registry.get_tool.side_effect = lambda name: {
        "searxng_search": MockTool("searxng_search", searxng_response),
        "firecrawl_scrape": MockTool("firecrawl_scrape", firecrawl_response),
        "jina_reranker": MockTool("jina_reranker", jina_response),
    }.get(name)

    return registry


@pytest.fixture
def deep_search_agent(mock_tool_registry, mock_llm, mock_embeddings):
    """Fixture for Deep Search agent."""
    return DeepSearchAgent(
        tool_registry=mock_tool_registry,
        llm=mock_llm,
        embeddings=mock_embeddings,
        max_iterations=1,
    )


class TestDeepSearchAgent:
    """Test cases for Deep Search agent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, deep_search_agent):
        """Test agent initialization."""
        assert deep_search_agent.name == "deep_search"
        assert "deep web search" in deep_search_agent.description.lower()
        assert deep_search_agent.max_search_results == 5
        assert deep_search_agent.retrieval_k == 50
        assert deep_search_agent.rerank_top_n == 5

    @pytest.mark.asyncio
    async def test_generate_search_query(self, deep_search_agent):
        """Test search query generation."""
        user_query = "What are the latest developments in AI?"

        optimized_query = await deep_search_agent._generate_search_query(user_query)

        assert isinstance(optimized_query, str)
        assert len(optimized_query) > 0
        assert "artificial intelligence" in optimized_query.lower()

    @pytest.mark.asyncio
    async def test_search_and_scrape(self, deep_search_agent):
        """Test search and scrape functionality."""
        query = "artificial intelligence developments"

        documents = await deep_search_agent._search_and_scrape(query)

        assert isinstance(documents, list)
        assert len(documents) > 0

        for doc in documents:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0
            assert "source" in doc.metadata

    @pytest.mark.asyncio
    async def test_process_documents(self, deep_search_agent):
        """Test document processing and chunking."""
        documents = [
            Document(
                page_content="This is a long document about artificial intelligence. "
                * 50,
                metadata={"source": "https://example.com"},
            )
        ]

        chunks = await deep_search_agent._process_documents(documents)

        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should be split into multiple chunks

        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) <= 1000  # Chunk size limit
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    @pytest.mark.asyncio
    @patch("app.core.agents.deep_search_agent.MilvusClient")
    async def test_full_workflow(self, mock_milvus_client_class, deep_search_agent):
        """Test the complete Deep Search workflow."""
        # Mock Milvus client
        mock_milvus_client = AsyncMock()
        mock_milvus_client.connect.return_value = True
        mock_milvus_client.create_temporary_collection.return_value = "test_collection"
        mock_milvus_client.ingest_documents.return_value = 10
        mock_milvus_client.similarity_search.return_value = [
            (Document(page_content="AI content", metadata={"source": "test"}), 0.9)
        ]
        mock_milvus_client.drop_collection.return_value = True
        mock_milvus_client_class.return_value = mock_milvus_client

        # Initialize the agent's Milvus client
        deep_search_agent.milvus_client = mock_milvus_client

        # Test the full workflow
        user_query = "What are the latest developments in AI?"
        result = await deep_search_agent._process_message_impl(user_query)

        assert result.success
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        assert "artificial intelligence" in result.response.lower()

        # Verify Milvus operations were called
        mock_milvus_client.connect.assert_called_once()
        mock_milvus_client.create_temporary_collection.assert_called_once()
        mock_milvus_client.ingest_documents.assert_called_once()
        mock_milvus_client.similarity_search.assert_called_once()
        mock_milvus_client.drop_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_and_synthesize(self, deep_search_agent):
        """Test document retrieval and synthesis."""
        # Mock Milvus client
        mock_milvus_client = AsyncMock()
        mock_milvus_client.similarity_search.return_value = [
            (
                Document(page_content="AI development 1", metadata={"source": "test1"}),
                0.9,
            ),
            (
                Document(page_content="AI development 2", metadata={"source": "test2"}),
                0.8,
            ),
        ]
        deep_search_agent.milvus_client = mock_milvus_client

        documents = [
            Document(page_content="AI development 1", metadata={"source": "test1"}),
            Document(page_content="AI development 2", metadata={"source": "test2"}),
        ]

        answer = await deep_search_agent._retrieve_and_synthesize(
            "test_collection", "What is AI?", documents
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        mock_milvus_client.similarity_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_milvus_connection_failure(self, deep_search_agent):
        """Test handling of Milvus connection failure."""
        # Mock Milvus client to fail connection
        mock_milvus_client = AsyncMock()
        mock_milvus_client.connect.return_value = False
        deep_search_agent.milvus_client = mock_milvus_client

        user_query = "What are the latest developments in AI?"
        result = await deep_search_agent._process_message_impl(user_query)

        assert not result.success
        assert "error" in result.response.lower()

    @pytest.mark.asyncio
    @patch("app.core.agents.deep_search_agent.MilvusClient")
    async def test_no_search_results(self, mock_milvus_client_class, deep_search_agent):
        """Test handling when no search results are found."""
        # Mock Milvus client
        mock_milvus_client = AsyncMock()
        mock_milvus_client.connect.return_value = True
        mock_milvus_client_class.return_value = mock_milvus_client

        # Mock SearXNG tool to return no results
        deep_search_agent.searxng_tool = MockTool("searxng_search", {"results": []})

        user_query = "very specific obscure query"
        result = await deep_search_agent._process_message_impl(user_query)

        assert not result.success
        assert "couldn't find relevant information" in result.response.lower()

    @pytest.mark.asyncio
    async def test_scraping_failure(self, deep_search_agent):
        """Test handling of scraping failures."""
        # Mock Firecrawl tool to fail
        deep_search_agent.firecrawl_tool = MockTool("firecrawl_scrape", success=False)

        user_query = "test query"
        result = await deep_search_agent._process_message_impl(user_query)

        assert not result.success

    @pytest.mark.asyncio
    async def test_reranking_disabled(self, deep_search_agent):
        """Test workflow when Jina reranker is disabled."""
        # Disable reranker
        deep_search_agent.jina_reranker_tool = None

        # Mock Milvus client
        mock_milvus_client = AsyncMock()
        mock_milvus_client.connect.return_value = True
        mock_milvus_client.create_temporary_collection.return_value = "test_collection"
        mock_milvus_client.ingest_documents.return_value = 5
        mock_milvus_client.similarity_search.return_value = [
            (Document(page_content="AI content", metadata={"source": "test"}), 0.9)
        ]
        mock_milvus_client.drop_collection.return_value = True

        deep_search_agent.milvus_client = mock_milvus_client

        user_query = "What are the latest developments in AI?"
        result = await deep_search_agent._process_message_impl(user_query)

        assert result.success
        assert isinstance(result.response, str)
        assert len(result.response) > 0

    def test_agent_properties(self, deep_search_agent):
        """Test agent properties."""
        assert deep_search_agent.name == "deep_search"
        assert "deep web search" in deep_search_agent.description.lower()
        assert deep_search_agent.max_iterations == 1

    @pytest.mark.asyncio
    async def test_conversation_management(self, deep_search_agent):
        """Test conversation management functionality."""
        conversation_id = deep_search_agent.start_conversation()
        assert conversation_id is not None

        history = deep_search_agent.get_conversation_history(conversation_id)
        assert isinstance(history, list)
        assert len(history) == 0

        deep_search_agent.add_to_conversation("user", "test message")
        history = deep_search_agent.get_conversation_history(conversation_id)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "test message"
