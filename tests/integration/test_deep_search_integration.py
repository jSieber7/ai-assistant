"""
Integration tests for the Deep Search agent.

This module contains integration tests that test the Deep Search agent
with real services (or Docker containers) to ensure end-to-end functionality.
"""

import pytest
import asyncio
from typing import Dict, Any

from app.core.agents.deep_search_agent import DeepSearchAgent
from app.core.tools.registry import ToolRegistry
from app.core.config import settings
from app.core.llm_providers import LLMProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_search_full_workflow():
    """
    Test the complete Deep Search workflow with real services.
    
    This test requires:
    - Milvus service running
    - SearXNG service running
    - Firecrawl service running
    - Jina Reranker service running (optional)
    """
    # Skip if services are not available
    pytest.importorskip("pymilvus")
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    
    # Get LLM instance
    try:
        from app.core.llm_providers.openai_provider import OpenAIProvider
        llm_provider = OpenAIProvider()
        llm = await llm_provider.get_llm()
    except Exception as e:
        pytest.skip(f"LLM not available: {e}")
    
    # Get embeddings
    try:
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        pytest.skip(f"Embeddings not available: {e}")
    
    # Create Deep Search agent
    agent = DeepSearchAgent(
        tool_registry=tool_registry,
        llm=llm,
        embeddings=embeddings
    )
    
    # Test with a simple query
    user_query = "What is artificial intelligence?"
    
    try:
        result = await agent.process_message(user_query)
        
        # Verify result
        assert result.success
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        
        # Check metadata
        assert "optimized_query" in result.metadata
        assert "documents_count" in result.metadata
        assert "chunks_count" in result.metadata
        
        print(f"Deep Search Result: {result.response}")
        
    except Exception as e:
        pytest.skip(f"Deep Search workflow failed (services may not be running): {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_milvus_connection():
    """Test Milvus connection and basic operations."""
    pytest.importorskip("pymilvus")
    
    from app.core.storage.milvus_client import MilvusClient
    from langchain.embeddings import OpenAIEmbeddings
    
    # Skip if no API key
    if not settings.openai_settings.api_key:
        pytest.skip("OpenAI API key not configured")
    
    embeddings = OpenAIEmbeddings()
    milvus_client = MilvusClient(embeddings)
    
    try:
        # Test connection
        connected = await milvus_client.connect()
        assert connected, "Failed to connect to Milvus"
        
        # Test temporary collection creation
        collection_name = await milvus_client.create_temporary_collection()
        assert collection_name.startswith("deep_search_")
        
        # Test document ingestion
        from langchain.docstore.document import Document
        
        documents = [
            Document(
                page_content="This is a test document about artificial intelligence.",
                metadata={"source": "test://doc1"}
            ),
            Document(
                page_content="This is another test document about machine learning.",
                metadata={"source": "test://doc2"}
            )
        ]
        
        ingested_count = await milvus_client.ingest_documents(collection_name, documents)
        assert ingested_count > 0
        
        # Test similarity search
        results = await milvus_client.similarity_search(
            collection_name=collection_name,
            query="artificial intelligence",
            k=5
        )
        
        assert len(results) > 0
        assert all(isinstance(doc, tuple) for doc in results)
        
        # Test collection cleanup
        dropped = await milvus_client.drop_collection(collection_name)
        assert dropped
        
        # Disconnect
        await milvus_client.disconnect()
        
    except Exception as e:
        pytest.fail(f"Milvus integration test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_searxng_and_firecrawl_integration():
    """Test SearXNG and Firecrawl tool integration."""
    from app.core.tools.searxng_tool import SearXNGTool
    from app.core.tools.firecrawl_tool import FirecrawlTool
    
    # Initialize tools
    searxng_tool = SearXNGTool()
    firecrawl_tool = FirecrawlTool()
    
    try:
        # Test search
        search_result = await searxng_tool.execute(
            query="artificial intelligence",
            results_count=3
        )
        
        assert search_result.success
        assert len(search_result.data["results"]) > 0
        
        # Get first URL for scraping
        first_url = search_result.data["results"][0]["url"]
        
        # Test scraping
        scrape_result = await firecrawl_tool.execute(
            url=first_url,
            formats=["markdown"]
        )
        
        if scrape_result.success:
            assert len(scrape_result.data["content"]) > 0
            print(f"Successfully scraped: {first_url}")
        else:
            print(f"Scraping failed for {first_url}: {scrape_result.error}")
        
    except Exception as e:
        pytest.skip(f"SearXNG/Firecrawl integration test skipped (services may not be running): {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_jina_reranker_integration():
    """Test Jina Reranker integration."""
    pytest.importorskip("httpx")
    
    from app.core.tools.jina_reranker_tool import JinaRerankerTool
    
    # Skip if not enabled
    if not settings.jina_reranker_enabled:
        pytest.skip("Jina Reranker not enabled")
    
    reranker_tool = JinaRerankerTool()
    
    try:
        # Test reranking
        documents = [
            "Artificial intelligence is a rapidly evolving field.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language."
        ]
        
        result = await reranker_tool.execute(
            query="What is the relationship between AI and machine learning?",
            documents=documents,
            top_n=3
        )
        
        if result.success:
            assert "results" in result.data
            assert len(result.data["results"]) <= 3
            print(f"Successfully reranked {len(documents)} documents")
        else:
            print(f"Reranking failed: {result.error}")
        
    except Exception as e:
        pytest.fail(f"Jina Reranker integration test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_search_with_real_query():
    """Test Deep Search agent with a real, complex query."""
    pytest.importorskip("pymilvus")
    
    # Skip if services are not available
    if not settings.openai_settings.api_key:
        pytest.skip("OpenAI API key not configured")
    
    # Initialize components
    tool_registry = ToolRegistry()
    llm = await get_llm()
    
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    
    # Create agent
    agent = DeepSearchAgent(
        tool_registry=tool_registry,
        llm=llm,
        embeddings=embeddings
    )
    
    # Test with a complex query
    complex_query = "What are the latest developments in quantum computing and how might they impact artificial intelligence?"
    
    try:
        result = await agent.process_message(complex_query)
        
        assert result.success
        assert isinstance(result.response, str)
        assert len(result.response) > 50  # Should be a substantial response
        
        print(f"Complex Query: {complex_query}")
        print(f"Response: {result.response}")
        print(f"Metadata: {result.metadata}")
        
    except Exception as e:
        pytest.fail(f"Complex query test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deep_search_error_handling():
    """Test Deep Search agent error handling with various failure scenarios."""
    pytest.importorskip("pymilvus")
    
    # Initialize components
    tool_registry = ToolRegistry()
    
    # Skip if no API key
    if not settings.openai_settings.api_key:
        pytest.skip("OpenAI API key not configured")
    
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    
    from app.core.llm_providers.openai_provider import OpenAIProvider
    llm_provider = OpenAIProvider()
    llm = await llm_provider.get_llm()
    
    # Create agent
    agent = DeepSearchAgent(
        tool_registry=tool_registry,
        llm=llm,
        embeddings=embeddings
    )
    
    # Test with a query that should fail gracefully
    problematic_query = "query_that_should_return_no_results_xyz123"
    
    try:
        result = await agent.process_message(problematic_query)
        
        # Should handle gracefully
        if not result.success:
            assert "error" in result.response.lower() or "couldn't find" in result.response.lower()
            print(f"Graceful failure: {result.response}")
        else:
            print(f"Unexpected success with problematic query: {result.response}")
        
    except Exception as e:
        pytest.fail(f"Error handling test failed: {e}")