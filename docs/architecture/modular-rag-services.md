# Modular RAG Services Architecture

This document describes the modular RAG (Retrieval-Augmented Generation) services architecture that replaces the monolithic DeepSearchAgent implementation.

## Overview

The modular RAG architecture breaks down the complete RAG pipeline into independent, reusable services that can be used separately or combined through an orchestrator. This design provides better maintainability, testability, and flexibility.

## Architecture Diagram

```
User Query
    ↓
Query Processing Service
    ↓ (Optimized Search Query)
Search Service
    ↓ (Scraped Documents)
Ingestion Service
    ↓ (Document Chunks in Vector DB)
Retrieval Service
    ↓ (Re-ranked Documents)
Synthesis Service
    ↓ (Final Answer)
```

## Core Services

### 1. Query Processing Service (`app/core/services/query_processing.py`)

**Purpose**: Optimizes user queries for better search results.

**Key Features**:
- Generates optimized search queries using LLM
- Supports context-aware query processing
- Provides query validation and analysis
- Batch processing capabilities

**Usage Example**:
```python
from app.core.services.query_processing import QueryProcessingService

query_service = QueryProcessingService(llm)
optimized_query = await query_service.generate_search_query(
    "What is async programming in Python?"
)
```

### 2. Search Service (`app/core/services/search.py`)

**Purpose**: Handles web search and content scraping.

**Key Features**:
- Coordinates search and scraping tools
- Concurrent scraping with rate limiting
- URL validation and filtering
- Independent search-only and scrape-only modes

**Usage Example**:
```python
from app.core.services.search import SearchService

search_service = SearchService(tool_registry)
documents = await search_service.search_and_scrape(
    "Python async programming"
)

# Or use components independently
search_results = await search_service.search_only(query)
scraped_docs = await search_service.scrape_urls_only(urls)
```

### 3. Ingestion Service (`app/core/services/ingestion.py`)

**Purpose**: Handles document chunking, embedding, and vector storage.

**Key Features**:
- Configurable text chunking
- Batch embedding generation
- Temporary collection management
- Document validation

**Usage Example**:
```python
from app.core.services.ingestion import IngestionService

ingestion_service = IngestionService(embeddings, vector_client)
result = await ingestion_service.ingest_documents(
    documents, collection_name="temp_collection"
)
```

### 4. Retrieval Service (`app/core/services/retrieval.py`)

**Purpose**: Handles similarity search and document reranking.

**Key Features**:
- Vector similarity search
- Jina reranker integration
- Configurable retrieval parameters
- Independent retrieval and reranking modes

**Usage Example**:
```python
from app.core.services.retrieval import RetrievalService

retrieval_service = RetrievalService(vector_client, tool_registry)
reranked_docs = await retrieval_service.retrieve_and_rerank(
    query="Python async programming",
    collection_name="temp_collection",
    k=50,
    top_n=5
)

# Or use components independently
retrieved_docs = await retrieval_service.retrieve_only(query, collection_name)
reranked_docs = await retrieval_service.rerank_only(query, documents)
```

### 5. Synthesis Service (`app/core/services/synthesis.py`)

**Purpose**: Generates final answers from retrieved documents.

**Key Features**:
- LCEL chain-based answer generation
- Configurable prompt templates
- Context length management
- Source citation

**Usage Example**:
```python
from app.core.services.synthesis import SynthesisService

synthesis_service = SynthesisService(llm)
result = await synthesis_service.synthesize_answer(
    query="What is async programming?",
    documents=retrieved_docs
)
```

### 6. RAG Orchestrator (`app/core/services/rag_orchestrator.py`)

**Purpose**: Coordinates all services for the complete RAG pipeline.

**Key Features**:
- End-to-end pipeline execution
- Step-by-step progress tracking
- Automatic resource cleanup
- Comprehensive statistics

**Usage Example**:
```python
from app.core.services.rag_orchestrator import RAGOrchestrator

orchestrator = RAGOrchestrator(
    query_service=query_service,
    search_service=search_service,
    ingestion_service=ingestion_service,
    retrieval_service=retrieval_service,
    synthesis_service=synthesis_service
)

result = await orchestrator.process_query(
    "What is async programming in Python?"
)
```

## Independent Service Usage

One of the key benefits of the modular architecture is the ability to use services independently:

### Scraping Only
```python
from app.core.services.search import SearchService

search_service = SearchService(tool_registry)
documents = await search_service.scrape_urls_only([
    "https://example.com/article1",
    "https://example.com/article2"
])
```

### Document Reranking Only
```python
from app.core.services.retrieval import RetrievalService

retrieval_service = RetrievalService(vector_client, tool_registry)
reranked_docs = await retrieval_service.rerank_only(
    query="Best practices for Python",
    documents=documents
)
```

### Search and Scrape Only
```python
from app.core.services.search import SearchService

search_service = SearchService(tool_registry)
documents = await search_service.search_and_scrape(
    "Python async programming best practices"
)
```

## Configuration

### Service Configuration
Each service can be configured with specific parameters:

```python
# Search Service with custom configuration
search_service = SearchService(
    tool_registry=tool_registry,
    max_search_results=10,
    max_concurrent_scrapes=5,
    scrape_timeout=60
)

# Ingestion Service with custom chunking
ingestion_service = IngestionService(
    embeddings=embeddings,
    vector_client=vector_client,
    chunk_size=2000,
    chunk_overlap=400
)

# Retrieval Service with custom parameters
retrieval_service = RetrievalService(
    vector_client=vector_client,
    tool_registry=tool_registry,
    retrieval_k=100,
    rerank_top_n=10,
    enable_reranking=True
)
```

### Easy Orchestrator Creation
```python
from app.core.services.rag_orchestrator import RAGOrchestrator

orchestrator = await RAGOrchestrator.create_default(
    tool_registry=tool_registry,
    llm=llm,
    embeddings=embeddings
)
```

## DeepSearchAgent Integration

The DeepSearchAgent has been completely refactored to use the new modular services:

```python
# DeepSearchAgent now uses modular services
agent = DeepSearchAgent(
    tool_registry=tool_registry,
    llm=llm,
    embeddings=embeddings
)

# Direct access to services through the agent
documents = await agent.search_and_scrape("Python async programming")
reranked_docs = await agent.retrieve_and_rerank(query, collection_name)
answer = await agent.synthesize_answer(query, documents)
```

## Error Handling

The modular architecture includes comprehensive error handling:

1. **Service-level error handling**: Each service handles its own errors gracefully
2. **Orchestrator error handling**: The orchestrator manages errors across services
3. **Resource cleanup**: Automatic cleanup of temporary resources

## Statistics and Monitoring

Each service provides detailed statistics:

```python
# Get individual service stats
query_stats = query_service.get_service_stats()
search_stats = search_service.get_service_stats()

# Get orchestrator stats (includes all services)
orchestrator_stats = orchestrator.get_orchestrator_stats()
```

## Testing

The modular architecture includes comprehensive tests:

- Unit tests for each service
- Integration tests for service combinations
- End-to-end tests for the complete pipeline

## Benefits

1. **Reusability**: Services can be used independently in different contexts
2. **Testability**: Each service can be tested in isolation
3. **Maintainability**: Clear separation of concerns makes code easier to maintain
4. **Flexibility**: Services can be configured and combined as needed
5. **Performance**: Services can be optimized and scaled independently
6. **Debugging**: Issues can be isolated to specific services

## Future Enhancements

1. **Additional vector stores**: Support for other vector databases
2. **More reranking options**: Integration with additional reranking services
3. **Caching layers**: Add caching to improve performance
4. **Service discovery**: Dynamic service registration and discovery
5. **Distributed processing**: Support for distributed service execution