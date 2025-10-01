# RAG (Retrieval-Augmented Generation) System

The RAG system enhances the AI assistant's capabilities by providing access to a knowledge base of documents. This allows the assistant to answer questions based on specific content beyond its training data.

## Overview

Retrieval-Augmented Generation combines information retrieval with generative AI to provide accurate, context-aware responses. The RAG system:

- **Stores documents** in a vector database for efficient retrieval
- **Searches relevant content** based on user queries
- **Augments LLM responses** with retrieved information
- **Provides citations** for source material

## Architecture

### System Components

```
User Query → Embedding Model → Vector Search → Context Augmentation → LLM Response
```

1. **Document Processing**: Convert documents to vector embeddings
2. **Vector Storage**: Store embeddings in a vector database
3. **Query Processing**: Convert user queries to embeddings
4. **Similarity Search**: Find most relevant document chunks
5. **Response Generation**: Augment LLM prompt with retrieved context

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Vector database configuration
VECTOR_DB_URL=postgresql://user:pass@localhost:5432/rag_db
# or for SQLite (development)
VECTOR_DB_URL=sqlite:///./rag.db

# Embedding model settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# RAG settings
RAG_MAX_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
```

### Supported Vector Databases

- **PostgreSQL with pgvector** (recommended for production)
- **SQLite with vector extension** (development)
- **ChromaDB** (lightweight alternative)
- **Pinecone** (cloud-based)

## Implementation Status

**Current Status**: *Planned Feature*

This integration is part of the roadmap and will be implemented in future releases.

## Planned Features

### Document Ingestion
```python
# Planned implementation
async def ingest_document(file_path: str, metadata: dict = None) -> str:
    """
    Process and store a document in the RAG system
    
    Args:
        file_path: Path to document (PDF, TXT, DOCX, etc.)
        metadata: Optional document metadata
    
    Returns:
        Document ID for reference
    """
    pass
```

### Query Processing
```python
async def rag_query(query: str, top_k: int = 5) -> RAGResult:
    """
    Search knowledge base and augment LLM response
    
    Args:
        query: User question or search term
        top_k: Number of results to retrieve
    
    Returns:
        RAG result with context and citations
    """
    pass
```

## Document Support

### Supported Formats
- **Text files** (.txt, .md)
- **PDF documents** (.pdf)
- **Word documents** (.docx)
- **Web pages** (URL ingestion)
- **Code files** (.py, .js, .java, etc.)

### Processing Pipeline
1. **Text extraction** from various formats
2. **Chunking** into manageable pieces
3. **Embedding generation** using transformer models
4. **Vector storage** in database

## Usage Examples

### Basic RAG Integration
```python
# Planned usage in tool calling
@tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information
    
    Use this tool when you need to find specific information
    from uploaded documents or the knowledge base.
    """
    results = await rag_client.search(query, top_k=3)
    return format_rag_results(results)
```

### Document Management
```python
# Planned document management
async def add_document_to_kb(file_path: str, description: str) -> str:
    """
    Add a document to the knowledge base
    
    Returns document ID for future reference
    """
    doc_id = await rag_client.ingest_document(file_path)
    return f"Document added with ID: {doc_id}"
```

## Setup Instructions

### Option 1: PostgreSQL with pgvector (Production)

1. **Install PostgreSQL with pgvector**:
```bash
# Using Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password ankane/pgvector
```

2. **Create database**:
```sql
CREATE DATABASE rag_db;
\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
```

3. **Configure the AI assistant**:
```bash
echo "VECTOR_DB_URL=postgresql://user:password@localhost:5432/rag_db" >> .env
```

### Option 2: SQLite with vector extension (Development)

1. **Install SQLite vector extension**:
```bash
pip install sqlite-vector
```

2. **Configure for development**:
```bash
echo "VECTOR_DB_URL=sqlite:///./rag.db" >> .env
```

## Security Considerations

### Data Privacy
- Documents are stored locally or in controlled infrastructure
- No third-party data sharing unless explicitly configured
- Encryption at rest for sensitive documents

### Access Control
- Implement document-level permissions
- Role-based access to knowledge bases
- Audit trails for document access

### Content Validation
- Scan for malicious content before ingestion
- Validate document sources
- Implement content filtering

## Performance Optimization

### Indexing Strategy
- **Hierarchical indexing** for large document collections
- **Approximate nearest neighbor search** for speed
- **Caching** frequently accessed embeddings

### Chunking Optimization
- **Semantic chunking** based on content structure
- **Overlap management** to maintain context
- **Size tuning** for optimal retrieval

## Testing

### Unit Tests
```python
@pytest.mark.asyncio
async def test_rag_ingestion():
    """Test document ingestion functionality"""
    rag_client = RAGClient()
    doc_id = await rag_client.ingest_document("test_document.txt")
    assert doc_id is not None
```

### Integration Tests
```python
@pytest.mark.integration
async def test_rag_query():
    """Test RAG query functionality"""
    rag_client = RAGClient()
    results = await rag_client.search("test query")
    assert len(results) > 0
```

## Error Handling

### Common Issues
- **Database connectivity**: Implement retry logic with backoff
- **Document processing failures**: Fallback to alternative parsers
- **Embedding generation failures**: Use backup models

### Planned Error Handling
```python
class RAGClient:
    async def search_with_fallback(self, query: str) -> RAGResult:
        try:
            return await self.search(query)
        except RAGError as e:
            logger.warning(f"RAG search failed: {e}")
            # Fallback to basic keyword search or disable feature
            return await self.fallback_search(query)
```

## Roadmap

### Phase 1: Basic RAG
- [ ] Implement document ingestion pipeline
- [ ] Add vector database integration
- [ ] Create basic search functionality
- [ ] Add unit tests

### Phase 2: Advanced Features
- [ ] Implement multi-modal RAG (images, audio)
- [ ] Add cross-document reasoning
- [ ] Implement citation generation
- [ ] Add result ranking and filtering

### Phase 3: Production Ready
- [ ] Performance optimization for large datasets
- [ ] Advanced security features
- [ ] Monitoring and analytics
- [ ] Backup and recovery procedures

## Integration with AI Assistant

### Automatic Tool Selection
The AI agent will automatically determine when RAG is appropriate based on:
- Query specificity and domain knowledge requirements
- Availability of relevant documents in the knowledge base
- User preferences and conversation context

### Response Enhancement
RAG-enhanced responses will include:
- **Source citations** for retrieved information
- **Confidence scores** based on similarity matching
- **Contextual relevance** indicators

## Related Documentation

- [AI Assistant Architecture](architecture/overview.md)
- [Tool System Design](architecture/tools.md)
- [API Endpoints Reference](api/endpoints.md)
- [SearX Integration](tools/searx.md)

## Support

For issues with RAG integration:
1. Check database connectivity and permissions
2. Verify document processing pipeline
3. Review embedding model compatibility
4. Consult the [troubleshooting guide](development/troubleshooting.md)

## Best Practices

### Document Organization
- **Categorize documents** by topic or project
- **Maintain metadata** for better retrieval
- **Regularly update** the knowledge base

### Query Optimization
- **Use specific queries** for better results
- **Combine with filters** when appropriate
- **Iterative refinement** based on results

### Performance Monitoring
- **Track query response times**
- **Monitor retrieval accuracy**
- **Optimize chunking strategies**