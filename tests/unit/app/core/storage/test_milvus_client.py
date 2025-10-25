"""
Unit tests for Milvus Client.

This module tests the Milvus client for vector storage and similarity search.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, asynccontextmanager
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.storage.milvus_client import (
    MilvusClient,
    MilvusVectorStore,
    get_milvus_client
)


class TestMilvusClient:
    """Test MilvusClient class"""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings instance"""
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return embeddings

    @pytest.fixture
    def milvus_client(self, mock_embeddings):
        """Create a Milvus client instance"""
        return MilvusClient(embeddings=mock_embeddings)

    def test_client_init(self, milvus_client, mock_embeddings):
        """Test MilvusClient initialization"""
        assert milvus_client.embeddings == mock_embeddings
        assert milvus_client.client is None
        assert milvus_client.collection is None

    @pytest.mark.asyncio
    async def test_connect(self, milvus_client):
        """Test connecting to Milvus"""
        with patch('pymilvus.connections.connect') as mock_connect:
            await milvus_client.connect()
            
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_error(self, milvus_client):
        """Test connecting to Milvus with error"""
        with patch('pymilvus.connections.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception, match="Connection error"):
                await milvus_client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, milvus_client):
        """Test disconnecting from Milvus"""
        with patch('pymilvus.connections.disconnect') as mock_disconnect:
            await milvus_client.disconnect()
            
            mock_disconnect.assert_called_once()

    def test_create_collection_schema(self, milvus_client):
        """Test creating collection schema"""
        with patch('pymilvus.FieldSchema') as mock_field_schema, \
             patch('pymilvus.CollectionSchema') as mock_collection_schema:
            
            mock_id_field = Mock()
            mock_text_field = Mock()
            mock_vector_field = Mock()
            mock_field_schema.side_effect = [mock_id_field, mock_text_field, mock_vector_field]
            
            mock_schema = Mock()
            mock_collection_schema.return_value = mock_schema
            
            result = milvus_client._create_collection_schema()
            
            assert result == mock_schema
            assert mock_field_schema.call_count == 3
            mock_collection_schema.assert_called_once_with(
                fields=[mock_id_field, mock_text_field, mock_vector_field],
                description="Document collection for similarity search"
            )

    @pytest.mark.asyncio
    async def test_create_temporary_collection(self, milvus_client):
        """Test creating a temporary collection"""
        collection_name = "temp_collection_123"
        
        with patch('pymilvus.Collection') as mock_collection_class, \
             patch.object(milvus_client, '_create_collection_schema') as mock_create_schema:
            
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            mock_schema = Mock()
            mock_create_schema.return_value = mock_schema
            
            result = await milvus_client.create_temporary_collection(collection_name)
            
            assert result == mock_collection
            mock_collection_class.assert_called_once_with(
                name=collection_name,
                schema=mock_schema,
                using='default'
            )
            mock_collection.create_index.assert_called_once()
            mock_collection.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_temporary_collection_with_session_id(self, milvus_client):
        """Test creating a temporary collection with session ID"""
        session_id = "session-123"
        
        with patch.object(milvus_client, 'create_temporary_collection') as mock_create:
            mock_collection = Mock()
            mock_create.return_value = mock_collection
            
            result = await milvus_client.create_temporary_collection(session_id=session_id)
            
            assert result == mock_collection
            mock_create.assert_called_once()
            
            # Check that collection name includes session ID
            call_args = mock_create.call_args[0]
            assert session_id in call_args[0]

    @pytest.mark.asyncio
    async def test_create_temporary_collection_error(self, milvus_client):
        """Test creating a temporary collection with error"""
        collection_name = "temp_collection_123"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection_class.side_effect = Exception("Collection creation error")
            
            with pytest.raises(Exception, match="Collection creation error"):
                await milvus_client.create_temporary_collection(collection_name)

    @pytest.mark.asyncio
    async def test_ingest_documents(self, milvus_client):
        """Test ingesting documents"""
        collection_name = "test_collection"
        documents = [
            {"id": "doc1", "text": "First document", "metadata": {"source": "test"}},
            {"id": "doc2", "text": "Second document", "metadata": {"source": "test"}}
        ]
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            # Mock embeddings
            milvus_client.embeddings.embed_documents.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            
            await milvus_client.ingest_documents(collection_name, documents)
            
            # Check that collection was used
            mock_collection.insert.assert_called_once()
            mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_documents_error(self, milvus_client):
        """Test ingesting documents with error"""
        collection_name = "test_collection"
        documents = [{"id": "doc1", "text": "First document"}]
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            mock_collection.insert.side_effect = Exception("Insert error")
            
            with pytest.raises(Exception, match="Insert error"):
                await milvus_client.ingest_documents(collection_name, documents)

    @pytest.mark.asyncio
    async def test_similarity_search(self, milvus_client):
        """Test similarity search"""
        collection_name = "test_collection"
        query_text = "search query"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            # Mock search results
            mock_results = [
                [{"id": "doc1", "distance": 0.1}],
                [{"id": "doc2", "distance": 0.2}]
            ]
            mock_collection.search.return_value = mock_results
            
            result = await milvus_client.similarity_search(
                collection_name=collection_name,
                query_text=query_text,
                k=2
            )
            
            assert len(result) == 2
            assert result[0]["id"] == "doc1"
            assert result[0]["distance"] == 0.1
            assert result[1]["id"] == "doc2"
            assert result[1]["distance"] == 0.2
            
            # Check that search was called with correct parameters
            mock_collection.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, milvus_client):
        """Test similarity search with filter"""
        collection_name = "test_collection"
        query_text = "search query"
        filter_expr = "source == 'test'"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            mock_results = [[{"id": "doc1", "distance": 0.1}]]
            mock_collection.search.return_value = mock_results
            
            result = await milvus_client.similarity_search(
                collection_name=collection_name,
                query_text=query_text,
                filter_expr=filter_expr,
                k=1
            )
            
            assert len(result) == 1
            assert result[0]["id"] == "doc1"
            
            # Check that search was called with filter
            call_args = mock_collection.search.call_args
            assert filter_expr in call_args[0]

    @pytest.mark.asyncio
    async def test_similarity_search_error(self, milvus_client):
        """Test similarity search with error"""
        collection_name = "test_collection"
        query_text = "search query"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            mock_collection.search.side_effect = Exception("Search error")
            
            with pytest.raises(Exception, match="Search error"):
                await milvus_client.similarity_search(
                    collection_name=collection_name,
                    query_text=query_text
                )

    @pytest.mark.asyncio
    async def test_drop_collection(self, milvus_client):
        """Test dropping a collection"""
        collection_name = "test_collection"
        
        with patch('pymilvus.utility.has_collection') as mock_has_collection, \
             patch('pymilvus.utility.drop_collection') as mock_drop_collection:
            
            mock_has_collection.return_value = True
            
            result = await milvus_client.drop_collection(collection_name)
            
            assert result is True
            mock_has_collection.assert_called_once_with(collection_name)
            mock_drop_collection.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_drop_collection_not_exists(self, milvus_client):
        """Test dropping a collection that doesn't exist"""
        collection_name = "nonexistent_collection"
        
        with patch('pymilvus.utility.has_collection') as mock_has_collection:
            mock_has_collection.return_value = False
            
            result = await milvus_client.drop_collection(collection_name)
            
            assert result is False
            mock_has_collection.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_drop_collection_error(self, milvus_client):
        """Test dropping a collection with error"""
        collection_name = "test_collection"
        
        with patch('pymilvus.utility.has_collection') as mock_has_collection, \
             patch('pymilvus.utility.drop_collection') as mock_drop_collection:
            
            mock_has_collection.return_value = True
            mock_drop_collection.side_effect = Exception("Drop error")
            
            with pytest.raises(Exception, match="Drop error"):
                await milvus_client.drop_collection(collection_name)

    @pytest.mark.asyncio
    async def test_schedule_collection_cleanup(self, milvus_client):
        """Test scheduling collection cleanup"""
        collection_name = "temp_collection_123"
        
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            await milvus_client._schedule_collection_cleanup(collection_name)
            
            mock_create_task.assert_called_once()
            # Check that task is scheduled for 1 hour from now
            call_args = mock_create_task.call_args[0][0]
            assert asyncio.iscoroutinefunction(call_args)

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, milvus_client):
        """Test getting collection statistics"""
        collection_name = "test_collection"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            mock_collection.num_entities.return_value = 100
            
            result = await milvus_client.get_collection_stats(collection_name)
            
            assert result["name"] == collection_name
            assert result["num_entities"] == 100
            mock_collection.num_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_stats_error(self, milvus_client):
        """Test getting collection statistics with error"""
        collection_name = "test_collection"
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            mock_collection.num_entities.side_effect = Exception("Stats error")
            
            with pytest.raises(Exception, match="Stats error"):
                await milvus_client.get_collection_stats(collection_name)

    @pytest.mark.asyncio
    async def test_temporary_collection_context_manager(self, milvus_client):
        """Test temporary collection context manager"""
        session_id = "test-session"
        
        with patch.object(milvus_client, 'create_temporary_collection') as mock_create, \
             patch.object(milvus_client, 'drop_collection') as mock_drop, \
             patch.object(milvus_client, '_schedule_collection_cleanup') as mock_schedule:
            
            mock_collection = Mock()
            mock_create.return_value = mock_collection
            
            async with milvus_client.temporary_collection(session_id) as collection_name:
                assert collection_name is not None
                assert session_id in collection_name
            
            mock_create.assert_called_once_with(session_id=session_id)
            mock_schedule.assert_called_once_with(collection_name)


class TestMilvusVectorStore:
    """Test MilvusVectorStore class"""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings instance"""
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return embeddings

    @pytest.fixture
    def vector_store(self, mock_embeddings):
        """Create a MilvusVectorStore instance"""
        return MilvusVectorStore(
            embeddings=mock_embeddings,
            collection_name="test_collection"
        )

    def test_vector_store_init(self, vector_store, mock_embeddings):
        """Test MilvusVectorStore initialization"""
        assert vector_store.embeddings == mock_embeddings
        assert vector_store.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_create_temporary_collection(self, vector_store):
        """Test creating temporary collection"""
        session_id = "test-session"
        
        with patch('pymilvus.Collection') as mock_collection_class, \
             patch.object(vector_store, '_create_collection_schema') as mock_create_schema:
            
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            mock_schema = Mock()
            mock_create_schema.return_value = mock_schema
            
            result = await vector_store.create_temporary_collection(session_id)
            
            assert result == mock_collection
            mock_collection_class.assert_called_once()
            mock_collection.create_index.assert_called_once()
            mock_collection.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_documents(self, vector_store):
        """Test ingesting documents"""
        documents = [
            {"id": "doc1", "text": "First document", "metadata": {"source": "test"}},
            {"id": "doc2", "text": "Second document", "metadata": {"source": "test"}}
        ]
        
        with patch('pymilvus.Collection') as mock_collection_class:
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection
            
            # Mock embeddings
            vector_store.embeddings.embed_documents.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            
            await vector_store.ingest_documents(documents)
            
            # Check that collection was used
            mock_collection.insert.assert_called_once()
            mock_collection.flush.assert_called_once()


class TestMilvusClientHelpers:
    """Test helper functions for Milvus client"""

    @pytest.mark.asyncio
    async def test_get_milvus_client(self):
        """Test get_milvus_client function"""
        with patch('app.core.storage.milvus_client.MilvusClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            result = await get_milvus_client()
            
            assert result == mock_client
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_milvus_client_cached(self):
        """Test get_milvus_client function with caching"""
        with patch('app.core.storage.milvus_client.MilvusClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # First call should create and connect
            result1 = await get_milvus_client()
            assert result1 == mock_client
            mock_client.connect.assert_called_once()
            
            # Second call should return cached instance
            result2 = await get_milvus_client()
            assert result2 == mock_client
            # Should not connect again
            assert mock_client.connect.call_count == 1