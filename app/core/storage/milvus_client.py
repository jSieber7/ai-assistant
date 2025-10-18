"""
Milvus vector database client for Deep Search agent.

This module provides a high-level interface for interacting with Milvus
vector database, including collection management, document ingestion,
and similarity search operations.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Milvus as LangChainMilvus

from ..config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    High-level Milvus client for Deep Search operations.

    Provides functionality for creating temporary collections,
    ingesting documents, and performing similarity searches.
    """

    def __init__(self, embeddings: Embeddings):
        """
        Initialize Milvus client.

        Args:
            embeddings: Embedding model instance
        """
        self.embeddings = embeddings
        self.settings = settings.milvus_settings
        self._connection_alias = "deep_search_connection"
        self._active_collections: Dict[str, Collection] = {}
        self._collection_cleanup_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self) -> bool:
        """
        Establish connection to Milvus server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Connect to Milvus
            connections.connect(
                alias=self._connection_alias,
                host=self.settings.host,
                port=self.settings.port,
                user=self.settings.user,
                password=self.settings.password,
                db_name=self.settings.database,
                timeout=self.settings.connection_timeout,
            )

            # Test connection
            from pymilvus import utility

            utility.list_collections(using=self._connection_alias)

            logger.info(
                f"Connected to Milvus at {self.settings.host}:{self.settings.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from Milvus and cleanup resources."""
        try:
            # Cancel all cleanup tasks
            for task in self._collection_cleanup_tasks.values():
                if not task.done():
                    task.cancel()

            # Disconnect
            connections.disconnect(alias=self._connection_alias)
            logger.info("Disconnected from Milvus")

        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {str(e)}")

    def _create_collection_schema(self) -> CollectionSchema:
        """
        Create the schema for Deep Search collections.

        Returns:
            CollectionSchema instance
        """
        # Define fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                auto_id=False,
                description="Document ID",
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Document text content",
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=2048,
                description="Source URL or identifier",
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.settings.embedding_dimension,
                description="Document embedding vector",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Additional metadata as JSON",
            ),
        ]

        # Create schema
        schema = CollectionSchema(
            fields=fields, description="Deep Search temporary collection"
        )

        return schema

    async def create_temporary_collection(
        self, session_id: Optional[str] = None
    ) -> str:
        """
        Create a temporary collection for a search session.

        Args:
            session_id: Optional session ID, generates one if not provided

        Returns:
            Collection name
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        collection_name = f"{self.settings.collection_prefix}{session_id}"

        try:
            # Check if collection already exists
            if utility.has_collection(collection_name, using=self._connection_alias):
                logger.warning(
                    f"Collection {collection_name} already exists, dropping it"
                )
                utility.drop_collection(collection_name, using=self._connection_alias)

            # Create schema
            schema = self._create_collection_schema()

            # Create collection
            collection = Collection(
                name=collection_name, schema=schema, using=self._connection_alias
            )

            # Create index on embedding field
            index_params = {
                "metric_type": self.settings.metric_type,
                "index_type": self.settings.index_type,
                "params": {"M": 8, "efConstruction": 64}
                if self.settings.index_type == "HNSW"
                else {},
            }

            collection.create_index(
                field_name="embedding",
                index_params=index_params,
                using=self._connection_alias,
            )

            # Load collection
            collection.load()

            # Store collection reference
            self._active_collections[collection_name] = collection

            # Schedule cleanup if auto-cleanup is enabled
            if self.settings.auto_cleanup:
                cleanup_task = asyncio.create_task(
                    self._schedule_collection_cleanup(collection_name)
                )
                self._collection_cleanup_tasks[collection_name] = cleanup_task

            logger.info(f"Created temporary collection: {collection_name}")
            return collection_name

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            raise

    async def ingest_documents(
        self, collection_name: str, documents: List[Document], batch_size: int = 100
    ) -> int:
        """
        Ingest documents into a collection.

        Args:
            collection_name: Name of the collection
            documents: List of documents to ingest
            batch_size: Batch size for processing

        Returns:
            Number of documents ingested
        """
        if collection_name not in self._active_collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._active_collections[collection_name]
        ingested_count = 0

        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Prepare batch data
                ids = []
                texts = []
                sources = []
                embeddings = []
                metadata_list = []

                for doc in batch:
                    doc_id = str(uuid.uuid4())
                    ids.append(doc_id)
                    texts.append(doc.page_content)
                    sources.append(doc.metadata.get("source", ""))
                    metadata_list.append(doc.metadata)

                # Generate embeddings for the batch
                batch_texts = [doc.page_content for doc in batch]
                embedding_vectors = await self.embeddings.aembed_documents(batch_texts)

                # Prepare data for insertion
                insert_data = [ids, texts, sources, embedding_vectors, metadata_list]

                # Insert batch
                collection.insert(insert_data)
                ingested_count += len(batch)

                logger.debug(
                    f"Inserted batch of {len(batch)} documents into {collection_name}"
                )

            # Flush to ensure data is persisted
            collection.flush()

            logger.info(
                f"Successfully ingested {ingested_count} documents into {collection_name}"
            )
            return ingested_count

        except Exception as e:
            logger.error(f"Failed to ingest documents into {collection_name}: {str(e)}")
            raise

    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 50,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search in a collection.

        Args:
            collection_name: Name of the collection
            query: Query string
            k: Number of results to return
            search_params: Additional search parameters

        Returns:
            List of (Document, score) tuples
        """
        if collection_name not in self._active_collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._active_collections[collection_name]

        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)

            # Default search parameters
            if search_params is None:
                search_params = {
                    "metric_type": self.settings.metric_type,
                    "params": {"ef": 64} if self.settings.index_type == "HNSW" else {},
                }

            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["id", "text", "source", "metadata"],
                using=self._connection_alias,
            )

            # Process results
            documents_with_scores = []
            for hit in results[0]:  # First query's results
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "id": hit.entity.get("id", ""),
                        "source": hit.entity.get("source", ""),
                        **hit.entity.get("metadata", {}),
                    },
                )
                documents_with_scores.append((doc, hit.score))

            logger.debug(
                f"Found {len(documents_with_scores)} results for query in {collection_name}"
            )
            return documents_with_scores

        except Exception as e:
            logger.error(
                f"Failed to perform similarity search in {collection_name}: {str(e)}"
            )
            raise

    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection and cleanup resources.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cancel cleanup task if exists
            if collection_name in self._collection_cleanup_tasks:
                task = self._collection_cleanup_tasks[collection_name]
                if not task.done():
                    task.cancel()
                del self._collection_cleanup_tasks[collection_name]

            # Drop collection
            if utility.has_collection(collection_name, using=self._connection_alias):
                utility.drop_collection(collection_name, using=self._connection_alias)
                logger.info(f"Dropped collection: {collection_name}")

            # Remove from active collections
            if collection_name in self._active_collections:
                del self._active_collections[collection_name]

            return True

        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            return False

    async def _schedule_collection_cleanup(self, collection_name: str):
        """
        Schedule collection cleanup after delay.

        Args:
            collection_name: Name of the collection to cleanup
        """
        try:
            await asyncio.sleep(self.settings.cleanup_delay)
            await self.drop_collection(collection_name)
            logger.info(f"Auto-cleaned collection: {collection_name}")

        except asyncio.CancelledError:
            logger.debug(f"Collection cleanup cancelled for: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup collection {collection_name}: {str(e)}")

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection statistics
        """
        if collection_name not in self._active_collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._active_collections[collection_name]

        try:
            stats = collection.describe()
            stats["num_entities"] = collection.num_entities
            return stats

        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {str(e)}")
            return {}

    @asynccontextmanager
    async def temporary_collection(self, session_id: Optional[str] = None) -> str:
        """
        Context manager for temporary collection.

        Args:
            session_id: Optional session ID

        Yields:
            Collection name
        """
        collection_name = await self.create_temporary_collection(session_id)
        try:
            yield collection_name
        finally:
            await self.drop_collection(collection_name)


class MilvusVectorStore(LangChainMilvus):
    """
    LangChain-compatible Milvus vector store for Deep Search.

    Extends LangChain's Milvus integration with additional functionality
    for temporary collections and cleanup management.
    """

    def __init__(self, embeddings: Embeddings, collection_name: str):
        """
        Initialize vector store.

        Args:
            embeddings: Embedding model
            collection_name: Name of the collection
        """
        # Initialize Milvus client
        self.milvus_client = MilvusClient(embeddings)

        # Initialize LangChain Milvus with custom connection
        super().__init__(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                "host": settings.milvus_settings.host,
                "port": settings.milvus_settings.port,
                "user": settings.milvus_settings.user,
                "password": settings.milvus_settings.password,
            },
            auto_id=False,
            timeout=settings.milvus_settings.connection_timeout,
        )

    async def connect(self) -> bool:
        """Connect to Milvus."""
        return await self.milvus_client.connect()

    async def disconnect(self):
        """Disconnect from Milvus."""
        await self.milvus_client.disconnect()

    async def create_temporary_collection(
        self, session_id: Optional[str] = None
    ) -> str:
        """Create a temporary collection."""
        return await self.milvus_client.create_temporary_collection(session_id)

    async def ingest_documents(
        self, collection_name: str, documents: List[Document]
    ) -> int:
        """Ingest documents into collection."""
        return await self.milvus_client.ingest_documents(collection_name, documents)

    async def drop_collection(self, collection_name: str) -> bool:
        """Drop a collection."""
        return await self.milvus_client.drop_collection(collection_name)
