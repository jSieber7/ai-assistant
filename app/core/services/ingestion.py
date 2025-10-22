"""
Ingestion Service for RAG Pipeline

This service handles the ingestion phase:
- Chunk Documents → Embed Chunks → Vector DB (Postgres/pgvector or Milvus)
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..storage.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for handling document ingestion into vector storage.

    This service handles document chunking, embedding generation,
    and storage in vector databases for later retrieval.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        vector_client: Optional[MilvusClient] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100,
    ):
        """
        Initialize the ingestion service.

        Args:
            embeddings: Embedding model instance
            vector_client: Vector database client (Milvus or other)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for processing
        """
        self.embeddings = embeddings
        self.vector_client = vector_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Service statistics
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "batches_processed": 0,
            "total_ingestion_time": 0.0,
            "avg_chunk_processing_time": 0.0,
        }

    async def ingest_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process and store documents in vector database.

        Args:
            documents: List of documents to ingest
            collection_name: Name of the collection (creates temporary if None)
            metadata: Additional metadata for the ingestion batch

        Returns:
            Dictionary with ingestion results
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return {"success": False, "error": "No documents provided"}

        start_time = time.time()

        try:
            # Create temporary collection if not provided
            if not collection_name:
                if not self.vector_client:
                    raise ValueError(
                        "Vector client required for automatic collection creation"
                    )

                session_id = str(uuid.uuid4())
                collection_name = await self.vector_client.create_temporary_collection(
                    session_id
                )
                logger.info(f"Created temporary collection: {collection_name}")

            # Step 1: Split documents into chunks
            chunks = await self._chunk_documents(documents)

            if not chunks:
                logger.warning("No chunks created from documents")
                return {"success": False, "error": "No chunks created"}

            # Step 2: Generate embeddings for chunks
            embedded_chunks = await self._generate_embeddings(chunks)

            # Step 3: Store in vector database
            if self.vector_client:
                ingested_count = await self.vector_client.ingest_documents(
                    collection_name, embedded_chunks, self.batch_size
                )
            else:
                logger.warning("No vector client configured, skipping storage")
                ingested_count = len(embedded_chunks)

            # Update statistics
            execution_time = time.time() - start_time
            self._update_stats(len(documents), len(chunks), execution_time)

            result = {
                "success": True,
                "collection_name": collection_name,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "chunks_ingested": ingested_count,
                "execution_time": execution_time,
                "metadata": metadata or {},
            }

            logger.info(
                f"Ingestion completed: {len(documents)} documents → {len(chunks)} chunks "
                f"→ {ingested_count} ingested in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Document ingestion failed: {str(e)}"
            logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "documents_attempted": len(documents),
                "execution_time": execution_time,
            }

    async def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of document chunks
        """
        chunks = []

        for doc in documents:
            try:
                # Split document into chunks
                doc_chunks = self.text_splitter.split_documents([doc])

                # Add chunk metadata
                for i, chunk in enumerate(doc_chunks):
                    chunk.metadata.update(
                        {
                            "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                            "chunk_index": i,
                            "total_chunks": len(doc_chunks),
                            "parent_document": doc.metadata.get("source", ""),
                            "chunking_method": "recursive_character",
                            "chunk_size": len(chunk.page_content),
                            "created_at": time.time(),
                        }
                    )
                    chunks.append(chunk)

                logger.debug(f"Split document into {len(doc_chunks)} chunks")

            except Exception as e:
                logger.error(
                    f"Error chunking document {doc.metadata.get('source', 'unknown')}: {str(e)}"
                )
                continue

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    async def _generate_embeddings(self, chunks: List[Document]) -> List[Document]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            return []

        try:
            # Process chunks in batches
            embedded_chunks = []

            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                # Generate embeddings for the batch
                batch_texts = [chunk.page_content for chunk in batch]
                embedding_vectors = await self.embeddings.aembed_documents(batch_texts)

                # Add embeddings to chunks
                for chunk, embedding in zip(batch, embedding_vectors):
                    chunk.metadata["embedding"] = embedding
                    chunk.metadata["embedding_model"] = getattr(
                        self.embeddings, "model", "unknown"
                    )
                    chunk.metadata["embedding_dimension"] = len(embedding)
                    embedded_chunks.append(chunk)

                logger.debug(f"Generated embeddings for batch of {len(batch)} chunks")

            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            return embedded_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def create_temporary_collection(
        self, session_id: Optional[str] = None
    ) -> str:
        """
        Create a temporary collection for ingestion.

        Args:
            session_id: Optional session ID

        Returns:
            Collection name
        """
        if not self.vector_client:
            raise ValueError("Vector client required for collection creation")

        if not session_id:
            session_id = str(uuid.uuid4())

        collection_name = await self.vector_client.create_temporary_collection(
            session_id
        )
        logger.info(f"Created temporary collection: {collection_name}")
        return collection_name

    async def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection.

        Args:
            collection_name: Name of the collection to drop

        Returns:
            True if successful
        """
        if not self.vector_client:
            logger.warning("No vector client configured, cannot drop collection")
            return False

        return await self.vector_client.drop_collection(collection_name)

    def _update_stats(
        self, documents_count: int, chunks_count: int, execution_time: float
    ):
        """Update service statistics."""
        self._stats["documents_processed"] += documents_count
        self._stats["chunks_created"] += chunks_count
        self._stats["embeddings_generated"] += chunks_count
        self._stats["batches_processed"] += (
            chunks_count + self.batch_size - 1
        ) // self.batch_size
        self._stats["total_ingestion_time"] += execution_time
        self._stats["avg_chunk_processing_time"] = self._stats[
            "total_ingestion_time"
        ] / max(self._stats["chunks_created"], 1)

    async def validate_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Validate documents before ingestion.

        Args:
            documents: List of documents to validate

        Returns:
            Validation results
        """
        validation_results = {
            "valid_documents": [],
            "invalid_documents": [],
            "warnings": [],
            "total_documents": len(documents),
        }

        for i, doc in enumerate(documents):
            doc_validation = {
                "index": i,
                "source": doc.metadata.get("source", f"document_{i}"),
                "valid": True,
                "issues": [],
            }

            # Check content
            if not doc.page_content or not doc.page_content.strip():
                doc_validation["valid"] = False
                doc_validation["issues"].append("Empty content")

            # Check content length
            content_length = len(doc.page_content)
            if content_length < 50:
                doc_validation["issues"].append("Very short content")
            elif content_length > 100000:
                doc_validation["issues"].append("Very long content")

            # Check metadata
            if not doc.metadata.get("source"):
                doc_validation["issues"].append("Missing source metadata")

            # Add to appropriate list
            if doc_validation["valid"]:
                validation_results["valid_documents"].append(doc_validation)
            else:
                validation_results["invalid_documents"].append(doc_validation)

            # Add warnings
            if doc_validation["issues"]:
                validation_results["warnings"].extend(
                    [
                        f"Document {i} ({doc_validation['source']}): {issue}"
                        for issue in doc_validation["issues"]
                    ]
                )

        return validation_results

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service_name": "IngestionService",
            "documents_processed": self._stats["documents_processed"],
            "chunks_created": self._stats["chunks_created"],
            "embeddings_generated": self._stats["embeddings_generated"],
            "batches_processed": self._stats["batches_processed"],
            "total_ingestion_time": self._stats["total_ingestion_time"],
            "avg_chunk_processing_time": self._stats["avg_chunk_processing_time"],
            "avg_chunks_per_document": (
                self._stats["chunks_created"]
                / max(self._stats["documents_processed"], 1)
            ),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "batch_size": self.batch_size,
                "vector_client_configured": self.vector_client is not None,
            },
        }

    def reset_stats(self):
        """Reset service statistics."""
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "batches_processed": 0,
            "total_ingestion_time": 0.0,
            "avg_chunk_processing_time": 0.0,
        }

    async def cleanup_temporary_collections(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary collections.

        Args:
            max_age_hours: Maximum age in hours for collections to keep

        Returns:
            Number of collections cleaned up
        """
        # This would need to be implemented based on the vector client's capabilities
        # For now, return 0 as placeholder
        logger.info("Temporary collection cleanup not yet implemented")
        return 0
