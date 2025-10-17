"""
Modular RAG Pipeline Services

This package provides modular services for building RAG (Retrieval-Augmented Generation) pipelines.
Each service can be used independently or combined through the RAG orchestrator.

Services:
- QueryProcessingService: Handles query optimization and generation
- SearchService: Handles web search and content scraping
- IngestionService: Handles document chunking and embedding
- RetrievalService: Handles similarity search and reranking
- SynthesisService: Handles final answer generation
- RAGOrchestrator: Coordinates all services for complete pipeline
"""

from .query_processing import QueryProcessingService
from .search import SearchService
from .ingestion import IngestionService
from .retrieval import RetrievalService
from .synthesis import SynthesisService
from .rag_orchestrator import RAGOrchestrator

__all__ = [
    "QueryProcessingService",
    "SearchService", 
    "IngestionService",
    "RetrievalService",
    "SynthesisService",
    "RAGOrchestrator",
]