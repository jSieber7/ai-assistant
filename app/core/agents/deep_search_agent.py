"""
Deep Search Agent for advanced web search and RAG pipeline.

This agent implements a sophisticated search workflow that combines web search,
content scraping, vector storage, and re-ranking to provide comprehensive
answers to user queries.

This agent now uses the modular RAG services architecture for better
maintainability and reusability.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

from .base import BaseAgent, AgentResult, AgentState
from ..tools.registry import ToolRegistry
from ..tools.dynamic_executor import DynamicToolExecutor, TaskRequest, TaskType
from ..storage.milvus_client import MilvusClient
from ..services.rag_orchestrator import RAGOrchestrator
from ..services.query_processing import QueryProcessingService
from ..services.search import SearchService
from ..services.ingestion import IngestionService
from ..services.retrieval import RetrievalService
from ..services.synthesis import SynthesisService
from ..config import settings

logger = logging.getLogger(__name__)


class CustomRetriever(BaseRetriever):
    """
    Custom retriever that returns pre-retrieved documents.
    """
    
    def __init__(self, documents: List[Document]):
        super().__init__()
        self._documents = documents
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return the stored documents."""
        return self._documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._documents


class DeepSearchAgent(BaseAgent):
    """
    Advanced agent for deep web search and RAG-based answer generation.
    
    This agent implements the following workflow using modular services:
    1. Generate optimized search query from user input (QueryProcessingService)
    2. Search using SearXNG tool and scrape with Firecrawl (SearchService)
    3. Split content and create embeddings (IngestionService)
    4. Store in temporary Milvus collection (IngestionService)
    5. Retrieve with Milvus retriever (RetrievalService)
    6. Re-rank with Jina Reranker (RetrievalService)
    7. Generate final response using LCEL chain (SynthesisService)
    8. Clean up temporary collection (RAGOrchestrator)
    
    This agent maintains backward compatibility while using the new modular
    architecture internally.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm,
        embeddings: Embeddings,
        max_iterations: int = 1,
        use_modular_services: bool = True,
    ):
        super().__init__(tool_registry, max_iterations)
        self.llm = llm
        self.embeddings = embeddings
        self.milvus_client: Optional[MilvusClient] = None
        self.use_modular_services = use_modular_services
        
        # Initialize modular services if enabled
        if self.use_modular_services:
            self._initialize_modular_services()
        else:
            # Legacy initialization
            self.dynamic_executor = DynamicToolExecutor(tool_registry)
            
            # Text splitter configuration
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
        
        # Search configuration
        self.max_search_results = 5
        self.retrieval_k = 50
        self.rerank_top_n = 5

    def _initialize_modular_services(self):
        """Initialize the modular RAG services."""
        try:
            # Initialize Milvus client
            if self.milvus_client is None:
                self.milvus_client = MilvusClient(self.embeddings)
            
            # Create services
            self.query_service = QueryProcessingService(self.llm)
            self.search_service = SearchService(self.tool_registry)
            self.ingestion_service = IngestionService(self.embeddings, self.milvus_client)
            self.retrieval_service = RetrievalService(self.milvus_client, self.tool_registry)
            self.synthesis_service = SynthesisService(self.llm)
            
            # Create orchestrator
            self.rag_orchestrator = RAGOrchestrator(
                query_service=self.query_service,
                search_service=self.search_service,
                ingestion_service=self.ingestion_service,
                retrieval_service=self.retrieval_service,
                synthesis_service=self.synthesis_service,
                auto_cleanup=True
            )
            
            logger.info("Modular RAG services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize modular services: {str(e)}")
            # Fallback to legacy mode
            self.use_modular_services = False
            self.dynamic_executor = DynamicToolExecutor(self.tool_registry)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

    @property
    def name(self) -> str:
        return "deep_search"

    @property
    def description(self) -> str:
        return "Advanced agent for deep web search and comprehensive answer generation using RAG pipeline"

    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Process a message using the Deep Search workflow.
        """
        start_time = time.time()
        self.state = AgentState.THINKING

        if conversation_id:
            self._current_conversation_id = conversation_id
        else:
            self.start_conversation()

        # Add user message to conversation history
        self.add_to_conversation("user", message, {"timestamp": start_time})

        try:
            # Use modular services if enabled
            if self.use_modular_services:
                return await self._process_with_modular_services(message, context, start_time)
            else:
                return await self._process_with_legacy_implementation(message, context, start_time)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Deep search processing failed: {str(e)}"
            logger.error(error_msg)
            self.state = AgentState.ERROR

            return AgentResult(
                success=False,
                response="I encountered an error while processing your deep search request. Please try again.",
                tool_results=[],
                error=error_msg,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={"error_type": type(e).__name__},
            )

    async def _process_with_modular_services(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        start_time: float
    ) -> AgentResult:
        """
        Process message using modular RAG services.
        """
        try:
            # Initialize Milvus connection if needed
            if not await self._initialize_milvus():
                raise RuntimeError("Failed to initialize Milvus connection")

            # Prepare options for the orchestrator
            options = {
                "search_params": {
                    "results_count": self.max_search_results,
                    "category": "general"
                },
                "retrieval_k": self.retrieval_k,
                "rerank_top_n": self.rerank_top_n
            }

            # Process using the RAG orchestrator
            self.state = AgentState.EXECUTING_TOOL
            result = await self.rag_orchestrator.process_query(message, context, options)

            # Convert orchestrator result to AgentResult
            if result["success"]:
                execution_time = time.time() - start_time

                # Add assistant response to conversation history
                self.add_to_conversation(
                    "assistant",
                    result["answer"],
                    {
                        "execution_time": execution_time,
                        "search_query": result.get("metadata", {}).get("optimized_query", message),
                        "documents_processed": result.get("metadata", {}).get("documents_used", 0),
                        "collection_name": result.get("metadata", {}).get("collection_name", ""),
                        "pipeline_steps": result.get("pipeline_steps", {}),
                    },
                )

                self._usage_count += 1
                self._last_used = time.time()
                self.state = AgentState.IDLE

                return AgentResult(
                    success=True,
                    response=result["answer"],
                    tool_results=[],
                    agent_name=self.name,
                    execution_time=execution_time,
                    conversation_id=self._current_conversation_id,
                    metadata={
                        "optimized_query": result.get("metadata", {}).get("optimized_query", message),
                        "documents_count": result.get("metadata", {}).get("documents_used", 0),
                        "chunks_count": result.get("metadata", {}).get("chunks_created", 0),
                        "collection_name": result.get("metadata", {}).get("collection_name", ""),
                        "pipeline_steps": result.get("pipeline_steps", {}),
                        "sources": result.get("sources", []),
                    },
                )
            else:
                return AgentResult(
                    success=False,
                    response=result.get("answer", "I couldn't process your request. Please try again."),
                    tool_results=[],
                    error=result.get("error", "Unknown error"),
                    agent_name=self.name,
                    execution_time=time.time() - start_time,
                    conversation_id=self._current_conversation_id,
                    metadata={"error": result.get("error", "Unknown error")},
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Modular services processing failed: {str(e)}"
            logger.error(error_msg)
            
            # Fallback to legacy implementation
            logger.info("Falling back to legacy implementation")
            return await self._process_with_legacy_implementation(message, context, start_time)

    async def _process_with_legacy_implementation(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        start_time: float
    ) -> AgentResult:
        """
        Process message using legacy implementation (fallback).
        """
        # Initialize Milvus connection
        if not await self._initialize_milvus():
            raise RuntimeError("Failed to initialize Milvus connection")

        # Step 1: Generate optimized search query
        self.state = AgentState.THINKING
        optimized_query = await self._generate_search_query(message)
        logger.info(f"Generated optimized search query: {optimized_query}")

        # Step 2: Search and scrape
        self.state = AgentState.EXECUTING_TOOL
        scraped_documents = await self._search_and_scrape(optimized_query)
        
        if not scraped_documents:
            return AgentResult(
                success=False,
                response="I couldn't find relevant information for your query. Please try rephrasing or providing more specific terms.",
                tool_results=[],
                agent_name=self.name,
                execution_time=time.time() - start_time,
                conversation_id=self._current_conversation_id,
                metadata={"error": "No search results found"},
            )

        # Step 3: Create temporary Milvus collection
        session_id = str(uuid.uuid4())
        collection_name = await self.milvus_client.create_temporary_collection(session_id)
        logger.info(f"Created temporary collection: {collection_name}")

        try:
            # Step 4: Process and ingest documents
            self.state = AgentState.THINKING
            processed_docs = await self._process_documents(scraped_documents)
            await self.milvus_client.ingest_documents(collection_name, processed_docs)
            logger.info(f"Ingested {len(processed_docs)} document chunks")

            # Step 5: Retrieve, re-rank, and synthesize
            self.state = AgentState.RESPONDING
            final_answer = await self._retrieve_and_synthesize(
                collection_name, message, processed_docs
            )

            execution_time = time.time() - start_time

            # Add assistant response to conversation history
            self.add_to_conversation(
                "assistant",
                final_answer,
                {
                    "execution_time": execution_time,
                    "search_query": optimized_query,
                    "documents_processed": len(processed_docs),
                    "collection_name": collection_name,
                },
            )

            self._usage_count += 1
            self._last_used = time.time()
            self.state = AgentState.IDLE

            return AgentResult(
                success=True,
                response=final_answer,
                tool_results=[],
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={
                    "optimized_query": optimized_query,
                    "documents_count": len(scraped_documents),
                    "chunks_count": len(processed_docs),
                    "collection_name": collection_name,
                },
            )

        finally:
            # Step 6: Cleanup
            await self.milvus_client.drop_collection(collection_name)
            logger.info(f"Cleaned up collection: {collection_name}")

    async def _initialize_milvus(self) -> bool:
        """Initialize Milvus connection."""
        try:
            if self.milvus_client is None:
                self.milvus_client = MilvusClient(self.embeddings)
            
            return await self.milvus_client.connect()
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {str(e)}")
            return False

    async def _generate_search_query(self, user_query: str) -> str:
        """
        Generate an optimized search query from user input.
        
        Args:
            user_query: Original user query
            
        Returns:
            Optimized search query
        """
        # Use modular service if available
        if self.use_modular_services and hasattr(self, 'query_service'):
            return await self.query_service.generate_search_query(user_query)
        
        # Legacy implementation
        prompt = ChatPromptTemplate.from_template("""
        You are an expert at crafting effective search queries. Given the user's question, 
        generate an optimized search query that will return the most relevant and comprehensive results.
        
        User Query: {user_query}
        
        Guidelines:
        1. Use specific, relevant keywords
        2. Include context terms if helpful
        3. Keep it concise but comprehensive
        4. Focus on factual and informational content
        5. Avoid overly broad or generic terms
        
        Optimized Search Query:
        """)
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"user_query": user_query})
        
        # Extract the optimized query from the response
        if hasattr(response, 'content'):
            optimized_query = response.content.strip()
        else:
            optimized_query = str(response).strip()
        
        # Fallback to original query if generation fails
        if not optimized_query or len(optimized_query) < 3:
            optimized_query = user_query
            
        return optimized_query

    async def _search_and_scrape(self, query: str) -> List[Document]:
        """
        Search and scrape using dynamic tool execution.
        
        Args:
            query: Search query
            
        Returns:
            List of scraped documents
        """
        # Use modular service if available
        if self.use_modular_services and hasattr(self, 'search_service'):
            return await self.search_service.search_and_scrape(query)
        
        # Legacy implementation
        try:
            # Step 1: Execute search task using dynamic executor
            search_request = TaskRequest(
                task_type=TaskType.SEARCH,
                query=query,
                context={"results_count": self.max_search_results, "category": "general"},
                required_tools=["searxng_search"],
                max_tools=1
            )
            
            search_result = await self.dynamic_executor.execute_task(search_request)
            
            if not search_result.success:
                logger.error(f"Search task failed: {search_result.error}")
                return []
            
            # Extract URLs from search results
            search_data = search_result.data
            urls = []
            if isinstance(search_data, dict) and "results" in search_data:
                for result in search_data.get("results", []):
                    url = result.get("url", "")
                    if url:
                        urls.append(url)
            
            if not urls:
                logger.warning("No URLs found in search results")
                return []
            
            logger.info(f"Found {len(urls)} URLs to scrape")
            
            # Step 2: Scrape each URL using dynamic executor
            scraped_documents = []
            
            for url in urls:
                try:
                    scrape_request = TaskRequest(
                        task_type=TaskType.SCRAPE,
                        query=f"Scrape content from {url}",
                        context={"url": url},
                        required_tools=["firecrawl_scrape"],
                        parameters={
                            "formats": ["markdown"],
                            "wait_for": 2000,
                            "timeout": 30
                        },
                        max_tools=1
                    )
                    
                    scrape_result = await self.dynamic_executor.execute_task(scrape_request)
                    
                    if scrape_result.success and scrape_result.tool_results:
                        # Get the first successful tool result
                        tool_result = scrape_result.tool_results[0]
                        if tool_result.success:
                            scrape_data = tool_result.data
                            content = scrape_data.get("content", "")
                            title = scrape_data.get("title", "")
                            
                            if content and len(content.strip()) > 100:  # Filter out very short content
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": url,
                                        "title": title,
                                        "scraped_at": time.time(),
                                    }
                                )
                                scraped_documents.append(doc)
                                logger.debug(f"Successfully scraped: {url}")
                        else:
                            logger.warning(f"Failed to scrape {url}: {tool_result.error}")
                    else:
                        logger.warning(f"Scrape task failed for {url}: {scrape_result.error}")
                        
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
                    continue
            
            logger.info(f"Successfully scraped {len(scraped_documents)} documents")
            return scraped_documents
            
        except Exception as e:
            logger.error(f"Search and scrape failed: {str(e)}")
            return []

    async def _process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of document chunks
        """
        # Use modular service if available
        if self.use_modular_services and hasattr(self, 'ingestion_service'):
            # The ingestion service handles chunking internally
            # For legacy compatibility, we'll still return chunks
            return await self.ingestion_service._chunk_documents(documents)
        
        # Legacy implementation
        chunks = []
        
        for doc in documents:
            # Split document into chunks
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                })
                chunks.append(chunk)
        
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    async def _retrieve_and_synthesize(
        self,
        collection_name: str,
        original_query: str,
        documents: List[Document]
    ) -> str:
        """
        Retrieve relevant documents, re-rank, and synthesize final answer.
        
        Args:
            collection_name: Milvus collection name
            original_query: Original user query
            documents: All processed documents
            
        Returns:
            Final synthesized answer
        """
        # Use modular services if available
        if self.use_modular_services and hasattr(self, 'retrieval_service') and hasattr(self, 'synthesis_service'):
            try:
                # Retrieve and rerank documents
                retrieved_docs = await self.retrieval_service.retrieve_and_rerank(
                    original_query, collection_name, self.retrieval_k, self.rerank_top_n
                )
                
                # Synthesize answer
                synthesis_result = await self.synthesis_service.synthesize_answer(
                    original_query, retrieved_docs
                )
                
                return synthesis_result.get("answer", "I couldn't generate a comprehensive answer from the search results.")
                
            except Exception as e:
                logger.error(f"Modular retrieval and synthesis failed: {str(e)}")
                # Fallback to legacy implementation
        
        # Legacy implementation
        try:
            # Step 1: Retrieve documents from Milvus
            retrieved_docs_with_scores = await self.milvus_client.similarity_search(
                collection_name=collection_name,
                query=original_query,
                k=self.retrieval_k
            )
            
            retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
            logger.info(f"Retrieved {len(retrieved_docs)} documents from Milvus")
            
            # Step 2: Re-rank with Jina Reranker using dynamic executor
            if settings.jina_reranker_enabled:
                try:
                    # Extract content for reranking
                    doc_contents = [doc.page_content for doc in retrieved_docs]
                    
                    rerank_request = TaskRequest(
                        task_type=TaskType.RERANK,
                        query=original_query,
                        context={"documents": doc_contents},
                        required_tools=["jina_reranker"],
                        parameters={
                            "documents": doc_contents,
                            "top_n": self.rerank_top_n
                        },
                        max_tools=1
                    )
                    
                    rerank_result = await self.dynamic_executor.execute_task(rerank_request)
                    
                    if rerank_result.success and rerank_result.tool_results:
                        tool_result = rerank_result.tool_results[0]
                        if tool_result.success:
                            rerank_data = tool_result.data
                            reranked_indices = [item.get("index") for item in rerank_data.get("results", [])]
                            
                            # Reorder documents based on reranking
                            reranked_docs = [retrieved_docs[i] for i in reranked_indices if 0 <= i < len(retrieved_docs)]
                            retrieved_docs = reranked_docs[:self.rerank_top_n]
                            logger.info(f"Re-ranked to {len(retrieved_docs)} top documents")
                        else:
                            logger.warning(f"Jina reranking failed: {tool_result.error}")
                    else:
                        logger.warning(f"Rerank task failed: {rerank_result.error}")
                        
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}")
            
            # Step 3: Synthesize final answer using LCEL chain
            return await self._synthesize_answer(original_query, retrieved_docs)
            
        except Exception as e:
            logger.error(f"Error in retrieve and synthesize: {str(e)}")
            return "I encountered an error while processing the search results. Please try again."

    async def _synthesize_answer(self, query: str, documents: List[Document]) -> str:
        """
        Synthesize final answer from retrieved documents using LCEL chain.
        
        Args:
            query: Original user query
            documents: Retrieved and reranked documents
            
        Returns:
            Synthesized answer
        """
        # Use modular service if available
        if self.use_modular_services and hasattr(self, 'synthesis_service'):
            synthesis_result = await self.synthesis_service.synthesize_answer(query, documents)
            return synthesis_result.get("answer", "I couldn't generate a comprehensive answer from the search results.")
        
        # Legacy implementation
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Based on the provided documents,
        generate a comprehensive and accurate answer to the user's question.
        
        User Question: {input}
        
        Context Documents:
        {context}
        
        Guidelines:
        1. Use only information from the provided documents
        2. Synthesize information from multiple sources when relevant
        3. Be comprehensive but concise
        4. Cite sources when possible using the provided URLs
        5. If the documents don't fully answer the question, acknowledge the limitations
        6. Use a clear, structured format
        
        Answer:
        """)
        
        # Create document processing chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retriever = CustomRetriever(documents)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Invoke the chain
        response = await retrieval_chain.ainvoke({"input": query})
        
        return response.get("answer", "I couldn't generate a comprehensive answer from the search results.")

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics including service stats."""
        base_stats = {
            "agent_name": self.name,
            "use_modular_services": self.use_modular_services,
            "max_search_results": self.max_search_results,
            "retrieval_k": self.retrieval_k,
            "rerank_top_n": self.rerank_top_n,
        }
        
        # Add service stats if using modular services
        if self.use_modular_services and hasattr(self, 'rag_orchestrator'):
            base_stats["service_stats"] = self.rag_orchestrator.get_orchestrator_stats()
        
        return base_stats

    async def cleanup_resources(self):
        """Clean up agent resources."""
        try:
            # Cleanup orchestrator resources
            if self.use_modular_services and hasattr(self, 'rag_orchestrator'):
                await self.rag_orchestrator.cleanup_all_collections()
            
            # Cleanup Milvus connection
            if self.milvus_client:
                await self.milvus_client.disconnect()
                
            logger.info("DeepSearchAgent resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")