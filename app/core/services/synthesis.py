"""
Synthesis Service for RAG Pipeline

This service handles the synthesis phase:
- Final LLM Prompt → Final Synthesized Answer
"""

import logging
import time
from typing import Dict, Any, List, Optional
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document

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


class SynthesisService:
    """
    Service for synthesizing final answers from retrieved documents.
    
    This service takes retrieved documents and user queries to generate
    comprehensive, accurate answers using language models.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: Optional[str] = None,
        max_context_length: int = 8000
    ):
        """
        Initialize the synthesis service.
        
        Args:
            llm: Language model for answer generation
            prompt_template: Custom prompt template (uses default if None)
            max_context_length: Maximum context length for documents
        """
        self.llm = llm
        self.max_context_length = max_context_length
        
        # Create prompt template
        if prompt_template:
            self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        else:
            self.prompt_template = self._create_default_prompt()
        
        # Service statistics
        self._stats = {
            "syntheses_performed": 0,
            "total_documents_processed": 0,
            "total_synthesis_time": 0.0,
            "avg_synthesis_time": 0.0,
            "context_length_truncations": 0
        }
    
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """Create the default prompt template for answer synthesis."""
        return ChatPromptTemplate.from_template("""
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
        7. Include specific details and examples from the sources
        8. Maintain objectivity and avoid speculation
        
        Answer:
        """)
    
    async def synthesize_answer(
        self,
        query: str,
        documents: List[Document],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a final answer from retrieved documents.
        
        Args:
            query: Original user query
            documents: Retrieved and reranked documents
            context: Additional context for synthesis
            
        Returns:
            Dictionary with synthesized answer and metadata
        """
        start_time = time.time()
        
        try:
            if not documents:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "success": False,
                    "error": "No documents provided",
                    "metadata": {"query": query}
                }
            
            # Step 1: Prepare documents for synthesis
            prepared_docs = await self._prepare_documents(documents)
            
            if not prepared_docs:
                return {
                    "answer": "The retrieved documents don't contain sufficient information to answer your question.",
                    "sources": self._extract_sources(documents),
                    "success": False,
                    "error": "No valid documents after preparation",
                    "metadata": {"query": query}
                }
            
            # Step 2: Generate answer
            answer = await self._generate_answer(query, prepared_docs, context)
            
            # Step 3: Extract sources and create metadata
            sources = self._extract_sources(prepared_docs)
            metadata = self._create_metadata(query, prepared_docs, context)
            
            # Update statistics
            synthesis_time = time.time() - start_time
            self._update_stats(len(documents), synthesis_time)
            
            result = {
                "answer": answer,
                "sources": sources,
                "success": True,
                "metadata": metadata,
                "synthesis_time": synthesis_time
            }
            
            logger.info(f"Answer synthesis completed in {synthesis_time:.2f}s "
                       f"using {len(prepared_docs)} documents")
            
            return result
            
        except Exception as e:
            synthesis_time = time.time() - start_time
            error_msg = f"Answer synthesis failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "answer": "I encountered an error while generating an answer to your question. Please try again.",
                "sources": self._extract_sources(documents) if documents else [],
                "success": False,
                "error": error_msg,
                "metadata": {"query": query, "synthesis_time": synthesis_time}
            }
    
    async def _prepare_documents(self, documents: List[Document]) -> List[Document]:
        """
        Prepare documents for synthesis by truncating and formatting.
        
        Args:
            documents: List of documents to prepare
            
        Returns:
            List of prepared documents
        """
        prepared_docs = []
        current_length = 0
        
        for doc in documents:
            content = doc.page_content
            
            # Check if adding this document would exceed the limit
            if current_length + len(content) > self.max_context_length:
                # Truncate the document to fit
                remaining_space = self.max_context_length - current_length
                if remaining_space > 100:  # Only include if meaningful content remains
                    content = content[:remaining_space] + "..."
                    self._stats["context_length_truncations"] += 1
                    
                    # Create truncated document
                    truncated_doc = Document(
                        page_content=content,
                        metadata={
                            **doc.metadata,
                            "truncated": True,
                            "original_length": len(doc.page_content)
                        }
                    )
                    prepared_docs.append(truncated_doc)
                    current_length += len(content)
                break
            else:
                prepared_docs.append(doc)
                current_length += len(content)
        
        logger.debug(f"Prepared {len(prepared_docs)} documents for synthesis "
                    f"(total length: {current_length} characters)")
        
        return prepared_docs
    
    async def _generate_answer(
        self,
        query: str,
        documents: List[Document],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate the final answer using LCEL chain.
        
        Args:
            query: Original user query
            documents: Prepared documents
            context: Additional context
            
        Returns:
            Generated answer
        """
        try:
            # Create document processing chain
            document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
            
            # Create custom retriever
            retriever = CustomRetriever(documents)
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Prepare input with context
            chain_input = {"input": query}
            if context:
                chain_input.update(context)
            
            # Invoke the chain
            response = await retrieval_chain.ainvoke(chain_input)
            
            answer = response.get("answer", "")
            
            if not answer:
                logger.warning("Empty answer generated from chain")
                return "I couldn't generate a comprehensive answer from the provided documents."
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer with chain: {str(e)}")
            # Fallback to simple LLM call
            try:
                fallback_prompt = f"""
                Based on these documents, please answer the question: {query}
                
                Documents:
                {chr(10).join([f"- {doc.page_content[:500]}..." for doc in documents[:3]])}
                
                Provide a concise answer:
                """
                
                response = await self.llm.ainvoke(fallback_prompt)
                if hasattr(response, 'content'):
                    return response.content.strip()
                else:
                    return str(response).strip()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback answer generation also failed: {str(fallback_error)}")
                return "I encountered difficulties generating an answer. Please try rephrasing your question."
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of source information
        """
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get("source", "")
            title = doc.metadata.get("title", "")
            
            if source and source not in seen_sources:
                sources.append({
                    "source": source,
                    "title": title,
                    "similarity_score": doc.metadata.get("similarity_score"),
                    "rerank_score": doc.metadata.get("rerank_score")
                })
                seen_sources.add(source)
        
        return sources
    
    def _create_metadata(
        self,
        query: str,
        documents: List[Document],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create metadata for the synthesis result.
        
        Args:
            query: Original query
            documents: Documents used
            context: Additional context
            
        Returns:
            Metadata dictionary
        """
        return {
            "query": query,
            "documents_used": len(documents),
            "total_content_length": sum(len(doc.page_content) for doc in documents),
            "sources_count": len(set(doc.metadata.get("source", "") for doc in documents)),
            "has_context": context is not None,
            "synthesis_method": "lcel_chain",
            "model": getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown')),
            "max_context_length": self.max_context_length
        }
    
    async def batch_synthesize(
        self,
        queries: List[str],
        documents_list: List[List[Document]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesize answers for multiple queries.
        
        Args:
            queries: List of queries
            documents_list: List of document lists for each query
            context: Additional context for all syntheses
            
        Returns:
            List of synthesis results
        """
        if len(queries) != len(documents_list):
            raise ValueError("Number of queries must match number of document lists")
        
        results = []
        
        for query, documents in zip(queries, documents_list):
            result = await self.synthesize_answer(query, documents, context)
            results.append(result)
        
        return results
    
    def _update_stats(self, documents_count: int, synthesis_time: float):
        """Update service statistics."""
        self._stats["syntheses_performed"] += 1
        self._stats["total_documents_processed"] += documents_count
        self._stats["total_synthesis_time"] += synthesis_time
        self._stats["avg_synthesis_time"] = (
            self._stats["total_synthesis_time"] / max(self._stats["syntheses_performed"], 1)
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service_name": "SynthesisService",
            "syntheses_performed": self._stats["syntheses_performed"],
            "total_documents_processed": self._stats["total_documents_processed"],
            "total_synthesis_time": self._stats["total_synthesis_time"],
            "avg_synthesis_time": self._stats["avg_synthesis_time"],
            "context_length_truncations": self._stats["context_length_truncations"],
            "avg_documents_per_synthesis": (
                self._stats["total_documents_processed"] / max(self._stats["syntheses_performed"], 1)
            ),
            "config": {
                "max_context_length": self.max_context_length,
                "llm_configured": self.llm is not None,
                "prompt_template_loaded": self.prompt_template is not None
            }
        }
    
    def reset_stats(self):
        """Reset service statistics."""
        self._stats = {
            "syntheses_performed": 0,
            "total_documents_processed": 0,
            "total_synthesis_time": 0.0,
            "avg_synthesis_time": 0.0,
            "context_length_truncations": 0
        }
    
    def update_prompt_template(self, new_template: str):
        """
        Update the prompt template.
        
        Args:
            new_template: New prompt template string
        """
        self.prompt_template = ChatPromptTemplate.from_template(new_template)
        logger.info("Updated synthesis prompt template")