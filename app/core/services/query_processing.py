"""
Query Processing Service for RAG Pipeline

This service handles the initial query processing phase:
- User Query → Agent LLM → Search Query Generation
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class QueryProcessingService:
    """
    Service for processing and optimizing user queries for search.

    This service takes user queries and generates optimized search queries
    that will return the most relevant and comprehensive results.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the query processing service.

        Args:
            llm: Language model for query optimization
        """
        self.llm = llm
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for query optimization."""
        return ChatPromptTemplate.from_template("""
        You are an expert at crafting effective search queries. Given the user's question, 
        generate an optimized search query that will return the most relevant and comprehensive results.
        
        User Query: {user_query}
        
        Guidelines:
        1. Use specific, relevant keywords
        2. Include context terms if helpful
        3. Keep it concise but comprehensive
        4. Focus on factual and informational content
        5. Avoid overly broad or generic terms
        6. Use quotes for exact phrases when helpful
        7. Include site: operators for specific sources if relevant
        
        Optimized Search Query:
        """)

    async def generate_search_query(
        self, user_query: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an optimized search query from user input.

        Args:
            user_query: Original user query
            context: Additional context for query optimization

        Returns:
            Optimized search query
        """
        try:
            # Prepare context information
            context_info = ""
            if context:
                if context.get("domain"):
                    context_info += f"Domain focus: {context['domain']}\n"
                if context.get("time_range"):
                    context_info += f"Time range: {context['time_range']}\n"
                if context.get("content_type"):
                    context_info += f"Content type: {context['content_type']}\n"

            # Create enhanced prompt if context is provided
            if context_info:
                enhanced_prompt = ChatPromptTemplate.from_template(f"""
                You are an expert at crafting effective search queries. Given the user's question and context,
                generate an optimized search query that will return the most relevant and comprehensive results.
                
                User Query: {{user_query}}
                
                Context:
                {context_info}
                
                Guidelines:
                1. Use specific, relevant keywords
                2. Include context terms if helpful
                3. Keep it concise but comprehensive
                4. Focus on factual and informational content
                5. Avoid overly broad or generic terms
                6. Use quotes for exact phrases when helpful
                7. Include site: operators for specific sources if relevant
                
                Optimized Search Query:
                """)
                chain = enhanced_prompt | self.llm
            else:
                chain = self._prompt_template | self.llm

            # Generate optimized query
            response = await chain.ainvoke({"user_query": user_query})

            # Extract the optimized query from the response
            if hasattr(response, "content"):
                optimized_query = response.content.strip()
            else:
                optimized_query = str(response).strip()

            # Fallback to original query if generation fails
            if not optimized_query or len(optimized_query) < 3:
                logger.warning(
                    f"Query optimization failed, using original query: {user_query}"
                )
                return user_query

            logger.info(
                f"Generated optimized search query: '{optimized_query}' from '{user_query}'"
            )
            return optimized_query

        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            # Fallback to original query
            return user_query

    async def batch_generate_queries(
        self, user_queries: list[str], context: Optional[Dict[str, Any]] = None
    ) -> list[str]:
        """
        Generate optimized search queries for multiple user queries.

        Args:
            user_queries: List of original user queries
            context: Additional context for query optimization

        Returns:
            List of optimized search queries
        """
        optimized_queries = []

        for query in user_queries:
            optimized_query = await self.generate_search_query(query, context)
            optimized_queries.append(optimized_query)

        return optimized_queries

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and analyze a search query.

        Args:
            query: Search query to validate

        Returns:
            Dictionary with validation results
        """
        analysis = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "query_type": "general",
            "complexity": "simple",
        }

        # Check for empty query
        if not query or not query.strip():
            analysis["valid"] = False
            analysis["issues"].append("Query is empty")
            return analysis

        query = query.strip()

        # Check query length
        if len(query) < 3:
            analysis["issues"].append("Query is too short")
            analysis["valid"] = False

        if len(query) > 200:
            analysis["issues"].append("Query is too long")
            analysis["suggestions"].append("Consider shortening the query")

        # Analyze query complexity
        if '"' in query or "'" in query:
            analysis["complexity"] = "medium"

        if any(
            operator in query.lower()
            for operator in ["site:", "filetype:", "intitle:", "inurl:"]
        ):
            analysis["complexity"] = "advanced"
            analysis["query_type"] = "advanced"

        # Check for common issues
        if query.count('"') % 2 != 0:
            analysis["issues"].append("Unmatched quotes")
            analysis["suggestions"].append("Check quote matching")

        # Provide suggestions
        if not any(
            word in query.lower()
            for word in ["what", "how", "why", "when", "where", "who"]
        ):
            if "?" not in query:
                analysis["suggestions"].append(
                    "Consider adding question words for better results"
                )

        return analysis

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and configuration."""
        return {
            "service_name": "QueryProcessingService",
            "llm_configured": self.llm is not None,
            "prompt_template_loaded": self._prompt_template is not None,
            "supported_features": [
                "query_optimization",
                "batch_processing",
                "query_validation",
                "context_aware_processing",
            ],
        }
