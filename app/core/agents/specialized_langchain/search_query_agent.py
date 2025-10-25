"""
SearchQueryAgent - A specialized agent for generating optimized search queries using LangChain.

This agent can generate effective search queries, analyze search intent,
and optimize queries for different search engines and purposes.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class SearchQueryState(BaseModel):
    """State for search query generation workflow"""
    original_query: str = Field(description="Original user query")
    search_context: Optional[str] = Field(default=None, description="Additional context for search")
    search_purpose: str = Field(default="general", description="Purpose of search: general, research, shopping, news, academic")
    target_engine: str = Field(default="google", description="Target search engine: google, bing, duckduckgo, searx")
    query_count: int = Field(default=5, description="Number of queries to generate")
    language: str = Field(default="english", description="Language for search queries")
    location: Optional[str] = Field(default=None, description="Geographic location for localized search")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    generated_queries: List[str] = Field(default_factory=list, description="Generated search queries")
    query_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of original query")
    optimized_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Optimized queries with metadata")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether query generation was successful")


class SearchQueryAgent:
    """
    A specialized agent for generating optimized search queries using LangChain.
    
    This agent can analyze search intent, generate multiple query variations,
    and optimize queries for different search engines and purposes.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SearchQueryAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "search_query_agent"
        self.description = "Specialized agent for generating optimized search queries"
        
        # LangGraph workflow
        self.workflow = None
        self.checkpointer = MemorySaver()
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize components asynchronously"""
        try:
            if not self.llm:
                self.llm = await get_llm()
            
            # Create LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("SearchQueryAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SearchQueryAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for search query generation"""
        workflow = StateGraph(SearchQueryState)
        
        # Define nodes
        def analyze_query(state: SearchQueryState) -> SearchQueryState:
            """Analyze the original query to understand intent and extract key concepts"""
            try:
                # Create analysis prompt
                prompt = f"""
                Analyze the following search query to understand the user's intent and extract key information:
                
                Query: "{state.original_query}"
                Context: {state.search_context or 'None'}
                Purpose: {state.search_purpose}
                Language: {state.language}
                
                Please provide:
                1. Search intent (informational, navigational, transactional, commercial)
                2. Key entities and concepts
                3. Query type (question, phrase, keywords, comparison)
                4. Specific requirements (recent, local, academic, etc.)
                5. Suggested improvements
                
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    # Parse analysis as JSON
                    import json
                    analysis = json.loads(response.content)
                    state.query_analysis = analysis
                    state.messages.append(SystemMessage(content="Query analysis completed"))
                except json.JSONDecodeError:
                    # Fallback analysis
                    state.query_analysis = {
                        "intent": "informational",
                        "entities": [],
                        "query_type": "keywords",
                        "requirements": [],
                        "suggestions": [],
                        "raw_analysis": response.content
                    }
                    state.messages.append(SystemMessage(content="Query analysis completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error analyzing query: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_base_queries(state: SearchQueryState) -> SearchQueryState:
            """Generate base search query variations"""
            try:
                # Create query generation prompt
                prompt = f"""
                Generate {state.query_count} effective search queries based on the following:
                
                Original Query: "{state.original_query}"
                Analysis: {state.query_analysis}
                Purpose: {state.search_purpose}
                Target Engine: {state.target_engine}
                Language: {state.language}
                
                Guidelines:
                1. Create variations with different keyword combinations
                2. Use appropriate search operators for the target engine
                3. Include both broad and specific queries
                4. Consider synonyms and related terms
                5. Optimize for the search purpose
                
                Generate exactly {state.query_count} queries, one per line.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                # Parse queries
                queries = [q.strip() for q in response.content.split('\n') if q.strip()]
                state.generated_queries = queries[:state.query_count]
                state.messages.append(SystemMessage(content=f"Generated {len(state.generated_queries)} base queries"))
                
            except Exception as e:
                state.error = f"Error generating queries: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def optimize_for_engine(state: SearchQueryState) -> SearchQueryState:
            """Optimize queries for the target search engine"""
            try:
                optimized = []
                
                for query in state.generated_queries:
                    # Create optimization prompt
                    prompt = f"""
                    Optimize the following search query for {state.target_engine}:
                    
                    Query: "{query}"
                    Purpose: {state.search_purpose}
                    Language: {state.language}
                    Location: {state.location or 'Not specified'}
                    
                    Provide:
                    1. Optimized query with proper syntax
                    2. Expected improvement
                    3. Alternative variations
                    
                    Respond with JSON format.
                    """
                    
                    messages = [HumanMessage(content=prompt)]
                    response = self.llm.invoke(messages)
                    
                    try:
                        import json
                        optimization = json.loads(response.content)
                        
                        optimized_query = {
                            "original": query,
                            "optimized": optimization.get("optimized", query),
                            "improvement": optimization.get("improvement", "General optimization"),
                            "alternatives": optimization.get("alternatives", []),
                            "engine": state.target_engine,
                        }
                        optimized.append(optimized_query)
                        
                    except json.JSONDecodeError:
                        # Fallback optimization
                        optimized_query = {
                            "original": query,
                            "optimized": query,
                            "improvement": "Basic optimization",
                            "alternatives": [],
                            "engine": state.target_engine,
                        }
                        optimized.append(optimized_query)
                
                state.optimized_queries = optimized
                state.messages.append(SystemMessage(content=f"Optimized {len(optimized)} queries for {state.target_engine}"))
                
            except Exception as e:
                state.error = f"Error optimizing queries: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def rank_queries(state: SearchQueryState) -> SearchQueryState:
            """Rank queries by expected effectiveness"""
            try:
                if not state.optimized_queries:
                    state.success = False
                    return state
                
                # Create ranking prompt
                queries_text = "\n".join([
                    f"{i+1}. {q['optimized']}" for i, q in enumerate(state.optimized_queries)
                ])
                
                prompt = f"""
                Rank the following search queries by expected effectiveness for {state.search_purpose} search:
                
                {queries_text}
                
                Consider:
                1. Relevance to original query
                2. Likelihood of finding good results
                3. Specificity and focus
                4. Search engine compatibility
                
                Rank from best (1) to worst ({len(state.optimized_queries)}).
                Provide ranking numbers and brief justification.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                # Simple ranking extraction (could be made more sophisticated)
                rankings = []
                lines = response.content.split('\n')
                
                for line in lines:
                    if line.strip() and any(char.isdigit() for char in line):
                        # Extract ranking
                        import re
                        match = re.search(r'(\d+)', line)
                        if match:
                            rank = int(match.group(1)) - 1  # Convert to 0-based
                            if 0 <= rank < len(state.optimized_queries):
                                rankings.append(rank)
                
                # Apply rankings
                if rankings:
                    ranked_queries = []
                    for rank in rankings:
                        if rank < len(state.optimized_queries):
                            query = state.optimized_queries[rank]
                            query["rank"] = rankings.index(rank) + 1
                            ranked_queries.append(query)
                    
                    # Add unranked queries
                    for i, query in enumerate(state.optimized_queries):
                        if i not in rankings:
                            query["rank"] = len(ranked_queries) + 1
                            ranked_queries.append(query)
                    
                    state.optimized_queries = ranked_queries
                    state.messages.append(SystemMessage(content="Queries ranked by effectiveness"))
                
            except Exception as e:
                state.messages.append(SystemMessage(content=f"Query ranking failed: {str(e)}"))
            
            state.success = len(state.optimized_queries) > 0
            return state
        
        # Add nodes to workflow
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("generate_base_queries", generate_base_queries)
        workflow.add_node("optimize_for_engine", optimize_for_engine)
        workflow.add_node("rank_queries", rank_queries)
        
        # Set up workflow
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "generate_base_queries")
        workflow.add_edge("generate_base_queries", "optimize_for_engine")
        workflow.add_edge("optimize_for_engine", "rank_queries")
        workflow.add_edge("rank_queries", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def generate_queries(
        self,
        query: str,
        search_context: Optional[str] = None,
        search_purpose: str = "general",
        target_engine: str = "google",
        query_count: int = 5,
        language: str = "english",
        location: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate optimized search queries.
        
        Args:
            query: Original search query
            search_context: Additional context for search
            search_purpose: Purpose of search (general, research, shopping, news, academic)
            target_engine: Target search engine (google, bing, duckduckgo, searx)
            query_count: Number of queries to generate
            language: Language for search queries
            location: Geographic location for localized search
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing generated queries and metadata
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize SearchQueryAgent workflow")
        
        try:
            # Create initial state
            state = SearchQueryState(
                original_query=query,
                search_context=search_context,
                search_purpose=search_purpose,
                target_engine=target_engine,
                query_count=query_count,
                language=language,
                location=location,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "original_query": result.original_query,
                "query_analysis": result.query_analysis,
                "generated_queries": result.generated_queries,
                "optimized_queries": result.optimized_queries,
                "total_queries": len(result.optimized_queries),
                "search_purpose": result.search_purpose,
                "target_engine": result.target_engine,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in query generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_query": query,
                "optimized_queries": [],
            }
    
    async def stream_generate_queries(
        self,
        query: str,
        search_context: Optional[str] = None,
        search_purpose: str = "general",
        target_engine: str = "google",
        query_count: int = 5,
        language: str = "english",
        location: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query generation process for real-time updates.
        
        Args:
            query: Original search query
            search_context: Additional context for search
            search_purpose: Purpose of search
            target_engine: Target search engine
            query_count: Number of queries to generate
            language: Language for search queries
            location: Geographic location for localized search
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize SearchQueryAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = SearchQueryState(
                original_query=query,
                search_context=search_context,
                search_purpose=search_purpose,
                target_engine=target_engine,
                query_count=query_count,
                language=language,
                location=location,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield analysis updates
                if "analyze_query" in event:
                    yield {
                        "type": "analysis_complete",
                        "analysis": list(event.values())[0].query_analysis,
                    }
                
                # Yield query generation updates
                if "generate_base_queries" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'generated_queries') and node_state.generated_queries:
                        yield {
                            "type": "queries_generated",
                            "queries": node_state.generated_queries,
                            "count": len(node_state.generated_queries),
                        }
                
                # Yield optimization updates
                if "optimize_for_engine" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'optimized_queries') and node_state.optimized_queries:
                        yield {
                            "type": "queries_optimized",
                            "optimized_queries": node_state.optimized_queries,
                            "engine": node_state.target_engine,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "generation_complete",
                        "success": final_state.success,
                        "optimized_queries": final_state.optimized_queries,
                        "query_analysis": final_state.query_analysis,
                        "total_queries": len(final_state.optimized_queries),
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming query generation: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }