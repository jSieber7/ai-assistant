"""
FactCheckerAgent - A specialized agent for fact verification and validation using LangChain.

This agent can verify claims, check facts against reliable sources,
identify misinformation, and provide evidence-based assessments.
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class FactClaim(BaseModel):
    """Fact claim to be verified"""
    claim: str = Field(description="The claim statement to verify")
    claim_type: str = Field(description="Type of claim: statistical, factual, historical, scientific, etc.")
    key_entities: List[str] = Field(description="Key entities mentioned in the claim")
    confidence: float = Field(description="Initial confidence in claim validity (0.0-1.0)")


class EvidenceSource(BaseModel):
    """Evidence source for fact checking"""
    source: str = Field(description="Source name or URL")
    title: str = Field(description="Title of the source content")
    content: str = Field(description="Relevant content from the source")
    reliability_score: float = Field(description="Reliability score (0.0-1.0)")
    publication_date: Optional[str] = Field(default=None, description="Publication date")
    author: Optional[str] = Field(default=None, description="Author or organization")


class FactCheckResult(BaseModel):
    """Result of fact checking a claim"""
    claim: str = Field(description="The claim that was checked")
    verdict: str = Field(description="Verdict: true, false, mostly_true, mostly_false, half_true, misleading, unverified")
    confidence: float = Field(description="Confidence in verdict (0.0-1.0)")
    explanation: str = Field(description="Explanation for the verdict")
    supporting_evidence: List[EvidenceSource] = Field(description="Evidence supporting the verdict")
    contradictory_evidence: List[EvidenceSource] = Field(description="Evidence contradicting the claim")
    context: str = Field(description="Additional context for understanding the claim")
    limitations: List[str] = Field(description="Limitations of the fact check")


class FactCheckingState(BaseModel):
    """State for fact checking workflow"""
    text: str = Field(description="Text containing claims to verify")
    claims: List[FactClaim] = Field(default_factory=list, description="Extracted claims")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    fact_check_results: List[FactCheckResult] = Field(default_factory=list, description="Fact checking results")
    overall_assessment: Optional[str] = Field(default=None, description="Overall assessment of the text")
    search_queries: List[str] = Field(default_factory=list, description="Generated search queries")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether fact checking was successful")


class FactCheckerAgent:
    """
    A specialized agent for fact verification and validation using LangChain.
    
    This agent can extract claims from text, search for evidence,
    verify facts, and provide evidence-based assessments.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FactCheckerAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "fact_checker_agent"
        self.description = "Specialized agent for fact verification and validation"
        
        # Reliable source patterns for verification
        self.reliable_domains = [
            "reuters.com", "ap.org", "bbc.com", "npr.org", "pbs.org",
            "nature.com", "science.org", "nejm.org", "thelancet.com",
            "who.int", "cdc.gov", "nih.gov", "gov.uk", "europa.eu",
            "factcheck.org", "snopes.com", "politifact.com", "washingtonpost.com"
        ]
        
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
            
            logger.info("FactCheckerAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FactCheckerAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for fact checking"""
        workflow = StateGraph(FactCheckingState)
        
        # Define nodes
        def extract_claims(state: FactCheckingState) -> FactCheckingState:
            """Extract factual claims from the text"""
            try:
                # Create claim extraction prompt
                prompt = f"""
                Extract factual claims from the following text that can be verified:
                
                Text: "{state.text}"
                
                Please identify:
                1. Specific factual claims (not opinions or predictions)
                2. Statistical claims with numbers
                3. Historical claims
                4. Scientific claims
                5. Claims about specific events or people
                
                For each claim, provide:
                - The exact claim statement
                - Type of claim (statistical, factual, historical, scientific)
                - Key entities mentioned
                - Initial confidence in claim validity
                
                Focus on claims that can be objectively verified.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    claims_data = json.loads(response.content)
                    claims = []
                    
                    for claim_data in claims_data.get("claims", []):
                        claim = FactClaim(
                            claim=claim_data.get("claim", ""),
                            claim_type=claim_data.get("claim_type", "factual"),
                            key_entities=claim_data.get("key_entities", []),
                            confidence=claim_data.get("confidence", 0.5)
                        )
                        claims.append(claim)
                    
                    state.claims = claims
                    state.messages.append(SystemMessage(content=f"Extracted {len(claims)} claims"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback claim extraction
                    state.claims = []
                    state.messages.append(SystemMessage(content="Claim extraction completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error extracting claims: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_search_queries(state: FactCheckingState) -> FactCheckingState:
            """Generate search queries for fact checking"""
            try:
                # Create search query generation prompt
                claims_text = "\n".join([
                    f"- {claim.claim} (Type: {claim.claim_type})"
                    for claim in state.claims
                ])
                
                prompt = f"""
                Generate effective search queries to verify the following claims:
                
                Claims:
                {claims_text}
                
                For each claim, create 2-3 search queries that:
                1. Target reliable sources (news organizations, academic journals, government sites)
                2. Use specific terms and entities from the claim
                3. Include fact-checking terms like "fact check", "verification", "evidence"
                4. Are optimized for finding contradictory evidence if it exists
                
                Focus on queries that will return authoritative sources.
                Respond with JSON format listing all search queries.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    queries_data = json.loads(response.content)
                    search_queries = queries_data.get("search_queries", [])
                    
                    state.search_queries = search_queries
                    state.messages.append(SystemMessage(content=f"Generated {len(search_queries)} search queries"))
                    
                except json.JSONDecodeError:
                    # Fallback query generation
                    fallback_queries = [
                        f"fact check: {claim.claim[:50]}..."
                        for claim in state.claims[:3]
                    ]
                    state.search_queries = fallback_queries
                    state.messages.append(SystemMessage(content="Search queries generated (fallback)"))
                
            except Exception as e:
                state.error = f"Error generating search queries: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def search_for_evidence(state: FactCheckingState) -> FactCheckingState:
            """Search for evidence (simulated for this implementation)"""
            try:
                # In a real implementation, this would use search tools
                # For now, we'll simulate evidence gathering
                evidence_sources = []
                
                for claim in state.claims:
                    # Simulate finding evidence sources
                    for i, query in enumerate(state.search_queries[:2]):  # Limit for demo
                        # Simulate reliable source
                        reliable_domains = self.reliable_domains[:5]
                        domain = reliable_domains[i % len(reliable_domains)]
                        
                        # Determine if evidence supports or contradicts claim
                        # For demo, alternate between supporting and contradicting
                        is_supporting = (i % 2 == 0)
                        
                        evidence = EvidenceSource(
                            source=f"https://{domain}/article/{hash(query) % 10000}",
                            title=f"Evidence related to: {query[:30]}...",
                            content=f"Simulated content that {'supports' if is_supporting else 'contradicts'} aspects of the claim: {claim.claim[:50]}...",
                            reliability_score=0.8,
                            publication_date="2023-01-01",
                            author=f"Author from {domain}"
                        )
                        evidence_sources.append(evidence)
                
                # Store evidence for later use
                state.evidence_sources = evidence_sources
                state.messages.append(SystemMessage(content=f"Found {len(evidence_sources)} evidence sources"))
                
            except Exception as e:
                state.error = f"Error searching for evidence: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def verify_claims(state: FactCheckingState) -> FactCheckingState:
            """Verify claims against evidence"""
            try:
                # Create claim verification prompt
                claims_text = "\n".join([
                    f"- {claim.claim} (Type: {claim.claim_type})"
                    for claim in state.claims
                ])
                
                evidence_text = "\n".join([
                    f"- {evidence.title}: {evidence.content[:100]}... (Source: {evidence.source})"
                    for evidence in state.get("evidence_sources", [])
                ])
                
                prompt = f"""
                Verify the following claims using the provided evidence:
                
                Claims:
                {claims_text}
                
                Evidence:
                {evidence_text}
                
                For each claim, provide:
                1. Verdict (true, false, mostly_true, mostly_false, half_true, misleading, unverified)
                2. Confidence in verdict (0.0-1.0)
                3. Detailed explanation
                4. Supporting evidence from the sources
                5. Contradictory evidence if any
                6. Additional context
                7. Limitations of the verification
                
                Consider the reliability of sources and quality of evidence.
                Be objective and evidence-based in your assessment.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    results_data = json.loads(response.content)
                    fact_check_results = []
                    
                    for result_data in results_data.get("results", []):
                        # Parse supporting evidence
                        supporting_evidence = []
                        for ev_data in result_data.get("supporting_evidence", []):
                            evidence = EvidenceSource(
                                source=ev_data.get("source", ""),
                                title=ev_data.get("title", ""),
                                content=ev_data.get("content", ""),
                                reliability_score=ev_data.get("reliability_score", 0.5),
                                publication_date=ev_data.get("publication_date"),
                                author=ev_data.get("author")
                            )
                            supporting_evidence.append(evidence)
                        
                        # Parse contradictory evidence
                        contradictory_evidence = []
                        for ev_data in result_data.get("contradictory_evidence", []):
                            evidence = EvidenceSource(
                                source=ev_data.get("source", ""),
                                title=ev_data.get("title", ""),
                                content=ev_data.get("content", ""),
                                reliability_score=ev_data.get("reliability_score", 0.5),
                                publication_date=ev_data.get("publication_date"),
                                author=ev_data.get("author")
                            )
                            contradictory_evidence.append(evidence)
                        
                        fact_check_result = FactCheckResult(
                            claim=result_data.get("claim", ""),
                            verdict=result_data.get("verdict", "unverified"),
                            confidence=result_data.get("confidence", 0.5),
                            explanation=result_data.get("explanation", ""),
                            supporting_evidence=supporting_evidence,
                            contradictory_evidence=contradictory_evidence,
                            context=result_data.get("context", ""),
                            limitations=result_data.get("limitations", [])
                        )
                        fact_check_results.append(fact_check_result)
                    
                    state.fact_check_results = fact_check_results
                    state.messages.append(SystemMessage(content=f"Verified {len(fact_check_results)} claims"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback verification
                    fallback_results = []
                    for claim in state.claims:
                        fallback_result = FactCheckResult(
                            claim=claim.claim,
                            verdict="unverified",
                            confidence=0.3,
                            explanation="Unable to verify due to insufficient evidence",
                            supporting_evidence=[],
                            contradictory_evidence=[],
                            context="",
                            limitations=["Insufficient evidence sources"]
                        )
                        fallback_results.append(fallback_result)
                    
                    state.fact_check_results = fallback_results
                    state.messages.append(SystemMessage(content="Claim verification completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error verifying claims: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_overall_assessment(state: FactCheckingState) -> FactCheckingState:
            """Generate overall assessment of the text"""
            try:
                # Create overall assessment prompt
                results_summary = "\n".join([
                    f"- Claim: {result.claim}\n  Verdict: {result.verdict} (Confidence: {result.confidence})"
                    for result in state.fact_check_results
                ])
                
                prompt = f"""
                Provide an overall assessment of the text based on the fact checking results:
                
                Original Text: "{state.text}"
                
                Fact Check Results:
                {results_summary}
                
                Please provide:
                1. Overall credibility assessment (highly credible, credible, mixed, misleading, not credible)
                2. Summary of findings
                3. Key concerns or red flags
                4. Recommendations for the reader
                5. Limitations of this assessment
                
                Be balanced and objective in your assessment.
                Respond with a detailed paragraph (200-300 words).
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                state.overall_assessment = response.content
                state.messages.append(SystemMessage(content="Overall assessment generated"))
                
            except Exception as e:
                state.error = f"Error generating overall assessment: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def finalize_fact_check(state: FactCheckingState) -> FactCheckingState:
            """Finalize the fact checking process"""
            try:
                # Validate results
                if not state.fact_check_results:
                    state.error = "No fact check results generated"
                    state.success = False
                    return state
                
                state.success = True
                
                # Create completion message
                true_count = sum(1 for r in state.fact_check_results if r.verdict in ["true", "mostly_true"])
                false_count = sum(1 for r in state.fact_check_results if r.verdict in ["false", "mostly_false"])
                unverified_count = sum(1 for r in state.fact_check_results if r.verdict in ["unverified", "misleading"])
                
                completion_msg = f"Fact checking completed: {true_count} true, {false_count} false, {unverified_count} unverified"
                state.messages.append(SystemMessage(content=completion_msg))
                
            except Exception as e:
                state.error = f"Error finalizing fact check: {str(e)}"
                state.success = False
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("extract_claims", extract_claims)
        workflow.add_node("generate_search_queries", generate_search_queries)
        workflow.add_node("search_for_evidence", search_for_evidence)
        workflow.add_node("verify_claims", verify_claims)
        workflow.add_node("generate_overall_assessment", generate_overall_assessment)
        workflow.add_node("finalize_fact_check", finalize_fact_check)
        
        # Set up workflow
        workflow.set_entry_point("extract_claims")
        workflow.add_edge("extract_claims", "generate_search_queries")
        workflow.add_edge("generate_search_queries", "search_for_evidence")
        workflow.add_edge("search_for_evidence", "verify_claims")
        workflow.add_edge("verify_claims", "generate_overall_assessment")
        workflow.add_edge("generate_overall_assessment", "finalize_fact_check")
        workflow.add_edge("finalize_fact_check", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def check_facts(
        self,
        text: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check facts in the provided text.
        
        Args:
            text: Text containing claims to verify
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing fact checking results
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize FactCheckerAgent workflow")
        
        try:
            # Create initial state
            state = FactCheckingState(
                text=text,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "claims": [claim.dict() for claim in result.claims],
                "fact_check_results": [result.dict() for result in result.fact_check_results],
                "overall_assessment": result.overall_assessment,
                "search_queries": result.search_queries,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in fact checking: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "claims": [],
                "fact_check_results": [],
                "overall_assessment": None,
            }
    
    async def stream_check_facts(
        self,
        text: str,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream fact checking process for real-time updates.
        
        Args:
            text: Text containing claims to verify
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize FactCheckerAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = FactCheckingState(
                text=text,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield claim extraction updates
                if "extract_claims" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'claims') and node_state.claims:
                        yield {
                            "type": "claims_extracted",
                            "claims": [claim.dict() for claim in node_state.claims],
                            "count": len(node_state.claims),
                        }
                
                # Yield search query generation updates
                if "generate_search_queries" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'search_queries') and node_state.search_queries:
                        yield {
                            "type": "search_queries_generated",
                            "search_queries": node_state.search_queries,
                            "count": len(node_state.search_queries),
                        }
                
                # Yield evidence search updates
                if "search_for_evidence" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'evidence_sources'):
                        yield {
                            "type": "evidence_found",
                            "evidence_count": len(node_state.get("evidence_sources", [])),
                        }
                
                # Yield claim verification updates
                if "verify_claims" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'fact_check_results') and node_state.fact_check_results:
                        yield {
                            "type": "claims_verified",
                            "fact_check_results": [result.dict() for result in node_state.fact_check_results],
                            "count": len(node_state.fact_check_results),
                        }
                
                # Yield overall assessment updates
                if "generate_overall_assessment" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'overall_assessment') and node_state.overall_assessment:
                        yield {
                            "type": "assessment_generated",
                            "overall_assessment": node_state.overall_assessment,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "fact_check_complete",
                        "success": final_state.success,
                        "claims": [claim.dict() for claim in final_state.claims],
                        "fact_check_results": [result.dict() for result in final_state.fact_check_results],
                        "overall_assessment": final_state.overall_assessment,
                        "search_queries": final_state.search_queries,
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming fact checking: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }