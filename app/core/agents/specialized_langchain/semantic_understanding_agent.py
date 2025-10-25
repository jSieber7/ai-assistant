"""
SemanticUnderstandingAgent - A specialized agent for sentiment analysis and semantic understanding using LangChain.

This agent can analyze text for sentiment, extract semantic meaning,
identify entities, understand relationships, and provide deep text insights.
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


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    overall_sentiment: str = Field(description="Overall sentiment: positive, negative, neutral, mixed")
    sentiment_score: float = Field(description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    confidence: float = Field(description="Confidence in sentiment analysis (0.0-1.0)")
    emotions: List[str] = Field(description="Detected emotions: joy, anger, fear, sadness, surprise, etc.")
    emotion_scores: Dict[str, float] = Field(description="Scores for each emotion (0.0-1.0)")


class SemanticEntity(BaseModel):
    """Semantic entity extracted from text"""
    text: str = Field(description="Entity text as it appears in the source")
    label: str = Field(description="Entity type: PERSON, ORG, GPE, PRODUCT, EVENT, etc.")
    description: str = Field(description="Description of the entity")
    confidence: float = Field(description="Confidence in entity identification (0.0-1.0)")
    start_pos: int = Field(description="Start position in the original text")
    end_pos: int = Field(description="End position in the original text")


class SemanticRelation(BaseModel):
    """Semantic relation between entities"""
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Type of relation")
    object: str = Field(description="Object entity")
    confidence: float = Field(description="Confidence in relation extraction (0.0-1.0)")


class TextSummary(BaseModel):
    """Text summary with key points"""
    summary: str = Field(description="Concise summary of the text")
    key_points: List[str] = Field(description="Key points extracted from the text")
    main_topics: List[str] = Field(description="Main topics covered in the text")
    complexity_score: float = Field(description="Text complexity score (0.0-1.0)")


class SemanticUnderstandingState(BaseModel):
    """State for semantic understanding workflow"""
    text: str = Field(description="Text to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: sentiment, entities, relations, summary, comprehensive")
    language: str = Field(default="en", description="Language code for the text")
    domain: Optional[str] = Field(default=None, description="Domain context: medical, legal, technical, etc.")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    sentiment_analysis: Optional[SentimentAnalysis] = Field(default=None, description="Sentiment analysis results")
    entities: List[SemanticEntity] = Field(default_factory=list, description="Extracted entities")
    relations: List[SemanticRelation] = Field(default_factory=list, description="Extracted relations")
    text_summary: Optional[TextSummary] = Field(default=None, description="Text summary")
    semantic_insights: Optional[str] = Field(default=None, description="General semantic insights")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether analysis was successful")


class SemanticUnderstandingAgent:
    """
    A specialized agent for sentiment analysis and semantic understanding using LangChain.
    
    This agent can perform deep semantic analysis of text including sentiment,
    entity extraction, relation identification, and summarization.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SemanticUnderstandingAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "semantic_understanding_agent"
        self.description = "Specialized agent for sentiment analysis and semantic understanding"
        
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
            
            logger.info("SemanticUnderstandingAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticUnderstandingAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for semantic understanding"""
        workflow = StateGraph(SemanticUnderstandingState)
        
        # Define nodes
        def analyze_sentiment(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Analyze sentiment of the text"""
            try:
                # Create sentiment analysis prompt
                prompt = f"""
                Analyze the sentiment of the following text:
                
                Text: "{state.text}"
                Language: {state.language}
                Domain: {state.domain or "general"}
                
                Please provide:
                1. Overall sentiment (positive, negative, neutral, mixed)
                2. Sentiment score from -1.0 (very negative) to 1.0 (very positive)
                3. Confidence in the analysis (0.0-1.0)
                4. Detected emotions with scores (joy, anger, fear, sadness, surprise, disgust, trust, anticipation)
                
                Consider the language and domain context in your analysis.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    sentiment_data = json.loads(response.content)
                    
                    sentiment_analysis = SentimentAnalysis(
                        overall_sentiment=sentiment_data.get("overall_sentiment", "neutral"),
                        sentiment_score=sentiment_data.get("sentiment_score", 0.0),
                        confidence=sentiment_data.get("confidence", 0.5),
                        emotions=sentiment_data.get("emotions", []),
                        emotion_scores=sentiment_data.get("emotion_scores", {})
                    )
                    
                    state.sentiment_analysis = sentiment_analysis
                    state.messages.append(SystemMessage(content="Sentiment analysis completed"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback sentiment analysis
                    fallback_sentiment = SentimentAnalysis(
                        overall_sentiment="neutral",
                        sentiment_score=0.0,
                        confidence=0.5,
                        emotions=[],
                        emotion_scores={}
                    )
                    state.sentiment_analysis = fallback_sentiment
                    state.messages.append(SystemMessage(content="Sentiment analysis completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error analyzing sentiment: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def extract_entities(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Extract semantic entities from the text"""
            try:
                # Create entity extraction prompt
                prompt = f"""
                Extract semantic entities from the following text:
                
                Text: "{state.text}"
                Language: {state.language}
                Domain: {state.domain or "general"}
                
                Please identify and extract:
                1. Named entities (PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, etc.)
                2. Key concepts and terms
                3. Dates and temporal expressions
                4. Quantities and measurements
                5. Any other significant entities
                
                For each entity, provide:
                - Exact text as it appears
                - Entity type/label
                - Brief description
                - Confidence score (0.0-1.0)
                - Start and end positions in the text
                
                Consider the language and domain context.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    entities_data = json.loads(response.content)
                    entities = []
                    
                    for entity_data in entities_data.get("entities", []):
                        entity = SemanticEntity(
                            text=entity_data.get("text", ""),
                            label=entity_data.get("label", ""),
                            description=entity_data.get("description", ""),
                            confidence=entity_data.get("confidence", 0.5),
                            start_pos=entity_data.get("start_pos", 0),
                            end_pos=entity_data.get("end_pos", 0)
                        )
                        entities.append(entity)
                    
                    state.entities = entities
                    state.messages.append(SystemMessage(content=f"Extracted {len(entities)} entities"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback entity extraction
                    state.entities = []
                    state.messages.append(SystemMessage(content="Entity extraction completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error extracting entities: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def extract_relations(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Extract semantic relations between entities"""
            try:
                # Create relation extraction prompt
                entities_text = "\n".join([
                    f"- {entity.text} ({entity.label})"
                    for entity in state.entities
                ])
                
                prompt = f"""
                Extract semantic relations between entities in the following text:
                
                Text: "{state.text}"
                Language: {state.language}
                Domain: {state.domain or "general"}
                
                Identified Entities:
                {entities_text}
                
                Please identify:
                1. Subject-predicate-object relationships
                2. Causal relationships
                3. Temporal relationships
                4. Spatial relationships
                5. Part-whole relationships
                6. Any other meaningful relations
                
                For each relation, provide:
                - Subject entity
                - Type of relation
                - Object entity
                - Confidence score (0.0-1.0)
                
                Focus on clear, meaningful relationships.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    relations_data = json.loads(response.content)
                    relations = []
                    
                    for relation_data in relations_data.get("relations", []):
                        relation = SemanticRelation(
                            subject=relation_data.get("subject", ""),
                            relation=relation_data.get("relation", ""),
                            object=relation_data.get("object", ""),
                            confidence=relation_data.get("confidence", 0.5)
                        )
                        relations.append(relation)
                    
                    state.relations = relations
                    state.messages.append(SystemMessage(content=f"Extracted {len(relations)} relations"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback relation extraction
                    state.relations = []
                    state.messages.append(SystemMessage(content="Relation extraction completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error extracting relations: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_summary(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Generate text summary with key points"""
            try:
                # Create summary prompt
                prompt = f"""
                Create a comprehensive summary of the following text:
                
                Text: "{state.text}"
                Language: {state.language}
                Domain: {state.domain or "general"}
                
                Please provide:
                1. A concise summary (2-3 sentences)
                2. Key points extracted from the text
                3. Main topics covered
                4. Text complexity score (0.0-1.0, where 0.0 is very simple and 1.0 is very complex)
                
                The summary should capture the main ideas and important details.
                Consider the language and domain context.
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    summary_data = json.loads(response.content)
                    
                    text_summary = TextSummary(
                        summary=summary_data.get("summary", ""),
                        key_points=summary_data.get("key_points", []),
                        main_topics=summary_data.get("main_topics", []),
                        complexity_score=summary_data.get("complexity_score", 0.5)
                    )
                    
                    state.text_summary = text_summary
                    state.messages.append(SystemMessage(content="Text summary generated"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback summary
                    fallback_summary = TextSummary(
                        summary="Unable to generate summary",
                        key_points=[],
                        main_topics=[],
                        complexity_score=0.5
                    )
                    state.text_summary = fallback_summary
                    state.messages.append(SystemMessage(content="Text summary generated (fallback)"))
                
            except Exception as e:
                state.error = f"Error generating summary: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_insights(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Generate general semantic insights"""
            try:
                # Create insights prompt
                prompt = f"""
                Provide deep semantic insights about the following text:
                
                Text: "{state.text}"
                Language: {state.language}
                Domain: {state.domain or "general"}
                
                Consider the analysis results:
                - Sentiment: {state.sentiment_analysis.overall_sentiment if state.sentiment_analysis else "Not analyzed"}
                - Entities: {len(state.entities)} entities found
                - Relations: {len(state.relations)} relations found
                - Summary: {state.text_summary.summary if state.text_summary else "Not summarized"}
                
                Please provide insights on:
                1. Overall meaning and intent
                2. Implicit assumptions or biases
                3. Communication style and tone
                4. Key implications or consequences
                5. Areas of ambiguity or uncertainty
                6. Contextual understanding
                
                Focus on deeper semantic understanding beyond surface-level analysis.
                Respond with a detailed paragraph (200-300 words).
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                state.semantic_insights = response.content
                state.messages.append(SystemMessage(content="Semantic insights generated"))
                
            except Exception as e:
                state.error = f"Error generating insights: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def finalize_analysis(state: SemanticUnderstandingState) -> SemanticUnderstandingState:
            """Finalize the semantic understanding analysis"""
            try:
                # Validate analysis results
                if not any([
                    state.sentiment_analysis,
                    state.entities,
                    state.relations,
                    state.text_summary,
                    state.semantic_insights
                ]):
                    state.error = "No analysis results generated"
                    state.success = False
                    return state
                
                state.success = True
                
                # Create completion message
                completed_analyses = []
                if state.sentiment_analysis:
                    completed_analyses.append("sentiment")
                if state.entities:
                    completed_analyses.append(f"{len(state.entities)} entities")
                if state.relations:
                    completed_analyses.append(f"{len(state.relations)} relations")
                if state.text_summary:
                    completed_analyses.append("summary")
                if state.semantic_insights:
                    completed_analyses.append("insights")
                
                completion_msg = f"Semantic understanding completed: {', '.join(completed_analyses)}"
                state.messages.append(SystemMessage(content=completion_msg))
                
            except Exception as e:
                state.error = f"Error finalizing analysis: {str(e)}"
                state.success = False
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("analyze_sentiment", analyze_sentiment)
        workflow.add_node("extract_entities", extract_entities)
        workflow.add_node("extract_relations", extract_relations)
        workflow.add_node("generate_summary", generate_summary)
        workflow.add_node("generate_insights", generate_insights)
        workflow.add_node("finalize_analysis", finalize_analysis)
        
        # Set up workflow based on analysis type
        workflow.set_entry_point("analyze_sentiment")
        
        # Add conditional edges based on analysis type
        def route_after_sentiment(state: SemanticUnderstandingState):
            """Route to next step based on analysis type"""
            if state.analysis_type == "sentiment":
                return "finalize_analysis"
            else:
                return "extract_entities"
        
        def route_after_entities(state: SemanticUnderstandingState):
            """Route to next step based on analysis type"""
            if state.analysis_type == "entities":
                return "finalize_analysis"
            else:
                return "extract_relations"
        
        def route_after_relations(state: SemanticUnderstandingState):
            """Route to next step based on analysis type"""
            if state.analysis_type == "relations":
                return "finalize_analysis"
            else:
                return "generate_summary"
        
        def route_after_summary(state: SemanticUnderstandingState):
            """Route to next step based on analysis type"""
            if state.analysis_type == "summary":
                return "finalize_analysis"
            else:
                return "generate_insights"
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "analyze_sentiment",
            route_after_sentiment,
            {
                "finalize_analysis": "finalize_analysis",
                "extract_entities": "extract_entities"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_entities",
            route_after_entities,
            {
                "finalize_analysis": "finalize_analysis",
                "extract_relations": "extract_relations"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_relations",
            route_after_relations,
            {
                "finalize_analysis": "finalize_analysis",
                "generate_summary": "generate_summary"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_summary",
            route_after_summary,
            {
                "finalize_analysis": "finalize_analysis",
                "generate_insights": "generate_insights"
            }
        )
        
        workflow.add_edge("generate_insights", "finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def understand_text(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        language: str = "en",
        domain: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform semantic understanding of text.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, entities, relations, summary, comprehensive)
            language: Language code for the text
            domain: Domain context (medical, legal, technical, etc.)
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing semantic understanding results
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize SemanticUnderstandingAgent workflow")
        
        try:
            # Create initial state
            state = SemanticUnderstandingState(
                text=text,
                analysis_type=analysis_type,
                language=language,
                domain=domain,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "sentiment_analysis": result.sentiment_analysis.dict() if result.sentiment_analysis else None,
                "entities": [entity.dict() for entity in result.entities],
                "relations": [relation.dict() for relation in result.relations],
                "text_summary": result.text_summary.dict() if result.text_summary else None,
                "semantic_insights": result.semantic_insights,
                "analysis_type": result.analysis_type,
                "language": result.language,
                "domain": result.domain,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in semantic understanding: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sentiment_analysis": None,
                "entities": [],
                "relations": [],
                "text_summary": None,
                "semantic_insights": None,
            }
    
    async def stream_understand_text(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        language: str = "en",
        domain: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream semantic understanding process for real-time updates.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis
            language: Language code for the text
            domain: Domain context
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize SemanticUnderstandingAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = SemanticUnderstandingState(
                text=text,
                analysis_type=analysis_type,
                language=language,
                domain=domain,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield sentiment analysis updates
                if "analyze_sentiment" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'sentiment_analysis') and node_state.sentiment_analysis:
                        yield {
                            "type": "sentiment_analyzed",
                            "sentiment_analysis": node_state.sentiment_analysis.dict(),
                        }
                
                # Yield entity extraction updates
                if "extract_entities" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'entities') and node_state.entities:
                        yield {
                            "type": "entities_extracted",
                            "entities": [entity.dict() for entity in node_state.entities],
                            "count": len(node_state.entities),
                        }
                
                # Yield relation extraction updates
                if "extract_relations" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'relations') and node_state.relations:
                        yield {
                            "type": "relations_extracted",
                            "relations": [relation.dict() for relation in node_state.relations],
                            "count": len(node_state.relations),
                        }
                
                # Yield summary generation updates
                if "generate_summary" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'text_summary') and node_state.text_summary:
                        yield {
                            "type": "summary_generated",
                            "text_summary": node_state.text_summary.dict(),
                        }
                
                # Yield insights generation updates
                if "generate_insights" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'semantic_insights') and node_state.semantic_insights:
                        yield {
                            "type": "insights_generated",
                            "semantic_insights": node_state.semantic_insights,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "understanding_complete",
                        "success": final_state.success,
                        "sentiment_analysis": final_state.sentiment_analysis.dict() if final_state.sentiment_analysis else None,
                        "entities": [entity.dict() for entity in final_state.entities],
                        "relations": [relation.dict() for relation in final_state.relations],
                        "text_summary": final_state.text_summary.dict() if final_state.text_summary else None,
                        "semantic_insights": final_state.semantic_insights,
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming semantic understanding: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }