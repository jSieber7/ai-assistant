"""
LangGraph-based writer agent for content generation
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging
import json

logger = logging.getLogger(__name__)


class WriterState(BaseModel):
    """State for the writer agent workflow"""
    task: str = ""
    source_content: Dict[str, Any] = {}
    style_guide: Optional[Dict[str, Any]] = {}
    specialty: str = "general"
    model: str = "claude-3.5-sonnet"
    
    # Workflow state
    analysis: Optional[Dict[str, Any]] = {}
    content: Optional[str] = ""
    refined_content: Optional[str] = ""
    confidence_score: float = 0.0
    word_count: int = 0
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


class WriterAgent:
    """LangGraph-based writer agent for content generation"""

    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.config = config or {}
        self.specialty = self.config.get("specialty", "general")
        self.model = self.config.get("model", "claude-3.5-sonnet")
        self.workflow = None
        self._initialized = False

    async def _initialize_async(self):
        """Initialize components asynchronously"""
        if self._initialized:
            return

        if not self.llm:
            from app.core.langchain.integration import langchain_integration
            await langchain_integration.initialize()
            self.llm = await langchain_integration.get_llm(self.model)

        self.workflow = self._create_workflow()
        self._initialized = True

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for content generation"""
        workflow = StateGraph(WriterState)

        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("generate_content", self._generate_content)
        workflow.add_node("refine_content", self._refine_content)
        workflow.add_node("finalize_content", self._finalize_content)

        # Add edges
        workflow.set_entry_point("analyze_task")
        workflow.add_edge("analyze_task", "generate_content")
        workflow.add_edge("generate_content", "refine_content")
        workflow.add_edge("refine_content", "finalize_content")
        workflow.add_edge("finalize_content", END)

        # Add conditional edges
        workflow.add_conditional_edges(
            "refine_content",
            self._should_refine,
            {
                "refine": "refine_content",
                "finalize": "finalize_content"
            }
        )

        return workflow.compile()

    async def _analyze_task(self, state: WriterState) -> WriterState:
        """Analyze the writing task and requirements"""
        try:
            prompt = f"""
            Analyze the following writing task and provide a structured analysis:

            Task: {state.task}
            
            Source Material:
            Title: {state.source_content.get("title", "")}
            Content: {state.source_content.get("content", "")[:2000]}...
            Key Points: {", ".join(state.source_content.get("key_points", [])[:5])}
            
            Style Guide: {json.dumps(state.style_guide or {})}
            Writer Specialty: {state.specialty}
            
            Provide a JSON analysis with:
            1. content_type: Type of content to create (article, blog, report, etc.)
            2. target_audience: Primary audience for the content
            3. key_themes: Main themes to focus on
            4. structure: Suggested content structure
            5. tone: Appropriate tone for the content
            6. length_guidance: Guidance on content length
            7. special_considerations: Any special considerations based on specialty
            """

            response = await self.llm.ainvoke([
                {"role": "system", "content": "You are a content analysis expert. Analyze writing tasks and provide structured guidance."},
                {"role": "user", "content": prompt}
            ])

            try:
                analysis = json.loads(response.content)
                state.analysis = analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                state.analysis = {
                    "content_type": "article",
                    "target_audience": "general",
                    "key_themes": ["general"],
                    "structure": ["introduction", "body", "conclusion"],
                    "tone": "professional",
                    "length_guidance": "medium",
                    "special_considerations": []
                }

            return state

        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            state.error = f"Task analysis failed: {str(e)}"
            return state

    async def _generate_content(self, state: WriterState) -> WriterState:
        """Generate initial content based on analysis"""
        try:
            analysis = state.analysis or {}
            
            system_prompt = self._create_writer_prompt(state.specialty, state.style_guide)
            
            user_prompt = f"""
            Task: {state.task}
            
            Analysis:
            Content Type: {analysis.get("content_type", "article")}
            Target Audience: {analysis.get("target_audience", "general")}
            Key Themes: {", ".join(analysis.get("key_themes", []))}
            Structure: {", ".join(analysis.get("structure", []))}
            Tone: {analysis.get("tone", "professional")}
            Length: {analysis.get("length_guidance", "medium")}
            
            Source Material:
            Title: {state.source_content.get("title", "")}
            Content: {state.source_content.get("content", "")[:2000]}...
            Key Points: {", ".join(state.source_content.get("key_points", [])[:5])}
            
            Instructions:
            1. Write as a {state.specialty} writer
            2. Incorporate key information from sources
            3. Maintain factual accuracy
            4. Follow the analysis guidance
            5. Create engaging, well-structured content
            """

            response = await self.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            state.content = response.content
            state.word_count = len(response.content.split())
            state.confidence_score = self._calculate_confidence(state.content, state.source_content)

            return state

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            state.error = f"Content generation failed: {str(e)}"
            return state

    async def _refine_content(self, state: WriterState) -> WriterState:
        """Refine content if needed"""
        try:
            if not state.content or state.confidence_score >= 0.8:
                state.refined_content = state.content
                return state

            prompt = f"""
            Review and refine the following content to improve its quality:

            Original Content:
            {state.content}
            
            Task: {state.task}
            Specialty: {state.specialty}
            Current Confidence Score: {state.confidence_score}
            
            Source Material Key Points: {", ".join(state.source_content.get("key_points", []))}
            
            Please refine the content to:
            1. Better incorporate source material
            2. Improve structure and flow
            3. Enhance clarity and engagement
            4. Better match the {state.specialty} writing style
            5. Increase factual accuracy
            
            Return only the refined content without explanations.
            """

            response = await self.llm.ainvoke([
                {"role": "system", "content": "You are a content refinement expert. Improve content quality while preserving the core message."},
                {"role": "user", "content": prompt}
            ])

            state.refined_content = response.content
            
            # Recalculate confidence
            new_confidence = self._calculate_confidence(state.refined_content, state.source_content)
            if new_confidence > state.confidence_score:
                state.confidence_score = new_confidence

            return state

        except Exception as e:
            logger.error(f"Error refining content: {str(e)}")
            state.refined_content = state.content  # Fallback to original
            return state

    async def _finalize_content(self, state: WriterState) -> WriterState:
        """Finalize the content and prepare output"""
        try:
            final_content = state.refined_content or state.content
            
            if not final_content:
                state.error = "No content generated"
                return state

            state.metadata = {
                "specialty": state.specialty,
                "model_used": state.model,
                "sources_used": [state.source_content.get("url")],
                "word_count": len(final_content.split()),
                "confidence_score": state.confidence_score,
                "analysis": state.analysis or {},
                "task": state.task
            }

            # Update content with final version
            state.content = final_content
            state.word_count = len(final_content.split())

            return state

        except Exception as e:
            logger.error(f"Error finalizing content: {str(e)}")
            state.error = f"Content finalization failed: {str(e)}"
            return state

    def _should_refine(self, state: WriterState) -> str:
        """Determine if content should be refined"""
        if state.error:
            return "finalize"
        
        if not state.content:
            return "finalize"
        
        if state.confidence_score < 0.8:
            return "refine"
        
        return "finalize"

    def _create_writer_prompt(self, specialty: str, style_guide: Optional[Dict[str, Any]]) -> str:
        """Create system prompt based on writer specialty and style guide"""
        base_prompt = f"""
        You are a {specialty} writer creating high-quality content.
        """

        if specialty == "technical":
            base_prompt += """
            Focus on:
            - Accuracy and clarity
            - Technical precision
            - Structured explanations
            - Practical examples
            """
        elif specialty == "creative":
            base_prompt += """
            Focus on:
            - Engaging storytelling
            - Creative expression
            - Emotional connection
            - Vivid descriptions
            """
        elif specialty == "analytical":
            base_prompt += """
            Focus on:
            - Data-driven insights
            - Logical reasoning
            - Evidence-based claims
            - Balanced perspectives
            """

        if style_guide:
            base_prompt += f"""
            
            Style Guide:
            - Tone: {style_guide.get("tone", "professional")}
            - Audience: {style_guide.get("audience", "general")}
            - Length: {style_guide.get("length", "medium")}
            - Format: {style_guide.get("format", "article")}
            """

        return base_prompt

    def _calculate_confidence(self, content: str, source_content: Dict[str, Any]) -> float:
        """Calculate confidence score based on content quality and source usage"""
        score = 0.5  # Base score

        # Check if content mentions source material
        if source_content.get("title", "").lower() in content.lower():
            score += 0.1

        # Check word count
        word_count = len(content.split())
        if 100 <= word_count <= 1000:
            score += 0.2

        # Check for structure (paragraphs, headings)
        if "\n\n" in content and any(
            line.strip().endswith(":") for line in content.split("\n")
        ):
            score += 0.1

        # Check for key points integration
        key_points = source_content.get("key_points", [])
        integrated_points = sum(
            1 for point in key_points if point.lower() in content.lower()
        )
        if integrated_points > 0:
            score += min(0.1, integrated_points * 0.02)

        return min(1.0, score)

    async def generate_content(
        self,
        task: str,
        source_content: Dict[str, Any],
        style_guide: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate content using LangGraph workflow"""
        try:
            await self._initialize_async()

            initial_state = WriterState(
                task=task,
                source_content=source_content,
                style_guide=style_guide,
                specialty=self.specialty,
                model=self.model
            )

            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)

            if result.error:
                return {
                    "writer_id": f"{self.specialty}_writer",
                    "specialty": self.specialty,
                    "error": result.error,
                    "content": None
                }

            return {
                "writer_id": f"{self.specialty}_writer",
                "specialty": self.specialty,
                "content": result.content,
                "model_used": self.model,
                "sources_used": [source_content.get("url")],
                "word_count": result.word_count,
                "confidence_score": result.confidence_score,
                "metadata": result.metadata
            }

        except Exception as e:
            logger.error(f"Error in writer agent workflow: {str(e)}")
            return {
                "writer_id": f"{self.specialty}_writer",
                "specialty": self.specialty,
                "error": str(e),
                "content": None
            }


class MultiWriterOrchestrator:
    """Orchestrates multiple LangGraph-based writer agents"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.writers = {
            "technical_1": WriterAgent(None, {"specialty": "technical", "model": "claude-3.5-sonnet"}),
            "technical_2": WriterAgent(None, {"specialty": "technical", "model": "gpt-4-turbo"}),
            "creative_1": WriterAgent(None, {"specialty": "creative", "model": "claude-3.5-sonnet"}),
            "analytical_1": WriterAgent(None, {"specialty": "analytical", "model": "gpt-4-turbo"}),
        }

    async def generate_multiple_versions(
        self,
        task: str,
        source_content: Dict[str, Any],
        writer_ids: List[str] = None,
        style_guide: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple content versions using different writers"""
        import asyncio

        if writer_ids is None:
            writer_ids = list(self.writers.keys())

        tasks = []
        for writer_id in writer_ids:
            if writer_id in self.writers:
                task = self.writers[writer_id].generate_content(
                    task, source_content, style_guide
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            result
            for result in results
            if isinstance(result, dict) and result.get("content")
        ]

        return successful_results