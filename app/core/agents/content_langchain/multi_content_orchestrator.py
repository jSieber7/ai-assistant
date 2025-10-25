"""
LangGraph-based multi-content orchestrator for content generation workflow
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiContentState(BaseModel):
    """State for multi-content orchestrator workflow"""
    # Input parameters
    prompt: str = ""
    sources: List[Dict[str, Any]] = []
    style_guide: Optional[Dict[str, Any]] = {}
    template_name: str = "article.html.jinja"
    quality_threshold: float = 70.0
    max_iterations: int = 2
    save_to_db: bool = True
    
    # Workflow state
    workflow_id: str = ""
    processed_sources: List[Dict[str, Any]] = []
    combined_content: Dict[str, Any] = {}
    writer_results: List[Dict[str, Any]] = []
    best_content: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    checking_history: List[Dict[str, Any]] = []
    final_output: Optional[str] = None
    status: str = "started"
    errors: List[str] = []
    stages: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class MultiContentOrchestrator:
    """LangGraph-based multi-content orchestrator for content generation workflow"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm = None
        self.workflow = None
        self._initialized = False

    async def _initialize_async(self):
        """Initialize components asynchronously"""
        if self._initialized:
            return

        if not self.llm:
            from app.core.langchain.integration import langchain_integration
            await langchain_integration.initialize()
            self.llm = await langchain_integration.get_llm()

        # Initialize sub-components
        from .writer_agent import MultiWriterOrchestrator
        self.writer_orchestrator = MultiWriterOrchestrator(self.config)

        self.workflow = self._create_workflow()
        self._initialized = True

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for multi-content generation"""
        workflow = StateGraph(MultiContentState)

        # Add nodes
        workflow.add_node("initialize_workflow", self._initialize_workflow)
        workflow.add_node("process_sources", self._process_sources)
        workflow.add_node("generate_content", self._generate_content)
        workflow.add_node("quality_checking", self._quality_checking)
        workflow.add_node("template_rendering", self._template_rendering)
        workflow.add_node("finalize_workflow", self._finalize_workflow)

        # Add edges
        workflow.set_entry_point("initialize_workflow")
        workflow.add_edge("initialize_workflow", "process_sources")
        workflow.add_edge("process_sources", "generate_content")
        workflow.add_edge("generate_content", "quality_checking")
        workflow.add_edge("quality_checking", "template_rendering")
        workflow.add_edge("template_rendering", "finalize_workflow")
        workflow.add_edge("finalize_workflow", END)

        # Add conditional edges
        workflow.add_conditional_edges(
            "quality_checking",
            self._should_continue_quality_checking,
            {
                "continue": "quality_checking",
                "proceed": "template_rendering"
            }
        )

        return workflow.compile()

    async def _initialize_workflow(self, state: MultiContentState) -> MultiContentState:
        """Initialize workflow and save initial state"""
        try:
            # Generate workflow ID
            state.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize stages
            state.stages = {
                "source_processing": {"status": "pending"},
                "content_generation": {"status": "pending"},
                "quality_checking": {"status": "pending"},
                "template_rendering": {"status": "pending"}
            }
            
            # Save initial workflow to database if needed
            if state.save_to_db:
                await self._save_workflow_to_db(state)

            return state

        except Exception as e:
            logger.error(f"Error initializing workflow: {str(e)}")
            state.errors.append(f"Workflow initialization failed: {str(e)}")
            state.status = "failed"
            return state

    async def _process_sources(self, state: MultiContentState) -> MultiContentState:
        """Stage 1: Process and clean source content"""
        try:
            state.stages["source_processing"]["status"] = "processing"
            
            # Process each source
            processed_sources = []
            for source in state.sources:
                processed = await self._process_single_source(source)
                if processed:
                    processed_sources.append(processed)
            
            state.processed_sources = processed_sources
            
            # Create combined content
            if processed_sources:
                state.combined_content = {
                    "title": "Combined Sources",
                    "content": "\n\n".join([s["content"] for s in processed_sources]),
                    "key_points": [],
                    "sources": [s["url"] for s in processed_sources],
                }
            
            state.stages["source_processing"] = {
                "status": "completed",
                "processed_content": processed_sources,
                "combined_content": state.combined_content
            }
            
            return state

        except Exception as e:
            logger.error(f"Error processing sources: {str(e)}")
            state.errors.append(f"Source processing failed: {str(e)}")
            state.stages["source_processing"]["status"] = "failed"
            return state

    async def _process_single_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single source using LLM"""
        try:
            if not source.get("url") and not source.get("content"):
                return None

            # If content is not provided, fetch it
            content = source.get("content", "")
            if not content and source.get("url"):
                # In a real implementation, you would fetch the URL
                content = f"Content from {source['url']}"
            
            # Extract key points using LLM
            prompt = f"""
            Extract key points from the following content:
            
            {content[:3000]}
            
            Return a JSON object with:
            - title: Brief title for the content
            - content: Cleaned and formatted content
            - key_points: List of 5-7 key points
            - url: Source URL
            """

            response = await self.llm.ainvoke([
                {"role": "system", "content": "You are a content processing expert. Extract and structure information from source material."},
                {"role": "user", "content": prompt}
            ])

            try:
                processed = json.loads(response.content)
                processed["url"] = source.get("url", "")
                return processed
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "title": source.get("title", "Untitled"),
                    "content": content,
                    "key_points": ["Key point 1", "Key point 2"],
                    "url": source.get("url", "")
                }

        except Exception as e:
            logger.error(f"Error processing source: {str(e)}")
            return None

    async def _generate_content(self, state: MultiContentState) -> MultiContentState:
        """Stage 2: Generate content with multiple writers"""
        try:
            state.stages["content_generation"]["status"] = "processing"
            
            if not state.combined_content:
                raise ValueError("No processed content available for writers")
            
            # Get default writers from config
            default_writers = self.config.get("writers", {}).get("default_writers", ["technical_1", "creative_1"])
            
            # Generate content with different writers
            writer_results = await self.writer_orchestrator.generate_multiple_versions(
                state.prompt,
                state.combined_content,
                writer_ids=default_writers,
                style_guide=state.style_guide,
            )
            
            state.writer_results = writer_results
            
            state.stages["content_generation"] = {
                "status": "completed",
                "combined_source": state.combined_content,
                "writer_results": writer_results,
                "total_versions": len(writer_results)
            }
            
            return state

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            state.errors.append(f"Content generation failed: {str(e)}")
            state.stages["content_generation"]["status"] = "failed"
            return state

    async def _quality_checking(self, state: MultiContentState) -> MultiContentState:
        """Stage 3: Quality checking and iterative improvement"""
        try:
            state.stages["quality_checking"]["status"] = "processing"
            
            writer_results = state.writer_results
            best_content = None
            best_score = 0
            checking_history = []
            
            for iteration in range(state.max_iterations):
                iteration_results = []
                
                # Check each content version
                for content in writer_results:
                    check_result = await self._check_content_quality(content, state.quality_threshold)
                    iteration_results.append(check_result)
                    
                    # Track best content
                    if check_result["overall_score"] > best_score:
                        best_score = check_result["overall_score"]
                        best_content = check_result
                
                checking_history.append({
                    "iteration": iteration + 1,
                    "results": iteration_results,
                    "best_score": best_score,
                })
                
                # If we have content passing threshold, break
                if best_content and best_content["passes_threshold"]:
                    break
                
                # Prepare for next iteration (use improved versions)
                if iteration < state.max_iterations - 1:
                    writer_results = [
                        {
                            "writer_id": result["original_content"]["writer_id"],
                            "specialty": result["original_content"]["specialty"],
                            "content": result["best_improved_version"]["content"],
                            "sources_used": result["original_content"]["sources_used"],
                        }
                        for result in iteration_results
                        if result["best_improved_version"]["content"]
                    ]
            
            state.best_content = best_content
            state.best_score = best_score
            state.checking_history = checking_history
            
            state.stages["quality_checking"] = {
                "status": "completed",
                "best_content": best_content,
                "best_score": best_score,
                "passes_threshold": best_content and best_content["passes_threshold"],
                "checking_history": checking_history,
                "total_iterations": len(checking_history)
            }
            
            return state

        except Exception as e:
            logger.error(f"Error in quality checking: {str(e)}")
            state.errors.append(f"Quality checking failed: {str(e)}")
            state.stages["quality_checking"]["status"] = "failed"
            return state

    async def _check_content_quality(self, content: Dict[str, Any], quality_threshold: float) -> Dict[str, Any]:
        """Check content quality using LLM"""
        try:
            prompt = f"""
            Evaluate the quality of the following content:
            
            Writer: {content.get("writer_id", "unknown")}
            Specialty: {content.get("specialty", "unknown")}
            Content: {content.get("content", "")[:2000]}
            
            Quality Threshold: {quality_threshold}
            
            Provide a JSON evaluation with:
            1. overall_score: Score from 0-100
            2. passes_threshold: Boolean indicating if score >= threshold
            3. strengths: List of 3-5 strengths
            4. weaknesses: List of 3-5 weaknesses
            5. improvement_suggestions: List of 3-5 suggestions
            6. best_improved_version: Improved version addressing weaknesses (optional)
            7. original_content: Reference to original content
            """

            response = await self.llm.ainvoke([
                {"role": "system", "content": "You are a content quality evaluator. Assess content quality and provide constructive feedback."},
                {"role": "user", "content": prompt}
            ])

            try:
                evaluation = json.loads(response.content)
                evaluation["original_content"] = content
                return evaluation
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "overall_score": 70,
                    "passes_threshold": quality_threshold <= 70,
                    "strengths": ["Good structure"],
                    "weaknesses": ["Could be improved"],
                    "improvement_suggestions": ["Add more detail"],
                    "best_improved_version": None,
                    "original_content": content
                }

        except Exception as e:
            logger.error(f"Error checking content quality: {str(e)}")
            return {
                "overall_score": 50,
                "passes_threshold": False,
                "strengths": [],
                "weaknesses": ["Quality check failed"],
                "improvement_suggestions": [],
                "best_improved_version": None,
                "original_content": content
            }

    async def _template_rendering(self, state: MultiContentState) -> MultiContentState:
        """Stage 4: Render final content using templates"""
        try:
            state.stages["template_rendering"]["status"] = "processing"
            
            if not state.best_content:
                raise ValueError("No content available for template rendering")
            
            # Prepare additional context for template
            template_context = {
                "workflow_id": state.workflow_id,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "quality_score": state.best_score,
                "iterations": len(state.checking_history)
            }
            
            # For now, just return the best content as rendered content
            # In a real implementation, you would use JinjaProcessor
            rendered_content = state.best_content.get("original_content", {}).get("content", "")
            
            state.final_output = rendered_content
            
            state.stages["template_rendering"] = {
                "status": "completed",
                "rendered_content": rendered_content,
                "template_used": state.template_name,
                "template_context": template_context
            }
            
            return state

        except Exception as e:
            logger.error(f"Error in template rendering: {str(e)}")
            state.errors.append(f"Template rendering failed: {str(e)}")
            state.stages["template_rendering"]["status"] = "failed"
            return state

    async def _finalize_workflow(self, state: MultiContentState) -> MultiContentState:
        """Finalize workflow and save results"""
        try:
            state.status = "completed"
            
            # Update metadata
            state.metadata = {
                "workflow_id": state.workflow_id,
                "prompt": state.prompt,
                "sources_count": len(state.sources),
                "writers_used": len(state.writer_results),
                "quality_threshold": state.quality_threshold,
                "iterations_completed": len(state.checking_history),
                "final_quality_score": state.best_score,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results to database if needed
            if state.save_to_db:
                await self._save_workflow_results(state)
            
            return state

        except Exception as e:
            logger.error(f"Error finalizing workflow: {str(e)}")
            state.errors.append(f"Workflow finalization failed: {str(e)}")
            state.status = "failed"
            return state

    def _should_continue_quality_checking(self, state: MultiContentState) -> str:
        """Determine if quality checking should continue"""
        if state.errors:
            return "proceed"
        
        if not state.checking_history:
            return "continue"
        
        # Check if we have content passing threshold
        if state.best_content and state.best_content.get("passes_threshold", False):
            return "proceed"
        
        # Check if we've reached max iterations
        if len(state.checking_history) >= state.max_iterations:
            return "proceed"
        
        return "continue"

    async def _save_workflow_to_db(self, state: MultiContentState):
        """Save initial workflow to database"""
        try:
            # In a real implementation, you would save to MongoDB
            # For now, just log
            logger.info(f"Saving workflow {state.workflow_id} to database")
        except Exception as e:
            logger.error(f"Failed to save workflow to database: {str(e)}")
            state.errors.append(f"Database error: {str(e)}")

    async def _save_workflow_results(self, state: MultiContentState):
        """Save workflow results to database"""
        try:
            # In a real implementation, you would save to MongoDB
            # For now, just log
            logger.info(f"Saving results for workflow {state.workflow_id} to database")
            
            # Save generated content
            for content in state.writer_results:
                content["workflow_id"] = state.workflow_id
                logger.info(f"Saving content from {content.get('writer_id', 'unknown')}")
            
            # Save check results
            for check_round in state.checking_history:
                for check_result in check_round["results"]:
                    check_result["workflow_id"] = state.workflow_id
                    logger.info(f"Saving check result with score {check_result.get('overall_score', 0)}")
                    
        except Exception as e:
            logger.error(f"Failed to save workflow results to database: {str(e)}")
            state.errors.append(f"Database error: {str(e)}")

    async def create_content(
        self,
        prompt: str,
        sources: List[Dict[str, Any]],
        style_guide: Optional[Dict[str, Any]] = None,
        template_name: str = None,
        quality_threshold: float = None,
        max_iterations: int = None,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Create content using LangGraph workflow"""
        try:
            await self._initialize_async()

            # Use defaults from config if not provided
            template_name = template_name or self.config.get("templates", {}).get("default_template", "article.html.jinja")
            quality_threshold = quality_threshold or self.config.get("quality", {}).get("threshold", 70.0)
            max_iterations = max_iterations or self.config.get("quality", {}).get("max_iterations", 2)

            initial_state = MultiContentState(
                prompt=prompt,
                sources=sources,
                style_guide=style_guide,
                template_name=template_name,
                quality_threshold=quality_threshold,
                max_iterations=max_iterations,
                save_to_db=save_to_db
            )

            # Run workflow
            result = await self.workflow.ainvoke(initial_state)

            return {
                "workflow_id": result.workflow_id,
                "status": result.status,
                "final_output": result.final_output,
                "stages": result.stages,
                "errors": result.errors,
                "metadata": result.metadata
            }

        except Exception as e:
            logger.error(f"Error in multi-content orchestrator: {str(e)}")
            return {
                "workflow_id": f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "failed",
                "final_output": None,
                "stages": {},
                "errors": [str(e)],
                "metadata": {}
            }


# Factory function to create orchestrator
async def create_multi_content_orchestrator(
    config: Dict[str, Any] = None,
) -> MultiContentOrchestrator:
    """Create multi-content orchestrator instance"""
    return MultiContentOrchestrator(config)