"""
Main orchestrator for the multi-writer/checker system
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.core.agents.content_processor import ContentProcessor
from app.core.agents.writer_agent import MultiWriterOrchestrator
from app.core.agents.checker_agent import MultiCheckerOrchestrator
from app.core.templating.jinja_processor import JinjaProcessor
from app.core.storage.mongodb_client import get_mongodb_client
from app.core.multi_writer_config import (
    get_multi_writer_config,
    is_multi_writer_enabled,
)

logger = logging.getLogger(__name__)


class MultiContentOrchestrator:
    """Main orchestrator for the multi-writer/checker system"""

    def __init__(self, config: Dict[str, Any] = None):
        if not is_multi_writer_enabled():
            raise ValueError("Multi-writer system is not enabled")

        self.config = config or get_multi_writer_config()

        # Initialize components
        self.content_processor = ContentProcessor(self.config["firecrawl"]["api_key"])
        self.writer_orchestrator = MultiWriterOrchestrator()
        self.checker_orchestrator = MultiCheckerOrchestrator()
        self.jinja_processor = JinjaProcessor(self.config["templates"]["template_dir"])

        # MongoDB client will be initialized when needed
        self._mongodb_client = None

    async def get_mongodb_client(self):
        """Get MongoDB client instance"""
        if self._mongodb_client is None:
            self._mongodb_client = await get_mongodb_client()
        return self._mongodb_client

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
        """Complete content creation workflow"""

        # Use defaults from config if not provided
        template_name = template_name or self.config["templates"]["default_template"]
        quality_threshold = quality_threshold or self.config["quality"]["threshold"]
        max_iterations = max_iterations or self.config["quality"]["max_iterations"]

        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = {
            "workflow_id": workflow_id,
            "prompt": prompt,
            "sources": sources,
            "status": "started",
            "stages": {},
            "final_output": None,
            "errors": [],
        }

        # Save initial workflow to database
        if save_to_db:
            try:
                mongodb_client = await self.get_mongodb_client()
                await mongodb_client.save_workflow(result.copy())
            except Exception as e:
                logger.error(f"Failed to save workflow to database: {str(e)}")
                result["errors"].append(f"Database error: {str(e)}")

        try:
            # Stage 1: Process sources
            logger.info(
                f"Starting Stage 1: Source processing for workflow {workflow_id}"
            )
            result["stages"]["source_processing"] = await self._stage_1_process_sources(
                sources
            )

            # Stage 2: Generate content with multiple writers
            logger.info(
                f"Starting Stage 2: Content generation for workflow {workflow_id}"
            )
            result["stages"][
                "content_generation"
            ] = await self._stage_2_generate_content(
                prompt,
                result["stages"]["source_processing"]["processed_content"],
                style_guide,
            )

            # Stage 3: Quality checking and improvement
            logger.info(
                f"Starting Stage 3: Quality checking for workflow {workflow_id}"
            )
            result["stages"]["quality_checking"] = await self._stage_3_quality_checking(
                result["stages"]["content_generation"],
                quality_threshold,
                max_iterations,
            )

            # Stage 4: Template rendering
            logger.info(
                f"Starting Stage 4: Template rendering for workflow {workflow_id}"
            )
            result["stages"][
                "template_rendering"
            ] = await self._stage_4_template_rendering(
                result["stages"]["quality_checking"]["best_content"], template_name
            )

            # Final result
            result["final_output"] = result["stages"]["template_rendering"][
                "rendered_content"
            ]
            result["status"] = "completed"

            # Save content and check results to database
            if save_to_db:
                try:
                    mongodb_client = await self.get_mongodb_client()

                    # Save generated content
                    for content in result["stages"]["content_generation"][
                        "writer_results"
                    ]:
                        content["workflow_id"] = workflow_id
                        await mongodb_client.save_content(content)

                    # Save check results
                    for check in result["stages"]["quality_checking"][
                        "checking_history"
                    ]:
                        for check_result in check["results"]:
                            check_result["workflow_id"] = workflow_id
                            await mongodb_client.save_check_result(check_result)

                    # Update workflow status
                    await mongodb_client.update_workflow(
                        workflow_id,
                        {
                            "status": "completed",
                            "stages": result["stages"],
                            "final_output": result["final_output"],
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save workflow results to database: {str(e)}"
                    )
                    result["errors"].append(f"Database error: {str(e)}")

            logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")

            # Update workflow status in database
            if save_to_db:
                try:
                    mongodb_client = await self.get_mongodb_client()
                    await mongodb_client.update_workflow(
                        workflow_id, {"status": "failed", "errors": result["errors"]}
                    )
                except Exception as db_error:
                    logger.error(
                        f"Failed to update workflow status in database: {str(db_error)}"
                    )

        return result

    async def _stage_1_process_sources(
        self, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stage 1: Process and clean source content"""
        return await self.content_processor.process_sources(sources)

    async def _stage_2_generate_content(
        self,
        prompt: str,
        processed_content: List[Dict[str, Any]],
        style_guide: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stage 2: Generate content with multiple writers"""

        if not processed_content:
            raise ValueError("No processed content available for writers")

        # Combine all processed content
        combined_content = {
            "title": "Combined Sources",
            "content": "\n\n".join([c["content"] for c in processed_content]),
            "key_points": [],
            "sources": [c["url"] for c in processed_content],
        }

        # Get default writers from config
        default_writers = self.config["writers"]["default_writers"]

        # Generate content with different writers
        writer_results = await self.writer_orchestrator.generate_multiple_versions(
            prompt,
            combined_content,
            writer_ids=default_writers,
            style_guide=style_guide,
        )

        return {
            "combined_source": combined_content,
            "writer_results": writer_results,
            "total_versions": len(writer_results),
        }

    async def _stage_3_quality_checking(
        self,
        content_generation: Dict[str, Any],
        quality_threshold: float,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Stage 3: Quality checking and iterative improvement"""

        writer_results = content_generation["writer_results"]
        best_content = None
        best_score = 0
        checking_history = []

        for iteration in range(max_iterations):
            iteration_results = []

            # Check each content version
            for content in writer_results:
                check_result = await self.checker_orchestrator.comprehensive_check(
                    content, min_score_threshold=quality_threshold
                )

                iteration_results.append(check_result)

                # Track best content
                if check_result["overall_score"] > best_score:
                    best_score = check_result["overall_score"]
                    best_content = check_result

            checking_history.append(
                {
                    "iteration": iteration + 1,
                    "results": iteration_results,
                    "best_score": best_score,
                }
            )

            # If we have content passing threshold, break
            if best_content and best_content["passes_threshold"]:
                break

            # Prepare for next iteration (use improved versions)
            if iteration < max_iterations - 1:
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

        return {
            "best_content": best_content,
            "best_score": best_score,
            "passes_threshold": best_content and best_content["passes_threshold"],
            "checking_history": checking_history,
            "total_iterations": len(checking_history),
        }

    async def _stage_4_template_rendering(
        self, best_content: Dict[str, Any], template_name: str
    ) -> Dict[str, Any]:
        """Stage 4: Render final content using templates"""

        if not best_content:
            raise ValueError("No content available for template rendering")

        # Prepare additional context for template
        template_context = {
            "workflow_id": best_content.get("workflow_id"),
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

        # Render content
        render_result = await self.jinja_processor.render_content(
            template_name, best_content, template_context
        )

        return render_result


# Factory function to create orchestrator
async def create_multi_content_orchestrator(
    config: Dict[str, Any] = None,
) -> MultiContentOrchestrator:
    """Create multi-content orchestrator instance"""
    if not is_multi_writer_enabled():
        raise ValueError("Multi-writer system is not enabled")

    return MultiContentOrchestrator(config)
