"""
API routes for multi-writer/checker system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.core.agents.content.multi_content_orchestrator import create_multi_content_orchestrator
from app.core.storage.postgresql_client import get_postgresql_client
from app.core.multi_writer_config import (
    is_multi_writer_enabled,
    get_multi_writer_config,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Pydantic models for requests and responses
class ContentRequest(BaseModel):
    prompt: str = Field(..., description="Content generation prompt")
    sources: List[Dict[str, Any]] = Field(
        ..., description="List of source URLs or content"
    )
    style_guide: Optional[Dict[str, Any]] = Field(
        None, description="Style guide for content"
    )
    template_name: str = Field(
        "article.html.jinja", description="Template to use for rendering"
    )
    quality_threshold: float = Field(
        70.0, description="Minimum quality threshold (0-100)"
    )
    max_iterations: int = Field(
        2, description="Maximum iterations for quality improvement"
    )
    async_execution: bool = Field(False, description="Execute workflow asynchronously")
    save_to_db: bool = Field(True, description="Save results to database")


class ContentResponse(BaseModel):
    workflow_id: str
    status: str
    final_output: Optional[str] = None
    stages: Dict[str, Any] = {}
    errors: List[str] = []
    created_at: Optional[datetime] = None


class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: str
    stages: Dict[str, Any] = {}
    errors: List[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowStatusResponse]
    total: int


class StatisticsResponse(BaseModel):
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    success_rate: float
    total_content_generated: int
    total_checks_performed: int
    average_quality_score: float


# Background task storage (in production, use Redis or similar)
background_tasks = {}


async def check_multi_writer_enabled():
    """Check if multi-writer system is enabled"""
    if not is_multi_writer_enabled():
        raise HTTPException(
            status_code=503,
            detail="Multi-writer system is not enabled. Check configuration.",
        )


@router.post("/create", response_model=ContentResponse)
async def create_content(
    request: ContentRequest,
    background_tasks: BackgroundTasks,
    _=Depends(check_multi_writer_enabled),
):
    """Create content using multi-writer/checker system"""
    try:
        orchestrator = await create_multi_content_orchestrator()

        if request.async_execution:
            # Run in background
            workflow_id = f"async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Add background task
            background_tasks.add_task(
                _run_workflow_async, orchestrator, request, workflow_id
            )

            # Store task info
            background_tasks[workflow_id] = {
                "status": "queued",
                "created_at": datetime.utcnow(),
            }

            return ContentResponse(
                workflow_id=workflow_id,
                status="queued",
                stages={},
                errors=[],
                created_at=datetime.utcnow(),
            )
        else:
            # Run synchronously
            result = await orchestrator.create_content(
                prompt=request.prompt,
                sources=request.sources,
                style_guide=request.style_guide,
                template_name=request.template_name,
                quality_threshold=request.quality_threshold,
                max_iterations=request.max_iterations,
                save_to_db=request.save_to_db,
            )

            return ContentResponse(**result)

    except Exception as e:
        logger.error(f"Content creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str, _=Depends(check_multi_writer_enabled)):
    """Get status of async workflow"""
    try:
        # Check if it's a background task
        if workflow_id in background_tasks:
            task_info = background_tasks[workflow_id]
            return WorkflowStatusResponse(
                workflow_id=workflow_id,
                status=task_info["status"],
                stages=task_info.get("stages", {}),
                errors=task_info.get("errors", []),
                created_at=task_info.get("created_at"),
                updated_at=task_info.get("updated_at"),
            )

        # Otherwise, check database
        mongodb_client = await get_postgresql_client()
        workflow = await mongodb_client.get_workflow(workflow_id)

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return WorkflowStatusResponse(**workflow)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by workflow status"),
    limit: int = Query(
        50, ge=1, le=100, description="Maximum number of workflows to return"
    ),
    _=Depends(check_multi_writer_enabled),
):
    """List workflows with optional status filter"""
    try:
        mongodb_client = await get_postgresql_client()
        workflows = await mongodb_client.list_workflows(status=status, limit=limit)

        return WorkflowListResponse(
            workflows=[WorkflowStatusResponse(**workflow) for workflow in workflows],
            total=len(workflows),
        )

    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}/content")
async def get_workflow_content(workflow_id: str, _=Depends(check_multi_writer_enabled)):
    """Get all content generated for a workflow"""
    try:
        mongodb_client = await get_postgresql_client()
        content = await mongodb_client.get_workflow_content(workflow_id)

        if not content:
            raise HTTPException(status_code=404, detail="No content found for workflow")

        return {"workflow_id": workflow_id, "content": content}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}/check-results")
async def get_workflow_check_results(
    workflow_id: str, _=Depends(check_multi_writer_enabled)
):
    """Get all check results for a workflow"""
    try:
        mongodb_client = await get_postgresql_client()
        results = await mongodb_client.get_workflow_check_results(workflow_id)

        if not results:
            raise HTTPException(
                status_code=404, detail="No check results found for workflow"
            )

        return {"workflow_id": workflow_id, "check_results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow check results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str, _=Depends(check_multi_writer_enabled)):
    """Delete a workflow and all related data"""
    try:
        mongodb_client = await get_postgresql_client()
        success = await mongodb_client.delete_workflow(workflow_id)

        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {"message": f"Workflow {workflow_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(_=Depends(check_multi_writer_enabled)):
    """Get system statistics"""
    try:
        mongodb_client = await get_postgresql_client()
        stats = await mongodb_client.get_statistics()

        return StatisticsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config(_=Depends(check_multi_writer_enabled)):
    """Get current multi-writer configuration"""
    try:
        config = get_multi_writer_config()
        # Remove sensitive information
        config["firecrawl"]["api_key"] = (
            "***" if config["firecrawl"]["api_key"] else None
        )
        config["mongodb"]["connection_string"] = (
            "***" if config["mongodb"]["connection_string"] else None
        )

        return config

    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_workflow_async(orchestrator, request: ContentRequest, workflow_id: str):
    """Background task to run workflow asynchronously"""
    try:
        # Update status
        if workflow_id in background_tasks:
            background_tasks[workflow_id]["status"] = "processing"
            background_tasks[workflow_id]["updated_at"] = datetime.utcnow()

        # Run workflow
        result = await orchestrator.create_content(
            prompt=request.prompt,
            sources=request.sources,
            style_guide=request.style_guide,
            template_name=request.template_name,
            quality_threshold=request.quality_threshold,
            max_iterations=request.max_iterations,
            save_to_db=request.save_to_db,
        )

        # Update status
        if workflow_id in background_tasks:
            background_tasks[workflow_id]["status"] = result["status"]
            background_tasks[workflow_id]["stages"] = result["stages"]
            background_tasks[workflow_id]["errors"] = result["errors"]
            background_tasks[workflow_id]["updated_at"] = datetime.utcnow()

        # Clean up old tasks (in production, use proper cleanup)
        if len(background_tasks) > 100:
            oldest_tasks = sorted(
                background_tasks.items(),
                key=lambda x: x[1].get("created_at", datetime.min),
            )[:10]
            for task_id, _ in oldest_tasks:
                if background_tasks[task_id]["status"] in ["completed", "failed"]:
                    del background_tasks[task_id]

    except Exception as e:
        logger.error(f"Async workflow {workflow_id} failed: {str(e)}")

        # Update status with error
        if workflow_id in background_tasks:
            background_tasks[workflow_id]["status"] = "failed"
            background_tasks[workflow_id]["errors"] = [str(e)]
            background_tasks[workflow_id]["updated_at"] = datetime.utcnow()
