"""
PostgreSQL client for multi-writer/checker system storage
"""

from typing import Dict, Any, List, Optional
import asyncpg
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PostgreSQLClient:
    """PostgreSQL client for storing multi-writer/checker system data"""

    def __init__(
        self, connection_string: str, database_name: str = "multi_writer_system"
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.pool = None
        self._connected = False

    async def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self._connected = True
            logger.info(f"Connected to PostgreSQL database: {self.database_name}")

            # Create indexes for better performance
            await self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self._connected = False
            raise

    async def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("Disconnected from PostgreSQL")

    async def _create_indexes(self):
        """Create indexes for collections - already done in init script"""
        pass

    async def save_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Save workflow data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            # Convert JSON fields to strings
            sources_json = json.dumps(workflow_data.get("sources", []))
            style_guide_json = json.dumps(workflow_data.get("style_guide", {}))
            stages_json = json.dumps(workflow_data.get("stages", {}))
            errors_json = json.dumps(workflow_data.get("errors", []))

            result = await conn.fetchrow(
                """
                INSERT INTO multi_writer.workflows 
                (workflow_id, prompt, sources, style_guide, template_name, 
                 quality_threshold, max_iterations, status, stages, final_output, errors)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                workflow_data["workflow_id"],
                workflow_data["prompt"],
                sources_json,
                style_guide_json,
                workflow_data.get("template_name", "article.html.jinja"),
                workflow_data.get("quality_threshold", 70.0),
                workflow_data.get("max_iterations", 2),
                workflow_data.get("status", "pending"),
                stages_json,
                workflow_data.get("final_output"),
                errors_json
            )
            return str(result["id"])

    async def update_workflow(
        self, workflow_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update workflow data"""
        if not self._connected:
            await self.connect()

        # Build dynamic update query
        set_clauses = []
        values = []
        param_count = 1

        for key, value in update_data.items():
            if key in ["sources", "style_guide", "stages", "errors"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(json.dumps(value))
                param_count += 1
            elif key in ["prompt", "template_name", "status", "final_output"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1
            elif key in ["quality_threshold", "max_iterations"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1

        if not set_clauses:
            return False

        set_clauses.append("updated_at = NOW()")
        values.append(workflow_id)

        async with self.pool.acquire() as conn:
            query = f"""
                UPDATE multi_writer.workflows 
                SET {', '.join(set_clauses)}
                WHERE workflow_id = ${param_count}
            """
            result = await conn.execute(query, *values)
            return "UPDATE 1" in result

    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM multi_writer.workflows WHERE workflow_id = $1",
                workflow_id
            )
            if row:
                return {
                    "id": str(row["id"]),
                    "workflow_id": row["workflow_id"],
                    "prompt": row["prompt"],
                    "sources": json.loads(row["sources"]),
                    "style_guide": json.loads(row["style_guide"]),
                    "template_name": row["template_name"],
                    "quality_threshold": row["quality_threshold"],
                    "max_iterations": row["max_iterations"],
                    "status": row["status"],
                    "stages": json.loads(row["stages"]),
                    "final_output": row["final_output"],
                    "errors": json.loads(row["errors"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
        return None

    async def save_content(self, content_data: Dict[str, Any]) -> str:
        """Save content data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            metadata_json = json.dumps(content_data.get("metadata", {}))
            
            result = await conn.fetchrow(
                """
                INSERT INTO multi_writer.content 
                (workflow_id, writer_id, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                content_data["workflow_id"],
                content_data["writer_id"],
                content_data.get("content"),
                metadata_json
            )
            return str(result["id"])

    async def save_check_result(self, check_data: Dict[str, Any]) -> str:
        """Save check result data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            metadata_json = json.dumps(check_data.get("metadata", {}))
            
            result = await conn.fetchrow(
                """
                INSERT INTO multi_writer.check_results 
                (workflow_id, checker_id, score, feedback, passed, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                check_data["workflow_id"],
                check_data["checker_id"],
                check_data.get("score"),
                check_data.get("feedback"),
                check_data.get("passed", False),
                metadata_json
            )
            return str(result["id"])

    async def get_workflow_content(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all content for a workflow"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM multi_writer.content WHERE workflow_id = $1 ORDER BY created_at",
                workflow_id
            )
            return [
                {
                    "id": str(row["id"]),
                    "workflow_id": row["workflow_id"],
                    "writer_id": row["writer_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

    async def get_workflow_check_results(
        self, workflow_id: str
    ) -> List[Dict[str, Any]]:
        """Get all check results for a workflow"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM multi_writer.check_results WHERE workflow_id = $1 ORDER BY created_at",
                workflow_id
            )
            return [
                {
                    "id": str(row["id"]),
                    "workflow_id": row["workflow_id"],
                    "checker_id": row["checker_id"],
                    "score": row["score"],
                    "feedback": row["feedback"],
                    "passed": row["passed"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

    async def list_workflows(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List workflows with optional status filter"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM multi_writer.workflows 
                    WHERE status = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                    """,
                    status, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM multi_writer.workflows 
                    ORDER BY created_at DESC 
                    LIMIT $1
                    """,
                    limit
                )
            
            return [
                {
                    "id": str(row["id"]),
                    "workflow_id": row["workflow_id"],
                    "prompt": row["prompt"],
                    "sources": json.loads(row["sources"]),
                    "style_guide": json.loads(row["style_guide"]),
                    "template_name": row["template_name"],
                    "quality_threshold": row["quality_threshold"],
                    "max_iterations": row["max_iterations"],
                    "status": row["status"],
                    "stages": json.loads(row["stages"]),
                    "final_output": row["final_output"],
                    "errors": json.loads(row["errors"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and all related data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            # Delete related content
            await conn.execute(
                "DELETE FROM multi_writer.content WHERE workflow_id = $1",
                workflow_id
            )

            # Delete related check results
            await conn.execute(
                "DELETE FROM multi_writer.check_results WHERE workflow_id = $1",
                workflow_id
            )

            # Delete workflow
            result = await conn.execute(
                "DELETE FROM multi_writer.workflows WHERE workflow_id = $1",
                workflow_id
            )

            return "DELETE 1" in result

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            # Get workflow counts
            total_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM multi_writer.workflows"
            )
            completed_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM multi_writer.workflows WHERE status = 'completed'"
            )
            failed_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM multi_writer.workflows WHERE status = 'failed'"
            )

            # Get content and check counts
            total_content = await conn.fetchval(
                "SELECT COUNT(*) FROM multi_writer.content"
            )
            total_checks = await conn.fetchval(
                "SELECT COUNT(*) FROM multi_writer.check_results"
            )

            # Get average quality score
            avg_quality_score = await conn.fetchval(
                """
                SELECT AVG((stages::json->>'quality_checking'::json->>'best_score')::REAL)
                FROM multi_writer.workflows 
                WHERE status = 'completed' 
                AND stages::json ? 'quality_checking'
                """
            )

            return {
                "total_workflows": total_workflows,
                "completed_workflows": completed_workflows,
                "failed_workflows": failed_workflows,
                "success_rate": (
                    (completed_workflows / total_workflows * 100)
                    if total_workflows > 0
                    else 0
                ),
                "total_content_generated": total_content,
                "total_checks_performed": total_checks,
                "average_quality_score": avg_quality_score or 0,
            }


# Global PostgreSQL client instance
postgresql_client: Optional[PostgreSQLClient] = None


async def get_postgresql_client() -> PostgreSQLClient:
    """Get or create PostgreSQL client instance"""
    global postgresql_client

    if postgresql_client is None:
        from app.core.config import settings

        # Get PostgreSQL connection string from settings
        connection_string = getattr(settings, "database_url", None)
        if not connection_string:
            raise ValueError("PostgreSQL connection string not configured")

        postgresql_client = PostgreSQLClient(connection_string)
        await postgresql_client.connect()

    return postgresql_client


async def close_postgresql_connection():
    """Close PostgreSQL connection"""
    global postgresql_client

    if postgresql_client:
        await postgresql_client.disconnect()
        postgresql_client = None
