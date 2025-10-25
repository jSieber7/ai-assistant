"""
LangChain-specific PostgreSQL client for database operations
"""

from typing import Dict, Any, List, Optional, Union
import asyncpg
import logging
from datetime import datetime, timedelta
import json
import uuid

logger = logging.getLogger(__name__)


class LangChainClient:
    """PostgreSQL client for LangChain and LangGraph integration"""

    def __init__(
        self, connection_string: str, database_name: str = "langchain_system"
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
                command_timeout=60,
                server_settings={"search_path": "langchain, public"}
            )
            self._connected = True
            logger.info(f"Connected to LangChain database: {self.database_name}")

            # Create schema if it doesn't exist
            await self._ensure_schema_exists()

        except Exception as e:
            logger.error(f"Failed to connect to LangChain database: {str(e)}")
            self._connected = False
            raise

    async def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("Disconnected from LangChain database")

    async def _ensure_schema_exists(self):
        """Ensure langchain schema exists"""
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS langchain")
            await conn.execute("SET search_path TO langchain, public")

    # ============================================================================
    # Conversation Management
    # ============================================================================

    async def create_conversation(
        self, 
        conversation_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO conversations 
                (conversation_id, user_id, session_id, title, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                conversation_id,
                user_id,
                session_id,
                title,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id
            )
            if row:
                return {
                    "id": str(row["id"]),
                    "conversation_id": row["conversation_id"],
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "title": row["title"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "is_active": row["is_active"]
                }
        return None

    async def update_conversation(
        self, conversation_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update conversation data"""
        if not self._connected:
            await self.connect()

        set_clauses = []
        values = []
        param_count = 1

        for key, value in update_data.items():
            if key in ["title", "user_id", "session_id", "is_active"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1
            elif key == "metadata":
                set_clauses.append(f"metadata = ${param_count}")
                values.append(json.dumps(value))
                param_count += 1

        if not set_clauses:
            return False

        values.append(conversation_id)

        async with self.pool.acquire() as conn:
            query = f"""
                UPDATE conversations 
                SET {', '.join(set_clauses)}
                WHERE conversation_id = ${param_count}
            """
            result = await conn.execute(query, *values)
            return "UPDATE 1" in result

    async def list_conversations(
        self, 
        user_id: Optional[str] = None, 
        is_active: Optional[bool] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List conversations with optional filters"""
        if not self._connected:
            await self.connect()

        conditions = []
        values = []
        param_count = 1

        if user_id:
            conditions.append(f"user_id = ${param_count}")
            values.append(user_id)
            param_count += 1

        if is_active is not None:
            conditions.append(f"is_active = ${param_count}")
            values.append(is_active)
            param_count += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with self.pool.acquire() as conn:
            query = f"""
                SELECT * FROM conversations 
                {where_clause}
                ORDER BY updated_at DESC 
                LIMIT ${param_count}
            """
            rows = await conn.fetch(query, *values, limit)
            
            return [
                {
                    "id": str(row["id"]),
                    "conversation_id": row["conversation_id"],
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "title": row["title"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "is_active": row["is_active"]
                }
                for row in rows
            ]

    # ============================================================================
    # Chat Message Management
    # ============================================================================

    async def add_chat_message(
        self,
        conversation_id: str,
        message_id: str,
        message_type: str,
        content: str,
        message_sequence: int,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Add a chat message to a conversation"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO chat_messages 
                (conversation_id, message_id, message_type, content, message_sequence,
                 additional_kwargs, response_metadata, token_count, model_name, temperature, max_tokens)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                conversation_id,
                message_id,
                message_type,
                content,
                message_sequence,
                json.dumps(additional_kwargs or {}),
                json.dumps(response_metadata or {}),
                token_count or 0,
                model_name,
                temperature,
                max_tokens
            )
            return str(result["id"])

    async def get_conversation_messages(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all messages for a conversation"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            limit_clause = f"LIMIT {limit}" if limit else ""
            rows = await conn.fetch(
                f"""
                SELECT * FROM chat_messages 
                WHERE conversation_id = $1 
                ORDER BY message_sequence ASC
                {limit_clause}
                """,
                conversation_id
            )
            
            return [
                {
                    "id": str(row["id"]),
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "message_type": row["message_type"],
                    "content": row["content"],
                    "message_sequence": row["message_sequence"],
                    "additional_kwargs": json.loads(row["additional_kwargs"]),
                    "response_metadata": json.loads(row["response_metadata"]),
                    "created_at": row["created_at"],
                    "token_count": row["token_count"],
                    "model_name": row["model_name"],
                    "temperature": row["temperature"],
                    "max_tokens": row["max_tokens"]
                }
                for row in rows
            ]

    async def get_next_message_sequence(self, conversation_id: str) -> int:
        """Get the next message sequence number for a conversation"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COALESCE(MAX(message_sequence), 0) + 1
                FROM chat_messages 
                WHERE conversation_id = $1
                """,
                conversation_id
            )
            return result

    # ============================================================================
    # Memory Summary Management
    # ============================================================================

    async def add_memory_summary(
        self,
        conversation_id: str,
        summary_id: str,
        summary_type: str,
        content: str,
        message_range_start: Optional[int] = None,
        message_range_end: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> str:
        """Add a memory summary"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO memory_summaries 
                (conversation_id, summary_id, summary_type, content, message_range_start,
                 message_range_end, metadata, token_count, model_name)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                conversation_id,
                summary_id,
                summary_type,
                content,
                message_range_start,
                message_range_end,
                json.dumps(metadata or {}),
                token_count or 0,
                model_name
            )
            return str(result["id"])

    async def get_conversation_summaries(
        self, conversation_id: str, summary_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get memory summaries for a conversation"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            where_clause = "WHERE conversation_id = $1"
            values = [conversation_id]
            
            if summary_type:
                where_clause += " AND summary_type = $2"
                values.append(summary_type)

            rows = await conn.fetch(
                f"""
                SELECT * FROM memory_summaries 
                {where_clause}
                ORDER BY created_at DESC
                """,
                *values
            )
            
            return [
                {
                    "id": str(row["id"]),
                    "conversation_id": row["conversation_id"],
                    "summary_id": row["summary_id"],
                    "summary_type": row["summary_type"],
                    "content": row["content"],
                    "message_range_start": row["message_range_start"],
                    "message_range_end": row["message_range_end"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "token_count": row["token_count"],
                    "model_name": row["model_name"]
                }
                for row in rows
            ]

    # ============================================================================
    # Workflow Management
    # ============================================================================

    async def create_workflow(
        self,
        workflow_id: str,
        workflow_type: str,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO workflows 
                (workflow_id, workflow_type, input_data, config, user_id, session_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                workflow_id,
                workflow_type,
                json.dumps(input_data),
                json.dumps(config or {}),
                user_id,
                session_id,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def update_workflow(
        self, workflow_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update workflow data"""
        if not self._connected:
            await self.connect()

        set_clauses = []
        values = []
        param_count = 1

        for key, value in update_data.items():
            if key in ["status", "error_message"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1
            elif key in ["input_data", "output_data", "config", "metadata"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(json.dumps(value))
                param_count += 1
            elif key in ["started_at", "completed_at"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1

        if not set_clauses:
            return False

        values.append(workflow_id)

        async with self.pool.acquire() as conn:
            query = f"""
                UPDATE workflows 
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
                "SELECT * FROM workflows WHERE workflow_id = $1",
                workflow_id
            )
            if row:
                return {
                    "id": str(row["id"]),
                    "workflow_id": row["workflow_id"],
                    "workflow_type": row["workflow_type"],
                    "status": row["status"],
                    "input_data": json.loads(row["input_data"]),
                    "output_data": json.loads(row["output_data"]),
                    "config": json.loads(row["config"]),
                    "created_at": row["created_at"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "updated_at": row["updated_at"],
                    "error_message": row["error_message"],
                    "metadata": json.loads(row["metadata"]),
                    "user_id": row["user_id"],
                    "session_id": row["session_id"]
                }
        return None

    # ============================================================================
    # Workflow Step Management
    # ============================================================================

    async def add_workflow_step(
        self,
        workflow_id: str,
        step_id: str,
        step_name: str,
        step_type: str,
        step_sequence: int,
        input_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        parent_step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a workflow step"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO workflow_steps 
                (workflow_id, step_id, step_name, step_type, step_sequence,
                 input_data, config, parent_step_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                workflow_id,
                step_id,
                step_name,
                step_type,
                step_sequence,
                json.dumps(input_data or {}),
                json.dumps(config or {}),
                parent_step_id,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def update_workflow_step(
        self, step_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update workflow step data"""
        if not self._connected:
            await self.connect()

        set_clauses = []
        values = []
        param_count = 1

        for key, value in update_data.items():
            if key in ["status", "error_message", "execution_time_ms"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1
            elif key in ["input_data", "output_data", "config", "metadata"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(json.dumps(value))
                param_count += 1
            elif key in ["started_at", "completed_at"]:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1

        if not set_clauses:
            return False

        values.append(step_id)

        async with self.pool.acquire() as conn:
            query = f"""
                UPDATE workflow_steps 
                SET {', '.join(set_clauses)}
                WHERE step_id = ${param_count}
            """
            result = await conn.execute(query, *values)
            return "UPDATE 1" in result

    # ============================================================================
    # Checkpoint Management
    # ============================================================================

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        workflow_id: str,
        checkpoint_data: Dict[str, Any],
        thread_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a workflow checkpoint"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, workflow_id, thread_id, checkpoint_data, expires_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                checkpoint_id,
                workflow_id,
                thread_id,
                json.dumps(checkpoint_data),
                expires_at,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint by ID"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM checkpoints 
                WHERE checkpoint_id = $1 AND is_active = true
                """,
                checkpoint_id
            )
            if row:
                return {
                    "id": str(row["id"]),
                    "checkpoint_id": row["checkpoint_id"],
                    "workflow_id": row["workflow_id"],
                    "thread_id": row["thread_id"],
                    "checkpoint_data": json.loads(row["checkpoint_data"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "is_active": row["is_active"]
                }
        return None

    # ============================================================================
    # Tool Execution Management
    # ============================================================================

    async def start_tool_execution(
        self,
        execution_id: str,
        tool_name: str,
        tool_type: str,
        input_parameters: Dict[str, Any],
        workflow_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a tool execution"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO tool_executions 
                (execution_id, tool_name, tool_type, input_parameters, workflow_id, 
                 agent_session_id, user_id, metadata, started_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
                """,
                execution_id,
                tool_name,
                tool_type,
                json.dumps(input_parameters),
                workflow_id,
                agent_session_id,
                user_id,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def complete_tool_execution(
        self, execution_id: str, output_data: Dict[str, Any], 
        result_type: str = "success", raw_output: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None, token_count: Optional[int] = None
    ) -> bool:
        """Complete a tool execution"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Update execution record
                await conn.execute(
                    """
                    UPDATE tool_executions 
                    SET status = 'completed', completed_at = NOW(), execution_time_ms = 
                        EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000
                    WHERE execution_id = $1
                    """,
                    execution_id
                )

                # Add result record
                await conn.execute(
                    """
                    INSERT INTO tool_results 
                    (execution_id, result_type, output_data, raw_output, artifacts, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    execution_id,
                    result_type,
                    json.dumps(output_data),
                    raw_output,
                    json.dumps(artifacts or {}),
                    token_count or 0
                )

                return True

    async def fail_tool_execution(
        self, execution_id: str, error_message: str
    ) -> bool:
        """Mark a tool execution as failed"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE tool_executions 
                SET status = 'failed', error_message = $1, completed_at = NOW(),
                    execution_time_ms = EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000
                WHERE execution_id = $2
                """,
                error_message,
                execution_id
            )
            return "UPDATE 1" in result

    # ============================================================================
    # Agent Session Management
    # ============================================================================

    async def create_agent_session(
        self,
        session_id: str,
        agent_name: str,
        agent_type: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create an agent session"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO agent_sessions 
                (session_id, agent_name, agent_type, conversation_id, user_id, 
                 config, expires_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                session_id,
                agent_name,
                agent_type,
                conversation_id,
                user_id,
                json.dumps(config or {}),
                expires_at,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def update_agent_session_activity(
        self, session_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update agent session activity"""
        if not self._connected:
            await self.connect()

        set_clauses = []
        values = []
        param_count = 1

        for key, value in update_data.items():
            if key in ["status", "config"]:
                if key == "config":
                    set_clauses.append(f"{key} = ${param_count}")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = ${param_count}")
                    values.append(value)
                param_count += 1
            elif key == "metadata":
                set_clauses.append(f"{key} = ${param_count}")
                values.append(json.dumps(value))
                param_count += 1

        if not set_clauses:
            return False

        # Always update last_activity_at
        set_clauses.append("last_activity_at = NOW()")
        values.append(session_id)

        async with self.pool.acquire() as conn:
            query = f"""
                UPDATE agent_sessions 
                SET {', '.join(set_clauses)}
                WHERE session_id = ${param_count}
            """
            result = await conn.execute(query, *values)
            return "UPDATE 1" in result

    # ============================================================================
    # Metrics and Performance Logging
    # ============================================================================

    async def record_metric(
        self,
        metric_name: str,
        metric_type: str,
        value: float,
        component: Optional[str] = None,
        unit: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a metric"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO metrics 
                (metric_name, metric_type, value, component, unit, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                metric_name,
                metric_type,
                value,
                component,
                unit,
                json.dumps(tags or {}),
                json.dumps(metadata or {})
            )
            return str(result["id"])

    async def log_performance(
        self,
        log_level: str,
        component: str,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        token_count: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log performance data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO performance_logs 
                (log_level, component, operation, message, details, execution_time_ms,
                 token_count, user_id, session_id, workflow_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                log_level,
                component,
                operation,
                message,
                json.dumps(details or {}),
                execution_time_ms,
                token_count,
                user_id,
                session_id,
                workflow_id,
                json.dumps(metadata or {})
            )
            return str(result["id"])

    # ============================================================================
    # Cleanup and Maintenance
    # ============================================================================

    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            cleanup_stats = {}

            # Clean up expired checkpoints
            result = await conn.execute(
                """
                DELETE FROM checkpoints 
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )
            cleanup_stats["expired_checkpoints"] = int(result.split()[1]) if result else 0

            # Clean up old performance logs (older than 30 days)
            result = await conn.execute(
                """
                DELETE FROM performance_logs 
                WHERE created_at < NOW() - INTERVAL '30 days'
                """
            )
            cleanup_stats["old_performance_logs"] = int(result.split()[1]) if result else 0

            # Clean up old metrics (older than 90 days)
            result = await conn.execute(
                """
                DELETE FROM metrics 
                WHERE created_at < NOW() - INTERVAL '90 days'
                """
            )
            cleanup_stats["old_metrics"] = int(result.split()[1]) if result else 0

            logger.info(f"LangChain cleanup completed: {cleanup_stats}")
            return cleanup_stats

    async def get_statistics(self) -> Dict[str, Any]:
        """Get LangChain system statistics"""
        if not self._connected:
            await self.connect()

        async with self.pool.acquire() as conn:
            # Get conversation stats
            total_conversations = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations"
            )
            active_conversations = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE is_active = true"
            )

            # Get message stats
            total_messages = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_messages"
            )
            total_tokens = await conn.fetchval(
                "SELECT COALESCE(SUM(token_count), 0) FROM chat_messages"
            )

            # Get workflow stats
            total_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM workflows"
            )
            completed_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM workflows WHERE status = 'completed'"
            )
            failed_workflows = await conn.fetchval(
                "SELECT COUNT(*) FROM workflows WHERE status = 'failed'"
            )

            # Get tool execution stats
            total_tool_executions = await conn.fetchval(
                "SELECT COUNT(*) FROM tool_executions"
            )
            successful_tool_executions = await conn.fetchval(
                "SELECT COUNT(*) FROM tool_executions WHERE status = 'completed'"
            )

            # Get agent session stats
            active_agent_sessions = await conn.fetchval(
                "SELECT COUNT(*) FROM agent_sessions WHERE status = 'active'"
            )

            return {
                "conversations": {
                    "total": total_conversations,
                    "active": active_conversations
                },
                "messages": {
                    "total": total_messages,
                    "total_tokens": total_tokens
                },
                "workflows": {
                    "total": total_workflows,
                    "completed": completed_workflows,
                    "failed": failed_workflows,
                    "success_rate": (
                        (completed_workflows / total_workflows * 100)
                        if total_workflows > 0 else 0
                    )
                },
                "tool_executions": {
                    "total": total_tool_executions,
                    "successful": successful_tool_executions,
                    "success_rate": (
                        (successful_tool_executions / total_tool_executions * 100)
                        if total_tool_executions > 0 else 0
                    )
                },
                "agent_sessions": {
                    "active": active_agent_sessions
                }
            }


# Global LangChain client instance
langchain_client: Optional[LangChainClient] = None


async def get_langchain_client() -> LangChainClient:
    """Get or create LangChain client instance"""
    global langchain_client

    if langchain_client is None:
        from app.core.config import settings

        # Get PostgreSQL connection string from settings
        connection_string = getattr(settings, "database_url", None)
        if not connection_string:
            raise ValueError("PostgreSQL connection string not configured")

        langchain_client = LangChainClient(connection_string)
        await langchain_client.connect()

    return langchain_client


async def close_langchain_connection():
    """Close LangChain connection"""
    global langchain_client

    if langchain_client:
        await langchain_client.disconnect()
        langchain_client = None