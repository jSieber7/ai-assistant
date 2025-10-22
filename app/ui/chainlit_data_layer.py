"""
Chainlit Data Layer implementation for PostgreSQL/Supabase integration.

This module provides the data layer implementation for Chainlit to persist
chat data, manage user sessions, and enable chat lifecycle features.
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.data.types import (
    User,
    Thread,
    Step,
    Element,
    Feedback,
    ThreadDict,
    StepDict,
    ElementDict,
    FeedbackDict,
    PersistedUser,
    PersistedThread,
    PersistedStep,
    PersistedElement,
    PersistedFeedback,
)
import asyncpg
import logging

logger = logging.getLogger(__name__)


class PostgreSQLDataLayer(BaseDataLayer):
    """PostgreSQL implementation of Chainlit's data layer."""

    def __init__(self, conninfo: str):
        """
        Initialize the PostgreSQL data layer.
        
        Args:
            conninfo: PostgreSQL connection string
        """
        self.conninfo = conninfo
        self.pool = None
        self._connected = False

    async def connect(self):
        """Establish connection to PostgreSQL."""
        if self._connected:
            return
            
        try:
            self.pool = await asyncpg.create_pool(
                self.conninfo,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self._connected = True
            logger.info("Connected to PostgreSQL for Chainlit data layer")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close PostgreSQL connection."""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("Disconnected from PostgreSQL")

    async def _ensure_connection(self):
        """Ensure database connection is established."""
        if not self._connected:
            await self.connect()

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        """Create a new user in the database."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                user_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(
                    """
                    INSERT INTO "User" (id, identifier, metadata, "createdAt", "updatedAt")
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (identifier) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        "updatedAt" = EXCLUDED."updatedAt"
                    RETURNING id
                    """,
                    user_id,
                    user.identifier,
                    json.dumps(user.metadata or {}),
                    now,
                    now
                )
                
                return PersistedUser(
                    id=user_id,
                    identifier=user.identifier,
                    metadata=user.metadata or {},
                    createdAt=now
                )
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        """Get a user by identifier."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, identifier, metadata, "createdAt"
                    FROM "User"
                    WHERE identifier = $1
                    """,
                    identifier
                )
                
                if row:
                    return PersistedUser(
                        id=str(row["id"]),
                        identifier=row["identifier"],
                        metadata=row["metadata"],
                        createdAt=row["createdAt"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    async def update_user(self, identifier: str, metadata: Dict[str, Any]) -> Optional[PersistedUser]:
        """Update user metadata."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                now = datetime.now(timezone.utc).isoformat()
                
                result = await conn.execute(
                    """
                    UPDATE "User"
                    SET metadata = $2, "updatedAt" = $3
                    WHERE identifier = $1
                    RETURNING id, identifier, metadata, "createdAt"
                    """,
                    identifier,
                    json.dumps(metadata),
                    now
                )
                
                if "UPDATE 1" in result:
                    row = await conn.fetchrow(
                        """
                        SELECT id, identifier, metadata, "createdAt"
                        FROM "User"
                        WHERE identifier = $1
                        """,
                        identifier
                    )
                    
                    return PersistedUser(
                        id=str(row["id"]),
                        identifier=row["identifier"],
                        metadata=row["metadata"],
                        createdAt=row["createdAt"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return None

    async def delete_user(self, identifier: str) -> bool:
        """Delete a user."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM \"User\" WHERE identifier = $1",
                    identifier
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

    async def create_thread(self, thread: Thread) -> Optional[PersistedThread]:
        """Create a new thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                thread_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(
                    """
                    INSERT INTO "Thread" 
                    (id, "createdAt", name, "userId", "userIdentifier", tags, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    thread_id,
                    now,
                    thread.name,
                    thread.user_id,
                    thread.user_identifier,
                    thread.tags or [],
                    json.dumps(thread.metadata or {})
                )
                
                return PersistedThread(
                    id=thread_id,
                    createdAt=now,
                    name=thread.name,
                    userId=thread.user_id,
                    userIdentifier=thread.user_identifier,
                    tags=thread.tags or [],
                    metadata=thread.metadata or {}
                )
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            return None

    async def get_thread(self, thread_id: str) -> Optional[PersistedThread]:
        """Get a thread by ID."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, "createdAt", name, "userId", "userIdentifier", tags, metadata
                    FROM "Thread"
                    WHERE id = $1
                    """,
                    thread_id
                )
                
                if row:
                    return PersistedThread(
                        id=str(row["id"]),
                        createdAt=row["createdAt"],
                        name=row["name"],
                        userId=row["userId"],
                        userIdentifier=row["userIdentifier"],
                        tags=row["tags"],
                        metadata=row["metadata"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting thread: {e}")
            return None

    async def update_thread(self, thread_id: str, name: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[PersistedThread]:
        """Update thread metadata."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                set_clauses = []
                values = []
                param_count = 1
                
                if name is not None:
                    set_clauses.append(f"name = ${param_count}")
                    values.append(name)
                    param_count += 1
                
                if metadata is not None:
                    set_clauses.append(f"metadata = ${param_count}")
                    values.append(json.dumps(metadata))
                    param_count += 1
                
                if not set_clauses:
                    return await self.get_thread(thread_id)
                
                values.append(thread_id)
                
                query = f"""
                    UPDATE "Thread"
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_count}
                    RETURNING id, "createdAt", name, "userId", "userIdentifier", tags, metadata
                """
                
                row = await conn.fetchrow(query, *values)
                
                if row:
                    return PersistedThread(
                        id=str(row["id"]),
                        createdAt=row["createdAt"],
                        name=row["name"],
                        userId=row["userId"],
                        userIdentifier=row["userIdentifier"],
                        tags=row["tags"],
                        metadata=row["metadata"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error updating thread: {e}")
            return None

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all associated data."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM \"Thread\" WHERE id = $1",
                    thread_id
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting thread: {e}")
            return False

    async def list_threads(self, user_identifier: str, limit: int = 20, 
                         offset: int = 0) -> List[PersistedThread]:
        """List threads for a user."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, "createdAt", name, "userId", "userIdentifier", tags, metadata
                    FROM "Thread"
                    WHERE "userIdentifier" = $1
                    ORDER BY "createdAtTimestamp" DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_identifier,
                    limit,
                    offset
                )
                
                return [
                    PersistedThread(
                        id=str(row["id"]),
                        createdAt=row["createdAt"],
                        name=row["name"],
                        userId=row["userId"],
                        userIdentifier=row["userIdentifier"],
                        tags=row["tags"],
                        metadata=row["metadata"]
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing threads: {e}")
            return []

    async def create_step(self, step: Step) -> Optional[PersistedStep]:
        """Create a new step."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                step_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(
                    """
                    INSERT INTO "Step"
                    (id, name, type, "threadId", "parentId", streaming, "waitForAnswer", 
                     "isError", metadata, tags, input, output, "createdAt", command, 
                     start_time, end_time, generation, "showInput", language, indent, "defaultOpen")
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 
                            $15, $16, $17, $18, $19, $20, $21)
                    RETURNING id
                    """,
                    step_id,
                    step.name,
                    step.type,
                    step.thread_id,
                    step.parent_id,
                    step.streaming,
                    step.wait_for_answer,
                    step.is_error,
                    json.dumps(step.metadata or {}),
                    step.tags or [],
                    step.input,
                    step.output,
                    now,
                    step.command,
                    step.start,
                    step.end,
                    json.dumps(step.generation) if step.generation else None,
                    step.show_input,
                    step.language,
                    step.indent,
                    step.default_open
                )
                
                return PersistedStep(
                    id=step_id,
                    name=step.name,
                    type=step.type,
                    threadId=step.thread_id,
                    parentId=step.parent_id,
                    streaming=step.streaming,
                    waitForAnswer=step.wait_for_answer,
                    isError=step.is_error,
                    metadata=step.metadata or {},
                    tags=step.tags or [],
                    input=step.input,
                    output=step.output,
                    createdAt=now,
                    command=step.command,
                    start=step.start,
                    end=step.end,
                    generation=step.generation,
                    showInput=step.show_input,
                    language=step.language,
                    indent=step.indent,
                    defaultOpen=step.default_open
                )
        except Exception as e:
            logger.error(f"Error creating step: {e}")
            return None

    async def update_step(self, step_id: str, output: Optional[str] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[PersistedStep]:
        """Update a step."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                set_clauses = []
                values = []
                param_count = 1
                
                if output is not None:
                    set_clauses.append(f"output = ${param_count}")
                    values.append(output)
                    param_count += 1
                
                if metadata is not None:
                    set_clauses.append(f"metadata = ${param_count}")
                    values.append(json.dumps(metadata))
                    param_count += 1
                
                if not set_clauses:
                    return await self.get_step(step_id)
                
                values.append(step_id)
                
                query = f"""
                    UPDATE "Step"
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_count}
                    RETURNING id, name, type, "threadId", "parentId", streaming, 
                             "waitForAnswer", "isError", metadata, tags, input, output, 
                             "createdAt", command, start_time, end_time, generation, 
                             "showInput", language, indent, "defaultOpen"
                """
                
                row = await conn.fetchrow(query, *values)
                
                if row:
                    return PersistedStep(
                        id=str(row["id"]),
                        name=row["name"],
                        type=row["type"],
                        threadId=row["threadId"],
                        parentId=row["parentId"],
                        streaming=row["streaming"],
                        waitForAnswer=row["waitForAnswer"],
                        isError=row["isError"],
                        metadata=row["metadata"],
                        tags=row["tags"],
                        input=row["input"],
                        output=row["output"],
                        createdAt=row["createdAt"],
                        command=row["command"],
                        start=row["start_time"],
                        end=row["end_time"],
                        generation=row["generation"],
                        showInput=row["showInput"],
                        language=row["language"],
                        indent=row["indent"],
                        defaultOpen=row["defaultOpen"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error updating step: {e}")
            return None

    async def get_step(self, step_id: str) -> Optional[PersistedStep]:
        """Get a step by ID."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, name, type, "threadId", "parentId", streaming, 
                           "waitForAnswer", "isError", metadata, tags, input, output, 
                           "createdAt", command, start_time, end_time, generation, 
                           "showInput", language, indent, "defaultOpen"
                    FROM "Step"
                    WHERE id = $1
                    """,
                    step_id
                )
                
                if row:
                    return PersistedStep(
                        id=str(row["id"]),
                        name=row["name"],
                        type=row["type"],
                        threadId=row["threadId"],
                        parentId=row["parentId"],
                        streaming=row["streaming"],
                        waitForAnswer=row["waitForAnswer"],
                        isError=row["isError"],
                        metadata=row["metadata"],
                        tags=row["tags"],
                        input=row["input"],
                        output=row["output"],
                        createdAt=row["createdAt"],
                        command=row["command"],
                        start=row["start_time"],
                        end=row["end_time"],
                        generation=row["generation"],
                        showInput=row["showInput"],
                        language=row["language"],
                        indent=row["indent"],
                        defaultOpen=row["defaultOpen"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting step: {e}")
            return None

    async def delete_step(self, step_id: str) -> bool:
        """Delete a step."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM \"Step\" WHERE id = $1",
                    step_id
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting step: {e}")
            return False

    async def list_steps(self, thread_id: str) -> List[PersistedStep]:
        """List all steps in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, type, "threadId", "parentId", streaming, 
                           "waitForAnswer", "isError", metadata, tags, input, output, 
                           "createdAt", command, start_time, end_time, generation, 
                           "showInput", language, indent, "defaultOpen"
                    FROM "Step"
                    WHERE "threadId" = $1
                    ORDER BY "createdAtTimestamp" ASC
                    """,
                    thread_id
                )
                
                return [
                    PersistedStep(
                        id=str(row["id"]),
                        name=row["name"],
                        type=row["type"],
                        threadId=row["threadId"],
                        parentId=row["parentId"],
                        streaming=row["streaming"],
                        waitForAnswer=row["waitForAnswer"],
                        isError=row["isError"],
                        metadata=row["metadata"],
                        tags=row["tags"],
                        input=row["input"],
                        output=row["output"],
                        createdAt=row["createdAt"],
                        command=row["command"],
                        start=row["start_time"],
                        end=row["end_time"],
                        generation=row["generation"],
                        showInput=row["showInput"],
                        language=row["language"],
                        indent=row["indent"],
                        defaultOpen=row["defaultOpen"]
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing steps: {e}")
            return []

    async def create_element(self, element: Element) -> Optional[PersistedElement]:
        """Create a new element."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                element_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(
                    """
                    INSERT INTO chainlit.elements
                    (id, "threadId", type, url, "chainlitKey", name, display, 
                     "objectKey", size, page, language, "forId", mime, props, "createdAt")
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING id
                    """,
                    element_id,
                    element.thread_id,
                    element.type,
                    element.url,
                    element.chainlit_key,
                    element.name,
                    element.display,
                    element.object_key,
                    element.size,
                    element.page,
                    element.language,
                    element.for_id,
                    element.mime,
                    json.dumps(element.props or {}),
                    now
                )
                
                return PersistedElement(
                    id=element_id,
                    threadId=element.thread_id,
                    type=element.type,
                    url=element.url,
                    chainlitKey=element.chainlit_key,
                    name=element.name,
                    display=element.display,
                    objectKey=element.object_key,
                    size=element.size,
                    page=element.page,
                    language=element.language,
                    forId=element.for_id,
                    mime=element.mime,
                    props=element.props or {},
                    createdAt=now
                )
        except Exception as e:
            logger.error(f"Error creating element: {e}")
            return None

    async def get_element(self, element_id: str) -> Optional[PersistedElement]:
        """Get an element by ID."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, "threadId", type, url, "chainlitKey", name, display, 
                           "objectKey", size, page, language, "forId", mime, props, "createdAt"
                    FROM chainlit.elements
                    WHERE id = $1
                    """,
                    element_id
                )
                
                if row:
                    return PersistedElement(
                        id=str(row["id"]),
                        threadId=row["threadId"],
                        type=row["type"],
                        url=row["url"],
                        chainlitKey=row["chainlitKey"],
                        name=row["name"],
                        display=row["display"],
                        objectKey=row["objectKey"],
                        size=row["size"],
                        page=row["page"],
                        language=row["language"],
                        forId=row["forId"],
                        mime=row["mime"],
                        props=row["props"],
                        createdAt=row["createdAt"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting element: {e}")
            return None

    async def delete_element(self, element_id: str) -> bool:
        """Delete an element."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chainlit.elements WHERE id = $1",
                    element_id
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting element: {e}")
            return False

    async def list_elements(self, thread_id: str) -> List[PersistedElement]:
        """List all elements in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, "threadId", type, url, "chainlitKey", name, display, 
                           "objectKey", size, page, language, "forId", mime, props, "createdAt"
                    FROM chainlit.elements
                    WHERE "threadId" = $1
                    ORDER BY "createdAtTimestamp" ASC
                    """,
                    thread_id
                )
                
                return [
                    PersistedElement(
                        id=str(row["id"]),
                        threadId=row["threadId"],
                        type=row["type"],
                        url=row["url"],
                        chainlitKey=row["chainlitKey"],
                        name=row["name"],
                        display=row["display"],
                        objectKey=row["objectKey"],
                        size=row["size"],
                        page=row["page"],
                        language=row["language"],
                        forId=row["forId"],
                        mime=row["mime"],
                        props=row["props"],
                        createdAt=row["createdAt"]
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing elements: {e}")
            return []

    async def create_feedback(self, feedback: Feedback) -> Optional[PersistedFeedback]:
        """Create new feedback."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                feedback_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                
                await conn.execute(
                    """
                    INSERT INTO chainlit.feedbacks
                    (id, "forId", "threadId", value, comment, "createdAt")
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    feedback_id,
                    feedback.for_id,
                    feedback.thread_id,
                    feedback.value,
                    feedback.comment,
                    now
                )
                
                return PersistedFeedback(
                    id=feedback_id,
                    forId=feedback.for_id,
                    threadId=feedback.thread_id,
                    value=feedback.value,
                    comment=feedback.comment,
                    createdAt=now
                )
        except Exception as e:
            logger.error(f"Error creating feedback: {e}")
            return None

    async def update_feedback(self, feedback_id: str, value: Optional[int] = None, 
                            comment: Optional[str] = None) -> Optional[PersistedFeedback]:
        """Update feedback."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                set_clauses = []
                values = []
                param_count = 1
                
                if value is not None:
                    set_clauses.append(f"value = ${param_count}")
                    values.append(value)
                    param_count += 1
                
                if comment is not None:
                    set_clauses.append(f"comment = ${param_count}")
                    values.append(comment)
                    param_count += 1
                
                if not set_clauses:
                    return await self.get_feedback(feedback_id)
                
                values.append(feedback_id)
                
                query = f"""
                    UPDATE chainlit.feedbacks
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_count}
                    RETURNING id, "forId", "threadId", value, comment, "createdAt"
                """
                
                row = await conn.fetchrow(query, *values)
                
                if row:
                    return PersistedFeedback(
                        id=str(row["id"]),
                        forId=row["forId"],
                        threadId=row["threadId"],
                        value=row["value"],
                        comment=row["comment"],
                        createdAt=row["createdAt"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            return None

    async def get_feedback(self, feedback_id: str) -> Optional[PersistedFeedback]:
        """Get feedback by ID."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, "forId", "threadId", value, comment, "createdAt"
                    FROM chainlit.feedbacks
                    WHERE id = $1
                    """,
                    feedback_id
                )
                
                if row:
                    return PersistedFeedback(
                        id=str(row["id"]),
                        forId=row["forId"],
                        threadId=row["threadId"],
                        value=row["value"],
                        comment=row["comment"],
                        createdAt=row["createdAt"]
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting feedback: {e}")
            return None

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chainlit.feedbacks WHERE id = $1",
                    feedback_id
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting feedback: {e}")
            return False

    async def list_feedbacks(self, thread_id: str) -> List[PersistedFeedback]:
        """List all feedbacks in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, "forId", "threadId", value, comment, "createdAt"
                    FROM chainlit.feedbacks
                    WHERE "threadId" = $1
                    ORDER BY "createdAtTimestamp" ASC
                    """,
                    thread_id
                )
                
                return [
                    PersistedFeedback(
                        id=str(row["id"]),
                        forId=row["forId"],
                        threadId=row["threadId"],
                        value=row["value"],
                        comment=row["comment"],
                        createdAt=row["createdAt"]
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing feedbacks: {e}")
            return []

    async def get_thread_author(self, thread_id: str) -> Optional[str]:
        """Get the author (user identifier) of a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT "userIdentifier"
                    FROM "Thread"
                    WHERE id = $1
                    """,
                    thread_id
                )
                
                return row["userIdentifier"] if row else None
        except Exception as e:
            logger.error(f"Error getting thread author: {e}")
            return None

    async def delete_thread_elements(self, thread_id: str) -> bool:
        """Delete all elements in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chainlit.elements WHERE \"threadId\" = $1",
                    thread_id
                )
                return True  # We don't check count here as it might be 0
        except Exception as e:
            logger.error(f"Error deleting thread elements: {e}")
            return False

    async def delete_thread_steps(self, thread_id: str) -> bool:
        """Delete all steps in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM \"Step\" WHERE \"threadId\" = $1",
                    thread_id
                )
                return True  # We don't check count here as it might be 0
        except Exception as e:
            logger.error(f"Error deleting thread steps: {e}")
            return False

    async def delete_thread_feedbacks(self, thread_id: str) -> bool:
        """Delete all feedbacks in a thread."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chainlit.feedbacks WHERE \"threadId\" = $1",
                    thread_id
                )
                return True  # We don't check count here as it might be 0
        except Exception as e:
            logger.error(f"Error deleting thread feedbacks: {e}")
            return False

    async def get_user_threads(self, user_identifier: str) -> List[ThreadDict]:
        """Get all threads for a user with their steps."""
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Get threads
                threads = await conn.fetch(
                    """
                    SELECT id, "createdAt", name, "userId", "userIdentifier", tags, metadata
                    FROM "Thread"
                    WHERE "userIdentifier" = $1
                    ORDER BY "createdAtTimestamp" DESC
                    """,
                    user_identifier
                )
                
                result = []
                for thread in threads:
                    # Get steps for this thread
                    steps = await conn.fetch(
                        """
                        SELECT id, name, type, "threadId", "parentId", streaming, 
                               "waitForAnswer", "isError", metadata, tags, input, output, 
                               "createdAt", command, start_time, end_time, generation, 
                               "showInput", language, indent, "defaultOpen"
                        FROM "Step"
                        WHERE "threadId" = $1
                        ORDER BY "createdAtTimestamp" ASC
                        """,
                        str(thread["id"])
                    )
                    
                    thread_dict = ThreadDict(
                        id=str(thread["id"]),
                        name=thread["name"],
                        createdAt=thread["createdAt"],
                        userId=thread["userId"],
                        userIdentifier=thread["userIdentifier"],
                        tags=thread["tags"],
                        metadata=thread["metadata"],
                        steps=[
                            StepDict(
                                id=str(step["id"]),
                                name=step["name"],
                                type=step["type"],
                                threadId=step["threadId"],
                                parentId=step["parentId"],
                                streaming=step["streaming"],
                                waitForAnswer=step["waitForAnswer"],
                                isError=step["isError"],
                                metadata=step["metadata"],
                                tags=step["tags"],
                                input=step["input"],
                                output=step["output"],
                                createdAt=step["createdAt"],
                                command=step["command"],
                                start=step["start_time"],
                                end=step["end_time"],
                                generation=step["generation"],
                                showInput=step["showInput"],
                                language=step["language"],
                                indent=step["indent"],
                                defaultOpen=step["defaultOpen"]
                            )
                            for step in steps
                        ]
                    )
                    result.append(thread_dict)
                
                return result
        except Exception as e:
            logger.error(f"Error getting user threads: {e}")
            return []


# Global data layer instance
_data_layer: Optional[PostgreSQLDataLayer] = None


def get_data_layer() -> PostgreSQLDataLayer:
    """Get or create the global data layer instance."""
    global _data_layer
    
    if _data_layer is None:
        from app.core.config import settings
        
        # Get database URL from environment or settings
        database_url = os.getenv("CHAINLIT_DATABASE_URL") or getattr(settings, "database_url", None)
        
        if not database_url:
            raise ValueError("CHAINLIT_DATABASE_URL not configured")
        
        _data_layer = PostgreSQLDataLayer(conninfo=database_url)
    
    return _data_layer


# Register the data layer with Chainlit
@cl.data_layer
def get_chainlit_data_layer():
    """Chainlit data layer factory function."""
    return get_data_layer()