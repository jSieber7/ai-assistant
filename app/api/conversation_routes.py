"""
API routes for conversation management
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import logging
from datetime import datetime

from app.core.storage.postgresql_client import get_postgresql_client

logger = logging.getLogger(__name__)

router = APIRouter()


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    agent_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    user_id: Optional[str]
    model_id: Optional[str]
    agent_name: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0


class MessageCreate(BaseModel):
    conversation_id: str
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse] = []


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    db_client=Depends(get_postgresql_client)
):
    """Create a new conversation"""
    try:
        # Generate a title if not provided
        if not conversation.title:
            conversation.title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Insert conversation
        async with db_client.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO agent_memory.chat_conversations 
                (title, model_id, agent_name, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id, title, user_id, model_id, agent_name, metadata, created_at, updated_at
                """,
                conversation.title,
                conversation.model_id,
                conversation.agent_name,
                conversation.metadata or {}
            )
            
            return ConversationResponse(
                id=str(result["id"]),
                title=result["title"],
                user_id=result["user_id"],
                model_id=result["model_id"],
                agent_name=result["agent_name"],
                metadata=result["metadata"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
                message_count=0
            )
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    db_client=Depends(get_postgresql_client)
):
    """List conversations with message count"""
    try:
        async with db_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.*, COUNT(m.id) as message_count
                FROM agent_memory.chat_conversations c
                LEFT JOIN agent_memory.chat_messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )
            
            return [
                ConversationResponse(
                    id=str(row["id"]),
                    title=row["title"],
                    user_id=row["user_id"],
                    model_id=row["model_id"],
                    agent_name=row["agent_name"],
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=row["message_count"]
                )
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: str,
    db_client=Depends(get_postgresql_client)
):
    """Get a conversation with all its messages"""
    try:
        async with db_client.pool.acquire() as conn:
            # Get conversation
            conv_row = await conn.fetchrow(
                "SELECT * FROM agent_memory.chat_conversations WHERE id = $1",
                conversation_id
            )
            
            if not conv_row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Get messages
            msg_rows = await conn.fetch(
                "SELECT * FROM agent_memory.chat_messages WHERE conversation_id = $1 ORDER BY created_at",
                conversation_id
            )
            
            messages = [
                MessageResponse(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    content=row["content"],
                    metadata=row["metadata"],
                    created_at=row["created_at"]
                )
                for row in msg_rows
            ]
            
            return ConversationWithMessages(
                id=str(conv_row["id"]),
                title=conv_row["title"],
                user_id=conv_row["user_id"],
                model_id=conv_row["model_id"],
                agent_name=conv_row["agent_name"],
                metadata=conv_row["metadata"],
                created_at=conv_row["created_at"],
                updated_at=conv_row["updated_at"],
                message_count=len(messages),
                messages=messages
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversation")


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    conversation_id: str,
    message: MessageCreate,
    db_client=Depends(get_postgresql_client)
):
    """Add a message to a conversation"""
    try:
        # Verify conversation exists
        async with db_client.pool.acquire() as conn:
            conv_exists = await conn.fetchval(
                "SELECT id FROM agent_memory.chat_conversations WHERE id = $1",
                conversation_id
            )
            
            if not conv_exists:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Insert message
            result = await conn.fetchrow(
                """
                INSERT INTO agent_memory.chat_messages 
                (conversation_id, role, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id, conversation_id, role, content, metadata, created_at
                """,
                conversation_id,
                message.role,
                message.content,
                message.metadata or {}
            )
            
            # Update conversation's updated_at
            await conn.execute(
                "UPDATE agent_memory.chat_conversations SET updated_at = NOW() WHERE id = $1",
                conversation_id
            )
            
            return MessageResponse(
                id=str(result["id"]),
                conversation_id=str(result["conversation_id"]),
                role=result["role"],
                content=result["content"],
                metadata=result["metadata"],
                created_at=result["created_at"]
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add message")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db_client=Depends(get_postgresql_client)
):
    """Delete a conversation and all its messages"""
    try:
        async with db_client.pool.acquire() as conn:
            # Check if conversation exists
            conv_exists = await conn.fetchval(
                "SELECT id FROM agent_memory.chat_conversations WHERE id = $1",
                conversation_id
            )
            
            if not conv_exists:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Delete conversation (messages will be deleted due to CASCADE)
            result = await conn.execute(
                "DELETE FROM agent_memory.chat_conversations WHERE id = $1",
                conversation_id
            )
            
            if "DELETE 1" not in result:
                raise HTTPException(status_code=500, detail="Failed to delete conversation")
            
            return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_client=Depends(get_postgresql_client)
):
    """Update conversation metadata"""
    try:
        async with db_client.pool.acquire() as conn:
            # Check if conversation exists
            conv_exists = await conn.fetchval(
                "SELECT id FROM agent_memory.chat_conversations WHERE id = $1",
                conversation_id
            )
            
            if not conv_exists:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Build update query
            update_fields = []
            values = []
            param_count = 1
            
            if title is not None:
                update_fields.append(f"title = ${param_count}")
                values.append(title)
                param_count += 1
                
            if metadata is not None:
                update_fields.append(f"metadata = ${param_count}")
                values.append(metadata)
                param_count += 1
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            update_fields.append("updated_at = NOW()")
            values.append(conversation_id)
            
            # Execute update
            result = await conn.fetchrow(
                f"""
                UPDATE agent_memory.chat_conversations 
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
                RETURNING id, title, user_id, model_id, agent_name, metadata, created_at, updated_at
                """,
                *values
            )
            
            # Get message count
            message_count = await conn.fetchval(
                "SELECT COUNT(*) FROM agent_memory.chat_messages WHERE conversation_id = $1",
                conversation_id
            )
            
            return ConversationResponse(
                id=str(result["id"]),
                title=result["title"],
                user_id=result["user_id"],
                model_id=result["model_id"],
                agent_name=result["agent_name"],
                metadata=result["metadata"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
                message_count=message_count
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update conversation")