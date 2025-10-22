"""
Conversation manager for persisting and retrieving chat history
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.core.storage.postgresql_client import get_postgresql_client

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation persistence and retrieval"""

    @staticmethod
    async def create_conversation(
        title: Optional[str] = None,
        model_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation and return its ID"""
        try:
            db_client = await get_postgresql_client()
            
            # Generate a title if not provided
            if not title:
                title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            async with db_client.pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    INSERT INTO agent_memory.chat_conversations 
                    (title, model_id, agent_name, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    title,
                    model_id,
                    agent_name,
                    metadata or {}
                )
                
                return str(result["id"])
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise

    @staticmethod
    async def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation with all its messages"""
        try:
            db_client = await get_postgresql_client()
            
            async with db_client.pool.acquire() as conn:
                # Get conversation
                conv_row = await conn.fetchrow(
                    "SELECT * FROM agent_memory.chat_conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conv_row:
                    return None
                
                # Get messages
                msg_rows = await conn.fetch(
                    "SELECT * FROM agent_memory.chat_messages WHERE conversation_id = $1 ORDER BY created_at",
                    conversation_id
                )
                
                messages = [
                    {
                        "id": str(row["id"]),
                        "conversation_id": str(row["conversation_id"]),
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"]
                    }
                    for row in msg_rows
                ]
                
                return {
                    "id": str(conv_row["id"]),
                    "title": conv_row["title"],
                    "user_id": conv_row["user_id"],
                    "model_id": conv_row["model_id"],
                    "agent_name": conv_row["agent_name"],
                    "metadata": conv_row["metadata"],
                    "created_at": conv_row["created_at"],
                    "updated_at": conv_row["updated_at"],
                    "messages": messages
                }
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            raise

    @staticmethod
    async def add_message(
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to a conversation and return its ID"""
        try:
            db_client = await get_postgresql_client()
            
            async with db_client.pool.acquire() as conn:
                # Insert message
                result = await conn.fetchrow(
                    """
                    INSERT INTO agent_memory.chat_messages 
                    (conversation_id, role, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    conversation_id,
                    role,
                    content,
                    metadata or {}
                )
                
                # Update conversation's updated_at
                await conn.execute(
                    "UPDATE agent_memory.chat_conversations SET updated_at = NOW() WHERE id = $1",
                    conversation_id
                )
                
                return str(result["id"])
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise

    @staticmethod
    async def list_conversations(
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List conversations with message count"""
        try:
            db_client = await get_postgresql_client()
            
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
                    {
                        "id": str(row["id"]),
                        "title": row["title"],
                        "user_id": row["user_id"],
                        "model_id": row["model_id"],
                        "agent_name": row["agent_name"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "message_count": row["message_count"]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise

    @staticmethod
    async def delete_conversation(conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        try:
            db_client = await get_postgresql_client()
            
            async with db_client.pool.acquire() as conn:
                # Check if conversation exists
                conv_exists = await conn.fetchval(
                    "SELECT id FROM agent_memory.chat_conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conv_exists:
                    return False
                
                # Delete conversation (messages will be deleted due to CASCADE)
                result = await conn.execute(
                    "DELETE FROM agent_memory.chat_conversations WHERE id = $1",
                    conversation_id
                )
                
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise

    @staticmethod
    async def update_conversation(
        conversation_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update conversation metadata"""
        try:
            db_client = await get_postgresql_client()
            
            async with db_client.pool.acquire() as conn:
                # Check if conversation exists
                conv_exists = await conn.fetchval(
                    "SELECT id FROM agent_memory.chat_conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conv_exists:
                    return False
                
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
                    return True  # Nothing to update
                
                update_fields.append("updated_at = NOW()")
                values.append(conversation_id)
                
                # Execute update
                await conn.execute(
                    f"""
                    UPDATE agent_memory.chat_conversations 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    """,
                    *values
                )
                
                return True
        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            raise

    @staticmethod
    async def get_conversation_messages(
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation, optionally limited"""
        try:
            db_client = await get_postgresql_client()
            
            async with db_client.pool.acquire() as conn:
                query = """
                    SELECT * FROM agent_memory.chat_messages 
                    WHERE conversation_id = $1 
                    ORDER BY created_at
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query, conversation_id)
                
                return [
                    {
                        "id": str(row["id"]),
                        "conversation_id": str(row["conversation_id"]),
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting conversation messages: {str(e)}")
            raise