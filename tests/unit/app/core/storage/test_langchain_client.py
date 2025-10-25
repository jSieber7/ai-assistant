"""
Unit tests for LangChain Client.

This module tests the LangChain client for PostgreSQL integration with LangChain and LangGraph.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, asynccontextmanager
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.storage.langchain_client import (
    LangChainClient,
    get_langchain_client,
    close_langchain_connection
)


class TestLangChainClient:
    """Test LangChainClient class"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for LangChain client"""
        settings = Mock()
        settings.postgres_host = "localhost"
        settings.postgres_port = 5432
        settings.postgres_db = "test_langchain_db"
        settings.postgres_user = "test_user"
        settings.postgres_password = "test_password"
        return settings

    @pytest.fixture
    def langchain_client(self, mock_settings):
        """Create a LangChain client instance"""
        with patch('app.core.storage.langchain_client.settings', mock_settings):
            return LangChainClient()

    def test_client_init(self, langchain_client, mock_settings):
        """Test LangChainClient initialization"""
        assert langchain_client.host == "localhost"
        assert langchain_client.port == 5432
        assert langchain_client.database == "test_langchain_db"
        assert langchain_client.user == "test_user"
        assert langchain_client.password == "test_password"
        assert langchain_client.pool is None

    @pytest.mark.asyncio
    async def test_connect(self, langchain_client):
        """Test connecting to PostgreSQL"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await langchain_client.connect()
            
            assert langchain_client.pool == mock_pool
            mock_create_pool.assert_called_once_with(
                host="localhost",
                port=5432,
                database="test_langchain_db",
                user="test_user",
                password="test_password",
                min_size=1,
                max_size=10,
                command_timeout=60
            )

    @pytest.mark.asyncio
    async def test_connect_with_schema_creation(self, langchain_client):
        """Test connecting with schema creation"""
        with patch('asyncpg.create_pool') as mock_create_pool, \
             patch.object(langchain_client, '_ensure_schema_exists') as mock_ensure_schema:
            
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await langchain_client.connect()
            
            mock_ensure_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_error(self, langchain_client):
        """Test connecting to PostgreSQL with error"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception, match="Connection error"):
                await langchain_client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, langchain_client):
        """Test disconnecting from PostgreSQL"""
        mock_pool = AsyncMock()
        langchain_client.pool = mock_pool
        
        await langchain_client.disconnect()
        
        mock_pool.close.assert_called_once()
        assert langchain_client.pool is None

    @pytest.mark.asyncio
    async def test_ensure_schema_exists(self, langchain_client):
        """Test ensuring schema exists"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        await langchain_client._ensure_schema_exists()
        
        # Should have executed schema creation commands
        assert mock_conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_create_conversation(self, langchain_client):
        """Test creating a new conversation"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_data = {
            "id": "conv-123",
            "user_id": "user-456",
            "title": "Test Conversation",
            "metadata": {"source": "web"}
        }
        
        result = await langchain_client.create_conversation(conversation_data)
        
        assert result == "conv-123"
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO conversations" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_conversation(self, langchain_client):
        """Test getting conversation by ID"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "conv-123"
        expected_conversation = {
            "id": "conv-123",
            "user_id": "user-456",
            "title": "Test Conversation",
            "created_at": datetime.now()
        }
        mock_conn.fetchrow.return_value = expected_conversation
        
        result = await langchain_client.get_conversation(conversation_id)
        
        assert result == expected_conversation
        mock_conn.fetchrow.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchrow.call_args
        assert "SELECT.*FROM conversations" in call_args[0][0]
        assert conversation_id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, langchain_client):
        """Test getting conversation that doesn't exist"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "nonexistent"
        mock_conn.fetchrow.return_value = None
        
        result = await langchain_client.get_conversation(conversation_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_conversation(self, langchain_client):
        """Test updating conversation data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "conv-123"
        update_data = {
            "title": "Updated Conversation",
            "metadata": {"updated": True}
        }
        
        result = await langchain_client.update_conversation(conversation_id, update_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE conversations" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_conversations(self, langchain_client):
        """Test listing conversations"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        expected_conversations = [
            {
                "id": "conv-1",
                "user_id": "user-456",
                "title": "Conversation 1"
            },
            {
                "id": "conv-2",
                "user_id": "user-456",
                "title": "Conversation 2"
            }
        ]
        mock_conn.fetch.return_value = expected_conversations
        
        result = await langchain_client.list_conversations(user_id="user-456")
        
        assert result == expected_conversations
        mock_conn.fetch.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM conversations" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(self, langchain_client):
        """Test listing conversations with filters"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        expected_conversations = [
            {"id": "conv-1", "title": "Conversation 1"}
        ]
        mock_conn.fetch.return_value = expected_conversations
        
        result = await langchain_client.list_conversations(
            user_id="user-456",
            limit=10,
            offset=5,
            status="active"
        )
        
        assert result == expected_conversations
        mock_conn.fetch.assert_called_once()
        
        # Check SQL call includes filters
        call_args = mock_conn.fetch.call_args
        assert "LIMIT" in call_args[0][0]
        assert "OFFSET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_add_chat_message(self, langchain_client):
        """Test adding a chat message"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        message_data = {
            "conversation_id": "conv-123",
            "role": "user",
            "content": "Hello, how are you?",
            "metadata": {"tokens": 10}
        }
        
        result = await langchain_client.add_chat_message(message_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO chat_messages" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_conversation_messages(self, langchain_client):
        """Test getting messages for a conversation"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "conv-123"
        expected_messages = [
            {
                "id": "msg-1",
                "conversation_id": "conv-123",
                "role": "user",
                "content": "Hello"
            },
            {
                "id": "msg-2",
                "conversation_id": "conv-123",
                "role": "assistant",
                "content": "Hi there!"
            }
        ]
        mock_conn.fetch.return_value = expected_messages
        
        result = await langchain_client.get_conversation_messages(conversation_id)
        
        assert result == expected_messages
        mock_conn.fetch.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM chat_messages" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_next_message_sequence(self, langchain_client):
        """Test getting next message sequence number"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "conv-123"
        mock_conn.fetchval.return_value = 5
        
        result = await langchain_client.get_next_message_sequence(conversation_id)
        
        assert result == 5
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_summary(self, langchain_client):
        """Test adding a memory summary"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        summary_data = {
            "conversation_id": "conv-123",
            "summary": "User greeted and assistant responded",
            "metadata": {"model": "gpt-4"}
        }
        
        result = await langchain_client.add_memory_summary(summary_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO memory_summaries" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_conversation_summaries(self, langchain_client):
        """Test getting memory summaries for a conversation"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        conversation_id = "conv-123"
        expected_summaries = [
            {
                "id": "summary-1",
                "conversation_id": "conv-123",
                "summary": "First part of conversation"
            },
            {
                "id": "summary-2",
                "conversation_id": "conv-123",
                "summary": "Second part of conversation"
            }
        ]
        mock_conn.fetch.return_value = expected_summaries
        
        result = await langchain_client.get_conversation_summaries(conversation_id)
        
        assert result == expected_summaries
        mock_conn.fetch.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM memory_summaries" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_workflow(self, langchain_client):
        """Test creating a new workflow"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        workflow_data = {
            "id": "workflow-123",
            "type": "conversational",
            "status": "active",
            "metadata": {"user_id": "user-456"}
        }
        
        result = await langchain_client.create_workflow(workflow_data)
        
        assert result == "workflow-123"
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_workflow(self, langchain_client):
        """Test updating workflow data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        update_data = {
            "status": "completed",
            "metadata": {"completed_at": datetime.now()}
        }
        
        result = await langchain_client.update_workflow(workflow_id, update_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_workflow(self, langchain_client):
        """Test getting workflow by ID"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        expected_workflow = {
            "id": "workflow-123",
            "type": "conversational",
            "status": "active"
        }
        mock_conn.fetchrow.return_value = expected_workflow
        
        result = await langchain_client.get_workflow(workflow_id)
        
        assert result == expected_workflow
        mock_conn.fetchrow.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchrow.call_args
        assert "SELECT.*FROM workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_add_workflow_step(self, langchain_client):
        """Test adding a workflow step"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        step_data = {
            "workflow_id": "workflow-123",
            "step_name": "processing",
            "status": "completed",
            "input_data": {"query": "test"},
            "output_data": {"response": "result"}
        }
        
        result = await langchain_client.add_workflow_step(step_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO workflow_steps" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_workflow_step(self, langchain_client):
        """Test updating workflow step data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        step_id = "step-123"
        update_data = {
            "status": "completed",
            "output_data": {"result": "final"}
        }
        
        result = await langchain_client.update_workflow_step(step_id, update_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE workflow_steps" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, langchain_client):
        """Test saving a workflow checkpoint"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        checkpoint_data = {
            "workflow_id": "workflow-123",
            "checkpoint_id": "checkpoint-456",
            "checkpoint_data": {"state": "current"},
            "metadata": {"step": "processing"}
        }
        
        result = await langchain_client.save_checkpoint(checkpoint_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO checkpoints" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_checkpoint(self, langchain_client):
        """Test getting checkpoint by ID"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        checkpoint_id = "checkpoint-456"
        expected_checkpoint = {
            "id": "checkpoint-456",
            "workflow_id": "workflow-123",
            "checkpoint_data": {"state": "current"}
        }
        mock_conn.fetchrow.return_value = expected_checkpoint
        
        result = await langchain_client.get_checkpoint(checkpoint_id)
        
        assert result == expected_checkpoint
        mock_conn.fetchrow.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchrow.call_args
        assert "SELECT.*FROM checkpoints" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_start_tool_execution(self, langchain_client):
        """Test starting a tool execution"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        execution_data = {
            "tool_name": "search_tool",
            "input_data": {"query": "test"},
            "metadata": {"user_id": "user-456"}
        }
        
        result = await langchain_client.start_tool_execution(execution_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO tool_executions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_complete_tool_execution(self, langchain_client):
        """Test completing a tool execution"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        execution_id = "exec-123"
        completion_data = {
            "output_data": {"result": "success"},
            "metadata": {"duration": 1.5}
        }
        
        result = await langchain_client.complete_tool_execution(execution_id, completion_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE tool_executions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_fail_tool_execution(self, langchain_client):
        """Test marking a tool execution as failed"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        execution_id = "exec-123"
        error_data = {
            "error_message": "Tool execution failed",
            "metadata": {"error_code": 500}
        }
        
        result = await langchain_client.fail_tool_execution(execution_id, error_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE tool_executions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_agent_session(self, langchain_client):
        """Test creating an agent session"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        session_data = {
            "agent_id": "agent-123",
            "conversation_id": "conv-456",
            "metadata": {"model": "gpt-4"}
        }
        
        result = await langchain_client.create_agent_session(session_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO agent_sessions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_agent_session_activity(self, langchain_client):
        """Test updating agent session activity"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        session_id = "session-123"
        activity_data = {
            "last_activity": datetime.now(),
            "metadata": {"active": True}
        }
        
        result = await langchain_client.update_agent_session_activity(session_id, activity_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE agent_sessions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_record_metric(self, langchain_client):
        """Test recording a metric"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        metric_data = {
            "metric_name": "response_time",
            "metric_value": 1.5,
            "component_type": "llm",
            "component_id": "gpt-4",
            "metadata": {"request_id": "req-123"}
        }
        
        result = await langchain_client.record_metric(metric_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO metrics" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_log_performance(self, langchain_client):
        """Test logging performance data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        performance_data = {
            "component_type": "agent",
            "component_id": "conversational_agent",
            "metric_name": "execution_time",
            "metric_value": 2.5,
            "metadata": {"steps": 5}
        }
        
        result = await langchain_client.log_performance(performance_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO performance_logs" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_cleanup_expired_data(self, langchain_client):
        """Test cleaning up expired data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        # Mock cleanup results
        mock_conn.fetchval.side_effect = [10, 5, 3, 2]  # Deleted counts
        
        result = await langchain_client.cleanup_expired_data()
        
        assert result["expired_sessions"] == 10
        assert result["expired_checkpoints"] == 5
        assert result["old_metrics"] == 3
        assert result["old_performance_logs"] == 2
        assert mock_conn.fetchval.call_count == 4

    @pytest.mark.asyncio
    async def test_get_statistics(self, langchain_client):
        """Test getting LangChain system statistics"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        langchain_client.pool = mock_pool
        
        # Mock statistics results
        mock_conn.fetchrow.side_effect = [
            {"count": 100},  # Total conversations
            {"count": 25},   # Active workflows
            {"count": 500},  # Total messages
            {"count": 1000}, # Total tool executions
            {"count": 50},   # Active agent sessions
            {"count": 2000}  # Total metrics
        ]
        
        result = await langchain_client.get_statistics()
        
        assert result["total_conversations"] == 100
        assert result["active_workflows"] == 25
        assert result["total_messages"] == 500
        assert result["total_tool_executions"] == 1000
        assert result["active_agent_sessions"] == 50
        assert result["total_metrics"] == 2000
        assert mock_conn.fetchrow.call_count == 6


class TestLangChainClientHelpers:
    """Test helper functions for LangChain client"""

    @pytest.mark.asyncio
    async def test_get_langchain_client(self):
        """Test get_langchain_client function"""
        with patch('app.core.storage.langchain_client.LangChainClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await get_langchain_client()
            
            assert result == mock_client
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_langchain_client_cached(self):
        """Test get_langchain_client function with caching"""
        with patch('app.core.storage.langchain_client.LangChainClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # First call should create and connect
            result1 = await get_langchain_client()
            assert result1 == mock_client
            mock_client.connect.assert_called_once()
            
            # Second call should return cached instance
            result2 = await get_langchain_client()
            assert result2 == mock_client
            # Should not connect again
            assert mock_client.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_close_langchain_connection(self):
        """Test close_langchain_connection function"""
        with patch('app.core.storage.langchain_client._langchain_client') as mock_client:
            mock_client = AsyncMock()
            _langchain_client = mock_client
            
            await close_langchain_connection()
            
            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_langchain_connection_no_client(self):
        """Test close_langchain_connection when no client exists"""
        with patch('app.core.storage.langchain_client._langchain_client', None):
            # Should not raise error
            await close_langchain_connection()