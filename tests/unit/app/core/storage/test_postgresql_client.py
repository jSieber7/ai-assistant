"""
Unit tests for PostgreSQL Client.

This module tests the PostgreSQL client for storing multi-writer/checker system data.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, asynccontextmanager
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.storage.postgresql_client import (
    PostgreSQLClient,
    get_postgresql_client,
    close_postgresql_connection
)


class TestPostgreSQLClient:
    """Test PostgreSQLClient class"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for PostgreSQL client"""
        settings = Mock()
        settings.postgres_host = "localhost"
        settings.postgres_port = 5432
        settings.postgres_db = "test_db"
        settings.postgres_user = "test_user"
        settings.postgres_password = "test_password"
        return settings

    @pytest.fixture
    def postgresql_client(self, mock_settings):
        """Create a PostgreSQL client instance"""
        with patch('app.core.storage.postgresql_client.settings', mock_settings):
            return PostgreSQLClient()

    def test_client_init(self, postgresql_client, mock_settings):
        """Test PostgreSQLClient initialization"""
        assert postgresql_client.host == "localhost"
        assert postgresql_client.port == 5432
        assert postgresql_client.database == "test_db"
        assert postgresql_client.user == "test_user"
        assert postgresql_client.password == "test_password"
        assert postgresql_client.pool is None

    @pytest.mark.asyncio
    async def test_connect(self, postgresql_client):
        """Test connecting to PostgreSQL"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await postgresql_client.connect()
            
            assert postgresql_client.pool == mock_pool
            mock_create_pool.assert_called_once_with(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password="test_password",
                min_size=1,
                max_size=10,
                command_timeout=60
            )

    @pytest.mark.asyncio
    async def test_connect_error(self, postgresql_client):
        """Test connecting to PostgreSQL with error"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception, match="Connection error"):
                await postgresql_client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, postgresql_client):
        """Test disconnecting from PostgreSQL"""
        mock_pool = AsyncMock()
        postgresql_client.pool = mock_pool
        
        await postgresql_client.disconnect()
        
        mock_pool.close.assert_called_once()
        assert postgresql_client.pool is None

    @pytest.mark.asyncio
    async def test_disconnect_no_pool(self, postgresql_client):
        """Test disconnecting when no pool exists"""
        # Should not raise error
        await postgresql_client.disconnect()

    @pytest.mark.asyncio
    async def test_save_workflow(self, postgresql_client):
        """Test saving workflow data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_data = {
            "id": "workflow-123",
            "title": "Test Workflow",
            "description": "A test workflow",
            "status": "active",
            "metadata": {"key": "value"}
        }
        
        result = await postgresql_client.save_workflow(workflow_data)
        
        assert result == "workflow-123"
        mock_conn.fetchval.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_workflow(self, postgresql_client):
        """Test updating workflow data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        update_data = {
            "title": "Updated Workflow",
            "status": "completed"
        }
        
        result = await postgresql_client.update_workflow(workflow_id, update_data)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchval.call_args
        assert "UPDATE workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_workflow(self, postgresql_client):
        """Test getting workflow by ID"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        expected_workflow = {
            "id": "workflow-123",
            "title": "Test Workflow",
            "status": "active"
        }
        mock_conn.fetchrow.return_value = expected_workflow
        
        result = await postgresql_client.get_workflow(workflow_id)
        
        assert result == expected_workflow
        mock_conn.fetchrow.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchrow.call_args
        assert "SELECT.*FROM workflows" in call_args[0][0]
        assert workflow_id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, postgresql_client):
        """Test getting workflow that doesn't exist"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "nonexistent"
        mock_conn.fetchrow.return_value = None
        
        result = await postgresql_client.get_workflow(workflow_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_save_content(self, postgresql_client):
        """Test saving content data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        content_data = {
            "workflow_id": "workflow-123",
            "content_type": "text",
            "content": "Test content",
            "metadata": {"source": "user"}
        }
        
        result = await postgresql_client.save_content(content_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO content" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_check_result(self, postgresql_client):
        """Test saving check result data"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        check_data = {
            "workflow_id": "workflow-123",
            "checker_type": "grammar",
            "result": "passed",
            "score": 0.95,
            "details": {"errors": []}
        }
        
        result = await postgresql_client.save_check_result(check_data)
        
        assert isinstance(result, str)
        mock_conn.fetchval.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchval.call_args
        assert "INSERT INTO check_results" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_workflow_content(self, postgresql_client):
        """Test getting content for a workflow"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        expected_content = [
            {
                "id": "content-1",
                "workflow_id": "workflow-123",
                "content_type": "text",
                "content": "Test content 1"
            },
            {
                "id": "content-2",
                "workflow_id": "workflow-123",
                "content_type": "image",
                "content": "base64-image-data"
            }
        ]
        mock_conn.fetch.return_value = expected_content
        
        result = await postgresql_client.get_workflow_content(workflow_id)
        
        assert result == expected_content
        mock_conn.fetch.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM content" in call_args[0][0]
        assert workflow_id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_workflow_check_results(self, postgresql_client):
        """Test getting check results for a workflow"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        expected_results = [
            {
                "id": "check-1",
                "workflow_id": "workflow-123",
                "checker_type": "grammar",
                "result": "passed",
                "score": 0.95
            },
            {
                "id": "check-2",
                "workflow_id": "workflow-123",
                "checker_type": "style",
                "result": "warning",
                "score": 0.80
            }
        ]
        mock_conn.fetch.return_value = expected_results
        
        result = await postgresql_client.get_workflow_check_results(workflow_id)
        
        assert result == expected_results
        mock_conn.fetch.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM check_results" in call_args[0][0]
        assert workflow_id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_list_workflows(self, postgresql_client):
        """Test listing workflows"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        expected_workflows = [
            {
                "id": "workflow-1",
                "title": "Workflow 1",
                "status": "active"
            },
            {
                "id": "workflow-2",
                "title": "Workflow 2",
                "status": "completed"
            }
        ]
        mock_conn.fetch.return_value = expected_workflows
        
        result = await postgresql_client.list_workflows()
        
        assert result == expected_workflows
        mock_conn.fetch.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM workflows" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_workflows_with_status(self, postgresql_client):
        """Test listing workflows with status filter"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        expected_workflows = [
            {
                "id": "workflow-1",
                "title": "Workflow 1",
                "status": "active"
            }
        ]
        mock_conn.fetch.return_value = expected_workflows
        
        result = await postgresql_client.list_workflows(status="active")
        
        assert result == expected_workflows
        mock_conn.fetch.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetch.call_args
        assert "SELECT.*FROM workflows" in call_args[0][0]
        assert "status" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_workflows_with_limit(self, postgresql_client):
        """Test listing workflows with limit"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        expected_workflows = [
            {"id": "workflow-1", "title": "Workflow 1"}
        ]
        mock_conn.fetch.return_value = expected_workflows
        
        result = await postgresql_client.list_workflows(limit=10, offset=5)
        
        assert result == expected_workflows
        mock_conn.fetch.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetch.call_args
        assert "LIMIT" in call_args[0][0]
        assert "OFFSET" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_workflow(self, postgresql_client):
        """Test deleting a workflow"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "workflow-123"
        mock_conn.fetchval.return_value = 1  # Number of deleted rows
        
        result = await postgresql_client.delete_workflow(workflow_id)
        
        assert result is True
        mock_conn.fetchval.assert_called_once()
        
        # Check the SQL call
        call_args = mock_conn.fetchval.call_args
        assert "DELETE FROM workflows" in call_args[0][0]
        assert workflow_id in call_args[0][1]

    @pytest.mark.asyncio
    async def test_delete_workflow_not_found(self, postgresql_client):
        """Test deleting a workflow that doesn't exist"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        workflow_id = "nonexistent"
        mock_conn.fetchval.return_value = 0  # No rows deleted
        
        result = await postgresql_client.delete_workflow(workflow_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_statistics(self, postgresql_client):
        """Test getting system statistics"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        postgresql_client.pool = mock_pool
        
        mock_conn.fetchrow.side_effect = [
            {"count": 100},  # Total workflows
            {"count": 25},   # Active workflows
            {"count": 500},  # Total content items
            {"count": 1000}  # Total check results
        ]
        
        result = await postgresql_client.get_statistics()
        
        assert result["total_workflows"] == 100
        assert result["active_workflows"] == 25
        assert result["total_content_items"] == 500
        assert result["total_check_results"] == 1000
        assert mock_conn.fetchrow.call_count == 4

    @pytest.mark.asyncio
    async def test_save_workflow_with_transaction(self, postgresql_client):
        """Test saving workflow with transaction"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        postgresql_client.pool = mock_pool
        
        workflow_data = {
            "id": "workflow-123",
            "title": "Test Workflow",
            "status": "active"
        }
        
        await postgresql_client.save_workflow(workflow_data)
        
        # Should have used transaction
        mock_conn.transaction.assert_called_once()
        mock_transaction.fetchval.assert_called_once()


class TestPostgreSQLClientHelpers:
    """Test helper functions for PostgreSQL client"""

    @pytest.mark.asyncio
    async def test_get_postgresql_client(self):
        """Test get_postgresql_client function"""
        with patch('app.core.storage.postgresql_client.PostgreSQLClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await get_postgresql_client()
            
            assert result == mock_client
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_postgresql_client_cached(self):
        """Test get_postgresql_client function with caching"""
        with patch('app.core.storage.postgresql_client.PostgreSQLClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # First call should create and connect
            result1 = await get_postgresql_client()
            assert result1 == mock_client
            mock_client.connect.assert_called_once()
            
            # Second call should return cached instance
            result2 = await get_postgresql_client()
            assert result2 == mock_client
            # Should not connect again
            assert mock_client.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_close_postgresql_connection(self):
        """Test close_postgresql_connection function"""
        with patch('app.core.storage.postgresql_client._postgresql_client') as mock_client:
            mock_client = AsyncMock()
            _postgresql_client = mock_client
            
            await close_postgresql_connection()
            
            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_postgresql_connection_no_client(self):
        """Test close_postgresql_connection when no client exists"""
        with patch('app.core.storage.postgresql_client._postgresql_client', None):
            # Should not raise error
            await close_postgresql_connection()