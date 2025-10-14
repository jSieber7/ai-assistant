"""
MongoDB client for multi-writer/checker system storage
"""

from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for storing multi-writer/checker system data"""

    def __init__(
        self, connection_string: str, database_name: str = "multi_writer_system"
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self._connected = False

    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            # Test the connection
            await self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self._connected = True
            logger.info(f"Connected to MongoDB database: {self.database_name}")

            # Create indexes for better performance
            await self._create_indexes()

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self._connected = False
            raise

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self):
        """Create indexes for collections"""
        # Workflow collection indexes
        await self.db.workflows.create_index("workflow_id", unique=True)
        await self.db.workflows.create_index("status")
        await self.db.workflows.create_index("created_at")

        # Content collection indexes
        await self.db.content.create_index("workflow_id")
        await self.db.content.create_index("writer_id")
        await self.db.content.create_index("created_at")

        # Check results collection indexes
        await self.db.check_results.create_index("workflow_id")
        await self.db.check_results.create_index("checker_id")
        await self.db.check_results.create_index("created_at")

    async def save_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Save workflow data"""
        if not self._connected:
            await self.connect()

        workflow_data["created_at"] = datetime.utcnow()
        workflow_data["updated_at"] = datetime.utcnow()

        result = await self.db.workflows.insert_one(workflow_data)
        return str(result.inserted_id)

    async def update_workflow(
        self, workflow_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update workflow data"""
        if not self._connected:
            await self.connect()

        update_data["updated_at"] = datetime.utcnow()

        result = await self.db.workflows.update_one(
            {"workflow_id": workflow_id}, {"$set": update_data}
        )

        return result.modified_count > 0

    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        if not self._connected:
            await self.connect()

        workflow = await self.db.workflows.find_one({"workflow_id": workflow_id})
        if workflow:
            # Convert ObjectId to string
            workflow["_id"] = str(workflow["_id"])
        return workflow

    async def save_content(self, content_data: Dict[str, Any]) -> str:
        """Save content data"""
        if not self._connected:
            await self.connect()

        content_data["created_at"] = datetime.utcnow()

        result = await self.db.content.insert_one(content_data)
        return str(result.inserted_id)

    async def save_check_result(self, check_data: Dict[str, Any]) -> str:
        """Save check result data"""
        if not self._connected:
            await self.connect()

        check_data["created_at"] = datetime.utcnow()

        result = await self.db.check_results.insert_one(check_data)
        return str(result.inserted_id)

    async def get_workflow_content(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all content for a workflow"""
        if not self._connected:
            await self.connect()

        cursor = self.db.content.find({"workflow_id": workflow_id})
        content_list = []

        async for content in cursor:
            content["_id"] = str(content["_id"])
            content_list.append(content)

        return content_list

    async def get_workflow_check_results(
        self, workflow_id: str
    ) -> List[Dict[str, Any]]:
        """Get all check results for a workflow"""
        if not self._connected:
            await self.connect()

        cursor = self.db.check_results.find({"workflow_id": workflow_id})
        results_list = []

        async for result in cursor:
            result["_id"] = str(result["_id"])
            results_list.append(result)

        return results_list

    async def list_workflows(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List workflows with optional status filter"""
        if not self._connected:
            await self.connect()

        query = {}
        if status:
            query["status"] = status

        cursor = self.db.workflows.find(query).sort("created_at", -1).limit(limit)
        workflows = []

        async for workflow in cursor:
            workflow["_id"] = str(workflow["_id"])
            workflows.append(workflow)

        return workflows

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and all related data"""
        if not self._connected:
            await self.connect()

        # Delete workflow
        workflow_result = await self.db.workflows.delete_one(
            {"workflow_id": workflow_id}
        )

        # Delete related content
        content_result = await self.db.content.delete_many({"workflow_id": workflow_id})

        # Delete related check results
        check_result = await self.db.check_results.delete_many(
            {"workflow_id": workflow_id}
        )

        return (
            workflow_result.deleted_count > 0
            or content_result.deleted_count > 0
            or check_result.deleted_count > 0
        )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self._connected:
            await self.connect()

        total_workflows = await self.db.workflows.count_documents({})
        completed_workflows = await self.db.workflows.count_documents(
            {"status": "completed"}
        )
        failed_workflows = await self.db.workflows.count_documents({"status": "failed"})

        total_content = await self.db.content.count_documents({})
        total_checks = await self.db.check_results.count_documents({})

        avg_quality_score = await self.db.workflows.aggregate(
            [
                {
                    "$match": {
                        "status": "completed",
                        "stages.quality_checking.best_score": {"$exists": True},
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "avg_score": {"$avg": "$stages.quality_checking.best_score"},
                    }
                },
            ]
        ).to_list(length=1)

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
            "average_quality_score": (
                avg_quality_score[0]["avg_score"] if avg_quality_score else 0
            ),
        }


# Global MongoDB client instance
mongodb_client: Optional[MongoDBClient] = None


async def get_mongodb_client() -> MongoDBClient:
    """Get or create MongoDB client instance"""
    global mongodb_client

    if mongodb_client is None:
        from app.core.config import settings

        # Get MongoDB connection string from settings
        connection_string = getattr(settings, "mongodb_connection_string", None)
        if not connection_string:
            raise ValueError("MongoDB connection string not configured")

        database_name = getattr(
            settings, "mongodb_database_name", "multi_writer_system"
        )
        mongodb_client = MongoDBClient(connection_string, database_name)
        await mongodb_client.connect()

    return mongodb_client


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global mongodb_client

    if mongodb_client:
        await mongodb_client.disconnect()
        mongodb_client = None
