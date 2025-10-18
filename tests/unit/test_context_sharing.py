"""
Unit tests for context sharing system
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.agents.context_sharing import (
    ContextSharingSystem,
    SharedContext,
    ContextRelationship,
    ContextQuery,
    ContextSubscription,
    ContextType,
    ContextPriority,
)


@pytest.mark.unit
class TestContextSharingSystem:
    """Test context sharing system functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        with patch("app.core.agents.context_sharing.get_llm") as mock:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = AsyncMock(content="Mock response")
            mock.return_value = mock_llm_instance
            yield mock

    @pytest.fixture
    def context_system(self):
        """Create context sharing system for testing"""
        return ContextSharingSystem(max_context_items=100)

    @pytest.mark.asyncio
    async def test_share_insight(self, context_system):
        """Test sharing an insight"""
        agent_id = "test_agent"
        title = "Test Insight"
        content = "This is a test insight content"
        tags = ["test", "insight"]
        metadata = {"source": "test"}

        context_id = await context_system.share_insight(
            agent_id=agent_id,
            title=title,
            content=content,
            tags=tags,
            metadata=metadata,
        )

        assert isinstance(context_id, str)
        assert context_id in context_system.shared_context

        shared_context = context_system.shared_context[context_id]
        assert shared_context.agent_id == agent_id
        assert shared_context.title == title
        assert shared_context.content == content
        assert shared_context.context_type == ContextType.INSIGHT
        assert shared_context.priority == ContextPriority.MEDIUM
        assert tags.issubset(shared_context.tags)
        assert shared_context.metadata == metadata

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, context_system):
        """Test getting relevant context"""
        # Share some context first
        agent_id = "test_agent"
        context_id1 = await context_system.share_insight(
            agent_id=agent_id,
            title="Technical Insight",
            content="This is about AI technology",
            tags=["technical", "ai"],
            priority=ContextPriority.HIGH,
        )

        context_id2 = await context_system.share_insight(
            agent_id=agent_id,
            title="Creative Insight",
            content="This is about creative writing",
            tags=["creative", "writing"],
            priority=ContextPriority.LOW,
        )

        # Get relevant context for technical task
        task = "Write about AI technology"
        relevant_contexts = await context_system.get_relevant_context(
            agent_id="requesting_agent",
            task=task,
            context_types=[ContextType.INSIGHT],
            tags=["technical"],
            priority_threshold=ContextPriority.MEDIUM,
            max_results=5,
        )

        assert isinstance(relevant_contexts, list)
        assert len(relevant_contexts) >= 1

        # Should include technical insight
        technical_context_ids = [ctx.context_id for ctx in relevant_contexts]
        assert context_id1 in technical_context_ids

        # Should not include low priority creative context
        assert context_id2 not in technical_context_ids

    @pytest.mark.asyncio
    async def test_build_on_context(self, context_system, mock_llm):
        """Test building upon existing context"""
        # Share original context
        agent_id = "test_agent"
        original_context_id = await context_system.share_insight(
            agent_id=agent_id,
            title="Original Insight",
            content="Original content",
            tags=["original"],
        )

        # Build upon it
        new_content = "This builds upon the original insight"
        new_context_id = await context_system.build_on_context(
            agent_id=agent_id,
            source_context_id=original_context_id,
            new_content=new_content,
            relationship_type="builds_on",
        )

        assert new_context_id in context_system.shared_context

        # Check relationship was created
        original_context = context_system.shared_context[original_context_id]
        assert new_context_id in original_context.child_contexts

        new_context = context_system.shared_context[new_context_id]
        assert original_context_id in new_context.parent_contexts

        # Check relationship record
        relationships = [
            rel
            for rel in context_system.context_relationships
            if rel.source_context_id == new_context_id
            and rel.target_context_id == original_context_id
        ]
        assert len(relationships) == 1
        assert relationships[0].relationship_type == "builds_on"

    @pytest.mark.asyncio
    async def test_verify_context(self, context_system):
        """Test verifying context"""
        # Share context
        agent_id = "test_agent"
        context_id = await context_system.share_insight(
            agent_id=agent_id, title="Test Context", content="Test content"
        )

        # Verify it
        verification_threshold = 2
        context_system.verification_threshold = verification_threshold

        # First verification
        result1 = await context_system.verify_context(
            agent_id="verifier1", context_id=context_id, verification_result=True
        )
        assert result1 is False  # Not yet verified

        # Second verification
        result2 = await context_system.verify_context(
            agent_id="verifier2", context_id=context_id, verification_result=True
        )
        assert result2 is True  # Now verified

        # Check context is marked as verified
        context = context_system.shared_context[context_id]
        assert context.verified is True
        assert context.verification_count == 2

    @pytest.mark.asyncio
    async def test_subscribe_to_context(self, context_system):
        """Test subscribing to context updates"""
        agent_id = "test_agent"
        context_types = [ContextType.INSIGHT, ContextType.DISCOVERY]
        tags = ["technical", "ai"]

        subscription_id = await context_system.subscribe_to_context(
            agent_id=agent_id,
            context_types=context_types,
            tags=tags,
            priority_threshold=ContextPriority.MEDIUM,
        )

        assert isinstance(subscription_id, str)
        assert subscription_id in context_system.context_subscriptions

        subscription = context_system.context_subscriptions[subscription_id]
        assert subscription.agent_id == agent_id
        assert subscription.context_types == context_types
        assert subscription.tags == tags
        assert subscription.priority_threshold == ContextPriority.MEDIUM
        assert subscription.active is True

    @pytest.mark.asyncio
    async def test_get_context_graph(self, context_system):
        """Test getting context graph"""
        # Create related contexts
        agent_id = "test_agent"

        # Root context
        root_id = await context_system.share_insight(
            agent_id=agent_id, title="Root Context", content="Root content"
        )

        # Child contexts
        child1_id = await context_system.build_on_context(
            agent_id=agent_id, source_context_id=root_id, new_content="Child 1 content"
        )

        child2_id = await context_system.build_on_context(
            agent_id=agent_id, source_context_id=root_id, new_content="Child 2 content"
        )

        # Grandchild context
        grandchild_id = await context_system.build_on_context(
            agent_id=agent_id,
            source_context_id=child1_id,
            new_content="Grandchild content",
        )

        # Get graph
        graph = await context_system.get_context_graph(root_id, max_depth=3)

        assert "nodes" in graph
        assert "edges" in graph
        assert "starting_context" in graph
        assert graph["starting_context"] == root_id

        # Should include all contexts
        assert root_id in graph["nodes"]
        assert child1_id in graph["nodes"]
        assert child2_id in graph["nodes"]
        assert grandchild_id in graph["nodes"]

        # Should have relationships
        assert len(graph["edges"]) >= 3  # At least 3 relationships

    @pytest.mark.asyncio
    async def test_calculate_relevance(self, context_system):
        """Test calculating relevance score"""
        agent_id = "test_agent"
        task = "Write about AI technology and machine learning"

        # Create context
        context_id = await context_system.share_insight(
            agent_id=agent_id,
            title="AI Technology Insight",
            content="This is about AI technology and machine learning algorithms",
            priority=ContextPriority.HIGH,
        )

        context = context_system.shared_context[context_id]

        # Calculate relevance
        relevance = await context_system._calculate_relevance(context, task, agent_id)

        assert 0.0 <= relevance <= 1.0
        # Should be high due to content overlap
        assert relevance > 0.5

    @pytest.mark.asyncio
    async def test_calculate_text_similarity(self, context_system):
        """Test calculating text similarity"""
        text1 = "This is about AI technology and machine learning"
        text2 = "This discusses AI technology and deep learning models"
        text3 = "This is about creative writing and poetry"

        # High similarity
        similarity1 = context_system._calculate_text_similarity(text1, text2)
        assert 0.0 <= similarity1 <= 1.0
        assert similarity1 > 0.5

        # Low similarity
        similarity2 = context_system._calculate_text_similarity(text1, text3)
        assert 0.0 <= similarity2 <= 1.0
        assert similarity2 < 0.5

        # Identical text
        similarity3 = context_system._calculate_text_similarity(text1, text1)
        assert similarity3 == 1.0

    def test_update_usage_score(self, context_system):
        """Test updating usage score"""
        agent_id = "test_agent"
        context_id = "test_context"

        # Update usage score
        context_system._update_usage_score(context_id, agent_id)

        assert context_id in context_system.context_usage
        assert agent_id in context_system.context_usage[context_id]
        assert context_system.context_usage[context_id][agent_id] == 0.1

        # Update again
        context_system._update_usage_score(context_id, agent_id)
        assert context_system.context_usage[context_id][agent_id] == 0.2

        # Should not exceed 1.0
        for _ in range(20):
            context_system._update_usage_score(context_id, agent_id)
        assert context_system.context_usage[context_id][agent_id] == 1.0

    def test_generate_context_id(self, context_system):
        """Test generating context ID"""
        agent_id = "test_agent"
        title = "Test Title"
        content = "Test content"

        context_id1 = context_system._generate_context_id(agent_id, title, content)
        context_id2 = context_system._generate_context_id(agent_id, title, content)

        assert isinstance(context_id1, str)
        assert isinstance(context_id2, str)
        assert context_id1.startswith("ctx_")
        assert context_id2.startswith("ctx_")
        assert len(context_id1) == 20  # ctx_ + 16 char hash
        assert len(context_id2) == 20
        # Should be different due to timestamp
        assert context_id1 != context_id2

    @pytest.mark.asyncio
    async def test_cleanup_old_context(self, context_system):
        """Test cleaning up old context items"""
        # Set small max size for testing
        context_system.max_context_items = 5

        # Create more contexts than max
        agent_id = "test_agent"
        context_ids = []
        for i in range(10):
            context_id = await context_system.share_insight(
                agent_id=agent_id, title=f"Context {i}", content=f"Content {i}"
            )
            context_ids.append(context_id)

        # Should have cleaned up to max size
        assert len(context_system.shared_context) <= 5

        # Most recent contexts should be kept
        recent_contexts = list(context_system.shared_context.keys())
        assert context_ids[-1] in recent_contexts
        assert context_ids[-2] in recent_contexts

    def test_get_context_statistics(self, context_system):
        """Test getting context statistics"""
        # Create some context data
        agent_id = "test_agent"
        context_system.shared_context["ctx1"] = SharedContext(
            context_id="ctx1",
            agent_id=agent_id,
            context_type=ContextType.INSIGHT,
            priority=ContextPriority.HIGH,
            title="Test 1",
            content="Content 1",
            metadata={},
            tags={"test", "technical"},
            verified=True,
            verification_count=2,
        )

        context_system.shared_context["ctx2"] = SharedContext(
            context_id="ctx2",
            agent_id="other_agent",
            context_type=ContextType.DISCOVERY,
            priority=ContextPriority.MEDIUM,
            title="Test 2",
            content="Content 2",
            metadata={},
            tags={"test", "creative"},
            verified=False,
            verification_count=0,
        )

        # Create a relationship
        context_system.context_relationships.append(
            ContextRelationship(
                source_context_id="ctx1",
                target_context_id="ctx2",
                relationship_type="builds_on",
                strength=0.7,
            )
        )

        # Get statistics
        stats = context_system.get_context_statistics()

        assert "total_contexts" in stats
        assert "verified_contexts" in stats
        assert "verification_rate" in stats
        assert "context_types" in stats
        assert "priority_levels" in stats
        assert "agent_contributions" in stats
        assert "relationship_types" in stats
        assert "total_relationships" in stats
        assert "active_subscriptions" in stats
        assert "total_usage_events" in stats

        assert stats["total_contexts"] == 2
        assert stats["verified_contexts"] == 1
        assert stats["verification_rate"] == 0.5
        assert stats["context_types"]["insight"] == 1
        assert stats["context_types"]["discovery"] == 1
        assert stats["priority_levels"]["high"] == 1
        assert stats["priority_levels"]["medium"] == 1
        assert stats["agent_contributions"][agent_id] == 1
        assert stats["agent_contributions"]["other_agent"] == 1
        assert stats["total_relationships"] == 1
        assert stats["relationship_types"]["builds_on"] == 1

    def test_export_context_data(self, context_system):
        """Test exporting context data"""
        # Create some context data
        agent_id = "test_agent"
        context_system.shared_context["ctx1"] = SharedContext(
            context_id="ctx1",
            agent_id=agent_id,
            context_type=ContextType.INSIGHT,
            priority=ContextPriority.HIGH,
            title="Test 1",
            content="Content 1",
            metadata={},
            tags={"test"},
            verified=True,
            verification_count=2,
        )

        # Export data
        exported_data = context_system.export_context_data()

        assert "shared_context" in exported_data
        assert "context_relationships" in exported_data
        assert "context_subscriptions" in exported_data
        assert "context_usage" in exported_data
        assert "statistics" in exported_data

        assert "ctx1" in exported_data["shared_context"]
        assert exported_data["shared_context"]["ctx1"]["context_id"] == "ctx1"
        assert exported_data["shared_context"]["ctx1"]["agent_id"] == agent_id

    def test_import_context_data(self, context_system):
        """Test importing context data"""
        # Create test data
        import_data = {
            "shared_context": {
                "imported_ctx": {
                    "context_id": "imported_ctx",
                    "agent_id": "imported_agent",
                    "context_type": "insight",
                    "priority": "medium",
                    "title": "Imported Context",
                    "content": "Imported content",
                    "metadata": {},
                    "tags": ["imported"],
                    "timestamp": 1234567890,
                    "accessed_by": [],
                    "relevance_scores": {},
                    "parent_contexts": [],
                    "child_contexts": [],
                    "verified": False,
                    "verification_count": 0,
                }
            },
            "context_relationships": [],
            "context_subscriptions": {},
            "context_usage": {},
        }

        # Import data
        context_system.import_context_data(import_data)

        # Check data was imported
        assert "imported_ctx" in context_system.shared_context
        assert (
            context_system.shared_context["imported_ctx"].context_id == "imported_ctx"
        )
        assert (
            context_system.shared_context["imported_ctx"].agent_id == "imported_agent"
        )
        assert context_system.shared_context["imported_ctx"].title == "Imported Context"

        # Check indexes were updated
        assert "imported" in context_system.context_index
        assert "imported_ctx" in context_system.context_index["imported"]
        assert "imported_agent" in context_system.agent_contexts


@pytest.mark.unit
class TestSharedContext:
    """Test shared context data class"""

    def test_shared_context_creation(self):
        """Test shared context creation"""
        context = SharedContext(
            context_id="test_ctx",
            agent_id="test_agent",
            context_type=ContextType.INSIGHT,
            priority=ContextPriority.HIGH,
            title="Test Context",
            content="Test content",
            metadata={"source": "test"},
            tags={"test", "insight"},
            parent_contexts=["parent_ctx"],
            child_contexts=["child_ctx"],
        )

        assert context.context_id == "test_ctx"
        assert context.agent_id == "test_agent"
        assert context.context_type == ContextType.INSIGHT
        assert context.priority == ContextPriority.HIGH
        assert context.title == "Test Context"
        assert context.content == "Test content"
        assert context.metadata == {"source": "test"}
        assert context.tags == {"test", "insight"}
        assert context.parent_contexts == ["parent_ctx"]
        assert context.child_contexts == ["child_ctx"]
        assert context.timestamp > 0
        assert context.accessed_by == set()
        assert context.relevance_scores == {}
        assert context.verified is False
        assert context.verification_count == 0


@pytest.mark.unit
class TestContextRelationship:
    """Test context relationship data class"""

    def test_context_relationship_creation(self):
        """Test context relationship creation"""
        relationship = ContextRelationship(
            source_context_id="source_ctx",
            target_context_id="target_ctx",
            relationship_type="builds_on",
            strength=0.8,
        )

        assert relationship.source_context_id == "source_ctx"
        assert relationship.target_context_id == "target_ctx"
        assert relationship.relationship_type == "builds_on"
        assert relationship.strength == 0.8
        assert relationship.timestamp > 0


@pytest.mark.unit
class TestContextQuery:
    """Test context query data class"""

    def test_context_query_creation(self):
        """Test context query creation"""
        query = ContextQuery(
            query_id="test_query",
            agent_id="test_agent",
            query_text="Test query",
            context_types=[ContextType.INSIGHT],
            tags=["test"],
            priority_threshold=ContextPriority.MEDIUM,
            relevance_threshold=0.5,
            max_results=10,
        )

        assert query.query_id == "test_query"
        assert query.agent_id == "test_agent"
        assert query.query_text == "Test query"
        assert query.context_types == [ContextType.INSIGHT]
        assert query.tags == ["test"]
        assert query.priority_threshold == ContextPriority.MEDIUM
        assert query.relevance_threshold == 0.5
        assert query.max_results == 10
        assert query.timestamp > 0


@pytest.mark.unit
class TestContextSubscription:
    """Test context subscription data class"""

    def test_context_subscription_creation(self):
        """Test context subscription creation"""
        subscription = ContextSubscription(
            subscription_id="test_sub",
            agent_id="test_agent",
            context_types=[ContextType.INSIGHT],
            tags=["test"],
            priority_threshold=ContextPriority.MEDIUM,
        )

        assert subscription.subscription_id == "test_sub"
        assert subscription.agent_id == "test_agent"
        assert subscription.context_types == [ContextType.INSIGHT]
        assert subscription.tags == ["test"]
        assert subscription.priority_threshold == ContextPriority.MEDIUM
        assert subscription.active is True
        assert subscription.created_at > 0
