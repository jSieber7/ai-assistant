"""
Enhanced context sharing system for multi-writer/checker system.

This module implements a rich context sharing system that allows agents
to share insights, build on each other's discoveries, and maintain
shared understanding throughout the workflow.
"""

import logging
import time
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib

from app.core.config import get_llm

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be shared"""

    INSIGHT = "insight"
    DISCOVERY = "discovery"
    RECOMMENDATION = "recommendation"
    QUESTION = "question"
    CONCERN = "concern"
    RESOURCE = "resource"
    METHODOLOGY = "methodology"
    CONSTRAINT = "constraint"
    EXAMPLE = "example"


class ContextPriority(Enum):
    """Priority levels for shared context"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RelevanceScore(Enum):
    """Relevance scoring levels"""

    VERY_RELEVANT = 0.9
    RELEVANT = 0.7
    SOMEWHAT_RELEVANT = 0.5
    NOT_RELEVANT = 0.3


@dataclass
class SharedContext:
    """Represents a piece of shared context"""

    context_id: str
    agent_id: str
    context_type: ContextType
    priority: ContextPriority
    title: str
    content: str
    metadata: Dict[str, Any]
    tags: Set[str]
    timestamp: float = field(default_factory=time.time)
    accessed_by: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    parent_contexts: List[str] = field(default_factory=list)
    child_contexts: List[str] = field(default_factory=list)
    verified: bool = False
    verification_count: int = 0


@dataclass
class ContextRelationship:
    """Relationship between context items"""

    source_context_id: str
    target_context_id: str
    relationship_type: str  # "builds_on", "contradicts", "supports", "clarifies"
    strength: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContextQuery:
    """Query for searching shared context"""

    query_id: str
    agent_id: str
    query_text: str
    context_types: List[ContextType]
    tags: List[str]
    priority_threshold: ContextPriority
    relevance_threshold: float
    max_results: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContextSubscription:
    """Subscription to context updates"""

    subscription_id: str
    agent_id: str
    context_types: List[ContextType]
    tags: List[str]
    priority_threshold: ContextPriority
    active: bool = True
    created_at: float = field(default_factory=time.time)


class ContextSharingSystem:
    """Enhanced context sharing system for agent collaboration"""

    def __init__(
        self,
        max_context_items: int = 1000,
        relevance_decay_rate: float = 0.1,
        verification_threshold: int = 2,
        analysis_model: str = "claude-3.5-sonnet",
    ):
        self.max_context_items = max_context_items
        self.relevance_decay_rate = relevance_decay_rate
        self.verification_threshold = verification_threshold
        self.analysis_model = analysis_model
        self.llm = None  # Will be initialized when needed

        # Context storage
        self.shared_context: Dict[str, SharedContext] = {}
        self.context_relationships: List[ContextRelationship] = []
        self.context_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> context_ids
        self.agent_contexts: Dict[str, Set[str]] = defaultdict(
            set
        )  # agent_id -> context_ids

        # Subscriptions
        self.context_subscriptions: Dict[str, ContextSubscription] = {}

        # Usage tracking
        self.context_usage: Dict[str, Dict[str, float]] = defaultdict(
            dict
        )  # context_id -> agent_id -> usage_score
        self.agent_insights: Dict[str, List[SharedContext]] = defaultdict(list)

    async def _get_llm(self):
        """Initialize LLM if not already done"""
        if self.llm is None:
            self.llm = await get_llm(self.analysis_model)
        return self.llm

    async def share_insight(
        self,
        agent_id: str,
        title: str,
        content: str,
        context_type: ContextType = ContextType.INSIGHT,
        priority: ContextPriority = ContextPriority.MEDIUM,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        related_contexts: List[str] = None,
    ) -> str:
        """
        Share an insight with other agents.

        Args:
            agent_id: ID of the sharing agent
            title: Title of the insight
            content: Content of the insight
            context_type: Type of context
            priority: Priority level
            tags: Tags for categorization
            metadata: Additional metadata
            related_contexts: IDs of related context items

        Returns:
            ID of the created context item
        """
        context_id = self._generate_context_id(agent_id, title, content)

        # Create shared context
        shared_context = SharedContext(
            context_id=context_id,
            agent_id=agent_id,
            context_type=context_type,
            priority=priority,
            title=title,
            content=content,
            metadata=metadata or {},
            tags=set(tags or []),
            parent_contexts=related_contexts or [],
        )

        # Store context
        self.shared_context[context_id] = shared_context

        # Update indexes
        for tag in shared_context.tags:
            self.context_index[tag].add(context_id)

        self.agent_contexts[agent_id].add(context_id)

        # Create relationships with related contexts
        if related_contexts:
            for related_id in related_contexts:
                if related_id in self.shared_context:
                    relationship = ContextRelationship(
                        source_context_id=context_id,
                        target_context_id=related_id,
                        relationship_type="builds_on",
                        strength=0.7,
                    )
                    self.context_relationships.append(relationship)

        # Update parent contexts
        for related_id in related_contexts:
            if related_id in self.shared_context:
                self.shared_context[related_id].child_contexts.append(context_id)

        # Notify subscribers
        await self._notify_subscribers(shared_context)

        # Clean up if needed
        await self._cleanup_old_context()

        logger.info(f"Agent {agent_id} shared insight: {title} (ID: {context_id})")
        return context_id

    async def get_relevant_context(
        self,
        agent_id: str,
        task: str,
        context_types: List[ContextType] = None,
        tags: List[str] = None,
        priority_threshold: ContextPriority = ContextPriority.MEDIUM,
        max_results: int = 10,
    ) -> List[SharedContext]:
        """
        Get context relevant to a specific task.

        Args:
            agent_id: ID of the requesting agent
            task: Task description
            context_types: Types of context to include
            tags: Tags to filter by
            priority_threshold: Minimum priority level
            max_results: Maximum number of results

        Returns:
            List of relevant context items
        """
        # Start with all context items
        candidate_contexts = list(self.shared_context.values())

        # Filter by context type
        if context_types:
            candidate_contexts = [
                ctx for ctx in candidate_contexts if ctx.context_type in context_types
            ]

        # Filter by tags
        if tags:
            candidate_contexts = [
                ctx
                for ctx in candidate_contexts
                if any(tag in ctx.tags for tag in tags)
            ]

        # Filter by priority
        priority_order = {
            ContextPriority.CRITICAL: 4,
            ContextPriority.HIGH: 3,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 1,
        }

        min_priority_value = priority_order.get(priority_threshold, 2)
        candidate_contexts = [
            ctx
            for ctx in candidate_contexts
            if priority_order.get(ctx.priority, 2) >= min_priority_value
        ]

        # Calculate relevance scores
        for context in candidate_contexts:
            relevance = await self._calculate_relevance(context, task, agent_id)
            context.relevance_scores[agent_id] = relevance

        # Sort by relevance and priority
        candidate_contexts.sort(
            key=lambda ctx: (
                ctx.relevance_scores.get(agent_id, 0.5),
                priority_order.get(ctx.priority, 2),
            ),
            reverse=True,
        )

        # Limit results
        relevant_contexts = candidate_contexts[:max_results]

        # Update access tracking
        for context in relevant_contexts:
            context.accessed_by.add(agent_id)
            self._update_usage_score(context.context_id, agent_id)

        logger.info(
            f"Returned {len(relevant_contexts)} relevant contexts for agent {agent_id}"
        )
        return relevant_contexts

    async def build_on_context(
        self,
        agent_id: str,
        source_context_id: str,
        new_content: str,
        relationship_type: str = "builds_on",
    ) -> str:
        """
        Build upon existing context.

        Args:
            agent_id: ID of the building agent
            source_context_id: ID of the context to build upon
            new_content: New content to add
            relationship_type: Type of relationship

        Returns:
            ID of the new context item
        """
        if source_context_id not in self.shared_context:
            raise ValueError(f"Source context {source_context_id} not found")

        source_context = self.shared_context[source_context_id]

        # Create title based on relationship
        if relationship_type == "builds_on":
            title = f"Building on: {source_context.title}"
        elif relationship_type == "contradicts":
            title = f"Contradicting: {source_context.title}"
        elif relationship_type == "supports":
            title = f"Supporting: {source_context.title}"
        else:
            title = f"Related to: {source_context.title}"

        # Create new context
        new_context_id = await self.share_insight(
            agent_id=agent_id,
            title=title,
            content=new_content,
            context_type=ContextType.INSIGHT,
            priority=source_context.priority,
            tags=list(source_context.tags),
            metadata={"builds_on": source_context_id},
            related_contexts=[source_context_id],
        )

        # Create relationship
        relationship = ContextRelationship(
            source_context_id=new_context_id,
            target_context_id=source_context_id,
            relationship_type=relationship_type,
            strength=0.8,
        )
        self.context_relationships.append(relationship)

        logger.info(
            f"Agent {agent_id} built upon context {source_context_id} with {new_context_id}"
        )
        return new_context_id

    async def verify_context(
        self,
        agent_id: str,
        context_id: str,
        verification_result: bool,
        confidence: float = 1.0,
    ) -> bool:
        """
        Verify a piece of shared context.

        Args:
            agent_id: ID of the verifying agent
            context_id: ID of the context to verify
            verification_result: Whether the context is verified
            confidence: Confidence in the verification

        Returns:
            Whether the context is now considered verified
        """
        if context_id not in self.shared_context:
            return False

        context = self.shared_context[context_id]

        # Update verification count
        if verification_result:
            context.verification_count += 1

        # Check if verification threshold is met
        if context.verification_count >= self.verification_threshold:
            context.verified = True
            logger.info(f"Context {context_id} verified by {agent_id}")
            return True

        return False

    async def subscribe_to_context(
        self,
        agent_id: str,
        context_types: List[ContextType],
        tags: List[str] = None,
        priority_threshold: ContextPriority = ContextPriority.MEDIUM,
    ) -> str:
        """
        Subscribe to context updates.

        Args:
            agent_id: ID of the subscribing agent
            context_types: Types of context to subscribe to
            tags: Tags to filter by
            priority_threshold: Minimum priority level

        Returns:
            ID of the subscription
        """
        subscription_id = f"sub_{agent_id}_{int(time.time())}"

        subscription = ContextSubscription(
            subscription_id=subscription_id,
            agent_id=agent_id,
            context_types=context_types,
            tags=tags or [],
            priority_threshold=priority_threshold,
        )

        self.context_subscriptions[subscription_id] = subscription

        logger.info(
            f"Agent {agent_id} subscribed to context updates (ID: {subscription_id})"
        )
        return subscription_id

    async def get_context_graph(
        self, context_id: str, max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get a graph of related context items.

        Args:
            context_id: Starting context ID
            max_depth: Maximum depth of relationships to follow

        Returns:
            Graph representation of related contexts
        """
        if context_id not in self.shared_context:
            return {"error": "Context not found"}

        # Build graph using BFS
        graph = {"nodes": {}, "edges": [], "starting_context": context_id}

        visited = set()
        queue = [(context_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Add node
            if current_id in self.shared_context:
                context = self.shared_context[current_id]
                graph["nodes"][current_id] = {
                    "id": current_id,
                    "title": context.title,
                    "type": context.context_type.value,
                    "priority": context.priority.value,
                    "verified": context.verified,
                    "agent_id": context.agent_id,
                    "depth": depth,
                }

            # Find related contexts
            related_relationships = [
                rel
                for rel in self.context_relationships
                if rel.source_context_id == current_id
                or rel.target_context_id == current_id
            ]

            for relationship in related_relationships:
                # Determine related context ID
                if relationship.source_context_id == current_id:
                    related_id = relationship.target_context_id
                    direction = "outgoing"
                else:
                    related_id = relationship.source_context_id
                    direction = "incoming"

                # Add edge
                graph["edges"].append(
                    {
                        "from": current_id,
                        "to": related_id,
                        "type": relationship.relationship_type,
                        "strength": relationship.strength,
                        "direction": direction,
                    }
                )

                # Add to queue if not visited
                if related_id not in visited:
                    queue.append((related_id, depth + 1))

        return graph

    async def _calculate_relevance(
        self, context: SharedContext, task: str, agent_id: str
    ) -> float:
        """Calculate relevance score of context to task"""
        # Base relevance from content similarity
        content_similarity = await self._calculate_text_similarity(
            context.content, task
        )

        # Boost relevance if agent created the context
        if context.agent_id == agent_id:
            content_similarity += 0.2

        # Boost relevance if context is verified
        if context.verified:
            content_similarity += 0.1

        # Boost relevance based on priority
        priority_boost = {
            ContextPriority.CRITICAL: 0.2,
            ContextPriority.HIGH: 0.1,
            ContextPriority.MEDIUM: 0.0,
            ContextPriority.LOW: -0.1,
        }
        content_similarity += priority_boost.get(context.priority, 0.0)

        # Apply relevance decay based on age
        age = time.time() - context.timestamp
        age_factor = max(
            0.0, 1.0 - (age / (7 * 24 * 3600)) * self.relevance_decay_rate
        )  # Decay over 7 days
        content_similarity *= age_factor

        return min(1.0, max(0.0, content_similarity))

    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _update_usage_score(self, context_id: str, agent_id: str):
        """Update usage score for context"""
        if context_id not in self.context_usage:
            self.context_usage[context_id] = {}

        current_score = self.context_usage[context_id].get(agent_id, 0.0)
        self.context_usage[context_id][agent_id] = min(1.0, current_score + 0.1)

    def _generate_context_id(self, agent_id: str, title: str, content: str) -> str:
        """Generate unique context ID"""
        # Use SHA-256 instead of MD5 for better security
        content_hash = hashlib.sha256(
            f"{agent_id}{title}{content}{time.time()}".encode()
        ).hexdigest()
        return f"ctx_{content_hash[:16]}"

    async def _notify_subscribers(self, context: SharedContext):
        """Notify agents subscribed to this type of context"""
        for subscription in self.context_subscriptions.values():
            if not subscription.active:
                continue

            # Check if subscription matches
            if (
                context.context_type in subscription.context_types
                and context.priority.value >= subscription.priority_threshold.value
                and (
                    not subscription.tags
                    or any(tag in context.tags for tag in subscription.tags)
                )
            ):
                # In a real implementation, this would send a notification to the agent
                logger.debug(
                    f"Notified agent {subscription.agent_id} about context {context.context_id}"
                )

    async def _cleanup_old_context(self):
        """Clean up old context items if storage is full"""
        if len(self.shared_context) <= self.max_context_items:
            return

        # Sort by age and usage
        context_scores = []
        for context_id, context in self.shared_context.items():
            age = time.time() - context.timestamp
            usage_score = sum(self.context_usage.get(context_id, {}).values())

            # Lower score is more likely to be removed
            score = age - (usage_score * 1000)  # Weight usage heavily
            context_scores.append((context_id, score))

        # Sort by score (ascending = oldest/least used)
        context_scores.sort(key=lambda x: x[1])

        # Remove oldest/least used contexts
        to_remove = (
            len(self.shared_context) - self.max_context_items + 10
        )  # Remove extra to avoid frequent cleanup
        for context_id, _ in context_scores[:to_remove]:
            await self._remove_context(context_id)

        logger.info(f"Cleaned up {to_remove} old context items")

    async def _remove_context(self, context_id: str):
        """Remove a context item and its relationships"""
        if context_id not in self.shared_context:
            return

        context = self.shared_context[context_id]

        # Remove from indexes
        for tag in context.tags:
            if tag in self.context_index and context_id in self.context_index[tag]:
                self.context_index[tag].remove(context_id)
                if not self.context_index[tag]:
                    del self.context_index[tag]

        # Remove from agent contexts
        if context.agent_id in self.agent_contexts:
            self.agent_contexts[context.agent_id].discard(context_id)

        # Remove relationships
        self.context_relationships = [
            rel
            for rel in self.context_relationships
            if rel.source_context_id != context_id
            and rel.target_context_id != context_id
        ]

        # Remove from parent contexts
        for parent_id in context.parent_contexts:
            if parent_id in self.shared_context:
                self.shared_context[parent_id].child_contexts.remove(context_id)

        # Remove from child contexts
        for child_id in context.child_contexts:
            if child_id in self.shared_context:
                self.shared_context[child_id].parent_contexts.remove(context_id)

        # Remove from usage tracking
        if context_id in self.context_usage:
            del self.context_usage[context_id]

        # Remove context
        del self.shared_context[context_id]

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about shared context"""
        total_contexts = len(self.shared_context)
        verified_contexts = sum(
            1 for ctx in self.shared_context.values() if ctx.verified
        )

        # Count by type
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)

        for context in self.shared_context.values():
            type_counts[context.context_type.value] += 1
            priority_counts[context.priority.value] += 1

        # Count by agent
        agent_counts = defaultdict(int)
        for context in self.shared_context.values():
            agent_counts[context.agent_id] += 1

        # Count relationships
        relationship_counts = defaultdict(int)
        for rel in self.context_relationships:
            relationship_counts[rel.relationship_type] += 1

        return {
            "total_contexts": total_contexts,
            "verified_contexts": verified_contexts,
            "verification_rate": verified_contexts / max(1, total_contexts),
            "context_types": dict(type_counts),
            "priority_levels": dict(priority_counts),
            "agent_contributions": dict(agent_counts),
            "relationship_types": dict(relationship_counts),
            "total_relationships": len(self.context_relationships),
            "active_subscriptions": len(
                [s for s in self.context_subscriptions.values() if s.active]
            ),
            "total_usage_events": sum(
                len(usage) for usage in self.context_usage.values()
            ),
        }

    def export_context_data(self) -> Dict[str, Any]:
        """Export all context data for backup or analysis"""
        return {
            "shared_context": {
                context_id: {
                    "context_id": ctx.context_id,
                    "agent_id": ctx.agent_id,
                    "context_type": ctx.context_type.value,
                    "priority": ctx.priority.value,
                    "title": ctx.title,
                    "content": ctx.content,
                    "metadata": ctx.metadata,
                    "tags": list(ctx.tags),
                    "timestamp": ctx.timestamp,
                    "accessed_by": list(ctx.accessed_by),
                    "relevance_scores": ctx.relevance_scores,
                    "parent_contexts": ctx.parent_contexts,
                    "child_contexts": ctx.child_contexts,
                    "verified": ctx.verified,
                    "verification_count": ctx.verification_count,
                }
                for context_id, ctx in self.shared_context.items()
            },
            "context_relationships": [
                {
                    "source_context_id": rel.source_context_id,
                    "target_context_id": rel.target_context_id,
                    "relationship_type": rel.relationship_type,
                    "strength": rel.strength,
                    "timestamp": rel.timestamp,
                }
                for rel in self.context_relationships
            ],
            "context_subscriptions": {
                sub_id: {
                    "subscription_id": sub.subscription_id,
                    "agent_id": sub.agent_id,
                    "context_types": [ct.value for ct in sub.context_types],
                    "tags": sub.tags,
                    "priority_threshold": sub.priority_threshold.value,
                    "active": sub.active,
                    "created_at": sub.created_at,
                }
                for sub_id, sub in self.context_subscriptions.items()
            },
            "context_usage": dict(self.context_usage),
            "statistics": self.get_context_statistics(),
        }

    def import_context_data(self, data: Dict[str, Any]):
        """Import context data from backup"""
        # Import shared context
        if "shared_context" in data:
            for context_id, ctx_data in data["shared_context"].items():
                self.shared_context[context_id] = SharedContext(
                    context_id=ctx_data["context_id"],
                    agent_id=ctx_data["agent_id"],
                    context_type=ContextType(ctx_data["context_type"]),
                    priority=ContextPriority(ctx_data["priority"]),
                    title=ctx_data["title"],
                    content=ctx_data["content"],
                    metadata=ctx_data["metadata"],
                    tags=set(ctx_data["tags"]),
                    timestamp=ctx_data["timestamp"],
                    accessed_by=set(ctx_data["accessed_by"]),
                    relevance_scores=ctx_data["relevance_scores"],
                    parent_contexts=ctx_data["parent_contexts"],
                    child_contexts=ctx_data["child_contexts"],
                    verified=ctx_data["verified"],
                    verification_count=ctx_data["verification_count"],
                )

                # Update indexes
                for tag in self.shared_context[context_id].tags:
                    self.context_index[tag].add(context_id)

                self.agent_contexts[ctx_data["agent_id"]].add(context_id)

        # Import relationships
        if "context_relationships" in data:
            for rel_data in data["context_relationships"]:
                relationship = ContextRelationship(
                    source_context_id=rel_data["source_context_id"],
                    target_context_id=rel_data["target_context_id"],
                    relationship_type=rel_data["relationship_type"],
                    strength=rel_data["strength"],
                    timestamp=rel_data["timestamp"],
                )
                self.context_relationships.append(relationship)

        # Import subscriptions
        if "context_subscriptions" in data:
            for sub_id, sub_data in data["context_subscriptions"].items():
                subscription = ContextSubscription(
                    subscription_id=sub_data["subscription_id"],
                    agent_id=sub_data["agent_id"],
                    context_types=[ContextType(ct) for ct in sub_data["context_types"]],
                    tags=sub_data["tags"],
                    priority_threshold=ContextPriority(sub_data["priority_threshold"]),
                    active=sub_data["active"],
                    created_at=sub_data["created_at"],
                )
                self.context_subscriptions[sub_id] = subscription

        # Import usage data
        if "context_usage" in data:
            self.context_usage.update(data["context_usage"])

        logger.info("Context data imported successfully")
