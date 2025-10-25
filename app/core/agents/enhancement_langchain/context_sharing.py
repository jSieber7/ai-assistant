"""
LangGraph-based context sharing system for multi-agent collaboration.

This module implements a context sharing system using LangGraph workflows
that allows agents to share insights, build on each other's discoveries,
and maintain shared understanding throughout the workflow.
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from langgraph.graph import StateGraph, END
try:
    from langgraph.graph import CompiledGraph
except ImportError:
    CompiledGraph = None
from pydantic import BaseModel

# Legacy integration layer removed - direct LLM provider access

logger = logging.getLogger(__name__)


class ContextType(str, Enum):
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


class ContextPriority(str, Enum):
    """Priority levels for shared context"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RelationshipType(str, Enum):
    """Types of relationships between contexts"""
    BUILDS_ON = "builds_on"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    CLARIFIES = "clarifies"


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
    relationship_type: RelationshipType
    strength: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)


class ContextSharingState(BaseModel):
    """State for context sharing workflow"""
    # Input parameters
    agent_id: str
    action: str  # "share", "retrieve", "build_on", "verify", "subscribe"
    task_description: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    
    # Context sharing data
    shared_context: Dict[str, SharedContext] = field(default_factory=dict)
    context_relationships: List[ContextRelationship] = field(default_factory=list)
    context_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    agent_contexts: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Query results
    relevant_contexts: List[SharedContext] = field(default_factory=list)
    context_graph: Dict[str, Any] = field(default_factory=dict)
    
    # Subscriptions
    subscriptions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Results
    result: Optional[str] = None
    context_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class LangGraphContextSharingSystem:
    """LangGraph-based context sharing system for agent collaboration"""
    
    def __init__(
        self,
        max_context_items: int = 1000,
        relevance_decay_rate: float = 0.1,
        verification_threshold: int = 2,
    ):
        self.max_context_items = max_context_items
        self.relevance_decay_rate = relevance_decay_rate
        self.verification_threshold = verification_threshold
        
        # Initialize the workflow
        self.workflow = self._create_workflow()
        self.compiled_workflow = None
        
        # Storage for context data
        self.shared_context: Dict[str, SharedContext] = {}
        self.context_relationships: List[ContextRelationship] = []
        self.context_index: Dict[str, Set[str]] = defaultdict(set)
        self.agent_contexts: Dict[str, Set[str]] = defaultdict(set)
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # Usage tracking
        self.context_usage: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("Initialized LangGraph Context Sharing System")
    
    def _create_workflow(self) -> StateGraph:
        """Create the context sharing workflow"""
        workflow = StateGraph(ContextSharingState)
        
        # Add nodes
        workflow.add_node("route_action", self._route_action)
        workflow.add_node("share_context", self._share_context)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("build_on_context", self._build_on_context)
        workflow.add_node("verify_context", self._verify_context)
        workflow.add_node("subscribe_to_context", self._subscribe_to_context)
        workflow.add_node("get_context_graph", self._get_context_graph)
        workflow.add_node("cleanup_old_context", self._cleanup_old_context)
        
        # Set entry point
        workflow.set_entry_point("route_action")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "route_action",
            self._determine_action,
            {
                "share": "share_context",
                "retrieve": "retrieve_context",
                "build_on": "build_on_context",
                "verify": "verify_context",
                "subscribe": "subscribe_to_context",
                "graph": "get_context_graph",
                "error": END
            }
        )
        
        # Add edges to cleanup and end
        workflow.add_edge("share_context", "cleanup_old_context")
        workflow.add_edge("build_on_context", "cleanup_old_context")
        workflow.add_edge("retrieve_context", END)
        workflow.add_edge("verify_context", END)
        workflow.add_edge("subscribe_to_context", END)
        workflow.add_edge("get_context_graph", END)
        workflow.add_edge("cleanup_old_context", END)
        
        return workflow
    
    async def _route_action(self, state: ContextSharingState) -> ContextSharingState:
        """Route the action based on the input"""
        try:
            action = state.action.lower()
            
            if action in ["share", "add", "create"]:
                state.action = "share"
            elif action in ["retrieve", "get", "find", "search"]:
                state.action = "retrieve"
            elif action in ["build_on", "build", "extend", "respond"]:
                state.action = "build_on"
            elif action in ["verify", "validate", "confirm"]:
                state.action = "verify"
            elif action in ["subscribe", "follow", "watch"]:
                state.action = "subscribe"
            elif action in ["graph", "network", "relationships"]:
                state.action = "graph"
            else:
                state.action = "error"
                state.error = f"Unknown action: {action}"
            
            # Copy current state to workflow state
            state.shared_context = self.shared_context.copy()
            state.context_relationships = self.context_relationships.copy()
            state.context_index = dict(self.context_index)
            state.agent_contexts = dict(self.agent_contexts)
            state.subscriptions = self.subscriptions.copy()
            
        except Exception as e:
            logger.error(f"Error routing action: {str(e)}")
            state.action = "error"
            state.error = str(e)
        
        return state
    
    def _determine_action(self, state: ContextSharingState) -> str:
        """Determine which action to take"""
        return state.action
    
    async def _share_context(self, state: ContextSharingState) -> ContextSharingState:
        """Share context with other agents"""
        try:
            if not state.context_data:
                state.error = "No context data provided for sharing"
                state.success = False
                return state
            
            # Extract context data
            title = state.context_data.get("title", "Untitled Context")
            content = state.context_data.get("content", "")
            context_type = ContextType(state.context_data.get("context_type", "insight"))
            priority = ContextPriority(state.context_data.get("priority", "medium"))
            tags = set(state.context_data.get("tags", []))
            metadata = state.context_data.get("metadata", {})
            related_contexts = state.context_data.get("related_contexts", [])
            
            # Generate context ID
            context_id = self._generate_context_id(state.agent_id, title, content)
            
            # Create shared context
            shared_context = SharedContext(
                context_id=context_id,
                agent_id=state.agent_id,
                context_type=context_type,
                priority=priority,
                title=title,
                content=content,
                metadata=metadata,
                tags=tags,
                parent_contexts=related_contexts,
            )
            
            # Store context
            state.shared_context[context_id] = shared_context
            
            # Update indexes
            for tag in shared_context.tags:
                state.context_index[tag].add(context_id)
            
            state.agent_contexts[state.agent_id].add(context_id)
            
            # Create relationships with related contexts
            if related_contexts:
                for related_id in related_contexts:
                    if related_id in state.shared_context:
                        relationship = ContextRelationship(
                            source_context_id=context_id,
                            target_context_id=related_id,
                            relationship_type=RelationshipType.BUILDS_ON,
                            strength=0.7,
                        )
                        state.context_relationships.append(relationship)
            
            # Update parent contexts
            for related_id in related_contexts:
                if related_id in state.shared_context:
                    state.shared_context[related_id].child_contexts.append(context_id)
            
            # Notify subscribers (simplified for LangGraph)
            await self._notify_subscribers(shared_context, state)
            
            state.context_id = context_id
            state.result = f"Successfully shared context: {title}"
            state.success = True
            
            logger.info(f"Agent {state.agent_id} shared context: {title} (ID: {context_id})")
            
        except Exception as e:
            logger.error(f"Error sharing context: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _retrieve_context(self, state: ContextSharingState) -> ContextSharingState:
        """Retrieve relevant context for a task"""
        try:
            if not state.task_description:
                state.error = "No task description provided for context retrieval"
                state.success = False
                return state
            
            # Extract query parameters
            context_types = state.context_data.get("context_types", []) if state.context_data else []
            tags = state.context_data.get("tags", []) if state.context_data else []
            priority_threshold = ContextPriority(
                state.context_data.get("priority_threshold", "medium") if state.context_data else "medium"
            )
            max_results = state.context_data.get("max_results", 10) if state.context_data else 10
            
            # Convert context types to enum
            if context_types:
                context_types = [ContextType(ct) for ct in context_types]
            
            # Start with all context items
            candidate_contexts = list(state.shared_context.values())
            
            # Filter by context type
            if context_types:
                candidate_contexts = [
                    ctx for ctx in candidate_contexts if ctx.context_type in context_types
                ]
            
            # Filter by tags
            if tags:
                candidate_contexts = [
                    ctx for ctx in candidate_contexts
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
                relevance = await self._calculate_relevance(
                    context, state.task_description, state.agent_id
                )
                context.relevance_scores[state.agent_id] = relevance
            
            # Sort by relevance and priority
            candidate_contexts.sort(
                key=lambda ctx: (
                    ctx.relevance_scores.get(state.agent_id, 0.5),
                    priority_order.get(ctx.priority, 2),
                ),
                reverse=True,
            )
            
            # Limit results
            relevant_contexts = candidate_contexts[:max_results]
            
            # Update access tracking
            for context in relevant_contexts:
                context.accessed_by.add(state.agent_id)
                self._update_usage_score(context.context_id, state.agent_id)
            
            state.relevant_contexts = relevant_contexts
            state.result = f"Retrieved {len(relevant_contexts)} relevant contexts"
            state.success = True
            
            logger.info(f"Retrieved {len(relevant_contexts)} contexts for agent {state.agent_id}")
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _build_on_context(self, state: ContextSharingState) -> ContextSharingState:
        """Build upon existing context"""
        try:
            if not state.context_data:
                state.error = "No context data provided for building on"
                state.success = False
                return state
            
            source_context_id = state.context_data.get("source_context_id")
            new_content = state.context_data.get("new_content", "")
            relationship_type = RelationshipType(
                state.context_data.get("relationship_type", "builds_on")
            )
            
            if not source_context_id:
                state.error = "No source context ID provided"
                state.success = False
                return state
            
            if source_context_id not in state.shared_context:
                state.error = f"Source context {source_context_id} not found"
                state.success = False
                return state
            
            source_context = state.shared_context[source_context_id]
            
            # Create title based on relationship
            if relationship_type == RelationshipType.BUILDS_ON:
                title = f"Building on: {source_context.title}"
            elif relationship_type == RelationshipType.CONTRADICTS:
                title = f"Contradicting: {source_context.title}"
            elif relationship_type == RelationshipType.SUPPORTS:
                title = f"Supporting: {source_context.title}"
            else:
                title = f"Related to: {source_context.title}"
            
            # Generate new context ID
            new_context_id = self._generate_context_id(state.agent_id, title, new_content)
            
            # Create new context
            new_context = SharedContext(
                context_id=new_context_id,
                agent_id=state.agent_id,
                context_type=ContextType.INSIGHT,
                priority=source_context.priority,
                title=title,
                content=new_content,
                metadata={"builds_on": source_context_id},
                tags=list(source_context.tags),
                parent_contexts=[source_context_id],
            )
            
            # Store context
            state.shared_context[new_context_id] = new_context
            
            # Update indexes
            for tag in new_context.tags:
                state.context_index[tag].add(new_context_id)
            
            state.agent_contexts[state.agent_id].add(new_context_id)
            
            # Create relationship
            relationship = ContextRelationship(
                source_context_id=new_context_id,
                target_context_id=source_context_id,
                relationship_type=relationship_type,
                strength=0.8,
            )
            state.context_relationships.append(relationship)
            
            state.context_id = new_context_id
            state.result = f"Successfully built upon context: {source_context.title}"
            state.success = True
            
            logger.info(f"Agent {state.agent_id} built upon context {source_context_id}")
            
        except Exception as e:
            logger.error(f"Error building on context: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _verify_context(self, state: ContextSharingState) -> ContextSharingState:
        """Verify a piece of shared context"""
        try:
            if not state.context_data:
                state.error = "No context data provided for verification"
                state.success = False
                return state
            
            context_id = state.context_data.get("context_id")
            verification_result = state.context_data.get("verification_result", True)
            confidence = state.context_data.get("confidence", 1.0)
            
            if not context_id:
                state.error = "No context ID provided for verification"
                state.success = False
                return state
            
            if context_id not in state.shared_context:
                state.error = f"Context {context_id} not found"
                state.success = False
                return state
            
            context = state.shared_context[context_id]
            
            # Update verification count
            if verification_result:
                context.verification_count += 1
            
            # Check if verification threshold is met
            verified = context.verification_count >= self.verification_threshold
            if verified:
                context.verified = True
            
            state.result = f"Context {context_id} verification updated. Verified: {verified}"
            state.success = True
            
            logger.info(f"Agent {state.agent_id} verified context {context_id}")
            
        except Exception as e:
            logger.error(f"Error verifying context: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _subscribe_to_context(self, state: ContextSharingState) -> ContextSharingState:
        """Subscribe to context updates"""
        try:
            if not state.context_data:
                state.error = "No context data provided for subscription"
                state.success = False
                return state
            
            context_types = state.context_data.get("context_types", [])
            tags = state.context_data.get("tags", [])
            priority_threshold = ContextPriority(
                state.context_data.get("priority_threshold", "medium")
            )
            
            # Convert context types to enum
            if context_types:
                context_types = [ContextType(ct) for ct in context_types]
            
            # Generate subscription ID
            subscription_id = f"sub_{state.agent_id}_{int(time.time())}"
            
            # Create subscription
            subscription = {
                "subscription_id": subscription_id,
                "agent_id": state.agent_id,
                "context_types": context_types,
                "tags": tags,
                "priority_threshold": priority_threshold,
                "active": True,
                "created_at": time.time(),
            }
            
            state.subscriptions[subscription_id] = subscription
            
            state.result = f"Successfully created subscription: {subscription_id}"
            state.success = True
            
            logger.info(f"Agent {state.agent_id} subscribed to context updates")
            
        except Exception as e:
            logger.error(f"Error subscribing to context: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _get_context_graph(self, state: ContextSharingState) -> ContextSharingState:
        """Get a graph of related context items"""
        try:
            if not state.context_data:
                state.error = "No context data provided for graph retrieval"
                state.success = False
                return state
            
            context_id = state.context_data.get("context_id")
            max_depth = state.context_data.get("max_depth", 3)
            
            if not context_id:
                state.error = "No context ID provided for graph retrieval"
                state.success = False
                return state
            
            if context_id not in state.shared_context:
                state.error = f"Context {context_id} not found"
                state.success = False
                return state
            
            # Build graph using BFS
            graph = {
                "nodes": {},
                "edges": [],
                "starting_context": context_id
            }
            
            visited = set()
            queue = [(context_id, 0)]
            
            while queue:
                current_id, depth = queue.pop(0)
                
                if current_id in visited or depth > max_depth:
                    continue
                
                visited.add(current_id)
                
                # Add node
                if current_id in state.shared_context:
                    context = state.shared_context[current_id]
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
                    for rel in state.context_relationships
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
                    graph["edges"].append({
                        "from": current_id,
                        "to": related_id,
                        "type": relationship.relationship_type.value,
                        "strength": relationship.strength,
                        "direction": direction,
                    })
                    
                    # Add to queue if not visited
                    if related_id not in visited:
                        queue.append((related_id, depth + 1))
            
            state.context_graph = graph
            state.result = f"Successfully generated context graph for {context_id}"
            state.success = True
            
        except Exception as e:
            logger.error(f"Error getting context graph: {str(e)}")
            state.error = str(e)
            state.success = False
        
        return state
    
    async def _cleanup_old_context(self, state: ContextSharingState) -> ContextSharingState:
        """Clean up old context items if storage is full"""
        try:
            if len(state.shared_context) <= self.max_context_items:
                return state
            
            # Sort by age and usage
            context_scores = []
            for context_id, context in state.shared_context.items():
                age = time.time() - context.timestamp
                usage_score = sum(self.context_usage.get(context_id, {}).values())
                
                # Lower score is more likely to be removed
                score = age - (usage_score * 1000)  # Weight usage heavily
                context_scores.append((context_id, score))
            
            # Sort by score (ascending = oldest/least used)
            context_scores.sort(key=lambda x: x[1])
            
            # Remove oldest/least used contexts
            to_remove = (
                len(state.shared_context) - self.max_context_items + 10
            )  # Remove extra to avoid frequent cleanup
            
            for context_id, _ in context_scores[:to_remove]:
                await self._remove_context(context_id, state)
            
            logger.info(f"Cleaned up {to_remove} old context items")
            
        except Exception as e:
            logger.error(f"Error cleaning up context: {str(e)}")
            # Don't fail the workflow for cleanup errors
        
        return state
    
    async def _remove_context(self, context_id: str, state: ContextSharingState):
        """Remove a context item and its relationships"""
        if context_id not in state.shared_context:
            return
        
        context = state.shared_context[context_id]
        
        # Remove from indexes
        for tag in context.tags:
            if tag in state.context_index and context_id in state.context_index[tag]:
                state.context_index[tag].remove(context_id)
                if not state.context_index[tag]:
                    del state.context_index[tag]
        
        # Remove from agent contexts
        if context.agent_id in state.agent_contexts:
            state.agent_contexts[context.agent_id].discard(context_id)
        
        # Remove relationships
        state.context_relationships = [
            rel
            for rel in state.context_relationships
            if rel.source_context_id != context_id
            and rel.target_context_id != context_id
        ]
        
        # Remove from parent contexts
        for parent_id in context.parent_contexts:
            if parent_id in state.shared_context:
                state.shared_context[parent_id].child_contexts.remove(context_id)
        
        # Remove from child contexts
        for child_id in context.child_contexts:
            if child_id in state.shared_context:
                state.shared_context[child_id].parent_contexts.remove(context_id)
        
        # Remove from usage tracking
        if context_id in self.context_usage:
            del self.context_usage[context_id]
        
        # Remove context
        del state.shared_context[context_id]
    
    async def _notify_subscribers(self, context: SharedContext, state: ContextSharingState):
        """Notify agents subscribed to this type of context"""
        for subscription in state.subscriptions.values():
            if not subscription.get("active", True):
                continue
            
            # Check if subscription matches
            if (
                context.context_type in subscription.get("context_types", [])
                and context.priority.value >= subscription.get("priority_threshold", ContextPriority.MEDIUM).value
                and (
                    not subscription.get("tags", [])
                    or any(tag in context.tags for tag in subscription.get("tags", []))
                )
            ):
                # In a real implementation, this would send a notification to the agent
                logger.debug(
                    f"Notified agent {subscription['agent_id']} about context {context.context_id}"
                )
    
    async def _calculate_relevance(
        self, context: SharedContext, task: str, agent_id: str
    ) -> float:
        """Calculate relevance score of context to task"""
        # Base relevance from content similarity
        content_similarity = await self._calculate_text_similarity(context.content, task)
        
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
        # Use SHA-256 for better security
        content_hash = hashlib.sha256(
            f"{agent_id}{title}{content}{time.time()}".encode()
        ).hexdigest()
        return f"ctx_{content_hash[:16]}"
    
    async def process_request(self, state: ContextSharingState) -> ContextSharingState:
        """Process a context sharing request"""
        # Compile workflow if not already done
        if self.compiled_workflow is None:
            try:
                self.compiled_workflow = self.workflow.compile()
            except Exception as e:
                logger.error(f"Failed to compile workflow: {str(e)}")
                # Fallback to direct execution without compilation
                return await self._fallback_execution(state)
        
        # Update internal state from workflow state
        self.shared_context = state.shared_context
        self.context_relationships = state.context_relationships
        self.context_index = state.context_index
        self.agent_contexts = state.agent_contexts
        self.subscriptions = state.subscriptions
        
        # Process the request
        result_state = await self.compiled_workflow.ainvoke(state.dict())
        
        # Update internal state from result
        self.shared_context = result_state.get("shared_context", {})
        self.context_relationships = result_state.get("context_relationships", [])
        self.context_index = result_state.get("context_index", {})
        self.agent_contexts = result_state.get("agent_contexts", {})
        self.subscriptions = result_state.get("subscriptions", {})
        
        return ContextSharingState(**result_state)
    
    async def _fallback_execution(self, state: ContextSharingState) -> ContextSharingState:
        """Fallback execution when workflow compilation fails"""
        try:
            # Simple direct execution based on action
            if state.action == "share":
                return await self._share_context(state)
            elif state.action == "retrieve":
                return await self._retrieve_context(state)
            elif state.action == "build_on":
                return await self._build_on_context(state)
            elif state.action == "verify":
                return await self._verify_context(state)
            elif state.action == "subscribe":
                return await self._subscribe_to_context(state)
            elif state.action == "graph":
                return await self._get_context_graph(state)
            else:
                state.error = f"Unknown action: {state.action}"
                state.success = False
                return state
        except Exception as e:
            logger.error(f"Fallback execution failed: {str(e)}")
            state.error = str(e)
            state.success = False
            return state
    
    def get_statistics(self) -> Dict[str, Any]:
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
            relationship_counts[rel.relationship_type.value] += 1
        
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
                [s for s in self.subscriptions.values() if s.get("active", True)]
            ),
            "total_usage_events": sum(
                len(usage) for usage in self.context_usage.values()
            ),
        }