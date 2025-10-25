"""
LangGraph-based debate orchestrator for multi-writer collaboration.

This module implements a structured debate system using LangGraph's StateGraph
where writer agents can critique, rebut, and build upon each other's ideas
to create better content through collaborative discourse.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from app.core.langchain.integration import get_integration
from app.core.agents.content_langchain.writer_agent import WriterAgent

logger = logging.getLogger(__name__)


class DebatePhase(Enum):
    """Debate phases for structured collaboration"""
    INITIAL_POSITIONS = "initial_positions"
    CRITIQUE_AND_REBUTTAL = "critique_and_rebuttal"
    SYNTHESIS = "synthesis"
    FINAL_COLLABORATION = "final_collaboration"


@dataclass
class DebatePosition:
    """Represents a writer's position in the debate"""
    writer_id: str
    content: str
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateCritique:
    """Represents a critique of another writer's position"""
    critic_id: str
    target_id: str
    critique_points: List[str]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    overall_assessment: str
    timestamp: float = field(default_factory=time.time)


class DebateState(BaseModel):
    """State for the debate workflow"""
    debate_id: str
    prompt: str
    phase: DebatePhase
    participants: List[str]
    positions: Dict[str, DebatePosition] = {}
    critiques: Dict[str, List[DebateCritique]] = {}
    synthesis: Optional[str] = None
    final_content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    current_round: int = 0
    max_critique_rounds: int = 2
    synthesis_threshold: float = 0.7
    collaboration_model: str = "claude-3.5-sonnet"
    error: Optional[str] = None


class LangGraphDebateOrchestrator:
    """LangGraph-based debate orchestrator for structured collaboration"""

    def __init__(
        self,
        max_critique_rounds: int = 2,
        synthesis_threshold: float = 0.7,
        collaboration_model: str = "claude-3.5-sonnet",
        memory: Optional[MemorySaver] = None,
    ):
        self.max_critique_rounds = max_critique_rounds
        self.synthesis_threshold = synthesis_threshold
        self.collaboration_model = collaboration_model
        self.memory = memory or MemorySaver()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        # Compile the workflow
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _create_workflow(self) -> StateGraph:
        """Create the debate workflow graph"""
        workflow = StateGraph(DebateState)
        
        # Add nodes for each phase
        workflow.add_node("initialize_debate", self._initialize_debate)
        workflow.add_node("gather_initial_positions", self._gather_initial_positions)
        workflow.add_node("conduct_critique_round", self._conduct_critique_round)
        workflow.add_node("synthesize_positions", self._synthesize_positions)
        workflow.add_node("final_collaboration", self._final_collaboration)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("initialize_debate")
        
        # Add edges
        workflow.add_edge("initialize_debate", "gather_initial_positions")
        workflow.add_edge("gather_initial_positions", "conduct_critique_round")
        workflow.add_conditional_edges(
            "conduct_critique_round",
            self._should_continue_critique,
            {
                "continue": "conduct_critique_round",
                "synthesize": "synthesize_positions"
            }
        )
        workflow.add_edge("synthesize_positions", "final_collaboration")
        workflow.add_edge("final_collaboration", END)
        workflow.add_edge("handle_error", END)
        
        return workflow

    async def _initialize_debate(self, state: DebateState) -> DebateState:
        """Initialize the debate with basic setup"""
        try:
            logger.info(f"Initializing debate {state.debate_id}")
            
            # Set initial phase and configuration
            state.phase = DebatePhase.INITIAL_POSITIONS
            state.max_critique_rounds = self.max_critique_rounds
            state.synthesis_threshold = self.synthesis_threshold
            state.collaboration_model = self.collaboration_model
            state.current_round = 0
            
            # Initialize empty structures
            state.positions = {}
            state.critiques = {}
            
            logger.info(f"Debate {state.debate_id} initialized with {len(state.participants)} participants")
            return state
            
        except Exception as e:
            logger.error(f"Failed to initialize debate {state.debate_id}: {str(e)}")
            state.error = str(e)
            return state

    async def _gather_initial_positions(self, state: DebateState) -> DebateState:
        """Gather initial positions from all participants"""
        try:
            logger.info(f"Gathering initial positions for debate {state.debate_id}")
            
            state.phase = DebatePhase.INITIAL_POSITIONS
            
            # Get integration for LLM access
            integration = get_integration()
            
            # Get writer agents
            writer_agents = []
            for participant_id in state.participants:
                writer = WriterAgent(
                    writer_id=participant_id,
                    specialty=state.metadata.get(f"{participant_id}_specialty", "analytical")
                )
                writer_agents.append(writer)
            
            # Create position prompt
            position_prompt = self._create_position_prompt(state.prompt, state.context)
            
            # Get positions from all participants
            tasks = []
            for writer in writer_agents:
                task = self._get_writer_position(writer, position_prompt, state.context, integration)
                tasks.append(task)
            
            positions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store positions
            for i, position in enumerate(positions):
                if isinstance(position, Exception):
                    logger.error(f"Writer {writer_agents[i].writer_id} failed: {str(position)}")
                    continue
                
                state.positions[writer_agents[i].writer_id] = position
            
            logger.info(f"Collected {len(state.positions)} initial positions")
            return state
            
        except Exception as e:
            logger.error(f"Failed to gather initial positions: {str(e)}")
            state.error = str(e)
            return state

    async def _conduct_critique_round(self, state: DebateState) -> DebateState:
        """Conduct a round of critiques and rebuttals"""
        try:
            logger.info(f"Conducting critique round {state.current_round + 1} for debate {state.debate_id}")
            
            state.phase = DebatePhase.CRITIQUE_AND_REBUTTAL
            
            # Get integration for LLM access
            integration = get_integration()
            
            # Get writer agents
            writer_agents = []
            for participant_id in state.participants:
                writer = WriterAgent(
                    writer_id=participant_id,
                    specialty=state.metadata.get(f"{participant_id}_specialty", "analytical")
                )
                writer_agents.append(writer)
            
            # Generate critiques for this round
            critiques = await self._generate_critique_round(
                state, writer_agents, state.context, state.current_round, integration
            )
            
            # Store critiques
            for critic_id, target_critiques in critiques.items():
                if critic_id not in state.critiques:
                    state.critiques[critic_id] = []
                state.critiques[critic_id].extend(target_critiques)
            
            # Increment round counter
            state.current_round += 1
            
            logger.info(f"Completed critique round {state.current_round}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to conduct critique round: {str(e)}")
            state.error = str(e)
            return state

    async def _synthesize_positions(self, state: DebateState) -> DebateState:
        """Synthesize the best elements from all positions"""
        try:
            logger.info(f"Synthesizing debate {state.debate_id}")
            
            state.phase = DebatePhase.SYNTHESIS
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(state)
            
            # Get integration for LLM access
            integration = get_integration()
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(state.collaboration_model)
            
            # Generate synthesis
            messages = [
                {"role": "system", "content": self._get_synthesis_system_prompt()},
                {"role": "user", "content": synthesis_prompt},
            ]
            
            response = await llm.ainvoke(messages)
            state.synthesis = response.content
            
            logger.info("Synthesis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            state.error = str(e)
            
            # Fallback to best position
            if state.positions:
                best_position = max(state.positions.values(), key=lambda p: p.confidence)
                state.synthesis = best_position.content
            
            return state

    async def _final_collaboration(self, state: DebateState) -> DebateState:
        """Final collaborative refinement"""
        try:
            logger.info(f"Final collaboration for debate {state.debate_id}")
            
            state.phase = DebatePhase.FINAL_COLLABORATION
            
            # Get integration for LLM access
            integration = get_integration()
            
            # Get writer agents
            writer_agents = []
            for participant_id in state.participants:
                writer = WriterAgent(
                    writer_id=participant_id,
                    specialty=state.metadata.get(f"{participant_id}_specialty", "analytical")
                )
                writer_agents.append(writer)
            
            # Create final collaboration prompt
            collaboration_prompt = self._create_collaboration_prompt(state)
            
            # Get final collaborations
            tasks = []
            for writer in writer_agents:
                task = self._get_final_collaboration(
                    writer, collaboration_prompt, state, state.context, integration
                )
                tasks.append(task)
            
            collaborations = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Select best collaboration
            best_collaboration = None
            best_score = 0.0
            
            for i, collaboration in enumerate(collaborations):
                if isinstance(collaboration, Exception):
                    continue
                
                if collaboration.get("score", 0) > best_score:
                    best_score = collaboration.get("score", 0)
                    best_collaboration = collaboration
            
            if best_collaboration:
                state.final_content = best_collaboration.get("content", state.synthesis)
            else:
                state.final_content = state.synthesis
            
            logger.info("Final collaboration completed")
            return state
            
        except Exception as e:
            logger.error(f"Final collaboration failed: {str(e)}")
            state.error = str(e)
            state.final_content = state.synthesis or "Failed to generate final content"
            return state

    async def _handle_error(self, state: DebateState) -> DebateState:
        """Handle errors in the debate workflow"""
        logger.error(f"Handling error in debate {state.debate_id}: {state.error}")
        return state

    def _should_continue_critique(self, state: DebateState) -> str:
        """Determine if we should continue with another critique round"""
        if state.error:
            return "synthesize"
        
        if state.current_round >= state.max_critique_rounds:
            return "synthesize"
        
        return "continue"

    async def _get_writer_position(
        self, writer: WriterAgent, prompt: str, context: Dict[str, Any], integration
    ) -> DebatePosition:
        """Get initial position from a writer"""
        try:
            # Create writer-specific prompt
            writer_prompt = self._enhance_prompt_for_writer(prompt, writer)
            
            # Generate content
            source_content = context.get(
                "source_content",
                {"title": "Debate Topic", "content": prompt, "key_points": []},
            )
            
            result = await writer.generate_content(
                writer_prompt, source_content, context.get("style_guide")
            )
            
            if result.get("error"):
                raise Exception(result["error"])
            
            # Extract confidence score
            confidence = result.get("confidence_score", 0.5)
            
            # Generate reasoning
            reasoning = await self._generate_position_reasoning(
                result.get("content", ""), writer.specialty, prompt, integration
            )
            
            return DebatePosition(
                writer_id=writer.writer_id,
                content=result.get("content", ""),
                confidence=confidence,
                reasoning=reasoning,
                metadata=result,
            )
            
        except Exception as e:
            logger.error(f"Failed to get position from {writer.writer_id}: {str(e)}")
            raise

    async def _generate_critique_round(
        self,
        state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
        round_num: int,
        integration,
    ) -> Dict[str, List[DebateCritique]]:
        """Generate critiques for a specific round"""
        critiques = {}
        
        for critic in participants:
            critic_critiques = []
            
            for target in participants:
                if critic.writer_id == target.writer_id:
                    continue  # Don't critique self
                
                try:
                    target_position = state.positions[target.writer_id]
                    critique = await self._generate_critique(
                        critic,
                        target,
                        target_position,
                        state,
                        context,
                        round_num,
                        integration,
                    )
                    critic_critiques.append(critique)
                    
                except Exception as e:
                    logger.error(f"Failed to generate critique from {critic.writer_id}: {str(e)}")
            
            critiques[critic.writer_id] = critic_critiques
        
        return critiques

    async def _generate_critique(
        self,
        critic: WriterAgent,
        target: WriterAgent,
        target_position: DebatePosition,
        state: DebateState,
        context: Dict[str, Any],
        round_num: int,
        integration,
    ) -> DebateCritique:
        """Generate a critique from one writer to another"""
        critique_prompt = self._create_critique_prompt(
            critic.specialty,
            target.specialty,
            target_position.content,
            state.prompt,
            round_num,
        )
        
        try:
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(state.collaboration_model)
            
            messages = [
                {
                    "role": "system",
                    "content": self._get_critique_system_prompt(critic.specialty),
                },
                {"role": "user", "content": critique_prompt},
            ]
            
            response = await llm.ainvoke(messages)
            
            # Parse the critique response
            critique_data = self._parse_critique_response(response.content)
            
            return DebateCritique(
                critic_id=critic.writer_id,
                target_id=target.writer_id,
                critique_points=critique_data.get("critique_points", []),
                strengths=critique_data.get("strengths", []),
                weaknesses=critique_data.get("weaknesses", []),
                suggestions=critique_data.get("suggestions", []),
                overall_assessment=critique_data.get("overall_assessment", ""),
            )
            
        except Exception as e:
            logger.error(f"Failed to generate critique: {str(e)}")
            # Return a basic critique
            return DebateCritique(
                critic_id=critic.writer_id,
                target_id=target.writer_id,
                critique_points=["Unable to generate detailed critique"],
                strengths=[],
                weaknesses=[],
                suggestions=[],
                overall_assessment="Critique generation failed",
            )

    async def _get_final_collaboration(
        self,
        writer: WriterAgent,
        prompt: str,
        state: DebateState,
        context: Dict[str, Any],
        integration,
    ) -> Dict[str, Any]:
        """Get final collaborative contribution from a writer"""
        try:
            # Create collaboration prompt
            collaboration_prompt = self._enhance_prompt_for_writer(prompt, writer)
            
            # Add debate context
            collaboration_prompt += f"\n\nDebate Context:\n{self._summarize_debate_for_writer(state, writer.writer_id)}"
            
            # Generate content
            source_content = {
                "title": "Collaborative Writing",
                "content": collaboration_prompt,
                "key_points": [],
            }
            
            result = await writer.generate_content(
                collaboration_prompt, source_content, context.get("style_guide")
            )
            
            if result.get("error"):
                raise Exception(result["error"])
            
            # Score the collaboration
            score = await self._score_collaboration(
                result.get("content", ""), state, writer.writer_id
            )
            
            return {
                "writer_id": writer.writer_id,
                "content": result.get("content", ""),
                "score": score,
                "metadata": result,
            }
            
        except Exception as e:
            logger.error(f"Failed to get collaboration from {writer.writer_id}: {str(e)}")
            raise

    def _create_position_prompt(self, original_prompt: str, context: Dict[str, Any]) -> str:
        """Create prompt for initial positions"""
        return f"""
You are participating in a structured debate to create the best possible content.

ORIGINAL PROMPT:
{original_prompt}

CONTEXT:
{context or "No additional context provided"}

INSTRUCTIONS:
1. Provide your initial position on this topic
2. Be clear and articulate your main points
3. Consider your specialty and perspective
4. This will be critiqued by other writers, so make it strong
5. Aim for approximately 300-500 words

Your position:
"""

    def _create_critique_prompt(
        self,
        critic_specialty: str,
        target_specialty: str,
        target_content: str,
        original_prompt: str,
        round_num: int,
    ) -> str:
        """Create prompt for critiquing another writer's position"""
        return f"""
You are a {critic_specialty} writer critiquing a {target_specialty} writer's position.

ORIGINAL PROMPT:
{original_prompt}

TARGET POSITION:
{target_content}

CRITIQUE ROUND: {round_num + 1}

INSTRUCTIONS:
Provide a constructive critique that includes:
1. Specific critique points (what could be improved)
2. Strengths of the position (what works well)
3. Weaknesses or gaps (what's missing)
4. Specific suggestions for improvement
5. Overall assessment

Be constructive and specific. Your goal is to help improve the final content.

Format your response as:
CRITIQUE POINTS:
[Your critique points]

STRENGTHS:
[Identified strengths]

WEAKNESSES:
[Identified weaknesses]

SUGGESTIONS:
[Specific suggestions]

OVERALL ASSESSMENT:
[Your overall assessment]
"""

    def _create_synthesis_prompt(self, state: DebateState) -> str:
        """Create prompt for synthesizing debate results"""
        positions_text = "\n\n".join(
            [
                f"Position from {writer_id} (Confidence: {pos.confidence:.2f}):\n{pos.content}"
                for writer_id, pos in state.positions.items()
            ]
        )
        
        critiques_text = ""
        for critic_id, critiques in state.critiques.items():
            for critique in critiques:
                critiques_text += f"\n{critic_id} critiqued {critique.target_id}:\n"
                critiques_text += f"Overall: {critique.overall_assessment}\n"
                critiques_text += f"Key points: {', '.join(critique.critique_points)}\n"
        
        return f"""
You are synthesizing a debate between multiple writers to create the best possible content.

ORIGINAL PROMPT:
{state.prompt}

POSITIONS:
{positions_text}

CRITIQUES:
{critiques_text}

INSTRUCTIONS:
1. Identify the strongest elements from each position
2. Address the main concerns raised in critiques
3. Create a synthesized version that incorporates the best ideas
4. Ensure the final content is coherent and well-structured
5. Aim for approximately 400-600 words

Synthesized content:
"""

    def _create_collaboration_prompt(self, state: DebateState) -> str:
        """Create prompt for final collaboration"""
        return f"""
Based on the debate and synthesis, create your final collaborative contribution.

ORIGINAL PROMPT:
{state.prompt}

SYNTHESIS:
{state.synthesis}

INSTRUCTIONS:
1. Build upon the synthesized content
2. Incorporate insights from the debate
3. Apply your specialty to enhance the content
4. Make final improvements based on critiques
5. Aim for 400-600 words of polished content

Your final collaborative contribution:
"""

    def _enhance_prompt_for_writer(self, prompt: str, writer: WriterAgent) -> str:
        """Enhance a prompt specifically for a writer"""
        specialty_instructions = {
            "technical": "Focus on accuracy, clarity, and technical precision.",
            "creative": "Focus on engagement, storytelling, and creative expression.",
            "analytical": "Focus on data-driven insights and logical reasoning.",
        }
        
        instruction = specialty_instructions.get(
            writer.specialty, "Focus on quality and clarity."
        )
        
        return f"{prompt}\n\nAs a {writer.specialty} writer: {instruction}"

    def _get_synthesis_system_prompt(self) -> str:
        """Get system prompt for synthesis phase"""
        return """
You are an expert content synthesizer. Your role is to analyze multiple perspectives
and critiques to create the best possible synthesized content. Look for common themes,
strong arguments, and valuable insights from all participants.
"""

    def _get_critique_system_prompt(self, specialty: str) -> str:
        """Get system prompt for critique phase"""
        return f"""
You are a {specialty} writer providing constructive critique. Be specific, 
constructive, and helpful. Your goal is to improve the final content through 
thoughtful feedback, not to tear down others' work.
"""

    def _parse_critique_response(self, response: str) -> Dict[str, List[str]]:
        """Parse critique response into structured data"""
        sections = {
            "critique_points": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "overall_assessment": "",
        }
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.upper().startswith("CRITIQUE POINTS:"):
                current_section = "critique_points"
            elif line.upper().startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.upper().startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line.upper().startswith("SUGGESTIONS:"):
                current_section = "suggestions"
            elif line.upper().startswith("OVERALL ASSESSMENT:"):
                current_section = "overall_assessment"
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                if current_section and current_section != "overall_assessment":
                    sections[current_section].append(line[1:].strip())
            elif current_section == "overall_assessment":
                sections["overall_assessment"] += line + " "
        
        return sections

    async def _generate_position_reasoning(
        self, content: str, specialty: str, prompt: str, integration
    ) -> str:
        """Generate reasoning for a position"""
        reasoning_prompt = f"""
Explain the reasoning behind this {specialty} writer's position:

PROMPT: {prompt}

CONTENT: {content}

Provide a brief explanation of the approach and key decisions (50-100 words):
"""
        
        try:
            llm_manager = integration.get_llm_manager()
            llm = await llm_manager.get_llm(self.collaboration_model)
            response = await llm.ainvoke(reasoning_prompt)
            return response.content.strip()
        except Exception:
            return f"Position created with {specialty} approach focusing on key aspects of the prompt."

    def _summarize_debate_for_writer(self, state: DebateState, writer_id: str) -> str:
        """Summarize debate for a specific writer"""
        summary = f"Debate Summary for {writer_id}:\n\n"
        
        # Add other positions
        for other_id, position in state.positions.items():
            if other_id != writer_id:
                summary += f"Position from {other_id}:\n{position.content[:200]}...\n\n"
        
        # Add critiques of this writer
        if writer_id in state.critiques:
            summary += "Critiques of your position:\n"
            for critique in state.critiques[writer_id]:
                summary += f"- {critique.overall_assessment}\n"
        
        # Add synthesis
        if state.synthesis:
            summary += f"\nSynthesis: {state.synthesis[:300]}...\n"
        
        return summary

    async def _score_collaboration(self, content: str, state: DebateState, writer_id: str) -> float:
        """Score a collaborative contribution"""
        # Base score
        score = 0.5
        
        # Length check
        word_count = len(content.split())
        if 300 <= word_count <= 800:
            score += 0.2
        
        # Incorporates synthesis
        if state.synthesis and state.synthesis[:100] in content:
            score += 0.2
        
        # Addresses critiques
        if writer_id in state.critiques:
            critiques_addressed = 0
            for critique in state.critiques[writer_id]:
                for suggestion in critique.suggestions:
                    if any(
                        word in content.lower()
                        for word in suggestion.lower().split()[:3]
                    ):
                        critiques_addressed += 1
                        break
            
            if critiques_addressed > 0:
                score += min(0.1, critiques_addressed * 0.05)
        
        return min(1.0, score)

    async def conduct_debate(
        self,
        prompt: str,
        participants: List[str],
        context: Dict[str, Any] = None,
        thread_id: Optional[str] = None,
    ) -> DebateState:
        """
        Conduct a structured debate between writer agents.

        Args:
            prompt: The original writing prompt
            participants: List of writer IDs to participate
            context: Additional context for the debate
            thread_id: Optional thread ID for conversation persistence

        Returns:
            DebateState with complete debate results
        """
        debate_id = thread_id or f"debate_{int(time.time())}"
        logger.info(f"Starting debate {debate_id} with {len(participants)} participants")
        
        # Initialize state
        initial_state = DebateState(
            debate_id=debate_id,
            prompt=prompt,
            phase=DebatePhase.INITIAL_POSITIONS,
            participants=participants,
            context=context or {},
            metadata={"context": context or {}},
        )
        
        # Configure the run
        config = {"configurable": {"thread_id": debate_id}}
        
        try:
            # Run the workflow
            result = await self.app.ainvoke(initial_state, config=config)
            
            logger.info(f"Debate {debate_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {str(e)}")
            initial_state.error = str(e)
            return initial_state

    async def get_debate_state(self, thread_id: str) -> Optional[DebateState]:
        """Get the current state of a debate"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = await self.app.aget_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Failed to get debate state for {thread_id}: {str(e)}")
            return None

    def get_debate_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get the history of a debate"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            history = []
            
            for state in self.app.get_state_history(config):
                history.append({
                    "step": state.config.get("step", 0),
                    "phase": state.values.get("phase"),
                    "timestamp": state.metadata.get("timestamp"),
                })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get debate history for {thread_id}: {str(e)}")
            return []