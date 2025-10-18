"""
Debate system for multi-writer collaboration.

This module implements a structured debate system where writer agents
can critique, rebut, and build upon each other's ideas to create
better content through collaborative discourse.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.core.agents.content.writer_agent import WriterAgent
from app.core.config import get_llm

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


@dataclass
class DebateState:
    """Tracks the state of an ongoing debate"""

    debate_id: str
    prompt: str
    phase: DebatePhase
    participants: List[str]
    positions: Dict[str, DebatePosition] = field(default_factory=dict)
    critiques: Dict[str, List[DebateCritique]] = field(default_factory=dict)
    synthesis: Optional[str] = None
    final_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateOrchestrator:
    """Orchestrates structured debates between writer agents"""

    def __init__(
        self,
        max_critique_rounds: int = 2,
        synthesis_threshold: float = 0.7,
        collaboration_model: str = "claude-3.5-sonnet",
    ):
        self.max_critique_rounds = max_critique_rounds
        self.synthesis_threshold = synthesis_threshold
        self.collaboration_model = collaboration_model
        self.llm = None  # Will be initialized when needed

    async def _get_llm(self):
        """Initialize LLM if not already done"""
        if self.llm is None:
            self.llm = await get_llm(self.collaboration_model)
        return self.llm

    async def conduct_debate(
        self,
        prompt: str,
        participants: List[WriterAgent],
        context: Dict[str, Any] = None,
    ) -> DebateState:
        """
        Conduct a structured debate between writer agents.

        Args:
            prompt: The original writing prompt
            participants: List of writer agents to participate
            context: Additional context for the debate

        Returns:
            DebateState with complete debate results
        """
        debate_id = f"debate_{int(time.time())}"
        logger.info(
            f"Starting debate {debate_id} with {len(participants)} participants"
        )

        # Initialize debate state
        debate_state = DebateState(
            debate_id=debate_id,
            prompt=prompt,
            phase=DebatePhase.INITIAL_POSITIONS,
            participants=[p.writer_id for p in participants],
            metadata={"context": context or {}},
        )

        try:
            # Phase 1: Initial positions
            debate_state = await self._phase_1_initial_positions(
                debate_state, participants, context
            )

            # Phase 2: Critique and rebuttal
            debate_state = await self._phase_2_critique_and_rebuttal(
                debate_state, participants, context
            )

            # Phase 3: Synthesis
            debate_state = await self._phase_3_synthesis(
                debate_state, participants, context
            )

            # Phase 4: Final collaboration
            debate_state = await self._phase_4_final_collaboration(
                debate_state, participants, context
            )

            logger.info(f"Debate {debate_id} completed successfully")
            return debate_state

        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {str(e)}")
            debate_state.metadata["error"] = str(e)
            return debate_state

    async def _phase_1_initial_positions(
        self,
        debate_state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
    ) -> DebateState:
        """Phase 1: Gather initial positions from all participants"""
        logger.info(
            f"Phase 1: Gathering initial positions for debate {debate_state.debate_id}"
        )

        debate_state.phase = DebatePhase.INITIAL_POSITIONS

        # Create initial position prompt
        position_prompt = self._create_position_prompt(debate_state.prompt, context)

        # Get initial positions from all participants
        tasks = []
        for participant in participants:
            task = self._get_writer_position(participant, position_prompt, context)
            tasks.append(task)

        positions = await asyncio.gather(*tasks, return_exceptions=True)

        # Store positions
        for i, position in enumerate(positions):
            if isinstance(position, Exception):
                logger.error(
                    f"Writer {participants[i].writer_id} failed: {str(position)}"
                )
                continue

            debate_state.positions[participants[i].writer_id] = position

        logger.info(f"Collected {len(debate_state.positions)} initial positions")
        return debate_state

    async def _phase_2_critique_and_rebuttal(
        self,
        debate_state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
    ) -> DebateState:
        """Phase 2: Conduct critique and rebuttal rounds"""
        logger.info(
            f"Phase 2: Starting critique and rebuttal for debate {debate_state.debate_id}"
        )

        debate_state.phase = DebatePhase.CRITIQUE_AND_REBUTTAL

        for round_num in range(self.max_critique_rounds):
            logger.info(
                f"Debate {debate_state.debate_id}, critique round {round_num + 1}"
            )

            # Generate critiques for this round
            critiques = await self._generate_critique_round(
                debate_state, participants, context, round_num
            )

            # Store critiques
            for critic_id, target_critiques in critiques.items():
                if critic_id not in debate_state.critiques:
                    debate_state.critiques[critic_id] = []
                debate_state.critiques[critic_id].extend(target_critiques)

        logger.info(f"Completed {self.max_critique_rounds} critique rounds")
        return debate_state

    async def _phase_3_synthesis(
        self,
        debate_state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
    ) -> DebateState:
        """Phase 3: Synthesize the best elements from all positions"""
        logger.info(f"Phase 3: Synthesizing debate {debate_state.debate_id}")

        debate_state.phase = DebatePhase.SYNTHESIS

        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(debate_state)

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._get_synthesis_system_prompt()},
                    {"role": "user", "content": synthesis_prompt},
                ]
            )

            debate_state.synthesis = response.content
            logger.info("Synthesis completed successfully")

        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            debate_state.synthesis = "Synthesis failed. Using best individual position."
            # Fallback to best position
            best_position = max(
                debate_state.positions.values(), key=lambda p: p.confidence
            )
            debate_state.synthesis = best_position.content

        return debate_state

    async def _phase_4_final_collaboration(
        self,
        debate_state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
    ) -> DebateState:
        """Phase 4: Final collaborative refinement"""
        logger.info(f"Phase 4: Final collaboration for debate {debate_state.debate_id}")

        debate_state.phase = DebatePhase.FINAL_COLLABORATION

        # Create final collaboration prompt
        collaboration_prompt = self._create_collaboration_prompt(debate_state)

        tasks = []
        for participant in participants:
            task = self._get_final_collaboration(
                participant, collaboration_prompt, debate_state, context
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
            debate_state.final_content = best_collaboration.get(
                "content", debate_state.synthesis
            )
        else:
            debate_state.final_content = debate_state.synthesis

        logger.info("Final collaboration completed")
        return debate_state

    async def _get_writer_position(
        self, writer: WriterAgent, prompt: str, context: Dict[str, Any]
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
                result.get("content", ""), writer.specialty, prompt
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
        debate_state: DebateState,
        participants: List[WriterAgent],
        context: Dict[str, Any],
        round_num: int,
    ) -> Dict[str, List[DebateCritique]]:
        """Generate critiques for a specific round"""
        critiques = {}

        for critic in participants:
            critic_critiques = []

            for target in participants:
                if critic.writer_id == target.writer_id:
                    continue  # Don't critique self

                try:
                    critique = await self._generate_critique(
                        critic,
                        target,
                        debate_state.positions[target.writer_id],
                        debate_state,
                        context,
                        round_num,
                    )
                    critic_critiques.append(critique)

                except Exception as e:
                    logger.error(
                        f"Failed to generate critique from {critic.writer_id}: {str(e)}"
                    )

            critiques[critic.writer_id] = critic_critiques

        return critiques

    async def _generate_critique(
        self,
        critic: WriterAgent,
        target: WriterAgent,
        target_position: DebatePosition,
        debate_state: DebateState,
        context: Dict[str, Any],
        round_num: int,
    ) -> DebateCritique:
        """Generate a critique from one writer to another"""
        critique_prompt = self._create_critique_prompt(
            critic.specialty,
            target.specialty,
            target_position.content,
            debate_state.prompt,
            round_num,
        )

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(
                [
                    {
                        "role": "system",
                        "content": self._get_critique_system_prompt(critic.specialty),
                    },
                    {"role": "user", "content": critique_prompt},
                ]
            )

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
        debate_state: DebateState,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get final collaborative contribution from a writer"""
        try:
            # Create collaboration prompt
            collaboration_prompt = self._enhance_prompt_for_writer(prompt, writer)

            # Add debate context
            collaboration_prompt += f"\n\nDebate Context:\n{self._summarize_debate_for_writer(debate_state, writer.writer_id)}"

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
                result.get("content", ""), debate_state, writer.writer_id
            )

            return {
                "writer_id": writer.writer_id,
                "content": result.get("content", ""),
                "score": score,
                "metadata": result,
            }

        except Exception as e:
            logger.error(
                f"Failed to get collaboration from {writer.writer_id}: {str(e)}"
            )
            raise

    def _create_position_prompt(
        self, original_prompt: str, context: Dict[str, Any]
    ) -> str:
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

    def _create_synthesis_prompt(self, debate_state: DebateState) -> str:
        """Create prompt for synthesizing debate results"""
        positions_text = "\n\n".join(
            [
                f"Position from {writer_id} (Confidence: {pos.confidence:.2f}):\n{pos.content}"
                for writer_id, pos in debate_state.positions.items()
            ]
        )

        critiques_text = ""
        for critic_id, critiques in debate_state.critiques.items():
            for critique in critiques:
                critiques_text += f"\n{critic_id} critiqued {critique.target_id}:\n"
                critiques_text += f"Overall: {critique.overall_assessment}\n"
                critiques_text += f"Key points: {', '.join(critique.critique_points)}\n"

        return f"""
You are synthesizing a debate between multiple writers to create the best possible content.

ORIGINAL PROMPT:
{debate_state.prompt}

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

    def _create_collaboration_prompt(self, debate_state: DebateState) -> str:
        """Create prompt for final collaboration"""
        return f"""
Based on the debate and synthesis, create your final collaborative contribution.

ORIGINAL PROMPT:
{debate_state.prompt}

SYNTHESIS:
{debate_state.synthesis}

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
        self, content: str, specialty: str, prompt: str
    ) -> str:
        """Generate reasoning for a position"""
        reasoning_prompt = f"""
Explain the reasoning behind this {specialty} writer's position:

PROMPT: {prompt}

CONTENT: {content}

Provide a brief explanation of the approach and key decisions (50-100 words):
"""

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(reasoning_prompt)
            return response.content.strip()
        except Exception:
            return f"Position created with {specialty} approach focusing on key aspects of the prompt."

    def _summarize_debate_for_writer(
        self, debate_state: DebateState, writer_id: str
    ) -> str:
        """Summarize debate for a specific writer"""
        summary = f"Debate Summary for {writer_id}:\n\n"

        # Add other positions
        for other_id, position in debate_state.positions.items():
            if other_id != writer_id:
                summary += f"Position from {other_id}:\n{position.content[:200]}...\n\n"

        # Add critiques of this writer
        if writer_id in debate_state.critiques:
            summary += "Critiques of your position:\n"
            for critique in debate_state.critiques[writer_id]:
                summary += f"- {critique.overall_assessment}\n"

        # Add synthesis
        if debate_state.synthesis:
            summary += f"\nSynthesis: {debate_state.synthesis[:300]}...\n"

        return summary

    async def _score_collaboration(
        self, content: str, debate_state: DebateState, writer_id: str
    ) -> float:
        """Score a collaborative contribution"""
        # Base score
        score = 0.5

        # Length check
        word_count = len(content.split())
        if 300 <= word_count <= 800:
            score += 0.2

        # Incorporates synthesis
        if debate_state.synthesis and debate_state.synthesis[:100] in content:
            score += 0.2

        # Addresses critiques
        if writer_id in debate_state.critiques:
            critiques_addressed = 0
            for critique in debate_state.critiques[writer_id]:
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
