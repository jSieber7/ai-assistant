"""
Writer agent implementation for multi-writer system
"""

from typing import List, Dict, Any, Optional
import asyncio
from app.core.agents.base.base import BaseAgent
from app.core.config import get_llm


class WriterAgent(BaseAgent):
    """AI writer agent specialized for different content types"""

    def __init__(
        self, writer_id: str, specialty: str, model: str = "claude-3.5-sonnet"
    ):
        # Initialize with empty tool registry since writers don't use tools directly
        from app.core.tools.execution.registry import ToolRegistry

        super().__init__(ToolRegistry(), max_iterations=1)
        self.writer_id = writer_id
        self.specialty = specialty  # e.g., "technical", "creative", "analytical"
        self.model = model

    @property
    def name(self) -> str:
        return f"writer_{self.writer_id}"

    @property
    def description(self) -> str:
        return f"{self.specialty} writer agent for content generation"

    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> Any:
        """Process message for writer agent - not used in this implementation"""
        pass

    async def generate_content(
        self,
        prompt: str,
        source_content: Dict[str, Any],
        style_guide: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate content based on sources and requirements"""

        # Create writer-specific prompt
        system_prompt = self._create_writer_prompt(style_guide)

        user_prompt = f"""
        Task: {prompt}
        
        Source Material:
        Title: {source_content.get("title", "")}
        Content: {source_content.get("content", "")[:2000]}...
        Key Points: {", ".join(source_content.get("key_points", [])[:5])}
        
        Instructions:
        1. Write as a {self.specialty} writer
        2. Incorporate key information from sources
        3. Maintain factual accuracy
        4. Follow the provided style guide
        5. Create engaging, well-structured content
        """

        try:
            client = await get_llm(self.model)

            response = await client.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            content = response.content

            return {
                "writer_id": self.writer_id,
                "specialty": self.specialty,
                "content": content,
                "model_used": self.model,
                "sources_used": [source_content.get("url")],
                "word_count": len(content.split()),
                "confidence_score": self._calculate_confidence(content, source_content),
            }

        except Exception as e:
            return {"writer_id": self.writer_id, "error": str(e), "content": None}

    def _create_writer_prompt(self, style_guide: Optional[Dict[str, Any]]) -> str:
        """Create system prompt based on writer specialty and style guide"""
        base_prompt = f"""
        You are a {self.specialty} writer creating high-quality content.
        """

        if self.specialty == "technical":
            base_prompt += """
            Focus on:
            - Accuracy and clarity
            - Technical precision
            - Structured explanations
            - Practical examples
            """
        elif self.specialty == "creative":
            base_prompt += """
            Focus on:
            - Engaging storytelling
            - Creative expression
            - Emotional connection
            - Vivid descriptions
            """
        elif self.specialty == "analytical":
            base_prompt += """
            Focus on:
            - Data-driven insights
            - Logical reasoning
            - Evidence-based claims
            - Balanced perspectives
            """

        if style_guide:
            base_prompt += f"""
            
            Style Guide:
            - Tone: {style_guide.get("tone", "professional")}
            - Audience: {style_guide.get("audience", "general")}
            - Length: {style_guide.get("length", "medium")}
            - Format: {style_guide.get("format", "article")}
            """

        return base_prompt

    def _calculate_confidence(
        self, content: str, source_content: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on content quality and source usage"""
        score = 0.5  # Base score

        # Check if content mentions source material
        if source_content.get("title", "").lower() in content.lower():
            score += 0.1

        # Check word count
        word_count = len(content.split())
        if 100 <= word_count <= 1000:
            score += 0.2

        # Check for structure (paragraphs, headings)
        if "\n\n" in content and any(
            line.strip().endswith(":") for line in content.split("\n")
        ):
            score += 0.1

        # Check for key points integration
        key_points = source_content.get("key_points", [])
        integrated_points = sum(
            1 for point in key_points if point.lower() in content.lower()
        )
        if integrated_points > 0:
            score += min(0.1, integrated_points * 0.02)

        return min(1.0, score)


class MultiWriterOrchestrator:
    """Orchestrates multiple writer agents"""

    def __init__(self):
        self.writers = {
            "technical_1": WriterAgent("technical_1", "technical", "claude-3.5-sonnet"),
            "technical_2": WriterAgent("technical_2", "technical", "gpt-4-turbo"),
            "creative_1": WriterAgent("creative_1", "creative", "claude-3.5-sonnet"),
            "analytical_1": WriterAgent("analytical_1", "analytical", "gpt-4-turbo"),
        }

    async def generate_multiple_versions(
        self,
        prompt: str,
        source_content: Dict[str, Any],
        writer_ids: List[str] = None,
        style_guide: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple content versions using different writers"""

        if writer_ids is None:
            writer_ids = list(self.writers.keys())

        tasks = []
        for writer_id in writer_ids:
            if writer_id in self.writers:
                task = self.writers[writer_id].generate_content(
                    prompt, source_content, style_guide
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            result
            for result in results
            if isinstance(result, dict) and result.get("content")
        ]

        return successful_results
