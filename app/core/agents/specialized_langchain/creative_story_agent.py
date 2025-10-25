"""
CreativeStoryAgent - A specialized agent for creative story generation using LangChain.

This agent can generate creative stories, develop characters, create plots,
and produce engaging narratives across various genres and styles.
"""

import logging
import asyncio
import json
import random
from typing import Dict, List, Optional, Any, AsyncGenerator
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class Character(BaseModel):
    """Story character definition"""
    name: str = Field(description="Character name")
    role: str = Field(description="Character's role in the story")
    personality: str = Field(description="Character's personality traits")
    background: str = Field(description="Character's background story")
    motivation: str = Field(description="Character's motivation or goal")


class PlotPoint(BaseModel):
    """Story plot point"""
    sequence: int = Field(description="Sequence in the story")
    description: str = Field(description="Description of the plot point")
    conflict: Optional[str] = Field(default=None, description="Conflict or challenge")
    resolution: Optional[str] = Field(default=None, description="Resolution of the conflict")


class CreativeStoryState(BaseModel):
    """State for creative story generation workflow"""
    prompt: str = Field(description="Story prompt or theme")
    genre: str = Field(default="fantasy", description="Story genre: fantasy, sci-fi, mystery, romance, horror, etc.")
    style: str = Field(default="descriptive", description="Writing style: descriptive, minimal, poetic, etc.")
    length: str = Field(default="medium", description="Story length: short, medium, long")
    target_audience: str = Field(default="general", description="Target audience: children, teens, adults, etc.")
    tone: str = Field(default="neutral", description="Story tone: humorous, serious, dark, etc.")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    characters: List[Character] = Field(default_factory=list, description="Story characters")
    plot_outline: List[PlotPoint] = Field(default_factory=list, description="Story plot outline")
    setting: Optional[str] = Field(default=None, description="Story setting and time period")
    theme: Optional[str] = Field(default=None, description="Central theme of the story")
    story_content: Optional[str] = Field(default=None, description="Generated story content")
    title: Optional[str] = Field(default=None, description="Story title")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether story generation was successful")


class CreativeStoryAgent:
    """
    A specialized agent for creative story generation using LangChain.
    
    This agent can create engaging stories with well-developed characters,
    compelling plots, and appropriate settings across various genres.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CreativeStoryAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "creative_story_agent"
        self.description = "Specialized agent for creative story generation"
        
        # Story element libraries for inspiration
        self.genre_elements = {
            "fantasy": {
                "settings": ["enchanted forest", "ancient castle", "magical kingdom", "mystical realm"],
                "creatures": ["dragon", "unicorn", "wizard", "elf", "goblin"],
                "objects": ["magic sword", "ancient tome", "crystal ball", "enchanted amulet"],
            },
            "sci-fi": {
                "settings": ["space station", "alien planet", "cyberpunk city", "generation ship"],
                "creatures": ["android", "alien", "cyborg", "AI entity"],
                "objects": ["laser pistol", "starship", "time device", "neural implant"],
            },
            "mystery": {
                "settings": ["victorian mansion", "rainy city street", "isolated island", "old library"],
                "creatures": ["detective", "suspect", "witness", "victim"],
                "objects": ["clue", "weapon", "secret document", "mysterious letter"],
            },
        }
        
        # LangGraph workflow
        self.workflow = None
        self.checkpointer = MemorySaver()
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize components asynchronously"""
        try:
            if not self.llm:
                self.llm = await get_llm()
            
            # Create LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("CreativeStoryAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CreativeStoryAgent: {str(e)}")
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for creative story generation"""
        workflow = StateGraph(CreativeStoryState)
        
        # Define nodes
        def analyze_prompt(state: CreativeStoryState) -> CreativeStoryState:
            """Analyze the story prompt to extract key elements"""
            try:
                # Create analysis prompt
                prompt = f"""
                Analyze the following story prompt to extract key elements for story generation:
                
                Prompt: "{state.prompt}"
                Genre: {state.genre}
                Style: {state.style}
                Length: {state.length}
                Target Audience: {state.target_audience}
                Tone: {state.tone}
                
                Please identify:
                1. Main theme or central idea
                2. Key character archetypes needed
                3. Setting requirements
                4. Plot structure suggestions
                5. Conflict types that would work well
                
                Respond with JSON format.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    analysis = json.loads(response.content)
                    state.theme = analysis.get("theme", "")
                    state.messages.append(SystemMessage(content="Prompt analysis completed"))
                except json.JSONDecodeError:
                    state.theme = "Story based on: " + state.prompt
                    state.messages.append(SystemMessage(content="Prompt analysis completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error analyzing prompt: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def develop_characters(state: CreativeStoryState) -> CreativeStoryState:
            """Develop characters for the story"""
            try:
                # Get genre-specific elements for inspiration
                genre_elements = self.genre_elements.get(state.genre, {})
                
                # Create character development prompt
                elements_text = json.dumps(genre_elements)
                
                prompt = f"""
                Create 3-4 compelling characters for a {state.genre} story with the following details:
                
                Theme: {state.theme or state.prompt}
                Target Audience: {state.target_audience}
                Tone: {state.tone}
                Length: {state.length}
                
                Genre Elements for Inspiration: {elements_text}
                
                For each character, provide:
                1. Name (appropriate for genre)
                2. Role in the story (protagonist, antagonist, supporting, etc.)
                3. Personality traits (3-4 distinct traits)
                4. Brief background story
                5. Motivation or goal in the story
                
                Ensure characters are diverse and create interesting dynamics.
                Respond with JSON array of characters.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    characters_data = json.loads(response.content)
                    characters = []
                    
                    for char_data in characters_data:
                        character = Character(
                            name=char_data.get("name", ""),
                            role=char_data.get("role", ""),
                            personality=char_data.get("personality", ""),
                            background=char_data.get("background", ""),
                            motivation=char_data.get("motivation", "")
                        )
                        characters.append(character)
                    
                    state.characters = characters
                    state.messages.append(SystemMessage(content=f"Developed {len(characters)} characters"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback character creation
                    fallback_character = Character(
                        name="Protagonist",
                        role="Main character",
                        personality="Brave and curious",
                        background="A mysterious individual with unknown origins",
                        motivation="To solve the central mystery"
                    )
                    state.characters = [fallback_character]
                    state.messages.append(SystemMessage(content="Character development completed (fallback)"))
                
            except Exception as e:
                state.error = f"Error developing characters: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def create_plot_outline(state: CreativeStoryState) -> CreativeStoryState:
            """Create a plot outline for the story"""
            try:
                # Create plot outline prompt
                characters_summary = "\n".join([
                    f"- {char.name}: {char.role} - {char.personality}"
                    for char in state.characters
                ])
                
                prompt = f"""
                Create a compelling plot outline for a {state.length} {state.genre} story:
                
                Theme: {state.theme or state.prompt}
                Characters:
                {characters_summary}
                
                Target Audience: {state.target_audience}
                Tone: {state.tone}
                
                Create 5-7 plot points that include:
                1. Introduction/setup
                2. Inciting incident
                3. Rising action (2-3 points)
                4. Climax
                5. Resolution
                
                Each plot point should have:
                - Clear description
                - Conflict or challenge
                - Potential resolution
                
                Ensure logical flow and appropriate pacing for the story length.
                Respond with JSON array of plot points.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                try:
                    plot_data = json.loads(response.content)
                    plot_points = []
                    
                    for i, point_data in enumerate(plot_data):
                        plot_point = PlotPoint(
                            sequence=i + 1,
                            description=point_data.get("description", ""),
                            conflict=point_data.get("conflict"),
                            resolution=point_data.get("resolution")
                        )
                        plot_points.append(plot_point)
                    
                    state.plot_outline = plot_points
                    state.messages.append(SystemMessage(content=f"Created plot outline with {len(plot_points)} points"))
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback plot structure
                    fallback_plot = [
                        PlotPoint(
                            sequence=1,
                            description="Introduction of characters and setting",
                            conflict="Establishing the initial situation",
                            resolution="Characters are introduced to the story world"
                        ),
                        PlotPoint(
                            sequence=2,
                            description="Main conflict emerges",
                            conflict="Central challenge is presented",
                            resolution="Characters must decide how to respond"
                        ),
                        PlotPoint(
                            sequence=3,
                            description="Climax and resolution",
                            conflict="Final confrontation",
                            resolution="Story reaches its conclusion"
                        )
                    ]
                    state.plot_outline = fallback_plot
                    state.messages.append(SystemMessage(content="Plot outline created (fallback)"))
                
            except Exception as e:
                state.error = f"Error creating plot outline: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def generate_setting(state: CreativeStoryState) -> CreativeStoryState:
            """Generate an appropriate setting for the story"""
            try:
                # Get genre-specific settings for inspiration
                genre_elements = self.genre_elements.get(state.genre, {})
                possible_settings = genre_elements.get("settings", [])
                
                # Create setting prompt
                settings_list = ", ".join(possible_settings) if possible_settings else "various imaginative settings"
                
                prompt = f"""
                Create an immersive setting for a {state.genre} story:
                
                Theme: {state.theme or state.prompt}
                Style: {state.style}
                Target Audience: {state.target_audience}
                Tone: {state.tone}
                
                Genre inspiration settings: {settings_list}
                
                Provide:
                1. Primary location (time and place)
                2. Atmosphere and mood
                3. Key environmental features
                4. Sensory details (sights, sounds, smells)
                5. How setting influences the story
                
                Make the setting vivid and appropriate for the genre.
                Respond with a descriptive paragraph (200-300 words).
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                state.setting = response.content
                state.messages.append(SystemMessage(content="Story setting generated"))
                
            except Exception as e:
                state.error = f"Error generating setting: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def write_story(state: CreativeStoryState) -> CreativeStoryState:
            """Write the complete story based on all elements"""
            try:
                # Create story writing prompt
                characters_summary = "\n".join([
                    f"{char.name}: {char.role} - {char.personality}"
                    for char in state.characters
                ])
                
                plot_summary = "\n".join([
                    f"{point.sequence}. {point.description}"
                    for point in state.plot_outline
                ])
                
                # Determine target word count based on length
                word_counts = {
                    "short": "500-1000",
                    "medium": "1500-2500", 
                    "long": "3000-5000"
                }
                target_words = word_counts.get(state.length, "1500-2500")
                
                prompt = f"""
                Write a compelling {state.genre} story incorporating the following elements:
                
                Original Prompt: {state.prompt}
                Theme: {state.theme}
                Setting: {state.setting}
                
                Characters:
                {characters_summary}
                
                Plot Outline:
                {plot_summary}
                
                Style: {state.style}
                Target Audience: {state.target_audience}
                Tone: {state.tone}
                Target Length: {target_words} words
                
                Requirements:
                1. Engaging opening that hooks the reader
                2. Smooth character development
                3. Follow the plot outline naturally
                4. Vivid descriptions that match the {state.style} style
                5. Appropriate dialogue for the {state.target_audience} audience
                6. Satisfying conclusion that resolves the main conflict
                7. Maintain consistent {state.tone} tone throughout
                
                Write the complete story with a clear beginning, middle, and end.
                Also suggest 3-5 appropriate titles for this story.
                """
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                # Extract title suggestions (usually at the end)
                content_parts = response.content.split("Title Suggestions:")
                story_content = content_parts[0].strip()
                
                # Parse title suggestions
                title_suggestions = []
                if len(content_parts) > 1:
                    titles_text = content_parts[1].strip()
                    title_lines = [line.strip() for line in titles_text.split('\n') if line.strip()]
                    title_suggestions = [line.lstrip('- ').strip() for line in title_lines[:5]]
                
                state.story_content = story_content
                state.title = title_suggestions[0] if title_suggestions else None
                state.title_suggestions = title_suggestions
                state.messages.append(SystemMessage(content="Story writing completed"))
                
            except Exception as e:
                state.error = f"Error writing story: {str(e)}"
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        def finalize_story(state: CreativeStoryState) -> CreativeStoryState:
            """Finalize the story and validate quality"""
            try:
                # Basic validation
                if not state.story_content:
                    state.error = "No story content generated"
                    state.success = False
                    return state
                
                # Count words (rough estimate)
                word_count = len(state.story_content.split())
                
                # Validate against target length
                length_requirements = {
                    "short": (300, 1200),
                    "medium": (1200, 3000),
                    "long": (2500, 6000)
                }
                
                min_words, max_words = length_requirements.get(state.length, (1200, 3000))
                length_appropriate = min_words <= word_count <= max_words
                
                state.word_count = word_count
                state.length_appropriate = length_appropriate
                state.success = True
                
                validation_msg = f"Story finalized - {word_count} words"
                if not length_appropriate:
                    validation_msg += f" (target: {min_words}-{max_words})"
                
                state.messages.append(SystemMessage(content=validation_msg))
                
            except Exception as e:
                state.error = f"Error finalizing story: {str(e)}"
                state.success = False
                state.messages.append(SystemMessage(content=state.error))
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("analyze_prompt", analyze_prompt)
        workflow.add_node("develop_characters", develop_characters)
        workflow.add_node("create_plot_outline", create_plot_outline)
        workflow.add_node("generate_setting", generate_setting)
        workflow.add_node("write_story", write_story)
        workflow.add_node("finalize_story", finalize_story)
        
        # Set up workflow
        workflow.set_entry_point("analyze_prompt")
        workflow.add_edge("analyze_prompt", "develop_characters")
        workflow.add_edge("develop_characters", "create_plot_outline")
        workflow.add_edge("create_plot_outline", "generate_setting")
        workflow.add_edge("generate_setting", "write_story")
        workflow.add_edge("write_story", "finalize_story")
        workflow.add_edge("finalize_story", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def generate_story(
        self,
        prompt: str,
        genre: str = "fantasy",
        style: str = "descriptive",
        length: str = "medium",
        target_audience: str = "general",
        tone: str = "neutral",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a creative story.
        
        Args:
            prompt: Story prompt or theme
            genre: Story genre (fantasy, sci-fi, mystery, romance, horror, etc.)
            style: Writing style (descriptive, minimal, poetic, etc.)
            length: Story length (short, medium, long)
            target_audience: Target audience (children, teens, adults, etc.)
            tone: Story tone (humorous, serious, dark, etc.)
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing the generated story and metadata
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize CreativeStoryAgent workflow")
        
        try:
            # Create initial state
            state = CreativeStoryState(
                prompt=prompt,
                genre=genre,
                style=style,
                length=length,
                target_audience=target_audience,
                tone=tone,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "story": result.story_content,
                "title": result.title,
                "title_suggestions": getattr(result, 'title_suggestions', []),
                "characters": [char.dict() for char in result.characters],
                "plot_outline": [point.dict() for point in result.plot_outline],
                "setting": result.setting,
                "theme": result.theme,
                "word_count": getattr(result, 'word_count', 0),
                "length_appropriate": getattr(result, 'length_appropriate', False),
                "genre": result.genre,
                "style": result.style,
                "target_audience": result.target_audience,
                "tone": result.tone,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in story generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "story": None,
                "characters": [],
            }
    
    async def stream_generate_story(
        self,
        prompt: str,
        genre: str = "fantasy",
        style: str = "descriptive",
        length: str = "medium",
        target_audience: str = "general",
        tone: str = "neutral",
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream story generation process for real-time updates.
        
        Args:
            prompt: Story prompt or theme
            genre: Story genre
            style: Writing style
            length: Story length
            target_audience: Target audience
            tone: Story tone
            thread_id: Thread ID for conversation tracking
            
        Yields:
            Dictionary containing intermediate results and updates
        """
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                yield {
                    "type": "error",
                    "error": "Failed to initialize CreativeStoryAgent workflow",
                    "success": False,
                }
                return
        
        try:
            # Create initial state
            state = CreativeStoryState(
                prompt=prompt,
                genre=genre,
                style=style,
                length=length,
                target_audience=target_audience,
                tone=tone,
            )
            
            # Stream workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            
            async for event in self.workflow.astream(state, config=config):
                # Yield character development updates
                if "develop_characters" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'characters') and node_state.characters:
                        yield {
                            "type": "characters_developed",
                            "characters": [char.dict() for char in node_state.characters],
                            "count": len(node_state.characters),
                        }
                
                # Yield plot outline updates
                if "create_plot_outline" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'plot_outline') and node_state.plot_outline:
                        yield {
                            "type": "plot_outline_created",
                            "plot_outline": [point.dict() for point in node_state.plot_outline],
                            "points": len(node_state.plot_outline),
                        }
                
                # Yield setting generation updates
                if "generate_setting" in event:
                    node_state = list(event.values())[0]
                    if hasattr(node_state, 'setting') and node_state.setting:
                        yield {
                            "type": "setting_generated",
                            "setting": node_state.setting,
                        }
                
                # Yield final result
                if "__end__" in event:
                    final_state = list(event.values())[0]
                    yield {
                        "type": "story_complete",
                        "success": final_state.success,
                        "story": final_state.story_content,
                        "title": final_state.title,
                        "title_suggestions": getattr(final_state, 'title_suggestions', []),
                        "characters": [char.dict() for char in final_state.characters],
                        "plot_outline": [point.dict() for point in final_state.plot_outline],
                        "setting": final_state.setting,
                        "theme": final_state.theme,
                        "word_count": getattr(final_state, 'word_count', 0),
                        "length_appropriate": getattr(final_state, 'length_appropriate', False),
                    }
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming story generation: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "success": False,
            }