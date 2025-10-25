"""
Unit tests for LangChain specialized agents.

This module tests the specialized agents that were implemented as part of the
LangChain integration, including their workflows, tool usage, and integration
with the LangChain ecosystem.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from app.core.langchain.specialized_agents import (
    SummarizeAgent,
    WebdriverAgent,
    ScraperAgent,
    SearchQueryAgent,
    ChainOfThoughtAgent,
    CreativeStoryAgent,
    ToolSelectionAgent,
    SemanticUnderstandingAgent,
    FactCheckerAgent
)
from app.core.langchain.llm_manager import LangChainLLMManager
from app.core.langchain.tool_registry import LangChainToolRegistry


class TestSummarizeAgent:
    """Test cases for SummarizeAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "This is a summary of the text."
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        return Mock(spec=LangChainToolRegistry)
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create SummarizeAgent instance"""
        return SummarizeAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_summarize_short_text(self, agent):
        """Test summarizing short text"""
        result = await agent.summarize("This is a short text that needs to be summarized.")
        
        assert "summary" in result
        assert "This is a summary of the text." in result["summary"]
        assert "word_count" in result
        assert "original_length" in result
        assert "compression_ratio" in result
    
    async def test_summarize_long_text(self, agent):
        """Test summarizing long text"""
        long_text = "This is a very long text. " * 100
        result = await agent.summarize(long_text)
        
        assert "summary" in result
        assert "word_count" in result
        assert "original_length" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] < 1.0
    
    async def test_summarize_with_custom_length(self, agent):
        """Test summarizing with custom target length"""
        text = "This is a text that needs to be summarized to a specific length."
        result = await agent.summarize(text, target_length=10)
        
        assert "summary" in result
        assert "target_length" in result
        assert result["target_length"] == 10
    
    async def test_summarize_empty_text(self, agent):
        """Test summarizing empty text"""
        result = await agent.summarize("")
        
        assert "summary" in result
        assert result["summary"] == ""
        assert "word_count" in result
        assert result["word_count"] == 0


class TestWebdriverAgent:
    """Test cases for WebdriverAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock(spec=LangChainToolRegistry)
        mock_tool = AsyncMock()
        mock_tool.return_value = {"success": True, "result": "Page loaded successfully"}
        registry.execute_tool.return_value = mock_tool
        return registry
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create WebdriverAgent instance"""
        return WebdriverAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_navigate_to_url(self, agent):
        """Test navigating to a URL"""
        result = await agent.navigate("https://example.com")
        
        assert "success" in result
        assert result["success"] is True
        assert "url" in result
        assert result["url"] == "https://example.com"
    
    async def test_click_element(self, agent):
        """Test clicking an element"""
        result = await agent.click("button_id")
        
        assert "success" in result
        assert "element" in result
        assert result["element"] == "button_id"
    
    async def test_type_text(self, agent):
        """Test typing text into an element"""
        result = await agent.type("input_id", "Hello, world!")
        
        assert "success" in result
        assert "element" in result
        assert result["element"] == "input_id"
        assert "text" in result
        assert result["text"] == "Hello, world!"
    
    async def test_extract_text(self, agent):
        """Test extracting text from page"""
        result = await agent.extract_text()
        
        assert "text" in result
        assert "length" in result
    
    async def test_take_screenshot(self, agent):
        """Test taking a screenshot"""
        result = await agent.screenshot()
        
        assert "screenshot_path" in result
        assert "timestamp" in result


class TestScraperAgent:
    """Test cases for ScraperAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock(spec=LangChainToolRegistry)
        mock_tool = AsyncMock()
        mock_tool.return_value = {
            "success": True,
            "content": "<html><body><h1>Page Title</h1><p>Page content</p></body></html>",
            "links": ["https://example.com/page1", "https://example.com/page2"]
        }
        registry.execute_tool.return_value = mock_tool
        return registry
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create ScraperAgent instance"""
        return ScraperAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_scrape_page(self, agent):
        """Test scraping a web page"""
        result = await agent.scrape("https://example.com")
        
        assert "success" in result
        assert result["success"] is True
        assert "content" in result
        assert "links" in result
        assert "metadata" in result
    
    async def test_scrape_with_selector(self, agent):
        """Test scraping with CSS selector"""
        result = await agent.scrape("https://example.com", selector="h1")
        
        assert "success" in result
        assert "content" in result
        assert "selector" in result
        assert result["selector"] == "h1"
    
    async def test_extract_links(self, agent):
        """Test extracting links from page"""
        result = await agent.extract_links("https://example.com")
        
        assert "links" in result
        assert isinstance(result["links"], list)
        assert len(result["links"]) >= 0
    
    async def test_scrape_multiple_pages(self, agent):
        """Test scraping multiple pages"""
        urls = ["https://example.com/page1", "https://example.com/page2"]
        result = await agent.scrape_multiple(urls)
        
        assert "results" in result
        assert "total_pages" in result
        assert "successful_pages" in result
        assert len(result["results"]) == len(urls)


class TestSearchQueryAgent:
    """Test cases for SearchQueryAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "best search query for the topic"
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        return Mock(spec=LangChainToolRegistry)
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create SearchQueryAgent instance"""
        return SearchQueryAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_generate_search_query(self, agent):
        """Test generating search query"""
        result = await agent.generate_query("machine learning applications")
        
        assert "query" in result
        assert "keywords" in result
        assert "intent" in result
        assert "original_topic" in result
        assert result["original_topic"] == "machine learning applications"
    
    async def test_generate_multiple_queries(self, agent):
        """Test generating multiple search queries"""
        result = await agent.generate_multiple_queries("climate change", count=3)
        
        assert "queries" in result
        assert len(result["queries"]) == 3
        assert "primary_query" in result
        assert "alternative_queries" in result
    
    async def test_optimize_query(self, agent):
        """Test optimizing existing query"""
        result = await agent.optimize_query("ml applications", "machine learning")
        
        assert "optimized_query" in result
        assert "original_query" in result
        assert "improvements" in result
    
    async def test_analyze_query_intent(self, agent):
        """Test analyzing query intent"""
        result = await agent.analyze_intent("how to learn python")
        
        assert "intent" in result
        assert "confidence" in result
        assert "suggested_improvements" in result


class TestChainOfThoughtAgent:
    """Test cases for ChainOfThoughtAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "Step 1: Analyze the problem\nStep 2: Consider solutions\nStep 3: Choose best approach\nFinal answer: The solution is..."
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        return Mock(spec=LangChainToolRegistry)
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create ChainOfThoughtAgent instance"""
        return ChainOfThoughtAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_solve_problem(self, agent):
        """Test solving a problem with chain of thought"""
        result = await agent.solve("What is the capital of France?")
        
        assert "answer" in result
        assert "reasoning_steps" in result
        assert "confidence" in result
        assert "final_answer" in result
    
    async def test_solve_with_constraints(self, agent):
        """Test solving with constraints"""
        constraints = ["Must be accurate", "Consider historical context"]
        result = await agent.solve("When was the Eiffel Tower built?", constraints=constraints)
        
        assert "answer" in result
        assert "constraints" in result
        assert len(result["constraints"]) == 2
    
    async def test_step_by_step_reasoning(self, agent):
        """Test step-by-step reasoning"""
        result = await agent.step_by_step("How do you calculate the area of a circle?")
        
        assert "steps" in result
        assert "final_result" in result
        assert "verification" in result
        assert len(result["steps"]) >= 1
    
    async def test_verify_reasoning(self, agent):
        """Test reasoning verification"""
        reasoning = "Step 1: Identify the formula\nStep 2: Apply the values\nStep 3: Calculate the result"
        result = await agent.verify_reasoning(reasoning, "What is 2+2?")
        
        assert "is_valid" in result
        assert "feedback" in result
        assert "suggestions" in result


class TestCreativeStoryAgent:
    """Test cases for CreativeStoryAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "Once upon a time, in a land far away, there lived a brave adventurer..."
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        return Mock(spec=LangChainToolRegistry)
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create CreativeStoryAgent instance"""
        return CreativeStoryAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_generate_story(self, agent):
        """Test generating a story"""
        result = await agent.generate("A magical forest", "fantasy")
        
        assert "story" in result
        assert "genre" in result
        assert result["genre"] == "fantasy"
        assert "word_count" in result
        assert "characters" in result
    
    async def test_continue_story(self, agent):
        """Test continuing an existing story"""
        existing_story = "Once upon a time, there was a brave knight..."
        result = await agent.continue(existing_story, "adventure")
        
        assert "continued_story" in result
        assert "original_length" in result
        assert "new_length" in result
        assert "continuation_point" in result
    
    async def test_generate_story_with_elements(self, agent):
        """Test generating story with specific elements"""
        elements = {
            "character": "A wise wizard",
            "setting": "An ancient library",
            "conflict": "A mysterious curse"
        }
        result = await agent.generate_with_elements(elements)
        
        assert "story" in result
        assert "elements_used" in result
        assert "character" in result["elements_used"]
        assert "setting" in result["elements_used"]
        assert "conflict" in result["elements_used"]
    
    async def test_create_story_outline(self, agent):
        """Test creating story outline"""
        result = await agent.create_outline("A sci-fi adventure", 3)
        
        assert "outline" in result
        assert "chapters" in result
        assert len(result["chapters"]) == 3
        assert "theme" in result
        assert "protagonist" in result


class TestToolSelectionAgent:
    """Test cases for ToolSelectionAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "search_tool, scraper_tool"
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock(spec=LangChainToolRegistry)
        registry.list_tools.return_value = [
            {"name": "search_tool", "description": "Search the web"},
            {"name": "scraper_tool", "description": "Scrape web pages"},
            {"name": "calculator_tool", "description": "Perform calculations"}
        ]
        return registry
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create ToolSelectionAgent instance"""
        return ToolSelectionAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_select_tools_for_task(self, agent):
        """Test selecting tools for a task"""
        result = await agent.select_tools("Research information about climate change")
        
        assert "selected_tools" in result
        assert "reasoning" in result
        assert "confidence" in result
        assert len(result["selected_tools"]) >= 1
    
    async def test_select_tools_with_constraints(self, agent):
        """Test selecting tools with constraints"""
        constraints = ["Must be fast", "Prefer free tools"]
        result = await agent.select_tools("Calculate compound interest", constraints=constraints)
        
        assert "selected_tools" in result
        assert "constraints" in result
        assert len(result["constraints"]) == 2
    
    async def test_rank_tools_by_suitability(self, agent):
        """Test ranking tools by suitability"""
        tools = ["search_tool", "scraper_tool", "calculator_tool"]
        result = await agent.rank_tools(tools, "Find information about quantum computing")
        
        assert "ranked_tools" in result
        assert "scores" in result
        assert len(result["ranked_tools"]) == 3
        assert result["ranked_tools"][0] in tools
    
    async def test_explain_tool_selection(self, agent):
        """Test explaining tool selection"""
        selected_tools = ["search_tool", "scraper_tool"]
        result = await agent.explain_selection(selected_tools, "Research AI trends")
        
        assert "explanation" in result
        assert "tool_benefits" in result
        assert "workflow" in result
        assert len(result["tool_benefits"]) == len(selected_tools)


class TestSemanticUnderstandingAgent:
    """Test cases for SemanticUnderstandingAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "positive"
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        return Mock(spec=LangChainToolRegistry)
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create SemanticUnderstandingAgent instance"""
        return SemanticUnderstandingAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_analyze_sentiment(self, agent):
        """Test sentiment analysis"""
        result = await agent.analyze_sentiment("I love this product! It's amazing.")
        
        assert "sentiment" in result
        assert "confidence" in result
        assert "emotions" in result
        assert "intensity" in result
    
    async def test_extract_entities(self, agent):
        """Test entity extraction"""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        result = await agent.extract_entities(text)
        
        assert "entities" in result
        assert "entity_types" in result
        assert "confidence_scores" in result
        assert len(result["entities"]) >= 1
    
    async def test_classify_text(self, agent):
        """Test text classification"""
        categories = ["technology", "sports", "politics", "entertainment"]
        result = await agent.classify("The latest iPhone features a powerful A15 chip", categories)
        
        assert "category" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert result["category"] in categories
    
    async def test_understand_intent(self, agent):
        """Test intent understanding"""
        result = await agent.understand_intent("Book a flight to New York for tomorrow")
        
        assert "intent" in result
        assert "entities" in result
        assert "confidence" in result
        assert "action_items" in result


class TestFactCheckerAgent:
    """Test cases for FactCheckerAgent"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager"""
        manager = Mock(spec=LangChainLLMManager)
        llm = AsyncMock()
        llm.ainvoke.return_value.content = "The claim is partially accurate with some caveats"
        manager.get_llm.return_value = llm
        return manager
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry"""
        registry = Mock(spec=LangChainToolRegistry)
        mock_tool = AsyncMock()
        mock_tool.return_value = {
            "sources": ["https://example.com/source1", "https://example.com/source2"],
            "evidence": ["Evidence supporting the claim", "Evidence contradicting the claim"]
        }
        registry.execute_tool.return_value = mock_tool
        return registry
    
    @pytest.fixture
    def agent(self, mock_llm_manager, mock_tool_registry):
        """Create FactCheckerAgent instance"""
        return FactCheckerAgent(mock_llm_manager, mock_tool_registry)
    
    async def test_check_fact(self, agent):
        """Test checking a fact"""
        result = await agent.check_fact("The Earth is round")
        
        assert "verdict" in result
        assert "confidence" in result
        assert "sources" in result
        assert "evidence" in result
        assert "explanation" in result
    
    async def test_check_fact_with_sources(self, agent):
        """Test checking a fact with provided sources"""
        sources = ["https://nasa.gov/earth", "https://scientific-journal.com/earth-shape"]
        result = await agent.check_fact("The Earth orbits the Sun", sources=sources)
        
        assert "verdict" in result
        assert "provided_sources" in result
        assert "additional_sources" in result
        assert len(result["provided_sources"]) == 2
    
    async def test_verify_multiple_claims(self, agent):
        """Test verifying multiple claims"""
        claims = [
            "Water boils at 100°C at sea level",
            "The Moon is made of cheese",
            "Humans have 5 senses"
        ]
        result = await agent.verify_multiple(claims)
        
        assert "results" in result
        assert "summary" in result
        assert "overall_accuracy" in result
        assert len(result["results"]) == 3
    
    async def test_assess_source_reliability(self, agent):
        """Test assessing source reliability"""
        source = "https://reputable-journal.com/scientific-article"
        result = await agent.assess_source(source)
        
        assert "reliability_score" in result
        assert "factors" in result
        assert "recommendation" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__])