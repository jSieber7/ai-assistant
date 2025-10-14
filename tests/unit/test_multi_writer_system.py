"""
Unit tests for multi-writer/checker system
"""

import pytest
from unittest.mock import patch
from app.core.agents.writer_agent import WriterAgent, MultiWriterOrchestrator
from app.core.agents.checker_agent import CheckerAgent, MultiCheckerOrchestrator
from app.core.agents.content_processor import ContentProcessor
from app.core.templating.jinja_processor import JinjaProcessor
from app.core.multi_writer_config import (
    MultiWriterSettings,
    validate_multi_writer_config,
)


@pytest.mark.unit
class TestMultiWriterSettings:
    """Test multi-writer configuration settings"""

    def test_default_settings(self):
        """Test default configuration values"""
        settings = MultiWriterSettings()

        assert not settings.enabled  # Disabled by default
        assert settings.quality_threshold == 70.0
        assert settings.max_iterations == 2
        assert settings.template_dir == "templates"
        assert settings.default_template == "article.html.jinja"

    def test_validate_config_disabled(self):
        """Test configuration validation when disabled"""
        settings = MultiWriterSettings(enabled=False)
        with patch("app.core.multi_writer_config.multi_writer_settings", settings):
            issues = validate_multi_writer_config()
            assert "Multi-writer system is disabled" in issues

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key"""
        settings = MultiWriterSettings(enabled=True, firecrawl_api_key=None)
        with patch("app.core.multi_writer_config.multi_writer_settings", settings):
            issues = validate_multi_writer_config()
            assert any("Firecrawl API key not configured" in issue for issue in issues)


@pytest.mark.unit
class TestWriterAgent:
    """Test writer agent functionality"""

    @pytest.fixture
    def writer_agent(self):
        """Create a writer agent for testing"""
        return WriterAgent("test_writer", "technical", "claude-3.5-sonnet")

    def test_writer_properties(self, writer_agent):
        """Test writer agent properties"""
        assert writer_agent.name == "writer_test_writer"
        assert writer_agent.specialty == "technical"
        assert writer_agent.writer_id == "test_writer"

    def test_create_writer_prompt(self, writer_agent):
        """Test writer prompt creation"""
        prompt = writer_agent._create_writer_prompt(None)
        assert "technical writer" in prompt.lower()
        assert "accuracy and clarity" in prompt.lower()

    def test_create_writer_prompt_with_style_guide(self, writer_agent):
        """Test writer prompt creation with style guide"""
        style_guide = {
            "tone": "casual",
            "audience": "developers",
            "length": "short",
            "format": "blog",
        }
        prompt = writer_agent._create_writer_prompt(style_guide)
        assert "tone: casual" in prompt.lower()
        assert "audience: developers" in prompt.lower()

    def test_calculate_confidence(self, writer_agent):
        """Test confidence score calculation"""
        content = "This is a test content with proper structure and source references."
        source_content = {
            "title": "Test Source",
            "key_points": ["Point 1", "Point 2", "Point 3"],
        }
        score = writer_agent._calculate_confidence(content, source_content)
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestCheckerAgent:
    """Test checker agent functionality"""

    @pytest.fixture
    def checker_agent(self):
        """Create a checker agent for testing"""
        return CheckerAgent("test_checker", "factual", "claude-3.5-sonnet")

    def test_checker_properties(self, checker_agent):
        """Test checker agent properties"""
        assert checker_agent.name == "checker_test_checker"
        assert checker_agent.focus_area == "factual"
        assert checker_agent.checker_id == "test_checker"

    def test_get_system_prompt(self, checker_agent):
        """Test system prompt generation"""
        prompt = checker_agent._get_system_prompt()
        assert "fact-checker" in prompt.lower()
        assert "accuracy" in prompt.lower()

    def test_parse_check_result_json(self, checker_agent):
        """Test parsing structured check result"""
        result = '{"score": 85, "issues": [], "improvements": [], "improved_content": "Test", "recommendations": []}'
        parsed = checker_agent._parse_check_result(result)
        assert parsed["score"] == 85
        assert parsed["improved_content"] == "Test"

    def test_parse_check_result_invalid_json(self, checker_agent):
        """Test parsing invalid JSON result"""
        result = "This is not valid JSON"
        parsed = checker_agent._parse_check_result(result)
        assert parsed["score"] == 70  # Default score
        assert parsed["improved_content"] == result


@pytest.mark.unit
class TestMultiWriterOrchestrator:
    """Test multi-writer orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create multi-writer orchestrator for testing"""
        return MultiWriterOrchestrator()

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert len(orchestrator.writers) == 4
        assert "technical_1" in orchestrator.writers
        assert "creative_1" in orchestrator.writers
        assert "analytical_1" in orchestrator.writers


@pytest.mark.unit
class TestMultiCheckerOrchestrator:
    """Test multi-checker orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create multi-checker orchestrator for testing"""
        return MultiCheckerOrchestrator()

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert len(orchestrator.checkers) == 4
        assert "factual_1" in orchestrator.checkers
        assert "style_1" in orchestrator.checkers
        assert "structure_1" in orchestrator.checkers
        assert "seo_1" in orchestrator.checkers

    def test_select_best_version(self, orchestrator):
        """Test selecting best version from checks"""
        checks = [
            {
                "score": 70,
                "improved_content": "Content A",
                "checker_id": "checker1",
                "focus_area": "style",
            },
            {
                "score": 85,
                "improved_content": "Content B",
                "checker_id": "checker2",
                "focus_area": "factual",
            },
            {
                "score": 75,
                "improved_content": "Content C",
                "checker_id": "checker3",
                "focus_area": "structure",
            },
        ]

        best = orchestrator._select_best_version(checks)
        assert best["score"] == 85
        assert best["content"] == "Content B"
        assert best["checker_id"] == "checker2"


@pytest.mark.unit
class TestContentProcessor:
    """Test content processor"""

    @pytest.fixture
    def processor(self):
        """Create content processor for testing"""
        with patch("app.core.agents.content_processor.FirecrawlTool"):
            return ContentProcessor("test_api_key")

    def test_clean_content(self, processor):
        """Test content cleaning"""
        raw_content = {
            "title": "Test Title",
            "markdown": "Test content with **bold** text.",
            "raw": "<p>Test content with <b>bold</b> text.</p>",
            "url": "https://example.com",
            "links": [{"url": "https://example.com"}],
            "images": [{"src": "image.jpg"}],
        }

        cleaned = processor._clean_content(raw_content)
        assert cleaned["title"] == "Test Title"
        assert (
            cleaned["word_count"] == 5
        )  # "Test", "content", "with", "**bold**", "text."
        assert cleaned["url"] == "https://example.com"
        assert len(cleaned["links"]) == 1
        assert len(cleaned["images"]) == 1

    def test_extract_key_points(self, processor):
        """Test key points extraction"""
        content = """
        # Main Title
        
        - First important point
        - Second important point
        * Another key point
        
        Regular text here.
        
        1. Numbered point
        2. Another numbered point
        """

        key_points = processor._extract_key_points(content)
        assert len(key_points) <= 10
        assert any("First important point" in point for point in key_points)


@pytest.mark.unit
class TestJinjaProcessor:
    """Test Jinja processor"""

    @pytest.fixture
    def processor(self, tmp_path):
        """Create Jinja processor for testing"""
        return JinjaProcessor(str(tmp_path))

    def test_wordcount_filter(self, processor):
        """Test word count filter"""
        text = "This is a test text with seven words."
        count = processor._wordcount_filter(text)
        assert (
            count == 8
        )  # "This", "is", "a", "test", "text", "with", "seven", "words."

    def test_reading_time_filter(self, processor):
        """Test reading time filter"""
        text = "word " * 400  # 400 words
        reading_time = processor._reading_time_filter(text)
        assert reading_time == 2  # 400 words / 200 words per minute

    def test_seo_slug_filter(self, processor):
        """Test SEO slug filter"""
        text = "This is a Test Title with Special Characters!"
        slug = processor._seo_slug_filter(text)
        assert slug == "this-is-a-test-title-with-special-characters"

    def test_generate_metadata(self, processor):
        """Test metadata generation"""
        rendered_content = "<h1>Test</h1>\n\n<p>Content with multiple words.</p>\n\n<a href='test'>Link</a>"
        content_data = {
            "original_content": {"writer_id": "test_writer"},
            "overall_score": 85.0,
            "template_used": "test.html.jinja",
        }

        metadata = processor._generate_metadata(rendered_content, content_data)
        assert metadata["word_count"] == 7
        assert metadata["has_headings"]
        assert metadata["has_links"]
        assert metadata["original_writer"] == "test_writer"
        assert metadata["quality_score"] == 85.0
