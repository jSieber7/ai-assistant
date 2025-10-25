"""
Unit tests for LangChain integration components.

This module tests the compatibility layers between our tool system
and LangChain agents and tools.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import the components we're testing
from app.core.tools.integration.langchain_integration import (
    LangChainToolWrapper,
    LangChainToolkit,
    ToolOutputParser,
    create_agent_with_tools,
    get_tool_descriptions,
    tool_selection_prompt,
    get_langchain_tools,
    is_tool_available,
    get_tool_registry,
)


class TestLangChainToolWrapper:
    """Test cases for LangChainToolWrapper class."""

    @pytest.fixture
    def mock_base_tool(self):
        """Create a mock BaseTool for testing."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "A test tool for testing"
        tool.execute_with_timeout = AsyncMock()
        return tool

    @pytest.fixture
    def langchain_tool_wrapper(self, mock_base_tool):
        """Create a LangChainToolWrapper instance for testing."""
        return LangChainToolWrapper(mock_base_tool)

    def test_init(self, langchain_tool_wrapper, mock_base_tool):
        """Test LangChainToolWrapper initialization."""
        assert langchain_tool_wrapper._tool == mock_base_tool
        assert langchain_tool_wrapper.name == "test_tool"
        assert langchain_tool_wrapper.description == "A test tool for testing"

    @pytest.mark.asyncio
    async def test_arun_success(self, langchain_tool_wrapper, mock_base_tool):
        """Test successful async tool execution."""
        # Mock the tool result
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Tool executed successfully"
        mock_base_tool.execute_with_timeout.return_value = mock_result

        # Execute the tool
        result = await langchain_tool_wrapper._arun(test_param="test_value")

        # Verify the result
        assert result == "Tool executed successfully"
        mock_base_tool.execute_with_timeout.assert_called_once_with(test_param="test_value")

    @pytest.mark.asyncio
    async def test_arun_failure(self, langchain_tool_wrapper, mock_base_tool):
        """Test async tool execution with failure."""
        # Mock the tool result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Tool execution failed"
        mock_base_tool.execute_with_timeout.return_value = mock_result

        # Execute the tool
        result = await langchain_tool_wrapper._arun(test_param="test_value")

        # Verify the result
        assert result == "Tool execution failed"
        mock_base_tool.execute_with_timeout.assert_called_once_with(test_param="test_value")

    def test_run_sync(self, langchain_tool_wrapper, mock_base_tool):
        """Test synchronous tool execution."""
        # Mock the tool result
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Tool executed successfully"
        mock_base_tool.execute_with_timeout.return_value = mock_result

        # Execute the tool
        result = langchain_tool_wrapper._run(test_param="test_value")

        # Verify the result
        assert result == "Tool executed successfully"
        mock_base_tool.execute_with_timeout.assert_called_once_with(test_param="test_value")


class TestLangChainToolkit:
    """Test cases for LangChainToolkit class."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock ToolRegistry for testing."""
        registry = Mock()
        registry.list_tools.return_value = []
        return registry

    @pytest.fixture
    def langchain_toolkit(self, mock_tool_registry):
        """Create a LangChainToolkit instance for testing."""
        return LangChainToolkit(mock_tool_registry)

    def test_init_with_registry(self, langchain_toolkit, mock_tool_registry):
        """Test LangChainToolkit initialization with registry."""
        assert langchain_toolkit.registry == mock_tool_registry

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    def test_init_without_registry(self, mock_get_tool_registry):
        """Test LangChainToolkit initialization without registry."""
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        toolkit = LangChainToolkit()

        assert toolkit.registry == mock_registry
        mock_get_tool_registry.assert_called_once()

    def test_get_tools_empty(self, langchain_toolkit, mock_tool_registry):
        """Test getting tools when registry is empty."""
        mock_tool_registry.list_tools.return_value = []

        tools = langchain_toolkit.get_tools()

        assert tools == []
        mock_tool_registry.list_tools.assert_called_once_with(enabled_only=True)

    def test_get_tools_with_enabled_tools(self, langchain_toolkit, mock_tool_registry):
        """Test getting tools with enabled tools."""
        # Create mock tools
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool1.enabled = True

        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Second tool"
        mock_tool2.enabled = True

        mock_tool_registry.list_tools.return_value = [mock_tool1, mock_tool2]

        # Mock the LangChainToolWrapper
        with patch("app.core.tools.integration.langchain_integration.LangChainToolWrapper") as mock_wrapper:
            mock_wrapper_instance1 = Mock()
            mock_wrapper_instance2 = Mock()
            mock_wrapper.side_effect = [mock_wrapper_instance1, mock_wrapper_instance2]

            tools = langchain_toolkit.get_tools()

            assert len(tools) == 2
            assert tools[0] == mock_wrapper_instance1
            assert tools[1] == mock_wrapper_instance2

            # Verify wrapper was called correctly
            assert mock_wrapper.call_count == 2
            mock_wrapper.assert_any_call(mock_tool1)
            mock_wrapper.assert_any_call(mock_tool2)

    def test_get_tools_with_disabled_tools(self, langchain_toolkit, mock_tool_registry):
        """Test getting tools with disabled tools."""
        # Create mock tools
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.enabled = False

        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.enabled = True

        mock_tool_registry.list_tools.return_value = [mock_tool1, mock_tool2]

        # Mock the LangChainToolWrapper
        with patch("app.core.tools.integration.langchain_integration.LangChainToolWrapper") as mock_wrapper:
            mock_wrapper_instance = Mock()
            mock_wrapper.return_value = mock_wrapper_instance

            tools = langchain_toolkit.get_tools()

            assert len(tools) == 1
            assert tools[0] == mock_wrapper_instance

            # Verify wrapper was called only for enabled tool
            mock_wrapper.assert_called_once_with(mock_tool2)

    def test_get_tools_with_conversion_error(self, langchain_toolkit, mock_tool_registry, caplog):
        """Test getting tools when conversion fails for some tools."""
        # Create mock tools
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.enabled = True

        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.enabled = True

        mock_tool_registry.list_tools.return_value = [mock_tool1, mock_tool2]

        # Mock the LangChainToolWrapper with exception for second tool
        with patch("app.core.tools.integration.langchain_integration.LangChainToolWrapper") as mock_wrapper:
            mock_wrapper_instance = Mock()
            mock_wrapper.side_effect = [mock_wrapper_instance, Exception("Conversion failed")]

            tools = langchain_toolkit.get_tools()

            # Should still return the successfully converted tool
            assert len(tools) == 1
            assert tools[0] == mock_wrapper_instance

            # Check that warning was logged
            assert "Failed to convert tool tool2 to LangChain" in caplog.text

    def test_from_registry(self):
        """Test creating toolkit from registry."""
        mock_registry = Mock()
        toolkit = LangChainToolkit.from_registry(mock_registry)

        assert toolkit.registry == mock_registry


class TestToolOutputParser:
    """Test cases for ToolOutputParser class."""

    @pytest.fixture
    def output_parser(self):
        """Create a ToolOutputParser instance for testing."""
        return ToolOutputParser()

    def test_parse(self, output_parser):
        """Test parsing tool output."""
        text = "This is the tool output"
        result = output_parser.parse(text)

        assert result == {"output": "This is the tool output"}

    def test_type_property(self, output_parser):
        """Test the _type property."""
        assert output_parser._type == "tool_output_parser"


class TestCreateAgentWithTools:
    """Test cases for create_agent_with_tools function."""

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    @patch("app.core.tools.integration.langchain_integration.create_agent")
    def test_create_agent_with_tools_available(self, mock_create_agent, mock_get_tool_registry):
        """Test creating an agent with available tools."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.enabled = True

        mock_registry.list_tools.return_value = [mock_tool]

        mock_langchain_tool = Mock()
        mock_langchain_tool.name = "test_tool"

        with patch("app.core.tools.integration.langchain_integration.LangChainToolkit") as mock_toolkit:
            mock_toolkit_instance = Mock()
            mock_toolkit_instance.get_tools.return_value = [mock_langchain_tool]
            mock_toolkit.return_value = mock_toolkit_instance

            mock_llm = Mock()
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            # Call the function
            result = create_agent_with_tools(mock_llm, mock_registry)

            # Verify the result
            assert result == mock_agent

            # Verify toolkit was created and used
            mock_toolkit.assert_called_once_with(mock_registry)
            mock_toolkit_instance.get_tools.assert_called_once()

            # Verify agent was created with correct parameters
            mock_create_agent.assert_called_once()
            args, kwargs = mock_create_agent.call_args
            assert kwargs["model"] == mock_llm
            assert kwargs["tools"] == [mock_langchain_tool]
            assert "test_tool" in kwargs["system_prompt"]
            assert kwargs["debug"] is True

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    @patch("app.core.tools.integration.langchain_integration.create_agent")
    def test_create_agent_no_tools_available(self, mock_create_agent, mock_get_tool_registry):
        """Test creating an agent with no available tools."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry
        mock_registry.list_tools.return_value = []

        with patch("app.core.tools.integration.langchain_integration.LangChainToolkit") as mock_toolkit:
            mock_toolkit_instance = Mock()
            mock_toolkit_instance.get_tools.return_value = []
            mock_toolkit.return_value = mock_toolkit_instance

            mock_llm = Mock()
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            # Call the function
            result = create_agent_with_tools(mock_llm, mock_registry)

            # Verify the result
            assert result == mock_agent

            # Verify agent was created with no tools
            mock_create_agent.assert_called_once()
            args, kwargs = mock_create_agent.call_args
            assert kwargs["model"] == mock_llm
            assert kwargs["tools"] == []
            assert "You are a helpful AI assistant" in kwargs["system_prompt"]
            assert kwargs["debug"] is True


class TestToolDescriptions:
    """Test cases for tool description functions."""

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    def test_get_tool_descriptions(self, mock_get_tool_registry):
        """Test getting tool descriptions."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {"param1": {"type": "string"}, "param2": {"type": "integer"}}
        mock_tool.categories = ["test", "utility"]
        mock_tool.enabled = True

        mock_registry.list_tools.return_value = [mock_tool]

        # Call the function
        descriptions = get_tool_descriptions(mock_registry)

        # Verify the result
        assert len(descriptions) == 1
        assert descriptions[0]["name"] == "test_tool"
        assert descriptions[0]["description"] == "A test tool"
        assert descriptions[0]["parameters"] == {"param1": {"type": "string"}, "param2": {"type": "integer"}}
        assert descriptions[0]["categories"] == ["test", "utility"]

        # Verify registry was called correctly
        mock_registry.list_tools.assert_called_once_with(enabled_only=True)

    def test_tool_selection_prompt_empty(self):
        """Test tool selection prompt with no tools."""
        prompt = tool_selection_prompt([])

        assert prompt == "No tools available."

    def test_tool_selection_prompt_with_tools(self):
        """Test tool selection prompt with available tools."""
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {"query": {"type": "string"}, "limit": {"type": "integer"}}
            },
            {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {"expression": {"type": "string"}}
            }
        ]

        prompt = tool_selection_prompt(tools)

        assert "Available tools:" in prompt
        assert "- search: Search the web [Parameters: query (string), limit (integer)]" in prompt
        assert "- calculate: Perform calculations [Parameters: expression (string)]" in prompt


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    @patch("app.core.tools.integration.langchain_integration.LangChainToolkit")
    def test_get_langchain_tools(self, mock_toolkit, mock_get_tool_registry):
        """Test getting LangChain tools."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        mock_toolkit_instance = Mock()
        mock_langchain_tool = Mock()
        mock_toolkit_instance.get_tools.return_value = [mock_langchain_tool]
        mock_toolkit.return_value = mock_toolkit_instance

        # Call the function
        tools = get_langchain_tools(mock_registry)

        # Verify the result
        assert tools == [mock_langchain_tool]
        mock_toolkit.assert_called_once_with(mock_registry)
        mock_toolkit_instance.get_tools.assert_called_once()

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    def test_is_tool_available_true(self, mock_get_tool_registry):
        """Test checking if a tool is available (true case)."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        mock_tool = Mock()
        mock_tool.enabled = True
        mock_registry.get_tool.return_value = mock_tool

        # Call the function
        result = is_tool_available("test_tool", mock_registry)

        # Verify the result
        assert result is True
        mock_registry.get_tool.assert_called_once_with("test_tool")

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    def test_is_tool_available_false_not_found(self, mock_get_tool_registry):
        """Test checking if a tool is available (not found case)."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry
        mock_registry.get_tool.return_value = None

        # Call the function
        result = is_tool_available("test_tool", mock_registry)

        # Verify the result
        assert result is False
        mock_registry.get_tool.assert_called_once_with("test_tool")

    @patch("app.core.tools.integration.langchain_integration.get_tool_registry")
    def test_is_tool_available_false_disabled(self, mock_get_tool_registry):
        """Test checking if a tool is available (disabled case)."""
        # Setup mocks
        mock_registry = Mock()
        mock_get_tool_registry.return_value = mock_registry

        mock_tool = Mock()
        mock_tool.enabled = False
        mock_registry.get_tool.return_value = mock_tool

        # Call the function
        result = is_tool_available("test_tool", mock_registry)

        # Verify the result
        assert result is False
        mock_registry.get_tool.assert_called_once_with("test_tool")

    @patch("app.core.tools.integration.langchain_integration.tool_registry")
    def test_get_tool_registry(self, mock_tool_registry):
        """Test getting the tool registry."""
        # Call the function
        result = get_tool_registry()

        # Verify the result
        assert result == mock_tool_registry