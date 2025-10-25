"""
Unit tests for Tool Utilities and Examples.

This module tests the tool utilities and example implementations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import os
import tempfile
import shutil

from app.core.tools.utilities.examples import (
    ToolExamples,
    ExampleTool,
    UtilityTool,
    HelperFunction,
    ToolValidator,
    ToolFormatter
)


class TestToolExamples:
    """Test ToolExamples class"""

    @pytest.fixture
    def tool_examples(self):
        """Create a ToolExamples instance"""
        return ToolExamples()

    def test_tool_examples_init(self, tool_examples):
        """Test ToolExamples initialization"""
        assert tool_examples.name == "tool_examples"
        assert tool_examples.description == "Collection of tool examples and utilities"
        assert isinstance(tool_examples.examples, dict)
        assert isinstance(tool_examples.utilities, dict)
        assert len(tool_examples.examples) == 0
        assert len(tool_examples.utilities) == 0

    def test_add_example(self, tool_examples):
        """Test adding a tool example"""
        example = {
            "name": "example_tool",
            "description": "An example tool",
            "code": "def example_function(): pass",
            "usage": "example_tool()"
        }
        
        tool_examples.add_example("example_tool", example)
        
        assert "example_tool" in tool_examples.examples
        assert tool_examples.examples["example_tool"] == example

    def test_add_example_with_duplicate(self, tool_examples):
        """Test adding example with duplicate name"""
        example1 = {"name": "example_tool", "description": "First example"}
        example2 = {"name": "example_tool", "description": "Second example"}
        
        tool_examples.add_example("example_tool", example1)
        
        with pytest.raises(ValueError, match="Example example_tool already exists"):
            tool_examples.add_example("example_tool", example2)

    def test_get_example(self, tool_examples):
        """Test getting a specific example"""
        example = {
            "name": "example_tool",
            "description": "An example tool",
            "code": "def example_function(): pass"
        }
        
        tool_examples.add_example("example_tool", example)
        
        retrieved_example = tool_examples.get_example("example_tool")
        assert retrieved_example == example

    def test_get_nonexistent_example(self, tool_examples):
        """Test getting non-existent example"""
        assert tool_examples.get_example("nonexistent_example") is None

    def test_list_examples(self, tool_examples):
        """Test listing all examples"""
        # Initially empty
        assert tool_examples.list_examples() == []
        
        # Add examples
        tool_examples.add_example("example1", {"name": "example1"})
        tool_examples.add_example("example2", {"name": "example2"})
        
        examples = tool_examples.list_examples()
        assert len(examples) == 2
        assert "example1" in examples
        assert "example2" in examples

    def test_search_examples(self, tool_examples):
        """Test searching examples"""
        # Add examples
        tool_examples.add_example("search_tool", {
            "name": "search_tool",
            "description": "A tool for searching",
            "tags": ["search", "web"]
        })
        tool_examples.add_example("data_tool", {
            "name": "data_tool",
            "description": "A tool for data processing",
            "tags": ["data", "processing"]
        })
        
        # Search by keyword
        results = tool_examples.search_examples("search")
        assert len(results) == 1
        assert "search_tool" in results
        
        # Search by tag
        results = tool_examples.search_examples("data")
        assert len(results) == 1
        assert "data_tool" in results

    def test_add_utility(self, tool_examples):
        """Test adding a utility function"""
        utility = {
            "name": "helper_function",
            "description": "A helper function",
            "code": "def helper_function(): pass",
            "parameters": ["param1", "param2"]
        }
        
        tool_examples.add_utility("helper_function", utility)
        
        assert "helper_function" in tool_examples.utilities
        assert tool_examples.utilities["helper_function"] == utility

    def test_get_utility(self, tool_examples):
        """Test getting a specific utility"""
        utility = {
            "name": "helper_function",
            "description": "A helper function",
            "code": "def helper_function(): pass"
        }
        
        tool_examples.add_utility("helper_function", utility)
        
        retrieved_utility = tool_examples.get_utility("helper_function")
        assert retrieved_utility == utility

    def test_list_utilities(self, tool_examples):
        """Test listing all utilities"""
        # Initially empty
        assert tool_examples.list_utilities() == []
        
        # Add utilities
        tool_examples.add_utility("utility1", {"name": "utility1"})
        tool_examples.add_utility("utility2", {"name": "utility2"})
        
        utilities = tool_examples.list_utilities()
        assert len(utilities) == 2
        assert "utility1" in utilities
        assert "utility2" in utilities

    def test_export_examples(self, tool_examples):
        """Test exporting examples to file"""
        # Add examples
        tool_examples.add_example("example1", {"name": "example1", "description": "First example"})
        tool_examples.add_example("example2", {"name": "example2", "description": "Second example"})
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            tool_examples.export_examples(temp_path)
            
            # Check file was created and contains expected content
            with open(temp_path, 'r') as f:
                content = json.load(f)
            
            assert "examples" in content
            assert len(content["examples"]) == 2
            assert "example1" in content["examples"]
            assert "example2" in content["examples"]
        finally:
            os.unlink(temp_path)

    def test_import_examples(self, tool_examples):
        """Test importing examples from file"""
        # Create test data
        test_data = {
            "examples": {
                "imported_example": {
                    "name": "imported_example",
                    "description": "Imported example"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(test_data, temp_file)
            temp_path = temp_file.name
        
        try:
            tool_examples.import_examples(temp_path)
            
            # Check example was imported
            assert "imported_example" in tool_examples.examples
            assert tool_examples.examples["imported_example"]["description"] == "Imported example"
        finally:
            os.unlink(temp_path)

    def test_should_use_with_examples_keywords(self, tool_examples):
        """Test should_use method with examples keywords"""
        assert tool_examples.should_use("show me an example tool") is True
        assert tool_examples.should_use("give me a utility function") is True
        assert tool_examples.should_use("tool example for searching") is True

    def test_should_use_without_examples_keywords(self, tool_examples):
        """Test should_use method without examples keywords"""
        assert tool_examples.should_use("create a file") is False
        assert tool_examples.should_use("calculate something") is False


class TestExampleTool:
    """Test ExampleTool class"""

    @pytest.fixture
    def example_tool(self):
        """Create an ExampleTool instance"""
        return ExampleTool()

    def test_example_tool_init(self, example_tool):
        """Test ExampleTool initialization"""
        assert example_tool.name == "example_tool"
        assert example_tool.description == "An example tool for demonstration"
        assert isinstance(example_tool.keywords, list)
        assert "example" in example_tool.keywords
        assert example_tool.category == "example"

    def test_example_tool_parameters(self, example_tool):
        """Test ExampleTool parameters"""
        parameters = example_tool.parameters
        
        assert isinstance(parameters, dict)
        assert "message" in parameters
        assert "action" in parameters
        
        # Check message parameter
        message_param = parameters["message"]
        assert message_param["type"] == "string"
        assert message_param["required"] is True
        
        # Check action parameter
        action_param = parameters["action"]
        assert action_param["type"] == "string"
        assert action_param["required"] is False
        assert "echo" in action_param["enum"]
        assert "reverse" in action_param["enum"]
        assert "uppercase" in action_param["enum"]

    @pytest.mark.asyncio
    async def test_example_tool_execute_echo(self, example_tool):
        """Test ExampleTool execution with echo action"""
        result = await example_tool.execute(
            message="Hello, World!",
            action="echo"
        )
        
        assert result.success is True
        assert result.data["result"] == "Hello, World!"
        assert result.data["action"] == "echo"

    @pytest.mark.asyncio
    async def test_example_tool_execute_reverse(self, example_tool):
        """Test ExampleTool execution with reverse action"""
        result = await example_tool.execute(
            message="Hello, World!",
            action="reverse"
        )
        
        assert result.success is True
        assert result.data["result"] == "!dlroW ,olleH"
        assert result.data["action"] == "reverse"

    @pytest.mark.asyncio
    async def test_example_tool_execute_uppercase(self, example_tool):
        """Test ExampleTool execution with uppercase action"""
        result = await example_tool.execute(
            message="Hello, World!",
            action="uppercase"
        )
        
        assert result.success is True
        assert result.data["result"] == "HELLO, WORLD!"
        assert result.data["action"] == "uppercase"

    @pytest.mark.asyncio
    async def test_example_tool_execute_default_action(self, example_tool):
        """Test ExampleTool execution with default action"""
        result = await example_tool.execute(message="Hello, World!")
        
        assert result.success is True
        assert result.data["result"] == "Hello, World!"
        assert result.data["action"] == "echo"  # Default action

    def test_example_tool_validate_parameters(self, example_tool):
        """Test ExampleTool parameter validation"""
        assert example_tool.validate_parameters(message="test") is True
        assert example_tool.validate_parameters(message="test", action="reverse") is True
        assert example_tool.validate_parameters() is False  # Missing required parameter
        assert example_tool.validate_parameters(message="test", action="invalid") is False  # Invalid action


class TestUtilityTool:
    """Test UtilityTool class"""

    @pytest.fixture
    def utility_tool(self):
        """Create a UtilityTool instance"""
        return UtilityTool()

    def test_utility_tool_init(self, utility_tool):
        """Test UtilityTool initialization"""
        assert utility_tool.name == "utility_tool"
        assert utility_tool.description == "A utility tool with helper functions"
        assert isinstance(utility_tool.keywords, list)
        assert "utility" in utility_tool.keywords
        assert utility_tool.category == "utility"

    def test_utility_tool_parameters(self, utility_tool):
        """Test UtilityTool parameters"""
        parameters = utility_tool.parameters
        
        assert isinstance(parameters, dict)
        assert "function" in parameters
        assert "parameters" in parameters
        
        # Check function parameter
        function_param = parameters["function"]
        assert function_param["type"] == "string"
        assert function_param["required"] is True
        assert "format_date" in function_param["enum"]
        assert "generate_id" in function_param["enum"]
        assert "calculate_hash" in function_param["enum"]

    @pytest.mark.asyncio
    async def test_utility_tool_format_date(self, utility_tool):
        """Test UtilityTool date formatting"""
        result = await utility_tool.execute(
            function="format_date",
            parameters={"date": "2023-01-01", "format": "%Y-%m-%d"}
        )
        
        assert result.success is True
        assert "2023-01-01" in result.data["result"]

    @pytest.mark.asyncio
    async def test_utility_tool_generate_id(self, utility_tool):
        """Test UtilityTool ID generation"""
        result = await utility_tool.execute(
            function="generate_id",
            parameters={"prefix": "test", "length": 8}
        )
        
        assert result.success is True
        assert result.data["result"].startswith("test_")
        assert len(result.data["result"]) >= len("test_") + 8

    @pytest.mark.asyncio
    async def test_utility_tool_calculate_hash(self, utility_tool):
        """Test UtilityTool hash calculation"""
        result = await utility_tool.execute(
            function="calculate_hash",
            parameters={"text": "hello world", "algorithm": "md5"}
        )
        
        assert result.success is True
        assert result.data["result"] == "5eb63bbbe01eeed093cb22bb8f5acdc3"


class TestHelperFunction:
    """Test HelperFunction class"""

    @pytest.fixture
    def helper_function(self):
        """Create a HelperFunction instance"""
        return HelperFunction()

    def test_helper_function_init(self, helper_function):
        """Test HelperFunction initialization"""
        assert helper_function.name == "helper_function"
        assert helper_function.description == "Collection of helper functions"
        assert isinstance(helper_function.functions, dict)

    def test_format_text(self, helper_function):
        """Test text formatting helper"""
        # Test uppercase
        result = helper_function.format_text("hello", "uppercase")
        assert result == "HELLO"
        
        # Test lowercase
        result = helper_function.format_text("HELLO", "lowercase")
        assert result == "hello"
        
        # Test title case
        result = helper_function.format_text("hello world", "title")
        assert result == "Hello World"

    def test_validate_email(self, helper_function):
        """Test email validation helper"""
        assert helper_function.validate_email("test@example.com") is True
        assert helper_function.validate_email("invalid-email") is False
        assert helper_function.validate_email("test@.com") is False

    def test_generate_uuid(self, helper_function):
        """Test UUID generation helper"""
        uuid1 = helper_function.generate_uuid()
        uuid2 = helper_function.generate_uuid()
        
        assert uuid1 != uuid2  # Should be unique
        assert len(uuid1) == 36  # Standard UUID length

    def test_parse_json(self, helper_function):
        """Test JSON parsing helper"""
        # Valid JSON
        result = helper_function.parse_json('{"key": "value"}')
        assert result["key"] == "value"
        
        # Invalid JSON
        with pytest.raises(ValueError):
            helper_function.parse_json('invalid json')


class TestToolValidator:
    """Test ToolValidator class"""

    @pytest.fixture
    def tool_validator(self):
        """Create a ToolValidator instance"""
        return ToolValidator()

    def test_tool_validator_init(self, tool_validator):
        """Test ToolValidator initialization"""
        assert tool_validator.name == "tool_validator"
        assert tool_validator.description == "Validates tool configurations and parameters"

    def test_validate_tool_name(self, tool_validator):
        """Test tool name validation"""
        assert tool_validator.validate_tool_name("valid_tool_name") is True
        assert tool_validator.validate_tool_name("invalid-tool-name") is False  # Contains hyphen
        assert tool_validator.validate_tool_name("123tool") is False  # Starts with number
        assert tool_validator.validate_tool_name("") is False  # Empty

    def test_validate_tool_parameters(self, tool_validator):
        """Test tool parameter validation"""
        # Valid parameters
        valid_params = {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search query"
            }
        }
        assert tool_validator.validate_tool_parameters(valid_params) is True
        
        # Invalid parameters (missing type)
        invalid_params = {
            "query": {
                "required": True,
                "description": "Search query"
            }
        }
        assert tool_validator.validate_tool_parameters(invalid_params) is False

    def test_validate_parameter_value(self, tool_validator):
        """Test parameter value validation"""
        # String validation
        assert tool_validator.validate_parameter_value("test", "string") is True
        assert tool_validator.validate_parameter_value(123, "string") is False
        
        # Integer validation
        assert tool_validator.validate_parameter_value(123, "integer") is True
        assert tool_validator.validate_parameter_value("123", "integer") is False
        
        # Boolean validation
        assert tool_validator.validate_parameter_value(True, "boolean") is True
        assert tool_validator.validate_parameter_value("true", "boolean") is False

    def test_validate_tool_dependencies(self, tool_validator):
        """Test tool dependency validation"""
        # Valid dependencies
        valid_deps = ["tool1", "tool2"]
        assert tool_validator.validate_tool_dependencies(valid_deps, ["tool1", "tool2", "tool3"]) is True
        
        # Invalid dependencies (missing tool)
        invalid_deps = ["tool1", "nonexistent_tool"]
        assert tool_validator.validate_tool_dependencies(invalid_deps, ["tool1", "tool2"]) is False


class TestToolFormatter:
    """Test ToolFormatter class"""

    @pytest.fixture
    def tool_formatter(self):
        """Create a ToolFormatter instance"""
        return ToolFormatter()

    def test_tool_formatter_init(self, tool_formatter):
        """Test ToolFormatter initialization"""
        assert tool_formatter.name == "tool_formatter"
        assert tool_formatter.description == "Formats tool outputs and documentation"

    def test_format_tool_description(self, tool_formatter):
        """Test tool description formatting"""
        tool_info = {
            "name": "example_tool",
            "description": "An example tool",
            "parameters": {
                "query": {"type": "string", "required": True, "description": "Search query"}
            }
        }
        
        formatted = tool_formatter.format_tool_description(tool_info)
        
        assert "example_tool" in formatted
        assert "An example tool" in formatted
        assert "query" in formatted
        assert "Search query" in formatted

    def test_format_tool_result(self, tool_formatter):
        """Test tool result formatting"""
        result = {
            "success": True,
            "data": {"result": "Operation completed"},
            "message": "Success",
            "execution_time": 0.5
        }
        
        formatted = tool_formatter.format_tool_result(result)
        
        assert "Success" in formatted
        assert "Operation completed" in formatted
        assert "0.5" in formatted

    def test_format_error_message(self, tool_formatter):
        """Test error message formatting"""
        error = {
            "type": "ValueError",
            "message": "Invalid parameter",
            "traceback": "Traceback..."
        }
        
        formatted = tool_formatter.format_error_message(error)
        
        assert "ValueError" in formatted
        assert "Invalid parameter" in formatted

    def test_format_parameter_list(self, tool_formatter):
        """Test parameter list formatting"""
        parameters = {
            "query": {"type": "string", "required": True, "description": "Search query"},
            "limit": {"type": "integer", "required": False, "description": "Result limit"}
        }
        
        formatted = tool_formatter.format_parameter_list(parameters)
        
        assert "query" in formatted
        assert "limit" in formatted
        assert "string" in formatted
        assert "integer" in formatted
        assert "required" in formatted

    def test_format_usage_example(self, tool_formatter):
        """Test usage example formatting"""
        example = {
            "tool": "search_tool",
            "parameters": {"query": "example search", "limit": 10},
            "description": "Search for example content"
        }
        
        formatted = tool_formatter.format_usage_example(example)
        
        assert "search_tool" in formatted
        assert "example search" in formatted
        assert "limit" in formatted
        assert "Search for example content" in formatted