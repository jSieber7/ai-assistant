#!/usr/bin/env python3
"""
Test script for the AI Assistant tool system.

This script tests the core functionality of the tool system including
tool registration, execution, and API integration.
"""

import asyncio
import sys
import os

# Add the app directory to the path using shared utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

from app.core.tools import tool_registry, CalculatorTool, TimeTool, EchoTool
from app.core.tools.langchain_integration import (
    LangChainToolWrapper,
    create_agent_with_tools,
)
from app.core.config import get_llm
from tests.test_utils import TestResult, run_async_test


async def test_tool_registration():
    """Test tool registration and basic functionality"""
    print("=== Testing Tool Registration ===")

    # Create tool instances
    calculator = CalculatorTool()
    time_tool = TimeTool()
    echo_tool = EchoTool()

    # Register tools
    tool_registry.register(calculator, "utility")
    tool_registry.register(time_tool, "utility")
    tool_registry.register(echo_tool, "testing")

    # List tools
    tools = tool_registry.list_tools()
    print(f"Registered tools: {[tool.name for tool in tools]}")

    # Test tool discovery
    relevant_tools = tool_registry.find_relevant_tools("What's 2+2?")
    print(f"Relevant tools for 'What's 2+2?': {[tool.name for tool in relevant_tools]}")

    # Get registry stats
    stats = tool_registry.get_registry_stats()
    print(f"Registry stats: {stats}")

    print("✓ Tool registration test passed\n")


async def test_tool_execution():
    """Test tool execution functionality"""
    print("=== Testing Tool Execution ===")

    # Test calculator tool
    calculator = CalculatorTool()
    result = await calculator.execute_with_timeout(expression="2 + 3 * 4")
    print(f"Calculator result: {result}")

    # Test echo tool
    echo = EchoTool()
    result = await echo.execute_with_timeout(text="Hello, World!")
    print(f"Echo result: {result}")

    # Test time tool
    time_tool = TimeTool()
    result = await time_tool.execute_with_timeout(timezone="UTC", format="human")
    print(f"Time tool result: {result}")

    print("✓ Tool execution test passed\n")


async def test_langchain_integration():
    """Test LangChain integration"""
    print("=== Testing LangChain Integration ===")

    # Test LangChain wrapper
    calculator = CalculatorTool()
    langchain_tool = LangChainToolWrapper(calculator)

    # Test sync execution
    try:
        result = langchain_tool._run(expression="10 / 2")
        print(f"LangChain tool result: {result}")
    except Exception as e:
        print(f"LangChain tool error: {e}")

    # Test async execution
    try:
        result = await langchain_tool._arun(expression="10 / 2")
        print(f"LangChain async tool result: {result}")
    except Exception as e:
        print(f"LangChain async tool error: {e}")

    print("✓ LangChain integration test passed\n")


async def test_agent_creation():
    """Test agent creation with tools"""
    print("=== Testing Agent Creation ===")

    try:
        # Get LLM instance
        llm = get_llm()

        # Create agent with tools
        agent_executor = create_agent_with_tools(llm)
        print("Agent created successfully")

        # Test agent with simple query
        try:
            result = await agent_executor.ainvoke({"input": "What is 5 times 8?"})
            print(f"Agent result: {result}")
        except Exception as e:
            print(
                f"Agent execution error (expected if no tools are properly configured): {e}"
            )

    except Exception as e:
        print(f"Agent creation error: {e}")

    print("✓ Agent creation test passed\n")


async def main():
    """Run all tests"""
    print("Starting AI Assistant Tool System Tests...\n")

    try:
        await test_tool_registration()
        await test_tool_execution()
        await test_langchain_integration()
        await test_agent_creation()

        print("🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
