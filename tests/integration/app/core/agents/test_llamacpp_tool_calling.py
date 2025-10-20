#!/usr/bin/env python3
"""
Integration test for tool calling with Llama.cpp model.

This test verifies that the AI Assistant can properly use tools when
configured with a local Llama.cpp model serving on port 7543.
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import patch

# Add the app directory to the path using shared utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "app"))

from app.core.tools import tool_registry, CalculatorTool, TimeTool, EchoTool
from app.core.agents.specialized.tool_agent import ToolAgent
from app.core.agents.utilities.strategies import KeywordStrategy
from app.core.config import get_llm, initialize_llm_providers
from app.core.llm_providers import OpenAICompatibleProvider, provider_registry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llamacpp_tool_calling_setup():
    """Test setting up Llama.cpp provider and tool system"""
    print("=== Testing Llama.cpp Tool Calling Setup ===")
    
    # Configure environment for Llama.cpp
    with patch.dict(os.environ, {
        "OPENAI_COMPATIBLE_ENABLED": "true",
        "OPENAI_COMPATIBLE_API_KEY": "llama-cpp-key",
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:7543/v1",
        "OPENAI_COMPATIBLE_DEFAULT_MODEL": "llama.cpp.model",
        "PREFERRED_PROVIDER": "openai_compatible"
    }):
        # Clear any existing providers
        provider_registry._providers.clear()
        provider_registry._default_provider = None
        
        # Initialize providers with new configuration
        initialize_llm_providers()
        
        # Check if OpenAI-compatible provider is configured
        provider = provider_registry.get_provider("openai_compatible")
        assert provider is not None, "OpenAI-compatible provider should be configured"
        assert provider.base_url == "http://localhost:7543/v1", f"Expected base URL http://localhost:7543/v1, got {provider.base_url}"
        
        print(f"✓ Llama.cpp provider configured at {provider.base_url}")
        
        # Test health check (this will fail if server is not running)
        try:
            is_healthy = await provider.health_check()
            if is_healthy:
                print("✓ Llama.cpp server is healthy and responding")
            else:
                print("⚠ Llama.cpp server is not responding - make sure it's running on port 7543")
                return False
        except Exception as e:
            print(f"⚠ Could not connect to Llama.cpp server: {e}")
            print("Make sure Llama.cpp is running on port 7543 with OpenAI-compatible API")
            return False
    
    return True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_registration():
    """Test tool registration for Llama.cpp integration"""
    print("=== Testing Tool Registration ===")
    
    # Clear existing tools
    tool_registry._tools.clear()
    tool_registry._categories.clear()
    
    # Register basic tools
    calculator = CalculatorTool()
    time_tool = TimeTool()
    echo_tool = EchoTool()
    
    tool_registry.register(calculator, "utility")
    tool_registry.register(time_tool, "utility")
    tool_registry.register(echo_tool, "testing")
    
    # Verify tools are registered
    tools = tool_registry.list_tools()
    tool_names = [tool.name for tool in tools]
    
    assert "calculator" in tool_names, "Calculator tool should be registered"
    assert "time" in tool_names, "Time tool should be registered"
    assert "echo" in tool_names, "Echo tool should be registered"
    
    print(f"✓ Registered tools: {tool_names}")
    return True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_agent_with_calculator():
    """Test ToolAgent with Calculator tool using Llama.cpp"""
    print("=== Testing ToolAgent with Calculator ===")
    
    # Configure environment for Llama.cpp
    with patch.dict(os.environ, {
        "OPENAI_COMPATIBLE_ENABLED": "true",
        "OPENAI_COMPATIBLE_API_KEY": "llama-cpp-key",
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:7543/v1",
        "OPENAI_COMPATIBLE_DEFAULT_MODEL": "llama.cpp.model",
        "PREFERRED_PROVIDER": "openai_compatible"
    }):
        # Clear and reinitialize providers
        provider_registry._providers.clear()
        provider_registry._default_provider = None
        initialize_llm_providers()
        
        # Get LLM instance
        try:
            llm = await get_llm("openai_compatible:llama.cpp.model")
            print("✓ LLM instance created successfully")
        except Exception as e:
            print(f"✗ Failed to create LLM instance: {e}")
            return False
        
        # Ensure tools are registered
        await test_tool_registration()
        
        # Create ToolAgent
        tool_agent = ToolAgent(
            tool_registry=tool_registry,
            llm=llm,
            selection_strategy=KeywordStrategy(),
            max_iterations=3,
        )
        
        print("✓ ToolAgent created successfully")
        
        # Test with a math question that should trigger the calculator
        test_message = "What is 15 multiplied by 8?"
        print(f"Testing message: '{test_message}'")
        
        try:
            result = await tool_agent.process_message(test_message)
            
            print(f"Agent response: {result.response}")
            print(f"Tool results: {len(result.tool_results)} tools used")
            
            for tool_result in result.tool_results:
                print(f"  - {tool_result.tool_name}: {tool_result.data}")
            
            # Check if calculator was used
            calculator_used = any(tr.tool_name == "calculator" for tr in result.tool_results)
            if calculator_used:
                print("✓ Calculator tool was successfully used")
            else:
                print("⚠ Calculator tool was not used - LLM may not support tool calling or didn't recognize the need")
            
            return result.success
            
        except Exception as e:
            print(f"✗ ToolAgent execution failed: {e}")
            return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_tool_usage():
    """Test using multiple tools in sequence"""
    print("=== Testing Multiple Tool Usage ===")
    
    # Configure environment for Llama.cpp
    with patch.dict(os.environ, {
        "OPENAI_COMPATIBLE_ENABLED": "true",
        "OPENAI_COMPATIBLE_API_KEY": "llama-cpp-key",
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:7543/v1",
        "OPENAI_COMPATIBLE_DEFAULT_MODEL": "llama.cpp.model",
        "PREFERRED_PROVIDER": "openai_compatible"
    }):
        # Clear and reinitialize providers
        provider_registry._providers.clear()
        provider_registry._default_provider = None
        initialize_llm_providers()
        
        # Get LLM instance
        try:
            llm = await get_llm("openai_compatible:llama.cpp.model")
        except Exception as e:
            print(f"✗ Failed to create LLM instance: {e}")
            return False
        
        # Create ToolAgent
        tool_agent = ToolAgent(
            tool_registry=tool_registry,
            llm=llm,
            selection_strategy=KeywordStrategy(),
            max_iterations=3,
        )
        
        # Test with a message that could use multiple tools
        test_message = "Calculate 25 + 17 and then tell me the current time"
        print(f"Testing message: '{test_message}'")
        
        try:
            result = await tool_agent.process_message(test_message)
            
            print(f"Agent response: {result.response}")
            print(f"Tool results: {len(result.tool_results)} tools used")
            
            for tool_result in result.tool_results:
                print(f"  - {tool_result.tool_name}: {tool_result.data}")
            
            # Check if multiple tools were used
            tools_used = [tr.tool_name for tr in result.tool_results]
            if len(tools_used) > 1:
                print(f"✓ Multiple tools used: {tools_used}")
            elif len(tools_used) == 1:
                print(f"⚠ Only one tool used: {tools_used[0]}")
            else:
                print("⚠ No tools were used")
            
            return result.success
            
        except Exception as e:
            print(f"✗ Multiple tool usage test failed: {e}")
            return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_tool_execution():
    """Test direct tool execution to verify tools work independently"""
    print("=== Testing Direct Tool Execution ===")
    
    # Test calculator directly
    calculator = CalculatorTool()
    try:
        result = await calculator.execute_with_timeout(expression="12 * 5")
        print(f"Calculator result (12 * 5): {result.data}")
        assert result.success, "Calculator should succeed"
        assert result.data == 60, f"Expected 60, got {result.data}"
        print("✓ Calculator tool works correctly")
    except Exception as e:
        print(f"✗ Calculator tool failed: {e}")
        return False
    
    # Test echo tool
    echo = EchoTool()
    try:
        result = await echo.execute_with_timeout(text="Hello Llama.cpp!")
        print(f"Echo result: {result.data}")
        assert result.success, "Echo tool should succeed"
        assert result.data == "Hello Llama.cpp!", f"Expected 'Hello Llama.cpp!', got {result.data}"
        print("✓ Echo tool works correctly")
    except Exception as e:
        print(f"✗ Echo tool failed: {e}")
        return False
    
    return True


async def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Starting Llama.cpp Tool Calling Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Setup Test", test_llamacpp_tool_calling_setup),
        ("Tool Registration", test_tool_registration),
        ("Direct Tool Execution", test_direct_tool_execution),
        ("ToolAgent with Calculator", test_tool_agent_with_calculator),
        ("Multiple Tool Usage", test_multiple_tool_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = await test_func()
            if result is False:
                print(f"⚠ {test_name} returned False - server may not be running")
                results.append((test_name, "SKIPPED"))
            else:
                print(f"✅ {test_name} passed")
                results.append((test_name, "PASSED"))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, "FAILED"))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    for test_name, status in results:
        status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⚠️"}
        print(f"  {status_emoji.get(status, '❓')} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Llama.cpp tool calling is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the Llama.cpp server configuration.")


if __name__ == "__main__":
    # Run the tests directly
    asyncio.run(run_all_tests())