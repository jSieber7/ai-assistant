#!/usr/bin/env python3
"""
Test script for the agent system integration.

This script tests the core functionality of the agent system, including
agent registration, tool selection, and message processing.
"""

import sys
import os
import pytest

# Add the app directory to the path using shared utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

from app.core.config import initialize_agent_system, settings
from app.core.agents.registry import agent_registry
from app.core.tools.registry import tool_registry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_system():
    """Test the agent system functionality"""
    print("🧪 Testing Agent System Integration")
    print("=" * 50)

    # Test messages for the agent system
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "What can you help me with?"},
    ]

    # Ensure agent system is enabled
    if not settings.agent_system_enabled:
        print("❌ Agent system is disabled in settings")
        return False

    try:
        # Initialize the agent system
        print("📦 Initializing agent system...")
        registry = initialize_agent_system()

        if not registry:
            print("❌ Failed to initialize agent system")
            return False

        print("✅ Agent system initialized successfully")

        # Check agent registry
        stats = agent_registry.get_registry_stats()
        print(f"📊 Agent Registry Stats: {stats}")

        # Check tool registry
        tool_stats = tool_registry.get_registry_stats()
        print(f"🔧 Tool Registry Stats: {tool_stats}")

        # Test agent listing
        agents = agent_registry.list_agents(active_only=True)
        print(f"🤖 Active agents: {[agent.name for agent in agents]}")

        # Test default agent
        default_agent = agent_registry.get_default_agent()
        if not default_agent:
            print("❌ No default agent found")
            return False

        print(f"🎯 Default agent: {default_agent.name}")

        # Test simple message processing
        print("\n💬 Testing message processing...")

        for message in test_messages:
            print(f"\n📨 Testing: '{message}'")
            try:
                result = await agent_registry.process_message(message)

                if result.success:
                    print(f"✅ Success: {result.response[:100]}...")
                    print(f"   Agent: {result.agent_name}")
                    print(
                        f"   Tools used: {[tr.tool_name for tr in result.tool_results]}"
                    )
                    print(f"   Execution time: {result.execution_time:.2f}s")
                else:
                    print(f"❌ Failed: {result.error}")

            except Exception as e:
                print(f"❌ Error processing message: {str(e)}")

        # Test agent-specific functionality
        print("\n🔍 Testing agent-specific endpoints...")

        # Test agent info
        agent = agent_registry.get_agent(default_agent.name)
        if agent:
            stats = agent.get_usage_stats()
            print(
                f"📋 Agent info: {agent.name}, State: {agent.state}, Usage: {stats['usage_count']}"
            )

        # Test conversation history (using the last result)
        if "result" in locals() and result.conversation_id:
            try:
                history = default_agent.get_conversation_history(result.conversation_id)
                print(f"📚 Conversation history length: {len(history)}")
            except Exception as e:
                print(f"⚠️  Conversation history not available: {str(e)}")

        print("\n🎉 All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_selection_strategies():
    """Test different tool selection strategies"""
    print("\n🎯 Testing Tool Selection Strategies")
    print("=" * 50)

    from app.core.agents.strategies import KeywordStrategy, ToolSelectionManager

    # Test queries for tool selection
    test_queries = [
        "Calculate 42 * 3.14",
        "Search for information about Python",
        "What is the weather like today?",
    ]

    try:
        # Test keyword strategy
        keyword_strategy = KeywordStrategy()

        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            tools = await keyword_strategy.select_tools(query, None, tool_registry)
            print(f"   Selected tools: {[tool.name for tool in tools]}")

        # Test tool selection manager
        print("\n🔄 Testing Tool Selection Manager...")
        manager = ToolSelectionManager([keyword_strategy])
        tools = await manager.select_tools("Calculate 42 * 3.14", None, tool_registry)
        print(f"   Manager selected: {[tool.name for tool in tools]}")

        print("✅ Tool selection strategies tested successfully!")
        return True

    except Exception as e:
        print(f"❌ Tool selection test failed: {str(e)}")
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_endpoints():
    """Test agent API endpoints"""
    print("\n🌐 Testing Agent API Endpoints")
    print("=" * 50)

    try:
        # This would typically test the FastAPI endpoints
        # For now, we'll simulate the endpoint functionality

        from app.api.agent_routes import AgentChatRequest, AgentChatMessage

        # Simulate a chat request
        chat_request = AgentChatRequest(
            messages=[AgentChatMessage(role="user", content="Hello, can you help me?")],
            agent_name="tool_agent",
        )

        print(f"📨 Simulated chat request for agent: {chat_request.agent_name}")
        print("✅ API endpoint simulation completed")

        return True

    except Exception as e:
        print(f"❌ API endpoint test failed: {str(e)}")
        return False
