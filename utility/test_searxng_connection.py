#!/usr/bin/env python3
"""
Test script to verify SearXNG connectivity and functionality.
This script tests both direct SearXNG access and the AI Assistant's SearXNG tool.
"""

import asyncio
import aiohttp
import sys
import os
from typing import Dict, Any

# Add the app directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def test_direct_searxng_access(searxng_url: str) -> Dict[str, Any]:
    """Test direct access to SearXNG API"""
    print(f"Testing direct SearXNG access at {searxng_url}...")

    try:
        # Test basic connectivity
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{searxng_url}/") as response:
                if response.status == 200:
                    print("✅ SearXNG is accessible")
                else:
                    print(f"❌ SearXNG returned status {response.status}")
                    return {"success": False, "error": f"HTTP {response.status}"}

        # Test search functionality
        search_url = f"{searxng_url}/search"
        params = {"q": "test query", "format": "json", "language": "en"}

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    if results.get("results"):
                        print(
                            f"✅ SearXNG search returned {len(results['results'])} results"
                        )
                        return {"success": True, "results": len(results["results"])}
                    else:
                        print("⚠️ SearXNG search returned no results")
                        return {"success": True, "results": 0}
                else:
                    error_text = await response.text()
                    print(
                        f"❌ SearXNG search failed with status {response.status}: {error_text}"
                    )
                    return {
                        "success": False,
                        "error": f"Search failed: HTTP {response.status}",
                    }

    except aiohttp.ClientError as e:
        print(f"❌ Failed to connect to SearXNG: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Unexpected error testing SearXNG: {e}")
        return {"success": False, "error": str(e)}


async def test_ai_assistant_searxng_tool() -> Dict[str, Any]:
    """Test the AI Assistant's SearXNG tool"""
    print("\nTesting AI Assistant SearXNG tool...")

    try:
        # Import the tool
        from app.core.tools.web.searxng_tool import SearXNGTool

        # Create an instance
        tool = SearXNGTool()

        # Execute a search
        result = await tool.execute(
            query="test search", category="general", language="en", results_count=5
        )

        if result.get("results"):
            print(
                f"✅ AI Assistant SearXNG tool returned {len(result['results'])} results"
            )
            return {"success": True, "results": len(result["results"])}
        else:
            print("⚠️ AI Assistant SearXNG tool returned no results")
            return {"success": True, "results": 0}

    except Exception as e:
        print(f"❌ AI Assistant SearXNG tool failed: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main test function"""
    print("=" * 60)
    print("SearXNG Connectivity Test")
    print("=" * 60)

    # Get SearXNG URL from environment or use default
    searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")

    # Test direct SearXNG access
    direct_result = await test_direct_searxng_access(searxng_url)

    # Test AI Assistant SearXNG tool
    tool_result = await test_ai_assistant_searxng_tool()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if direct_result["success"]:
        print("✅ Direct SearXNG access: PASSED")
    else:
        print(
            f"❌ Direct SearXNG access: FAILED - {direct_result.get('error', 'Unknown error')}"
        )

    if tool_result["success"]:
        print("✅ AI Assistant SearXNG tool: PASSED")
    else:
        print(
            f"❌ AI Assistant SearXNG tool: FAILED - {tool_result.get('error', 'Unknown error')}"
        )

    # Overall result
    if direct_result["success"] and tool_result["success"]:
        print("\n🎉 All tests PASSED! SearXNG is working correctly.")
        return 0
    else:
        print("\n💥 Some tests FAILED. Check the errors above.")
        return 1


if __name__ == "__main__":
    # Set environment for testing
    os.environ.setdefault("ENVIRONMENT", "testing")

    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
