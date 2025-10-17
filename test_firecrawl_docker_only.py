#!/usr/bin/env python3
"""
Test script to verify Firecrawl Docker-only configuration
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.tools.firecrawl_tool import FirecrawlTool
from app.core.config import settings


async def test_docker_only_configuration():
    """Test that Firecrawl only works with Docker configuration"""
    print("=" * 60)
    print("FIRECRAWL DOCKER-ONLY CONFIGURATION TEST")
    print("=" * 60)
    
    # Test 1: Verify configuration
    print("\n1. Testing configuration...")
    print(f"   Deployment mode: {settings.firecrawl_settings.deployment_mode}")
    print(f"   Docker URL: {settings.firecrawl_settings.docker_url}")
    print(f"   Effective URL: {settings.firecrawl_settings.effective_url}")
    print(f"   Effective API key: {'None' if settings.firecrawl_settings.effective_api_key is None else 'Set'}")
    
    if settings.firecrawl_settings.deployment_mode != "docker":
        print("   ❌ FAIL: Deployment mode is not 'docker'")
        return False
    else:
        print("   ✅ PASS: Deployment mode is 'docker'")
    
    # Test 2: Verify no fallback mechanism
    print("\n2. Testing fallback mechanism...")
    if hasattr(settings.firecrawl_settings, 'enable_fallback'):
        print("   ❌ FAIL: Fallback mechanism still exists in configuration")
        return False
    else:
        print("   ✅ PASS: Fallback mechanism removed from configuration")
    
    # Test 3: Test FirecrawlTool initialization
    print("\n3. Testing FirecrawlTool initialization...")
    try:
        tool = FirecrawlTool()
        print("   ✅ PASS: FirecrawlTool initialized without API key")
    except Exception as e:
        print(f"   ❌ FAIL: Failed to initialize FirecrawlTool: {e}")
        return False
    
    # Test 4: Test Docker health check
    print("\n4. Testing Docker health check...")
    try:
        is_healthy = await tool._check_docker_health()
        if is_healthy:
            print("   ✅ PASS: Docker Firecrawl instance is healthy")
        else:
            print("   ⚠️  WARNING: Docker Firecrawl instance is unhealthy (services may not be running)")
            print("   This is expected if Docker services are not started.")
    except Exception as e:
        print(f"   ❌ FAIL: Health check failed: {e}")
        return False
    
    # Test 5: Test scraping with Docker (if healthy)
    if await tool._check_docker_health():
        print("\n5. Testing scraping functionality...")
        try:
            result = await tool.execute(
                url="https://httpbin.org/html",
                formats=["markdown"],
                timeout=30
            )
            
            if 'title' in result and 'content' in result:
                print(f"   ✅ PASS: Successfully scraped content (Title: {result['title']})")
                print(f"   Content length: {len(result['content'])} characters")
            else:
                print("   ❌ FAIL: Scraped result missing expected fields")
                return False
        except Exception as e:
            print(f"   ❌ FAIL: Scraping failed: {e}")
            return False
    else:
        print("\n5. Skipping scraping test (Docker services not healthy)")
    
    # Test 6: Verify no external API methods
    print("\n6. Testing for external API methods...")
    if hasattr(tool, '_fallback_client'):
        print("   ❌ FAIL: Fallback client still exists")
        return False
    if hasattr(tool, '_get_fallback_client'):
        print("   ❌ FAIL: Fallback client method still exists")
        return False
    else:
        print("   ✅ PASS: No external API methods found")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ All tests passed! Firecrawl is configured for Docker-only mode.")
    print("\nTo use Firecrawl:")
    print("1. Start Docker services: docker compose --profile firecrawl up -d")
    print("2. Check service health: python utility/firecrawl_health_check.py")
    print("3. Use FirecrawlTool in your code without API keys")
    
    return True


async def main():
    """Main test function"""
    try:
        success = await test_docker_only_configuration()
        return 0 if success else 1
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)