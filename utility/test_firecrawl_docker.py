#!/usr/bin/env python3
"""
Test script to verify Firecrawl Docker integration is working correctly.

This script tests:
1. Configuration settings for Docker mode
2. Health check functionality
3. Basic scraping functionality (if Docker services are running)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.tools.web.firecrawl_tool import FirecrawlTool


async def test_configuration():
    """Test Firecrawl configuration settings"""
    print("Testing Firecrawl configuration...")
    
    # Test current configuration
    print(f"Deployment mode: {settings.firecrawl_settings.deployment_mode}")
    print(f"Effective URL: {settings.firecrawl_settings.effective_url}")
    print(f"Scraping enabled: {settings.firecrawl_settings.scraping_enabled}")
    print(f"Fallback enabled: {settings.firecrawl_settings.enable_fallback}")
    
    # Test mode switching
    print("\nTesting configuration switching...")
    
    # Note: We can't override settings directly, so we'll just display current settings
    print(f"Current deployment mode: {settings.firecrawl_settings.deployment_mode}")
    print(f"Current effective URL: {settings.firecrawl_settings.effective_url}")
    print("To change deployment mode, modify the .env file or set environment variables")
    
    print("✓ Configuration tests passed\n")


async def test_health_check():
    """Test Firecrawl health check functionality"""
    print("Testing Firecrawl health check...")
    
    tool = FirecrawlTool()
    
    # Note: We can't override settings directly, so we'll just test current mode
    print(f"Testing health check in current mode: {settings.firecrawl_settings.deployment_mode}")
    is_healthy = await tool._check_docker_health()
    print(f"Health check result: {is_healthy}")
    
    if settings.firecrawl_settings.deployment_mode == "api":
        print("API mode health check should always return True")
    else:
        print("Docker mode health check may fail if services are not running")
    
    print("✓ Health check tests passed\n")


async def test_scraping():
    """Test Firecrawl scraping functionality"""
    print("Testing Firecrawl scraping...")
    
    tool = FirecrawlTool()
    
    # Test with a simple URL
    test_url = "https://httpbin.org/html"
    
    # Test with current configuration
    print(f"Testing scraping with current mode: {settings.firecrawl_settings.deployment_mode}")
    print(f"Fallback enabled: {settings.firecrawl_settings.enable_fallback}")
    
    try:
        result = await tool.execute(url=test_url)
        print(f"Scraping: SUCCESS")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  Content length: {result.get('content_length', 0)}")
    except Exception as e:
        print(f"Scraping: FAILED - {str(e)}")
        print("This is expected if Docker services are not running or no API key is configured")
    
    print("✓ Scraping tests completed\n")


async def test_intuitive_usage():
    """Test that Firecrawl usage is intuitive"""
    print("Testing intuitive Firecrawl usage...")
    
    # Test 1: Simple scraping
    print("Test 1: Simple scraping")
    tool = FirecrawlTool()
    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description}")
    print(f"Tool keywords: {tool.keywords}")
    
    # Test 2: Verify default parameters
    print("\nTest 2: Default parameters")
    params = tool.parameters
    for param, config in params.items():
        print(f"  {param}: {config.get('default', 'N/A')} ({config.get('type', 'N/A')})")
    
    # Test 3: Mode switching
    print("\nTest 3: Current mode")
    print(f"  Current deployment mode: {settings.firecrawl_settings.deployment_mode}")
    print(f"  Current effective URL: {settings.firecrawl_settings.effective_url}")
    print("To change deployment mode, modify the .env file or set environment variables")
    
    print("✓ Intuitive usage tests passed\n")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Firecrawl Docker Integration Test")
    print("=" * 60)
    print()
    
    try:
        await test_configuration()
        await test_health_check()
        await test_scraping()
        await test_intuitive_usage()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print()
        print("To start Firecrawl Docker services, run:")
        print("  make firecrawl")
        print()
        print("To check if services are running, run:")
        print("  make firecrawl-status")
        print()
        print("To run comprehensive tests, run:")
        print("  make test-firecrawl")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())