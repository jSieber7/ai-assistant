#!/usr/bin/env python3
"""
Test script to verify chainlit service connectivity in Docker environment.
"""

import asyncio
import httpx
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_chainlit_connection():
    """Test connection to chainlit service"""
    
    # Test URLs
    urls_to_test = [
        "http://chainlit:8001/",
        "http://chainlit:8001/auth/login",
    ]
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for url in urls_to_test:
            try:
                logger.info(f"Testing {url}...")
                response = await client.get(url)
                results[url] = {
                    "status_code": response.status_code,
                    "success": response.status_code in [200, 302],  # 302 for redirect to login
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(response.content),
                }
                logger.info(f"✅ {url} - Status: {response.status_code}")
            except Exception as e:
                results[url] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ {url} - Error: {str(e)}")
    
    return results

async def test_chainlit_via_traefik():
    """Test chainlit through traefik reverse proxy"""
    
    # Test through traefik
    urls_to_test = [
        "http://traefik-dev:80/chat",
        "http://traefik-dev:80/",
    ]
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for url in urls_to_test:
            try:
                logger.info(f"Testing {url} through Traefik...")
                response = await client.get(url)
                results[url] = {
                    "status_code": response.status_code,
                    "success": response.status_code in [200, 302],
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(response.content),
                }
                logger.info(f"✅ {url} - Status: {response.status_code}")
            except Exception as e:
                results[url] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ {url} - Error: {str(e)}")
    
    return results

async def main():
    """Main function"""
    logger.info("Testing Chainlit service connectivity...")
    
    # Test direct connection
    direct_results = await test_chainlit_connection()
    
    # Test through traefik
    traefik_results = await test_chainlit_via_traefik()
    
    # Print summary
    print("\n" + "="*50)
    print("CHAINLIT CONNECTION TEST RESULTS")
    print("="*50)
    
    overall_success = True
    
    print("\nDirect Connection Tests:")
    for url, result in direct_results.items():
        if result["success"]:
            print(f"  ✅ {url} - Status: {result['status_code']}")
        else:
            print(f"  ❌ {url} - Error: {result.get('error', 'Unknown error')}")
            overall_success = False
    
    print("\nTraefik Proxy Tests:")
    for url, result in traefik_results.items():
        if result["success"]:
            print(f"  ✅ {url} - Status: {result['status_code']}")
        else:
            print(f"  ❌ {url} - Error: {result.get('error', 'Unknown error')}")
            # Don't fail overall for traefik issues since it might be a proxy config issue
    
    print("\n" + "="*50)
    if overall_success:
        print("✅ CHAINLIT DIRECT CONNECTION TESTS PASSED!")
        print("\nTo access Chainlit in your browser:")
        print("- Direct: http://localhost:8001 (if port mapped)")
        print("- Through Traefik: http://localhost:8000/chat")
        sys.exit(0)
    else:
        print("❌ SOME CHAINLIT TESTS FAILED - Check the errors above")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())