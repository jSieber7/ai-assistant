#!/usr/bin/env python3
"""
Test script to verify firecrawl service connectivity in Docker environment.
"""

import asyncio
import httpx
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_firecrawl_connection():
    """Test connection to firecrawl service"""
    
    # Test URLs
    urls_to_test = [
        "http://firecrawl:3002/health",
        "http://firecrawl:3002/version",
    ]
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in urls_to_test:
            try:
                logger.info(f"Testing {url}...")
                response = await client.get(url)
                results[url] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
                }
                logger.info(f"✅ {url} - Status: {response.status_code}")
            except Exception as e:
                results[url] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ {url} - Error: {str(e)}")
    
    # Test scraping functionality
    try:
        logger.info("Testing scrape endpoint...")
        scrape_payload = {
            "url": "https://httpbin.org/html",
            "options": {
                "formats": ["markdown"],
                "waitFor": 2000
            }
        }
        
        response = await client.post(
            "http://firecrawl:3002/v1/scrape",
            json=scrape_payload,
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            results["scrape_test"] = {
                "success": True,
                "title": data.get("data", {}).get("metadata", {}).get("title", "Unknown"),
                "content_length": len(data.get("data", {}).get("markdown", "")),
            }
            logger.info(f"✅ Scrape test successful - Title: {results['scrape_test']['title']}")
        else:
            results["scrape_test"] = {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:200]
            }
            logger.error(f"❌ Scrape test failed - Status: {response.status_code}")
    except Exception as e:
        results["scrape_test"] = {
            "success": False,
            "error": str(e)
        }
        logger.error(f"❌ Scrape test error: {str(e)}")
    
    return results

async def main():
    """Main function"""
    logger.info("Testing Firecrawl service connectivity...")
    
    results = await test_firecrawl_connection()
    
    # Print summary
    print("\n" + "="*50)
    print("FIRECRAWL CONNECTION TEST RESULTS")
    print("="*50)
    
    overall_success = True
    
    for url, result in results.items():
        if url == "scrape_test":
            print(f"\nScrape Test:")
            if result["success"]:
                print(f"  ✅ Success - Title: {result['title']}")
                print(f"  Content Length: {result['content_length']} characters")
            else:
                print(f"  ❌ Failed - {result.get('error', 'Unknown error')}")
                overall_success = False
        else:
            print(f"\n{url}:")
            if result["success"]:
                print(f"  ✅ Status: {result['status_code']}")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
                overall_success = False
    
    print("\n" + "="*50)
    if overall_success:
        print("✅ ALL TESTS PASSED - Firecrawl is working correctly!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check the errors above")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())