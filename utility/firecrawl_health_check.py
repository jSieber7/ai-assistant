#!/usr/bin/env python3
"""
Firecrawl Docker Health Check Utility

This script checks the health of the Firecrawl Docker services
and provides diagnostic information for troubleshooting.
"""

import asyncio
import httpx
import logging
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FirecrawlHealthChecker:
    """Health checker for Firecrawl Docker services"""
    
    def __init__(self, docker_url: str = "http://firecrawl:3002"):
        self.docker_url = docker_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def check_api_health(self) -> Dict[str, Any]:
        """Check Firecrawl API health"""
        try:
            response = await self.client.get(f"{self.docker_url}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_api_version(self) -> Dict[str, Any]:
        """Check Firecrawl API version"""
        try:
            response = await self.client.get(f"{self.docker_url}/version")
            return {
                "status": "success",
                "version": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_scraping(self, test_url: str = "https://httpbin.org/html") -> Dict[str, Any]:
        """Test basic scraping functionality"""
        try:
            payload = {
                "url": test_url,
                "options": {
                    "formats": ["markdown"],
                    "waitFor": 2000
                }
            }
            
            response = await self.client.post(
                f"{self.docker_url}/v1/scrape",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "title": data.get("data", {}).get("metadata", {}).get("title", "Unknown"),
                    "content_length": len(data.get("data", {}).get("markdown", "")),
                    "formats": list(data.get("data", {}).keys())
                }
            else:
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check Firecrawl dependencies (Redis, PostgreSQL, Playwright)"""
        results = {}
        
        # Check Redis
        try:
            redis_client = httpx.AsyncClient(timeout=5.0)
            response = await redis_client.get("http://firecrawl-redis:6379")
            results["redis"] = {
                "status": "error",
                "error": "Redis should not respond to HTTP requests"
            }
        except Exception:
            # Expected behavior - Redis doesn't speak HTTP
            results["redis"] = {
                "status": "expected_failure",
                "message": "Redis is not HTTP accessible (expected)"
            }
        
        # Check PostgreSQL
        try:
            postgres_client = httpx.AsyncClient(timeout=5.0)
            response = await postgres_client.get("http://firecrawl-postgres:5432")
            results["postgres"] = {
                "status": "error", 
                "error": "PostgreSQL should not respond to HTTP requests"
            }
        except Exception:
            # Expected behavior - PostgreSQL doesn't speak HTTP
            results["postgres"] = {
                "status": "expected_failure",
                "message": "PostgreSQL is not HTTP accessible (expected)"
            }
        
        # Check Playwright service
        try:
            playwright_client = httpx.AsyncClient(timeout=5.0)
            response = await playwright_client.get("http://firecrawl-playwright:3000/health")
            results["playwright"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code
            }
        except Exception as e:
            results["playwright"] = {
                "status": "error",
                "error": str(e)
            }
        
        return results
    
    async def run_full_check(self, test_url: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("Starting Firecrawl health check...")
        
        results = {
            "timestamp": asyncio.get_event_loop().time(),
            "docker_url": self.docker_url,
            "checks": {}
        }
        
        # API Health
        logger.info("Checking API health...")
        results["checks"]["api_health"] = await self.check_api_health()
        
        # API Version
        logger.info("Checking API version...")
        results["checks"]["api_version"] = await self.check_api_version()
        
        # Dependencies
        logger.info("Checking dependencies...")
        results["checks"]["dependencies"] = await self.check_dependencies()
        
        # Test scraping
        if test_url:
            logger.info(f"Testing scraping with {test_url}...")
            results["checks"]["scraping_test"] = await self.test_scraping(test_url)
        
        # Overall status
        api_healthy = results["checks"]["api_health"]["status"] == "healthy"
        scraping_works = not test_url or results["checks"]["scraping_test"]["status"] == "success"
        
        results["overall_status"] = "healthy" if api_healthy and scraping_works else "unhealthy"
        
        return results
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Firecrawl Docker Health Check")
    parser.add_argument("--url", default="http://firecrawl:3002",
                       help="Firecrawl API URL (default: http://firecrawl:3002)")
    parser.add_argument("--test-url", default="https://httpbin.org/html",
                       help="URL to test scraping (default: https://httpbin.org/html)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checker = FirecrawlHealthChecker(args.url)
    
    try:
        results = await checker.run_full_check(args.test_url)
        
        # Print results
        print("\n" + "="*50)
        print("FIRECRAWL HEALTH CHECK RESULTS")
        print("="*50)
        
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"API URL: {results['docker_url']}")
        
        print("\nAPI Health:")
        api_health = results["checks"]["api_health"]
        print(f"  Status: {api_health['status']}")
        if api_health["status"] == "healthy":
            print(f"  Status Code: {api_health['status_code']}")
        else:
            print(f"  Error: {api_health.get('error', 'Unknown error')}")
        
        if "api_version" in results["checks"]:
            print("\nAPI Version:")
            version = results["checks"]["api_version"]
            if version["status"] == "success":
                print(f"  Version: {version['version']}")
            else:
                print(f"  Error: {version.get('error', 'Unknown error')}")
        
        print("\nDependencies:")
        for service, status in results["checks"]["dependencies"].items():
            print(f"  {service.capitalize()}: {status['status']}")
            if "message" in status:
                print(f"    {status['message']}")
            elif "error" in status:
                print(f"    Error: {status['error']}")
        
        if "scraping_test" in results["checks"]:
            print("\nScraping Test:")
            test = results["checks"]["scraping_test"]
            if test["status"] == "success":
                print(f"  Status: {test['status']}")
                print(f"  Title: {test['title']}")
                print(f"  Content Length: {test['content_length']} characters")
                print(f"  Formats: {', '.join(test['formats'])}")
            else:
                print(f"  Status: {test['status']}")
                print(f"  Error: {test.get('error', 'Unknown error')}")
        
        print("\n" + "="*50)
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"] == "healthy" else 1)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        sys.exit(2)
    finally:
        await checker.close()


if __name__ == "__main__":
    asyncio.run(main())