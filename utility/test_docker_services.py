#!/usr/bin/env python3
"""
Comprehensive test script to verify all Docker services are working correctly.
"""

import asyncio
import httpx
import sys
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerServiceTester:
    """Test suite for Docker services"""
    
    def __init__(self):
        self.results = {}
    
    async def test_service(self, name: str, url: str, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single service"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                logger.info(f"Testing {name} at {url}...")
                response = await client.get(url)
                
                success = response.status_code == expected_status
                
                result = {
                    "name": name,
                    "url": url,
                    "status_code": response.status_code,
                    "success": success,
                    "content_type": response.headers.get("content-type", ""),
                }
                
                if success:
                    logger.info(f"✅ {name} - Status: {response.status_code}")
                else:
                    logger.error(f"❌ {name} - Status: {response.status_code} (expected {expected_status})")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ {name} - Error: {str(e)}")
            return {
                "name": name,
                "url": url,
                "status_code": None,
                "success": False,
                "error": str(e)
            }
    
    async def test_core_services(self):
        """Test core services"""
        services = [
            ("AI Assistant (Dev)", "http://ai-assistant-dev:8000/monitoring/health"),
            ("Redis", "http://redis:6379", None),  # Redis doesn't speak HTTP
            ("PostgreSQL", "http://postgres:5432", None),  # PostgreSQL doesn't speak HTTP
            ("SearXNG", "http://searxng:8080/"),
            ("Milvus", "http://milvus:9091/healthz"),
        ]
        
        logger.info("Testing core services...")
        
        for name, url, expected in services:
            if expected is None:
                # For non-HTTP services, just check if they're reachable via TCP
                self.results[name] = await self.test_tcp_connection(name, url.split(":")[1], int(url.split(":")[2]))
            else:
                self.results[name] = await self.test_service(name, url, expected)
    
    async def test_tcp_connection(self, name: str, host: str, port: int) -> Dict[str, Any]:
        """Test TCP connection for non-HTTP services"""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )
            writer.close()
            await writer.wait_closed()
            
            logger.info(f"✅ {name} - TCP connection successful")
            return {
                "name": name,
                "host": host,
                "port": port,
                "success": True,
                "type": "tcp"
            }
        except Exception as e:
            logger.error(f"❌ {name} - TCP connection failed: {str(e)}")
            return {
                "name": name,
                "host": host,
                "port": port,
                "success": False,
                "error": str(e),
                "type": "tcp"
            }
    
    async def test_firecrawl(self):
        """Test Firecrawl service"""
        logger.info("Testing Firecrawl service...")
        
        # Test health endpoint
        health_result = await self.test_service("Firecrawl Health", "http://firecrawl:3002/health")
        self.results["Firecrawl Health"] = health_result
        
        if health_result["success"]:
            # Test scraping functionality
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    payload = {
                        "url": "https://httpbin.org/html",
                        "options": {
                            "formats": ["markdown"],
                            "waitFor": 2000
                        }
                    }
                    
                    response = await client.post(
                        "http://firecrawl:3002/v1/scrape",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        self.results["Firecrawl Scrape"] = {
                            "name": "Firecrawl Scrape",
                            "success": True,
                            "title": data.get("data", {}).get("metadata", {}).get("title", "Unknown"),
                            "content_length": len(data.get("data", {}).get("markdown", "")),
                        }
                        logger.info(f"✅ Firecrawl Scrape - Success")
                    else:
                        self.results["Firecrawl Scrape"] = {
                            "name": "Firecrawl Scrape",
                            "success": False,
                            "status_code": response.status_code,
                            "error": response.text[:200]
                        }
                        logger.error(f"❌ Firecrawl Scrape - Status: {response.status_code}")
            except Exception as e:
                self.results["Firecrawl Scrape"] = {
                    "name": "Firecrawl Scrape",
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ Firecrawl Scrape - Error: {str(e)}")
    
    async def test_chainlit(self):
        """Test Chainlit service"""
        logger.info("Testing Chainlit service...")
        
        # Test direct connection
        chainlit_result = await self.test_service("Chainlit", "http://chainlit:8001/", 200)
        self.results["Chainlit"] = chainlit_result
        
        # Test through Traefik
        traefik_result = await self.test_service("Chainlit via Traefik", "http://traefik-dev:80/chat", 200)
        self.results["Chainlit via Traefik"] = traefik_result
    
    async def test_traefik(self):
        """Test Traefik reverse proxy"""
        logger.info("Testing Traefik...")
        
        # Test Traefik dashboard
        dashboard_result = await self.test_service("Traefik Dashboard", "http://traefik-dev:8080/ping", 200)
        self.results["Traefik Dashboard"] = dashboard_result
        
        # Test main proxy
        proxy_result = await self.test_service("Traefik Proxy", "http://traefik-dev:80/", 200)
        self.results["Traefik Proxy"] = proxy_result
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting comprehensive Docker service tests...")
        
        await self.test_core_services()
        await self.test_firecrawl()
        await self.test_chainlit()
        await self.test_traefik()
        
        return self.results
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("DOCKER SERVICE TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        
        print("\nDetailed Results:")
        print("-" * 40)
        
        for name, result in self.results.items():
            if result["success"]:
                if result.get("type") == "tcp":
                    print(f"✅ {name}: TCP connection successful")
                elif "title" in result:
                    print(f"✅ {name}: Working - Title: {result['title']}")
                else:
                    print(f"✅ {name}: Working (Status: {result.get('status_code', 'N/A')})")
            else:
                error_msg = result.get("error", f"Status: {result.get('status_code', 'N/A')}")
                print(f"❌ {name}: Failed - {error_msg}")
        
        print("\n" + "="*60)
        
        if failed_tests == 0:
            print("🎉 ALL SERVICES ARE WORKING CORRECTLY!")
            print("\nAccess URLs:")
            print("- AI Assistant: http://localhost:8000")
            print("- Chainlit Chat: http://localhost:8000/chat")
            print("- Traefik Dashboard: http://localhost:8080")
            print("- SearXNG Search: http://localhost:8000/search")
            print("- Redis Commander: http://localhost:8000/redis")
        else:
            print(f"⚠️  {failed_tests} SERVICE(S) FAILED - Check the errors above")
            print("\nTroubleshooting Tips:")
            print("1. Ensure all containers are running: docker compose --profile dev ps")
            print("2. Check container logs: docker compose --profile dev logs [service-name]")
            print("3. Restart services: docker compose --profile dev restart")
        
        return failed_tests == 0

async def main():
    """Main function"""
    tester = DockerServiceTester()
    
    try:
        await tester.run_all_tests()
        success = tester.print_summary()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())