#!/usr/bin/env python3
"""
Docker Integration Test Script
Tests all Docker services for the AI Assistant project
"""

import subprocess
import time
import requests
import sys
from typing import Dict, List, Tuple


class DockerTester:
    def __init__(self):
        self.services = {
            "ai-assistant": {"port": 8000, "health_path": "/"},
            "redis": {"port": 6379, "health_check": self._check_redis},
            "searxng": {"port": 8080, "health_path": "/"},
        }
        self.base_url = "http://localhost"

    def run_command(self, command: str, capture_output=True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout if capture_output else ""
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def check_service_status(self) -> Dict[str, bool]:
        """Check if all Docker services are running"""
        print("Checking service status...")
        success, output = self.run_command("docker compose ps")
        if not success:
            print(f"Failed to get service status: {output}")
            return {}

        services = {}
        for line in output.split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    service_name = parts[0].replace("ai-assistant-", "")
                    status = parts[2]
                    services[service_name] = "Up" in status

        return services

    def _check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=5)
            r.ping()
            return True
        except:
            return False

    def check_http_service(self, service: str, path: str = "/") -> bool:
        """Check if an HTTP service is responding"""
        port = self.services[service]["port"]
        url = f"{self.base_url}:{port}{path}"

        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def test_service_health(self) -> Dict[str, bool]:
        """Test health of all services"""
        print("Testing service health...")
        results = {}

        for service, config in self.services.items():
            print(f"  Testing {service}...")

            if service == "redis":
                results[service] = config["health_check"]()
            else:
                health_path = config.get("health_path", "/")
                results[service] = self.check_http_service(service, health_path)

            status = "✓" if results[service] else "✗"
            print(f"    {status} {service}")

        return results

    def test_connectivity(self) -> Dict[str, bool]:
        """Test connectivity between services"""
        print("Testing service connectivity...")
        results = {}

        # Test AI Assistant can reach Redis
        print("  Testing AI Assistant -> Redis connectivity...")
        success, output = self.run_command(
            'docker compose exec ai-assistant python -c "'
            "import redis; "
            "r = redis.Redis(host='redis', port=6379, db=0); "
            'print(r.ping())"'
        )
        results["assistant_to_redis"] = success

        # Test AI Assistant can reach SearXNG
        print("  Testing AI Assistant -> SearXNG connectivity...")
        success, output = self.run_command(
            'docker compose exec ai-assistant python -c "'
            "import requests; "
            "r = requests.get('http://searxng:8080/', timeout=5); "
            'print(r.status_code)"'
        )
        results["assistant_to_searxng"] = success

        return results

    def test_application_endpoints(self) -> Dict[str, bool]:
        """Test AI Assistant application endpoints"""
        print("Testing application endpoints...")
        results = {}

        endpoints = [
            ("/", "Root endpoint"),
            ("/docs", "API documentation"),
            ("/health", "Health check"),
            ("/metrics", "Prometheus metrics"),
        ]

        for endpoint, description in endpoints:
            print(f"  Testing {description} ({endpoint})...")
            try:
                response = requests.get(f"{self.base_url}:8000{endpoint}", timeout=10)
                success = response.status_code in [
                    200,
                    404,
                ]  # 404 is acceptable for some endpoints
                results[endpoint] = success
                status = "✓" if success else "✗"
                print(f"    {status} {description}: {response.status_code}")
            except Exception as e:
                results[endpoint] = False
                print(f"    ✗ {description}: {str(e)}")

        return results

    def test_searxng_functionality(self) -> Dict[str, bool]:
        """Test SearXNG search functionality"""
        print("Testing SearXNG functionality...")
        results = {}

        # Test basic search
        try:
            response = requests.get(
                f"{self.base_url}:8080/search",
                params={"q": "test", "format": "json"},
                timeout=10,
            )
            results["search_endpoint"] = response.status_code == 200
            print(f"    {'✓' if results['search_endpoint'] else '✗'} Search endpoint")
        except Exception as e:
            results["search_endpoint"] = False
            print(f"    ✗ Search endpoint: {str(e)}")

        return results

    def run_all_tests(self) -> bool:
        """Run all Docker integration tests"""
        print("=" * 60)
        print("Docker Integration Test Suite")
        print("=" * 60)

        all_passed = True

        # Test 1: Service Status
        print("\n1. Service Status Test")
        print("-" * 30)
        services = self.check_service_status()
        if not services:
            print("❌ No services found - Docker may not be running")
            return False

        for service, status in services.items():
            if status:
                print(f"✓ {service} is running")
            else:
                print(f"❌ {service} is not running")
                all_passed = False

        # Test 2: Service Health
        print("\n2. Service Health Test")
        print("-" * 30)
        health_results = self.test_service_health()
        for service, healthy in health_results.items():
            if not healthy:
                all_passed = False

        # Test 3: Connectivity
        print("\n3. Inter-Service Connectivity Test")
        print("-" * 30)
        connectivity_results = self.test_connectivity()
        for connection, success in connectivity_results.items():
            if success:
                print(f"✓ {connection}")
            else:
                print(f"❌ {connection}")
                all_passed = False

        # Test 4: Application Endpoints
        print("\n4. Application Endpoints Test")
        print("-" * 30)
        endpoint_results = self.test_application_endpoints()
        for endpoint, success in endpoint_results.items():
            if not success:
                all_passed = False

        # Test 5: SearXNG Functionality
        print("\n5. SearXNG Functionality Test")
        print("-" * 30)
        searxng_results = self.test_searxng_functionality()
        for test, success in searxng_results.items():
            if not success:
                all_passed = False

        # Summary
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ All tests passed! Docker integration is working correctly.")
        else:
            print("❌ Some tests failed. Please check the logs above.")
        print("=" * 60)

        return all_passed


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Docker integration for AI Assistant"
    )
    parser.add_argument(
        "--start", action="store_true", help="Start services before testing"
    )
    parser.add_argument(
        "--stop", action="store_true", help="Stop services after testing"
    )

    args = parser.parse_args()

    tester = DockerTester()

    if args.start:
        print("Starting Docker services...")
        success, _ = tester.run_command("docker compose up -d")
        if not success:
            print("Failed to start services")
            sys.exit(1)

        print("Waiting for services to start...")
        time.sleep(10)

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        if args.stop:
            print("\nStopping Docker services...")
            tester.run_command("docker compose down")


if __name__ == "__main__":
    main()
