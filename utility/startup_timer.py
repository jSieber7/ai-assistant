#!/usr/bin/env python3
"""
Startup timer script to measure and analyze AI Assistant container startup times.

This script helps identify bottlenecks during container startup by measuring
the time it takes for different components to become ready.
"""

import time
import subprocess
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

def run_command(cmd: List[str], timeout: int = 300) -> tuple:
    """Run a command and return (returncode, stdout, stderr, elapsed_time)"""
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        elapsed = time.time() - start_time
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return -1, "", f"Command timed out after {timeout}s", elapsed

def check_container_health(container_name: str) -> Dict[str, any]:
    """Check if a container is healthy and return status info"""
    cmd = ["docker", "compose", "--profile", "dev", "ps", container_name]
    returncode, stdout, stderr, elapsed = run_command(cmd)
    
    if returncode != 0:
        return {
            "status": "error",
            "message": f"Failed to check container: {stderr.strip()}",
            "elapsed": elapsed
        }
    
    lines = stdout.strip().split('\n')
    if len(lines) < 2:
        return {
            "status": "unknown",
            "message": "No container info found",
            "elapsed": elapsed
        }
    
    # Parse the container status line
    status_line = lines[1]
    parts = status_line.split()
    if len(parts) < 5:
        return {
            "status": "unknown",
            "message": f"Unable to parse status: {status_line}",
            "elapsed": elapsed
        }
    
    container_status = parts[4]
    if "healthy" in container_status:
        status = "healthy"
    elif "unhealthy" in container_status:
        status = "unhealthy"
    elif "starting" in container_status or "Up" in container_status:
        status = "starting"
    else:
        status = container_status
    
    return {
        "status": status,
        "message": f"Container status: {container_status}",
        "elapsed": elapsed
    }

def check_endpoint_health(url: str, timeout: int = 10) -> Dict[str, any]:
    """Check if an HTTP endpoint is responding"""
    cmd = ["curl", "-f", "-s", "-m", str(timeout), url]
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+5)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                return {
                    "status": "healthy",
                    "message": f"Endpoint responding (HTTP {data.get('status', 'unknown')})",
                    "elapsed": elapsed,
                    "response": data
                }
            except json.JSONDecodeError:
                return {
                    "status": "healthy",
                    "message": "Endpoint responding (non-JSON response)",
                    "elapsed": elapsed,
                    "response": result.stdout[:100]
                }
        else:
            return {
                "status": "unhealthy",
                "message": f"HTTP error: {result.stderr.strip()}",
                "elapsed": elapsed
            }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "status": "timeout",
            "message": f"Request timed out after {timeout}s",
            "elapsed": elapsed
        }

def measure_startup_times() -> Dict[str, any]:
    """Measure startup times for all components"""
    print("🚀 Measuring AI Assistant startup times...")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    results = {
        "start_time": datetime.now().isoformat(),
        "components": {},
        "summary": {}
    }
    
    # Start the containers
    print("📦 Starting containers...")
    start_time = time.time()
    returncode, stdout, stderr, elapsed = run_command([
        "docker", "compose", "--profile", "dev", "up", "-d"
    ], timeout=300)
    
    if returncode != 0:
        print(f"❌ Failed to start containers: {stderr}")
        results["summary"]["error"] = stderr
        return results
    
    startup_time = time.time() - start_time
    print(f"✅ Containers started in {startup_time:.2f}s")
    results["summary"]["container_startup"] = startup_time
    
    # Wait a bit for containers to initialize
    print("⏳ Waiting for containers to initialize...")
    time.sleep(10)
    
    # Check individual container health
    containers = [
        "ai-assistant-dev",
        "ai-assistant-redis", 
        "ai-assistant-searxng",
        "milvus-standalone",
        "ai-assistant-traefik-dev"
    ]
    
    for container in containers:
        print(f"🔍 Checking {container}...")
        health = check_container_health(container)
        results["components"][container] = health
        status_icon = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "starting" else "❌"
        print(f"   {status_icon} {health['message']} ({health['elapsed']:.2f}s)")
    
    # Check application endpoints
    endpoints = [
        ("Main App", "http://localhost:8000/"),
        ("Health Check", "http://localhost:8000/monitoring/health"),
        ("Traefik Dashboard", "http://localhost:8080/dashboard/")
    ]
    
    for name, url in endpoints:
        print(f"🌐 Checking {name} at {url}...")
        health = check_endpoint_health(url)
        results["components"][name] = health
        status_icon = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "timeout" else "❌"
        print(f"   {status_icon} {health['message']} ({health['elapsed']:.2f}s)")
    
    # Calculate total time
    total_time = time.time() - start_time
    results["summary"]["total_time"] = total_time
    
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Container startup: {startup_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    
    healthy_count = sum(1 for c in results["components"].values() if c["status"] == "healthy")
    total_count = len(results["components"])
    print(f"   Healthy components: {healthy_count}/{total_count}")
    
    if healthy_count == total_count:
        print("🎉 All components are healthy!")
    else:
        print("⚠️  Some components are not healthy yet")
    
    return results

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python3 startup_timer.py")
        print("Measures startup times for the AI Assistant development environment")
        sys.exit(0)
    
    results = measure_startup_times()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"startup_times_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {filename}")

if __name__ == "__main__":
    main()