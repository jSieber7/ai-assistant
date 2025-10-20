#!/usr/bin/env python3
"""
Simple health check script for the ai_assistant-ai-assistant-dev Docker container.

This script can be used to manually test the health check endpoint
or as a standalone health check tool.
"""

import sys
import json
import requests
from datetime import datetime

def check_health(base_url="http://localhost:8000", timeout=10):
    """
    Check the health of the AI Assistant application.
    
    Args:
        base_url (str): Base URL of the application
        timeout (int): Request timeout in seconds
        
    Returns:
        tuple: (status_code, response_data)
    """
    try:
        health_url = f"{base_url}/monitoring/health"
        response = requests.get(health_url, timeout=timeout)
        
        return response.status_code, response.json()
        
    except requests.exceptions.Timeout:
        return None, {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return None, {"error": "Connection failed - is the application running?"}
    except json.JSONDecodeError:
        return None, {"error": "Invalid JSON response"}
    except Exception as e:
        return None, {"error": f"Unexpected error: {str(e)}"}

def main():
    """Main function to run the health check."""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Checking health of {base_url}...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 50)
    
    status_code, data = check_health(base_url, timeout)
    
    if status_code is None:
        print(f"❌ Health check failed: {data.get('error', 'Unknown error')}")
        sys.exit(1)
    
    if status_code == 200:
        status = data.get("status", "unknown")
        message = data.get("message", "No message")
        uptime = data.get("uptime_seconds", 0)
        checks_performed = data.get("checks_performed", 0)
        
        print(f"✅ Health check successful (HTTP {status_code})")
        print(f"Status: {status}")
        print(f"Message: {message}")
        print(f"Uptime: {uptime:.2f} seconds")
        print(f"Checks performed: {checks_performed}")
        
        # Show detailed check results if available
        if "checks" in data:
            print("\nDetailed checks:")
            for check in data["checks"]:
                check_status = check.get("status", "unknown")
                check_name = check.get("check_name", "unnamed")
                check_message = check.get("message", "No message")
                response_time = check.get("response_time", 0)
                
                status_icon = "✅" if check_status == "healthy" else "⚠️" if check_status == "degraded" else "❌"
                print(f"  {status_icon} {check_name}: {check_message} ({response_time:.3f}s)")
        
        sys.exit(0)
    else:
        print(f"❌ Health check failed with HTTP {status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        sys.exit(1)

if __name__ == "__main__":
    main()