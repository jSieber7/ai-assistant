#!/usr/bin/env python3
"""
Simple script to run the Llama.cpp tool calling integration test.

This script provides an easy way to test if your Llama.cpp server on port 7543
can properly handle tool calling with the AI Assistant application.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_prerequisites():
    """Check if prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if we can import the required modules
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest is not installed. Please install it with: pip install pytest")
        return False
    
    # Check if the test file exists
    test_file = project_root / "tests" / "integration" / "app" / "core" / "agents" / "test_llamacpp_tool_calling.py"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print("✅ Test file found")
    return True

async def run_test_directly():
    """Run the test directly without pytest"""
    print("\n🚀 Running Llama.cpp tool calling test directly...")
    
    try:
        # Import and run the test module
        from tests.integration.app.core.agents.test_llamacpp_tool_calling import run_all_tests
        await run_all_tests()
        return True
    except Exception as e:
        print(f"❌ Direct test execution failed: {e}")
        return False

def run_test_with_pytest():
    """Run the test using pytest"""
    print("\n🚀 Running Llama.cpp tool calling test with pytest...")
    
    test_file = "tests/integration/app/core/agents/test_llamacpp_tool_calling.py"
    
    try:
        # Run pytest with the integration marker
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file,
            "-v",
            "-m", "integration",
            "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ pytest execution failed: {e}")
        return False

def check_llamacpp_server():
    """Check if Llama.cpp server is running on port 7543"""
    print("\n🔍 Checking if Llama.cpp server is running on port 7543...")
    
    try:
        import httpx
        import asyncio
        
        async def check_server():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:7543/v1/models", 
                                               headers={"Authorization": "Bearer llama-cpp-key"})
                    if response.status_code == 200:
                        print("✅ Llama.cpp server is responding on port 7543")
                        data = response.json()
                        if "data" in data and data["data"]:
                            model_names = [model.get("id", "unknown") for model in data["data"]]
                            print(f"📋 Available models: {model_names}")
                        return True
                    else:
                        print(f"⚠️ Llama.cpp server responded with status {response.status_code}")
                        return False
            except Exception as e:
                print(f"❌ Cannot connect to Llama.cpp server: {e}")
                return False
        
        return asyncio.run(check_server())
    except ImportError:
        print("⚠️ httpx not available, skipping server check")
        return None  # Unknown status

def main():
    """Main function"""
    print("🧪 Llama.cpp Tool Calling Test Runner")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check if Llama.cpp server is running
    server_status = check_llamacpp_server()
    if server_status is False:
        print("\n❌ Llama.cpp server is not running or not accessible on port 7543")
        print("Please start your Llama.cpp server with OpenAI-compatible API on port 7543")
        print("Example command:")
        print("  docker run -d --name llama-server \\")
        print("    --gpus all \\")
        print("    -p 7543:8080 \\")
        print("    -v /path/to/models:/models \\")
        print("    ghcr.io/ggml-org/llama.cpp:full-cuda \\")
        print("    -m /models/your-model.gguf \\")
        print("    --host 0.0.0.0 \\")
        print("    --port 8080 \\")
        print("    --api-key llama-cpp-key")
        print("\nContinuing with test anyway (it will fail if server is not running)...")
    
    # Ask user how to run the test
    print("\nHow would you like to run the test?")
    print("1. Direct execution (recommended for debugging)")
    print("2. With pytest (standard testing)")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user")
        sys.exit(0)
    
    success = False
    
    if choice in ["1", "3"]:
        print("\n" + "=" * 50)
        success = asyncio.run(run_test_directly())
    
    if choice in ["2", "3"]:
        if not success or choice == "3":
            print("\n" + "=" * 50)
            pytest_success = run_test_with_pytest()
            success = success or pytest_success
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
        print("Your Llama.cpp server is working with tool calling!")
    else:
        print("❌ Test failed!")
        print("Please check:")
        print("1. Llama.cpp server is running on port 7543")
        print("2. Server has OpenAI-compatible API enabled")
        print("3. API key is set to 'llama-cpp-key' (or update the test)")
        print("4. Model supports tool calling/function calling")

if __name__ == "__main__":
    main()