#!/usr/bin/env python3
"""
Simple test script to verify the fixes for the failing tests
"""

import subprocess
import sys

def run_test(test_path):
    """Run a specific test and return the result"""
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", test_path, "-v"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    """Run the fixed tests and report results"""
    print("Testing fixes for failing GitHub Actions tests...")
    print("=" * 50)
    
    # Test the configuration fix
    print("\n1. Testing configuration fix...")
    success, stdout, stderr = run_test("tests/unit/test_config.py::TestSettings::test_settings_defaults")
    if success:
        print("✅ Configuration test PASSED")
    else:
        print("❌ Configuration test FAILED")
        print(stderr)
    
    # Test the Docker integration fixes
    print("\n2. Testing Docker integration fixes...")
    success, stdout, stderr = run_test("tests/integration/test_docker.py::TestDockerIntegration::test_service_status")
    if success:
        print("✅ Docker service status test PASSED (likely skipped in CI)")
    else:
        print("❌ Docker service status test FAILED")
        print(stderr)
    
    success, stdout, stderr = run_test("tests/integration/test_docker.py::TestDockerIntegration::test_service_health")
    if success:
        print("✅ Docker service health test PASSED (likely skipped in CI)")
    else:
        print("❌ Docker service health test FAILED")
        print(stderr)
    
    success, stdout, stderr = run_test("tests/integration/test_docker.py::TestDockerIntegration::test_application_endpoints")
    if success:
        print("✅ Docker application endpoints test PASSED (likely skipped in CI)")
    else:
        print("❌ Docker application endpoints test FAILED")
        print(stderr)
    
    print("\n" + "=" * 50)
    print("Test verification complete!")

if __name__ == "__main__":
    main()