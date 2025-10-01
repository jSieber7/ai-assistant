#!/usr/bin/env python3
"""
Test script to validate GitHub Actions workflow configuration.

This script checks that the necessary files and configurations are in place
for the GitHub Actions workflows to function properly.
"""

import os
import yaml
import importlib.util


def check_package_available(spec):
    """Check if package is available and return status."""
    if importlib.util.find_spec(spec) is not None:
        print(f"✅ {spec} is available")
    else:
        print(f"❌ {spec} is not installed")


def check_file_exists(filepath):
    """Check if a file exists and return status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    return status, exists


def validate_yaml(filepath):
    """Validate YAML syntax."""
    try:
        with open(filepath, "r") as f:
            yaml.safe_load(f)
        return "✅", True
    except yaml.YAMLError as e:
        return f"❌ ({str(e)})", False


def check_workflow_configuration():
    """Check GitHub Actions workflow configuration."""
    print("🔍 Checking GitHub Actions Configuration")
    print("=" * 50)

    checks = []

    # Check workflow files
    workflow_files = [
        ".github/workflows/python-tests.yml",
        ".github/workflows/branch-protection.yml",
    ]

    for workflow_file in workflow_files:
        status, exists = check_file_exists(workflow_file)
        checks.append((workflow_file, status, exists))

        if exists:
            yaml_status, valid = validate_yaml(workflow_file)
            checks.append(("  YAML syntax", yaml_status, valid))

    # Check documentation files
    doc_files = [".github/branch-protection.md", "readme.md"]

    for doc_file in doc_files:
        status, exists = check_file_exists(doc_file)
        checks.append((doc_file, status, exists))

    # Check project structure
    required_dirs = [".github", ".github/workflows"]
    for directory in required_dirs:
        status, exists = check_file_exists(directory)
        checks.append((f"Directory: {directory}", status, exists))

    # Print results
    all_passed = True
    for check_name, status, passed in checks:
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("✅ All GitHub Actions configuration checks passed!")
        print("\nNext steps:")
        print("1. Push these changes to your repository")
        print("2. Go to GitHub repository Settings → Branches")
        print(
            "3. Set up branch protection rules as described in .github/branch-protection.md"
        )
        print("4. Create a pull request to test the workflow")
    else:
        print("❌ Some checks failed. Please review the configuration.")

    return all_passed


def check_python_dependencies():
    """Check that required Python dependencies are available."""
    print("\n🔍 Checking Python Dependencies")
    print("=" * 50)

    check_package_available("pytest")
    check_package_available("fastapi")
    check_package_available("langchain")


def main():
    """Run all validation checks."""
    print("GitHub Actions Configuration Validator")
    print("=" * 50)

    workflow_ok = check_workflow_configuration()
    check_python_dependencies()

    if workflow_ok:
        print("\n🎉 Configuration looks good! Ready to set up GitHub Actions.")
        return 0
    else:
        print("\n⚠️  Some issues found. Please fix them before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
