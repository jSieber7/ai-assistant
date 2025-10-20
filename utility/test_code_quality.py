#!/usr/bin/env python3
"""
Simple code quality test script
"""
import ast
import sys
import os


def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, "r") as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    """Main function to test code quality"""
    # Files to check
    files_to_check = [
        "app/main.py",
        "app/core/config.py",
        "app/api/routes.py",
        "Makefile",
        "docker-compose.yml",
        "Dockerfile",
    ]

    print("Checking code quality...")
    print("=" * 50)

    all_good = True

    for file_path in files_to_check:
        if os.path.exists(file_path):
            if file_path.endswith(".py"):
                # Check Python syntax
                is_valid, error = check_syntax(file_path)
                if is_valid:
                    print(f"✓ {file_path}: Syntax OK")
                else:
                    print(f"✗ {file_path}: {error}")
                    all_good = False
            else:
                print(f"✓ {file_path}: File exists")
        else:
            print(f"✗ {file_path}: File not found")
            all_good = False

    print("=" * 50)

    if all_good:
        print("All checks passed! ✅")
        return 0
    else:
        print("Some checks failed! ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
