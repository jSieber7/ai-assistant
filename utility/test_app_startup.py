#!/usr/bin/env python3
"""
Test script to verify the application can start without errors
"""
import sys
import os
import importlib.util


def test_import(module_name, file_path):
    """Test if a module can be imported without errors"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    """Main function to test application startup"""
    print("Testing application imports...")
    print("=" * 50)

    # Test core modules
    modules_to_test = [
        ("app.main", "app/main.py"),
        ("app.core.config", "app/core/config.py"),
        ("app.ui.gradio_app", "app/ui/gradio_app.py"),
        ("app.api.routes", "app/api/routes.py"),
    ]

    all_good = True

    for module_name, file_path in modules_to_test:
        if os.path.exists(file_path):
            is_valid, error = test_import(module_name, file_path)
            if is_valid:
                print(f"✓ {module_name}: Import OK")
            else:
                print(f"✗ {module_name}: {error}")
                all_good = False
        else:
            print(f"✗ {file_path}: File not found")
            all_good = False

    print("=" * 50)

    if all_good:
        print("All imports passed! ✅")
        return 0
    else:
        print("Some imports failed! ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
