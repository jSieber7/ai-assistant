#!/usr/bin/env python3
# scripts/docs.py - OPTIONAL local helper

import subprocess
import sys

def run_cmd(cmd, desc):
    """Simple local command runner"""
    print(f"📝 {desc}...")
    result = subprocess.run(cmd, shell=False)
    return result.returncode == 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/docs.py [serve|build|deploy]")
        return
    
    command = sys.argv[1]
    
    if command == "serve":
        run_cmd(["uv", "run", "mkdocs", "serve"], "Starting docs server")
    elif command == "build":
        run_cmd(["uv", "run", "mkdocs", "build"], "Building docs")
    elif command == "deploy":
        run_cmd(["uv", "run", "mkdocs", "gh-deploy", "--force"], "Deploying docs")
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()