#!/usr/bin/env python3
# scripts/docs.py - OPTIONAL local helper

import sys
from utils import run_cmd, merge_arguments


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/docs.py [serve|build|deploy] [--flags...]")
        return

    command = sys.argv[1]
    user_args = sys.argv[2:]  # Capture all additional arguments

    if command == "serve":
        default_args = ["-a", "localhost:8001", "-o"]
        cmd = ["uv", "run", "mkdocs", "serve"] + merge_arguments(
            default_args, user_args
        )
        run_cmd(cmd, "Starting docs server")
    elif command == "build":
        default_args = []
        cmd = ["uv", "run", "mkdocs", "build"] + merge_arguments(
            default_args, user_args
        )
        run_cmd(cmd, "Building docs")
    elif command == "deploy":
        default_args = ["--force"]
        cmd = ["uv", "run", "mkdocs", "gh-deploy"] + merge_arguments(
            default_args, user_args
        )
        run_cmd(cmd, "Deploying docs")
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
