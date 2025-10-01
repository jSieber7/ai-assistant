#!/usr/bin/env python3
# scripts/docs.py - OPTIONAL local helper

import subprocess
import sys


def run_cmd(cmd, desc):
    """Simple local command runner"""
    print(f"📝 {desc}...")
    print(f"   Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=False)
    return result.returncode == 0


def merge_arguments(default_args, user_args):
    """Merge default arguments with user arguments, with user args taking precedence"""
    merged = []
    default_flags = set()

    # Extract flags from default args (skip values that follow flags)
    i = 0
    while i < len(default_args):
        arg = default_args[i]
        if arg.startswith("-"):
            default_flags.add(arg.split("=")[0])  # Handle --flag=value format
            if i + 1 < len(default_args) and not default_args[i + 1].startswith("-"):
                i += 1  # Skip the value
        i += 1

    # Add user args first (they take precedence)
    for arg in user_args:
        if arg.startswith("-"):
            # Remove conflicting default flags
            flag = arg.split("=")[0]
            if flag in default_flags:
                default_flags.remove(flag)
        merged.append(arg)

    # Add remaining default args that weren't overridden
    i = 0
    while i < len(default_args):
        arg = default_args[i]
        flag = arg.split("=")[0] if arg.startswith("-") else None

        if flag and flag in default_flags:
            merged.append(arg)
            if i + 1 < len(default_args) and not default_args[i + 1].startswith("-"):
                merged.append(default_args[i + 1])
                i += 1  # Skip the value
        i += 1

    return merged


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
