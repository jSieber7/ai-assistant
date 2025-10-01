#!/usr/bin/env python3
# scripts/utils.py - Shared utilities for scripts

import subprocess
from typing import List, Tuple


def run_cmd(
    cmd: List[str], desc: str, capture_output: bool = False
) -> Tuple[bool, str]:
    """Run command with improved error handling and output capture.

    Args:
        cmd: Command to run as list of strings
        desc: Description of what the command does
        capture_output: Whether to capture and return output

    Returns:
        Tuple of (success: bool, output: str)
    """
    print(f"📝 {desc}...")
    print(f"   Command: {' '.join(cmd)}")

    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr
        else:
            result = subprocess.run(cmd)
            output = ""

        success = result.returncode == 0

        if success:
            print(f"✅ {desc} completed successfully")
        else:
            print(f"❌ {desc} failed with exit code {result.returncode}")
            if capture_output and output:
                print(f"   Output: {output}")

        return success, output

    except Exception as e:
        print(f"❌ {desc} failed with exception: {e}")
        return False, str(e)


def run_parallel_commands(
    commands: List[Tuple[List[str], str]], capture_output: bool = False
) -> List[Tuple[bool, str]]:
    """Run multiple commands in parallel and collect results.

    Args:
        commands: List of (command, description) tuples
        capture_output: Whether to capture output

    Returns:
        List of (success, output) tuples for each command
    """
    import concurrent.futures

    def run_single(cmd_desc):
        cmd, desc = cmd_desc
        return run_cmd(cmd, desc, capture_output)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_single, commands))

    return results


def merge_arguments(default_args: List[str], user_args: List[str]) -> List[str]:
    """Merge default arguments with user arguments, with user args taking precedence.

    Args:
        default_args: Default arguments
        user_args: User-provided arguments

    Returns:
        Merged argument list
    """
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


def print_summary(results: List[Tuple[bool, str]], tool_names: List[str]) -> bool:
    """Print a summary of tool results and return overall success.

    Args:
        results: List of (success, output) tuples
        tool_names: Names of the tools that were run

    Returns:
        True if all tools succeeded, False otherwise
    """
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_success = True
    for i, (success, output) in enumerate(results):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{tool_names[i]}: {status}")
        if not success and output:
            print(f"   Details: {output.strip()}")
        all_success = all_success and success

    print("=" * 50)
    if all_success:
        print("🎉 All checks passed!")
    else:
        print("💥 Some checks failed. Please review the output above.")

    return all_success
