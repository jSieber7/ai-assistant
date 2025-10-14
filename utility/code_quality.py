#!/usr/bin/env python3
# scripts/code_quality.py - Run black, ruff, and mypy with check/fix modes

import sys
from utils import run_parallel_commands, print_summary


def get_commands(mode: str) -> list:
    """Get the commands to run based on the mode.

    Args:
        mode: Either 'check' or 'fix'

    Returns:
        List of (command, description) tuples
    """
    if mode == "check":
        return [
            (["uv", "run", "black", "--check", "."], "Black formatting check"),
            (["uv", "run", "ruff", "check", "."], "Ruff linting check"),
            (
                ["uv", "run", "mypy", "app/", "--ignore-missing-imports"],
                "Mypy type checking",
            ),
        ]
    elif mode == "fix":
        return [
            (["uv", "run", "black", "."], "Black formatting fix"),
            (["uv", "run", "ruff", "check", "--fix", "."], "Ruff linting fix"),
            (
                ["uv", "run", "mypy", "app/", "--ignore-missing-imports"],
                "Mypy type checking",
            ),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["check", "fix"]:
        print("Usage: python scripts/code_quality.py <check|fix>")
        print("\nModes:")
        print("  check - Run all tools in verification mode (no changes)")
        print("  fix   - Run tools that can automatically fix issues")
        sys.exit(1)

    mode = sys.argv[1]

    print(f"🚀 Running code quality tools in {mode} mode...")
    print("=" * 60)

    # Get commands for the selected mode
    commands = get_commands(mode)
    tool_names = ["Black", "Ruff", "Mypy"]

    # Run all commands in parallel
    results = run_parallel_commands(commands, capture_output=True)

    # Print summary
    overall_success = print_summary(results, tool_names)

    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
