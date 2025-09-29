#!/usr/bin/env python3
"""
Test runner script for AI Assistant FastAPI application.

This script provides multiple ways to run tests:
- Run all tests
- Run only unit tests
- Run only integration tests
- Run tests with coverage
- Run tests in parallel
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for AI Assistant")
    parser.add_argument(
        '--unit', 
        action='store_true', 
        help='Run only unit tests'
    )
    parser.add_argument(
        '--integration', 
        action='store_true', 
        help='Run only integration tests'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true', 
        help='Run tests with coverage report'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true', 
        help='Run tests in parallel (requires pytest-xdist)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Verbose output'
    )
    parser.add_argument(
        '--slow', 
        action='store_true', 
        help='Include slow-running tests'
    )
    parser.add_argument(
        '--failed-first', 
        action='store_true', 
        help='Run failed tests first'
    )
    parser.add_argument(
        '--ci', 
        action='store_true', 
        help='CI mode: minimal output, generate reports'
    )

    args = parser.parse_args()
    
    # Build pytest command
    pytest_cmd = ['python', '-m', 'pytest']
    
    # Add options based on arguments
    if args.unit:
        pytest_cmd.extend(['-m', 'not integration and not slow'])
    elif args.integration:
        pytest_cmd.extend(['-m', 'integration'])
    else:
        # Run all tests by default, but exclude slow unless specified
        if not args.slow:
            pytest_cmd.extend(['-m', 'not slow'])
    
    if args.coverage:
        pytest_cmd.extend([
            '--cov=app',
            '--cov-report=term',
            '--cov-report=html:coverage_html',
            '--cov-report=xml:coverage.xml'
        ])
    
    if args.parallel:
        pytest_cmd.extend(['-n', 'auto'])
    
    if args.verbose:
        pytest_cmd.append('-v')
    
    if args.failed_first:
        pytest_cmd.append('--failed-first')

    if args.ci:
        pytest_cmd.extend([
            '--junitxml=junit.xml',
            '--cov-report=xml',
            '-q'  # Quiet mode for CI
        ])
    
    # Add test directory
    pytest_cmd.append('tests/')
    
    # Run the command
    success = run_command(pytest_cmd, "Test Suite")
    
    if success:
        print("\nAll tests passed.")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())