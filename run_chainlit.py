#!/usr/bin/env python3
"""
Script to run the Chainlit interface for the AI Assistant.

This script provides a convenient way to start the Chainlit interface
with default settings.
"""

import os
import sys
import subprocess
import argparse

def main():
    """Run the Chainlit interface"""
    parser = argparse.ArgumentParser(description="Run the Chainlit interface for AI Assistant")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to (default: 8001)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Build the command
    cmd = [
        "uv", "run", "chainlit", "run", "chainlit_app.py",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"Starting Chainlit interface on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping Chainlit interface...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Chainlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please install uv first.")
        sys.exit(1)

if __name__ == "__main__":
    main()