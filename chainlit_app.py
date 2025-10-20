"""
Chainlit application entry point for the AI Assistant.

This file serves as the main entry point for running the Chainlit interface.
It imports and configures the Chainlit app from the UI module.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.ui.chainlit_app import create_chainlit_app

# Initialize the Chainlit app
create_chainlit_app()

# The Chainlit app is defined by decorators in chainlit_app.py
# No additional setup needed here