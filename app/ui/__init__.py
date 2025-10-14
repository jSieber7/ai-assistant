"""
UI module for the AI Assistant application.

This module contains the Gradio interface for interacting with the AI Assistant.
"""

from .gradio_app import create_gradio_app, mount_gradio_app

__all__ = ["create_gradio_app", "mount_gradio_app"]