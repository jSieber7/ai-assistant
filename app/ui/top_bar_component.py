"""
Top Bar Component for Chainlit App

This module provides a top navigation bar that displays:
- Current model and provider information
- Agent button for agent management
- Current hosting address for OpenAI-compatible API
- Green indicator when model is being served via API
"""

import chainlit as cl
from typing import Dict, Any, Optional
from ..core.config import settings


async def display_top_bar(
    selected_provider: Optional[str] = None,
    selected_model: Optional[str] = None,
    is_api_serving: bool = False,
    output_api_endpoint: Optional[str] = None
) -> None:
    """
    Display a top bar with model info, API status, and agent button
    
    Args:
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
        is_api_serving: Whether the API is currently serving
        output_api_endpoint: Output API endpoint where users connect to use this app
    """
    # Determine status color
    status_color = "🟢" if is_api_serving else "🟡"
    status_text = "Serving" if is_api_serving else "Idle"
    
    # Format model display text
    if selected_provider and selected_model:
        model_text = f"📊 **Model:** {selected_provider}: {selected_model}"
    else:
        model_text = "📊 **Model:** No Model Selected"
    
    # Determine which endpoint to display
    if output_api_endpoint:
        endpoint_display = f"**OpenAI API:** {output_api_endpoint}"
    else:
        endpoint_display = "**API:** Not Available"
    
    # Create top bar content
    top_bar_content = f"""
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background-color: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin: -16px -16px 16px -16px;
">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
        ">
            {model_text}
        </div>
    </div>
    
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 13px; color: #6b7280;">
            {status_color} {endpoint_display} ({status_text})
        </span>
    </div>
    
    <div>
        <button id="agent-button" style="
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        ">
            👤 Agents
        </button>
    </div>
</div>

<script>
document.getElementById('agent-button').addEventListener('click', function() {
    // This would trigger an agent management action
    console.log('Agent button clicked');
});
</script>
    """
    
    # Send the top bar as a message
    await cl.Message(
        content=top_bar_content,
        author="System"
    ).send()


async def update_top_bar_status(is_serving: bool) -> None:
    """
    Update the API status in the top bar
    
    Args:
        is_serving: Whether the API is currently serving
    """
    status_color = "🟢" if is_serving else "🟡"
    status_text = "Serving" if is_serving else "Idle"
    
    # This would typically update an existing top bar
    # For now, we'll send a status update message
    await cl.Message(
        content=f"API Status: {status_color} {status_text}",
        author="System"
    ).send()


async def handle_agent_action() -> None:
    """
    Handle the agent button click action
    """
    # Display agent management options
    actions = [
        cl.Action(name="list_agents", value="list", label="📋 List Agents", payload={}),
        cl.Action(name="create_agent", value="create", label="➕ Create Agent", payload={}),
        cl.Action(name="configure_agent", value="configure", label="⚙️ Configure Agent", payload={}),
    ]
    
    await cl.Message(
        content="## Agent Management\n\nSelect an action:",
        actions=actions,
        author="System"
    ).send()