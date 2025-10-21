"""
Status Bar Component for Chainlit App

This module provides a status bar that displays below Chainlit's default top bar:
- Current model and provider information
- Agent button for agent management
- Current hosting address for OpenAI-compatible API
- Green indicator when model is being served via API

This is designed to work alongside Chainlit's existing top bar with theme toggle and new chat button.
"""

import chainlit as cl
from typing import Dict, Any, Optional
from ..core.config import settings


async def display_status_bar(
    selected_provider: Optional[str] = None,
    selected_model: Optional[str] = None,
    api_host: Optional[str] = None,
    is_api_serving: bool = False
) -> None:
    """
    Display a status bar with model info, API status, and agent button
    This appears below Chainlit's default top bar
    
    Args:
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
        api_host: Current API host address
        is_api_serving: Whether the API is currently serving
    """
    # Determine API host
    if not api_host:
        if settings.environment == "development":
            api_host = "localhost:8000"
        else:
            api_host = f"{settings.host}:{settings.port}"
    
    # Determine status color
    status_color = "🟢" if is_api_serving else "🟡"
    status_text = "Serving" if is_api_serving else "Idle"
    
    # Format model display text
    if selected_provider and selected_model:
        model_text = f"📊 **Model:** {selected_provider}: {selected_model}"
    else:
        model_text = "📊 **Model:** No Model Selected"
    
    # Create status bar content
    status_bar_content = f"""
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background-color: #f8f9fa;
    border-bottom: 1px solid #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    margin: 0 -16px 16px -16px;
    font-size: 13px;
">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        ">
            {model_text}
        </div>
    </div>
    
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="color: #6b7280;">
            {status_color} **API:** http://{api_host} ({status_text})
        </span>
    </div>
    
    <div>
        <button id="agent-button" style="
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 6px 10px;
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        ">
            👤 Agents
        </button>
    </div>
</div>

<script>
// Add event listener for agent button
document.addEventListener('DOMContentLoaded', function() {
    const agentButton = document.getElementById('agent-button');
    if (agentButton) {
        agentButton.addEventListener('click', function() {
            // Trigger a Chainlit action
            window.chainlit.sendMessage('/agent');
        });
    }
});
</script>
    """
    
    # Send the status bar as a message
    await cl.Message(
        content=status_bar_content,
        author="System"
    ).send()


async def update_status_bar_status(is_serving: bool) -> None:
    """
    Update the API status in the status bar
    
    Args:
        is_serving: Whether the API is currently serving
    """
    status_color = "🟢" if is_serving else "🟡"
    status_text = "Serving" if is_serving else "Idle"
    
    # This would typically update an existing status bar
    # For now, we'll send a status update message
    await cl.Message(
        content=f"API Status: {status_color} {status_text}",
        author="System"
    ).send()


async def handle_agent_command() -> None:
    """
    Handle the agent command triggered by the agent button
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