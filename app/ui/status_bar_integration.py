"""
Integration example for the Status Bar Component

This module shows how to integrate the Status Bar component into the main Chainlit app.
It demonstrates how to:
1. Display the status bar with current state
2. Update the status bar when settings change
3. Handle agent button clicks
"""

import chainlit as cl
from typing import Dict, Any, Optional
from .status_bar_component import display_status_bar, update_status_bar_status, handle_agent_command
from .chainlit_app import DropdownState


async def show_status_bar_with_state(state: DropdownState) -> None:
    """
    Display the status bar with current application state
    
    Args:
        state: Current dropdown state with provider and model information
    """
    # Determine API status (you might want to check actual API health here)
    is_api_serving = state.selected_provider is not None and state.selected_model is not None
    
    # Display the status bar
    await display_status_bar(
        selected_provider=state.selected_provider,
        selected_model=state.selected_model,
        is_api_serving=is_api_serving
    )


async def update_status_bar_on_settings_change(settings: Dict[str, Any], state: DropdownState) -> None:
    """
    Update the status bar when settings change
    
    Args:
        settings: Updated settings
        state: Current dropdown state
    """
    # Check if provider and model are configured
    is_api_serving = state.selected_provider is not None and state.selected_model is not None
    
    # Update the status bar status
    await update_status_bar_status(is_api_serving)
    
    # Display updated status bar
    await show_status_bar_with_state(state)


# Example of how to integrate with the main Chainlit app
"""
To integrate the Status Bar component into your main Chainlit app (app/ui/chainlit_app.py):

1. Import the necessary functions:
   from .status_bar_integration import show_status_bar_with_state, update_status_bar_on_settings_change
   from .status_bar_component import handle_agent_command

2. Modify the on_chat_start function to display the status bar:

@cl.on_chat_start
async def on_chat_start():
    # Initialize providers
    initialize_llm_providers()
    
    # Initialize dropdown state
    dropdown_state = await initialize_dropdown_state()
    
    # Send welcome message
    await cl.Message(
        content="# AI Assistant\n\nWelcome to the AI Assistant! Please configure your provider and model settings to get started.",
        author="System"
    ).send()
    
    # Display the status bar
    await show_status_bar_with_state(dropdown_state)
    
    # Store settings in session
    cl.user_session.set("dropdown_state", dropdown_state)
    
    # Show settings with dropdown widgets
    await show_settings_widget(dropdown_state)

3. Modify the on_settings_update function to update the status bar:

@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    # Get current state
    state = cl.user_session.get("dropdown_state")
    if not state:
        state = await initialize_dropdown_state()
        cl.user_session.set("dropdown_state", state)
    
    # Update state with new values
    old_provider = state.selected_provider
    state.selected_provider = settings.get("provider")
    state.selected_model = settings.get("model")
    state.search_query = settings.get("search", "")
    
    # If provider changed, update models list
    if old_provider != state.selected_provider:
        models = await get_models_for_provider(state.selected_provider or "")
        state.models[state.selected_provider or ""] = models
        
        # Update the model dropdown
        await update_model_dropdown(models, state.selected_model)
    
    # Apply search filter if search query changed
    if state.search_query:
        models = state.models.get(state.selected_provider or "", [])
        filtered_models = filter_models(models, state.search_query)
        await update_model_dropdown(filtered_models, state.selected_model)
    
    # Update the status bar
    await update_status_bar_on_settings_change(settings, state)
    
    # Store updated state
    cl.user_session.set("dropdown_state", state)
    
    # Show confirmation
    if state.selected_provider and state.selected_model:
        model_info = await get_model_info_for_provider(state.selected_provider, state.selected_model)
        
        info_text = f"""
## Configuration Updated

**Provider:** {state.selected_provider}
**Model:** {state.selected_model}
**Temperature:** {settings.get('temperature', 0.7)}
**Max Tokens:** {settings.get('max_tokens', 0) if settings.get('max_tokens', 0) > 0 else "Unlimited"}

### Model Details
- **Context Length:** {model_info.get('context_length', 'Unknown')}
- **Supports Streaming:** {'Yes' if model_info.get('supports_streaming', True) else 'No'}
- **Supports Tools:** {'Yes' if model_info.get('supports_tools', True) else 'No'}
- **Description:** {model_info.get('description', 'No description available')}

You can now start chatting with the AI assistant!
        """
        
        await cl.Message(
            content=info_text,
            author="System"
        ).send()

4. Add a command handler for the agent button:

@cl.on_message
async def on_message(message: cl.Message):
    # Handle agent command
    if message.content == "/agent":
        await handle_agent_command()
        return
    
    # Get current state
    state = cl.user_session.get("dropdown_state")
    
    # Check if provider and model are configured
    if not state or not state.selected_provider or not state.selected_model:
        await cl.Message(
            content="Please configure a provider and model first using the settings above.",
            author="System"
        ).send()
        return
    
    # Process other commands
    if message.content.startswith("/"):
        await handle_command(message.content)
        return
    
    # Rest of your existing message handling code...

5. Add action handlers for agent management:

@cl.action_callback("list_agents")
async def on_list_agents(action: cl.Action):
    """Handle list agents action"""
    await cl.Message(
        content="## Available Agents\n\nNo agents configured yet.",
        author="System"
    ).send()

@cl.action_callback("create_agent")
async def on_create_agent(action: cl.Action):
    """Handle create agent action"""
    await cl.Message(
        content="## Create New Agent\n\nAgent creation form would appear here.",
        author="System"
    ).send()

@cl.action_callback("configure_agent")
async def on_configure_agent(action: cl.Action):
    """Handle configure agent action"""
    await cl.Message(
        content="## Configure Agent\n\nAgent configuration options would appear here.",
        author="System"
    ).send()
"""