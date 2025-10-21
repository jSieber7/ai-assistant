"""
Chainlit interface for the AI Assistant application.

This module provides a web UI with:
1. An input box for testing queries
2. A nested, dynamic dropdown for provider and model selection using actual Select widgets
3. A search functionality for filtering models within providers
4. Provider testing with success/error messages
5. Proper state management for the dynamic dropdown system
6. A popup form for adding new providers
"""

import chainlit as cl
import httpx
import logging
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from ..core.config import settings, get_available_models, initialize_llm_providers
from ..core.tools import tool_registry
from ..core.llm_providers import provider_registry, ProviderType, ModelInfo
from chainlit.input_widget import Select, TextInput, Switch, Slider

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DropdownState:
    """State management for the dropdown system"""
    selected_provider: Optional[str] = None
    selected_model: Optional[str] = None
    providers: List[str] = None
    models: Dict[str, List[str]] = None
    provider_info: Dict[str, Dict[str, Any]] = None
    search_query: str = ""
    filtered_models: List[str] = None
    show_add_provider: bool = False
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = []
        if self.models is None:
            self.models = {}
        if self.provider_info is None:
            self.provider_info = {}
        if self.filtered_models is None:
            self.filtered_models = []


async def initialize_dropdown_state() -> DropdownState:
    """Initialize the dropdown state with providers and models"""
    state = DropdownState()
    
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()

        providers = provider_registry.list_configured_providers()
        
        # Extract provider names
        provider_names = [provider.name for provider in providers]
        
        state.providers = provider_names
        
        # Pre-fetch models for all providers
        for provider in providers:
            try:
                models = await provider.list_models()
                state.models[provider.name] = [model.name for model in models]
                
                # Store provider info
                state.provider_info[provider.name] = {
                    "type": provider.provider_type.value,
                    "is_healthy": provider.is_healthy(),
                    "is_configured": provider.is_configured,
                }
            except Exception as e:
                logger.debug(f"Error pre-fetching models for {provider.name}: {str(e)}")
                state.models[provider.name] = []
                state.provider_info[provider.name] = {
                    "type": provider.provider_type.value,
                    "is_healthy": False,
                    "is_configured": provider.is_configured,
                    "error": str(e),
                }
        
        return state
    
    except Exception as e:
        logger.debug(f"Error initializing dropdown state: {str(e)}")
        state.providers = []
        return state


async def get_providers_list() -> List[str]:
    """Get list of available providers for the dropdown"""
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()

        providers = provider_registry.list_configured_providers()
        
        if not providers:
            return []
        
        # Extract provider names
        provider_names = [provider.name for provider in providers]
        
        return provider_names
    
    except Exception as e:
        logger.debug(f"Error getting providers list (expected if no providers configured): {str(e)}")
        return []


async def get_models_for_provider(provider_name: str) -> List[str]:
    """Get list of available models for a specific provider"""
    if not provider_name:
        return []
    
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()
        
        # Find the provider by name
        providers = provider_registry.list_configured_providers()
        target_provider = None
        
        for provider in providers:
            if provider.name == provider_name:
                target_provider = provider
                break
        
        if not target_provider:
            return []
        
        # Get models from the provider
        models = await target_provider.list_models()
        return [model.name for model in models]
    
    except Exception as e:
        logger.debug(f"Error getting models for provider {provider_name} (expected if provider not configured): {str(e)}")
        return []


async def get_model_info_for_provider(provider_name: str, model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model from a provider"""
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()
        
        # Find the provider by name
        providers = provider_registry.list_configured_providers()
        target_provider = None
        
        for provider in providers:
            if provider.name == provider_name:
                target_provider = provider
                break
        
        if not target_provider:
            return {}
        
        # Get models from the provider
        models = await target_provider.list_models()
        for model in models:
            if model.name == model_name:
                return {
                    "context_length": model.context_length,
                    "supports_streaming": model.supports_streaming,
                    "supports_tools": model.supports_tools,
                    "description": model.description,
                }
        
        return {}
    
    except Exception as e:
        logger.debug(f"Error getting model info for {provider_name}:{model_name}: {str(e)}")
        return {}


async def test_provider_connection(provider_name: str) -> Tuple[bool, str]:
    """Test the connection to a provider"""
    if not provider_name:
        return False, "Invalid provider name"
    
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()
        
        # Find the provider by name
        providers = provider_registry.list_configured_providers()
        target_provider = None
        
        for provider in providers:
            if provider.name == provider_name:
                target_provider = provider
                break
        
        if not target_provider:
            return False, f"Provider '{provider_name}' not found"
        
        # Test health check
        is_healthy = await target_provider.health_check()
        
        if is_healthy:
            # Try to get models as an additional test
            models = await target_provider.list_models()
            return True, f"Connection successful. Found {len(models)} models."
        else:
            return False, "Health check failed"
    
    except Exception as e:
        logger.error(f"Error testing provider {provider_name}: {str(e)}")
        return False, f"Connection test failed: {str(e)}"


def filter_models(models: List[str], search_query: str) -> List[str]:
    """Filter models based on search query"""
    if not search_query:
        return models
    
    search_query = search_query.lower()
    filtered = []
    
    for model in models:
        if search_query in model.lower():
            filtered.append(model)
    
    return filtered


async def execute_query(
    message: str,
    provider_name: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Test a query against the AI assistant"""
    if not message or not message.strip():
        return "Error: Please enter a message to test"
    
    if not provider_name:
        return "Error: Please select a provider"
    
    if not model_name:
        return "Error: Please select a model"
    
    try:
        # Find the provider by name
        providers = provider_registry.list_configured_providers()
        target_provider = None
        
        for provider in providers:
            if provider.name == provider_name:
                target_provider = provider
                break
        
        if not target_provider:
            return f"Error: Provider '{provider_name}' not found"
        
        # Prepare the request payload
        payload = {
            "messages": [{"role": "user", "content": message.strip()}],
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens > 0 else None,
            "stream": False,
        }
        
        # Make the request to the FastAPI endpoint
        if settings.environment == "development":
            base_url = "http://localhost:8000"
        else:
            base_url = f"http://{settings.host}:{settings.port}"
        
        logger.info(f"Sending query to {base_url}/v1/chat/completions")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Validate response structure
                if "choices" not in result or not result["choices"]:
                    return "Error: Invalid response format from server"
                
                content = result["choices"][0]["message"]["content"]
                
                # Format response with additional information
                formatted_response = []
                
                # Add model info
                formatted_response.append(f"**Model:** {result.get('model', model_name)}")
                formatted_response.append(f"**Provider:** {provider_name}")
                formatted_response.append("\n--- Response ---\n")
                formatted_response.append(content)
                
                # Add usage info if available
                if "usage" in result:
                    usage = result["usage"]
                    formatted_response.append("\n\n--- Usage Information ---")
                    if "prompt_tokens" in usage:
                        formatted_response.append(f"Prompt tokens: {usage['prompt_tokens']}")
                    if "completion_tokens" in usage:
                        formatted_response.append(f"Completion tokens: {usage['completion_tokens']}")
                    if "total_tokens" in usage:
                        formatted_response.append(f"Total tokens: {usage['total_tokens']}")
                
                return "\n".join(formatted_response)
            
            except json.JSONDecodeError:
                return f"Error: Invalid JSON response from server\nRaw response: {response.text[:500]}"
        
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    error_msg += f": {error_detail['detail']}"
                else:
                    error_msg += f": {json.dumps(error_detail)}"
            except Exception:
                error_msg += ": Server error occurred"
            
            return f"Error: {error_msg}"
    
    except httpx.TimeoutException:
        return "Error: Request timed out. The server took too long to respond."
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {settings.host}:{settings.port}. Please ensure the server is running."
    except Exception as e:
        logger.error(f"Error testing query: {str(e)}", exc_info=True)
        return f"Error testing query: {str(e)}"


async def add_new_provider(provider_name: str, provider_type: str, api_key: str, base_url: str) -> str:
    """Add a new provider to the system"""
    if not provider_name or not provider_name.strip():
        return "Error: Please enter a provider name"
    
    if not provider_type:
        return "Error: Please select a provider type"
    
    if provider_type in ["openai_compatible", "openrouter"] and not api_key:
        return "Error: API key is required for this provider type"
    
    try:
        from ..core.llm_providers import OpenAICompatibleProvider, OllamaProvider
        
        if provider_type == "ollama":
            # Create Ollama provider
            new_provider = OllamaProvider(base_url=base_url or "http://localhost:11434")
        else:
            # Create OpenAI-compatible provider
            new_provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                provider_name=provider_name,
            )
        
        # Register the provider
        provider_registry.register_provider(new_provider)
        
        return f"Success: Provider '{provider_name}' has been added successfully!"
    
    except Exception as e:
        logger.error(f"Error adding provider: {str(e)}", exc_info=True)
        return f"Error adding provider: {str(e)}"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # Initialize providers
    initialize_llm_providers()
    
    # Initialize dropdown state
    dropdown_state = await initialize_dropdown_state()
    
    # Send welcome message
    await cl.Message(
        content="# AI Assistant\n\nWelcome to the AI Assistant! Please configure your provider and model settings to get started.",
        author="System"
    ).send()
    
    # Store settings in session
    cl.user_session.set("dropdown_state", dropdown_state)
    
    # Show settings with dropdown widgets
    await show_settings_widget(dropdown_state)


async def show_settings_widget(state: DropdownState):
    """Show settings with actual dropdown widgets"""
    # Get current providers list
    providers = await get_providers_list()
    
    # Create provider dropdown
    provider_select = Select(
        id="provider",
        label="Select Provider",
        values=providers,
        initial=state.selected_provider or (providers[0] if providers else None)
    )
    
    # Create model dropdown (will be updated when provider changes)
    models = []
    if state.selected_provider:
        models = await get_models_for_provider(state.selected_provider)
    
    # Only create model_select if there are models available
    if models:
        model_select = Select(
            id="model",
            label="Select Model",
            values=models,
            initial=state.selected_model or (models[0] if models else None)
        )
    else:
        # Create a placeholder widget when no models are available
        model_select = TextInput(
            id="model",
            label="Model",
            initial="No models available",
            disabled=True
        )
    
    # Create additional settings
    temperature_slider = Slider(
        id="temperature",
        label="Temperature",
        initial=0.7,
        min=0.0,
        max=2.0,
        step=0.1
    )
    
    max_tokens_slider = Slider(
        id="max_tokens",
        label="Max Tokens (0 for unlimited)",
        initial=0,
        min=0,
        max=4096,
        step=64
    )
    
    # Create search input for models
    search_input = TextInput(
        id="search",
        label="Search Models",
        initial=state.search_query,
        placeholder="Enter search terms to filter models"
    )
    
    # Create settings with all widgets
    settings = cl.ChatSettings(
        inputs=[provider_select, model_select, search_input, temperature_slider, max_tokens_slider]
    )
    
    # Store settings reference for later updates
    cl.user_session.set("settings_widget", settings)
    
    # Send settings
    await settings.send()
    
    # Show action buttons
    await show_action_buttons()


async def show_action_buttons():
    """Show action buttons for testing and adding providers"""
    actions = [
        cl.Action(name="test_provider", value="test", label="🧪 Test Provider", payload={}),
        cl.Action(name="add_provider", value="add", label="➕ Add New Provider", payload={}),
        cl.Action(name="refresh_providers", value="refresh", label="🔄 Refresh Providers", payload={}),
    ]
    
    await cl.Message(
        content="## Actions\n\nUse the buttons below to test your provider connection or add a new provider:",
        actions=actions,
        author="System"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    """Handle settings updates"""
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


async def update_model_dropdown(models: List[str], selected_model: Optional[str] = None):
    """Update the model dropdown with new values"""
    # Get the current settings widget
    settings_widget = cl.user_session.get("settings_widget")
    if not settings_widget:
        return
    
    # Find and update the model select widget
    for widget in settings_widget.inputs:
        if widget.id == "model":
            if isinstance(widget, Select):
                widget.values = models
                if selected_model and selected_model in models:
                    widget.initial = selected_model
                elif models:
                    widget.initial = models[0]
                else:
                    widget.initial = None
            break
    
    # Resend the settings
    await settings_widget.send()


@cl.action_callback("test_provider")
async def on_test_provider(action: cl.Action):
    """Handle provider testing"""
    # Get current state
    state = cl.user_session.get("dropdown_state")
    if not state:
        state = await initialize_dropdown_state()
        cl.user_session.set("dropdown_state", state)
    
    if not state.selected_provider:
        await cl.Message(
            content="Please select a provider first before testing.",
            author="System"
        ).send()
        return
    
    # Show testing message
    await cl.Message(
        content=f"Testing connection to {state.selected_provider}...",
        author="System"
    ).send()
    
    # Test provider
    success, message = await test_provider_connection(state.selected_provider)
    
    # Show result
    if success:
        result_content = f"✅ **Connection Test Successful**\n\n{message}"
    else:
        result_content = f"❌ **Connection Test Failed**\n\n{message}"
    
    await cl.Message(
        content=result_content,
        author="System"
    ).send()


@cl.action_callback("add_provider")
async def on_add_provider(action: cl.Action):
    """Handle add provider action"""
    await show_add_provider_form()


async def show_add_provider_form():
    """Show form to add a new provider"""
    # Create form widgets
    name_input = TextInput(
        id="provider_name",
        label="Provider Name",
        placeholder="Enter a unique name for this provider"
    )
    
    type_select = Select(
        id="provider_type",
        label="Provider Type",
        values=["openai_compatible", "ollama"],
        initial="openai_compatible"
    )
    
    api_key_input = TextInput(
        id="api_key",
        label="API Key",
        placeholder="Enter your API key (required for OpenAI-compatible providers)"
    )
    
    base_url_input = TextInput(
        id="base_url",
        label="Base URL",
        placeholder="Enter the base URL (optional, uses default if not provided)"
    )
    
    # Create form
    form = cl.ChatSettings(
        inputs=[name_input, type_select, api_key_input, base_url_input]
    )
    
    # Store form reference
    cl.user_session.set("add_provider_form", form)
    
    # Send form with submit button
    await form.send()
    
    # Show submit button
    await cl.Message(
        content="## Add New Provider\n\nFill in the form above and click submit:",
        actions=[
            cl.Action(name="submit_provider", value="submit", label="✅ Submit", payload={}),
            cl.Action(name="cancel_add_provider", value="cancel", label="❌ Cancel", payload={}),
        ],
        author="System"
    ).send()


@cl.action_callback("submit_provider")
async def on_submit_provider(action: cl.Action):
    """Handle provider form submission"""
    # Get form data
    form_data = cl.user_session.get("add_provider_form_data", {})
    
    # Extract values
    provider_name = form_data.get("provider_name", "")
    provider_type = form_data.get("provider_type", "")
    api_key = form_data.get("api_key", "")
    base_url = form_data.get("base_url", "")
    
    # Add the provider
    result = await add_new_provider(provider_name, provider_type, api_key, base_url)
    
    await cl.Message(
        content=f"## Provider Addition Result\n\n{result}",
        author="System"
    ).send()
    
    # Clear form data
    cl.user_session.set("add_provider_form_data", None)
    
    # Refresh dropdown state
    dropdown_state = await initialize_dropdown_state()
    cl.user_session.set("dropdown_state", dropdown_state)
    
    # Show updated settings
    await show_settings_widget(dropdown_state)


@cl.action_callback("cancel_add_provider")
async def on_cancel_add_provider(action: cl.Action):
    """Handle cancel add provider"""
    # Clear form data
    cl.user_session.set("add_provider_form_data", None)
    cl.user_session.set("add_provider_form", None)
    
    await cl.Message(
        content="Provider addition cancelled.",
        author="System"
    ).send()


@cl.action_callback("refresh_providers")
async def on_refresh_providers(action: cl.Action):
    """Handle refresh providers"""
    # Refresh dropdown state
    dropdown_state = await initialize_dropdown_state()
    cl.user_session.set("dropdown_state", dropdown_state)
    
    await cl.Message(
        content="Provider list refreshed.",
        author="System"
    ).send()
    
    # Show updated settings
    await show_settings_widget(dropdown_state)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    # Get current state
    state = cl.user_session.get("dropdown_state")
    
    # Check if provider and model are configured
    if not state or not state.selected_provider or not state.selected_model:
        await cl.Message(
            content="Please configure a provider and model first using the settings above.",
            author="System"
        ).send()
        return
    
    # Process commands
    if message.content.startswith("/"):
        await handle_command(message.content)
        return
    
    # Show typing indicator
    await cl.Message(
        content="⏳ Processing your query...",
        author="System"
    ).send()
    
    # Get current settings
    settings_widget = cl.user_session.get("settings_widget")
    settings_values = {}
    if settings_widget:
        for widget in settings_widget.inputs:
            settings_values[widget.id] = getattr(widget, 'initial', None)
    
    # Execute the query
    result = await execute_query(
        message.content,
        state.selected_provider,
        state.selected_model,
        settings_values.get("temperature", 0.7),
        settings_values.get("max_tokens", 0),
    )
    
    # Send the response
    await cl.Message(
        content=result,
        author="AI Assistant"
    ).send()


async def handle_command(command: str):
    """Handle special commands"""
    if command == "/settings":
        state = cl.user_session.get("dropdown_state")
        if state:
            await show_settings_widget(state)
    elif command == "/reset":
        await reset_settings()
    elif command == "/help":
        await show_help()
    else:
        await cl.Message(
            content=f"Unknown command: {command}\n\nType `/help` for available commands.",
            author="System"
        ).send()


async def reset_settings():
    """Reset settings and start over"""
    # Initialize new dropdown state
    dropdown_state = await initialize_dropdown_state()
    
    # Update session
    cl.user_session.set("dropdown_state", dropdown_state)
    
    await cl.Message(
        content="## Reset Configuration\n\nConfiguration has been reset. Please select a provider:",
        author="System"
    ).send()
    
    # Show settings
    await show_settings_widget(dropdown_state)


async def show_help():
    """Show help information"""
    help_text = """
## AI Assistant Help

### Commands
- `/settings` - Show current configuration
- `/reset` - Reset configuration and select new provider
- `/help` - Show this help message

### How to Use
1. Select a provider from the dropdown
2. Select a model from the available models
3. Adjust temperature and max tokens as needed
4. Start chatting with the AI assistant

### Adding New Providers
Click the "Add New Provider" button and fill in the form to add a new provider.

### Supported Provider Types
- **OpenAI Compatible** - For OpenAI API compatible services
- **Ollama** - For local Ollama instances
    """
    
    await cl.Message(
        content=help_text,
        author="System"
    ).send()


def create_chainlit_app():
    """Create the Chainlit application"""
    # This function is for compatibility with the existing code structure
    # The actual app is defined by the decorators above
    pass