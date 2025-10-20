"""
Simplified Gradio interface for the AI Assistant application.

This module provides a single-page web UI with:
1. An input box for testing queries
2. A dynamic dropdown for provider selection
3. A model dropdown that appears when a provider is selected
4. An "Add Provider" option
"""

import gradio as gr
import httpx
import logging
import json
from typing import List, Optional, Tuple, Dict
from ..core.config import settings, get_available_models, initialize_llm_providers
from ..core.tools import tool_registry
from ..core.llm_providers import provider_registry, ProviderType, ModelInfo

# Set up logging
logger = logging.getLogger(__name__)


def get_providers_list() -> List[str]:
    """Get list of available providers for the dropdown"""
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()

        providers = provider_registry.list_configured_providers()
        
        if not providers:
            return ["Add Provider"]
        
        # Extract provider names
        provider_names = [provider.name for provider in providers]
        provider_names.append("Add Provider")
        
        return provider_names
    
    except Exception as e:
        logger.debug(f"Error getting providers list (expected if no providers configured): {str(e)}")
        return ["Add Provider"]


def get_models_for_provider(provider_name: str) -> List[str]:
    """Get list of available models for a specific provider"""
    if provider_name == "Add Provider" or not provider_name:
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
        import asyncio
        
        def _get_models_sync():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(target_provider.list_models())
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Failed to get models from {provider_name}: {str(e)}")
                return []
        
        # Use threading to avoid event loop issues
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_get_models_sync)
            try:
                models = future.result(timeout=10)
                return [model.name for model in models]
            except Exception as e:
                logger.error(f"Failed to get models from thread: {str(e)}")
                return []
    
    except Exception as e:
        logger.debug(f"Error getting models for provider {provider_name} (expected if provider not configured): {str(e)}")
        return []


def update_model_dropdown(provider_name: str) -> gr.Dropdown:
    """Update the model dropdown based on the selected provider"""
    models = get_models_for_provider(provider_name)
    
    if not models:
        return gr.Dropdown(
            choices=[],
            value=None,
            visible=False,
            interactive=False
        )
    
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else None,
        visible=True,
        interactive=True
    )


async def execute_query_function(
    message: str,
    provider_name: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Test a query against the AI assistant"""
    if not message or not message.strip():
        return "Error: Please enter a message to test"
    
    if not provider_name or provider_name == "Add Provider":
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


def show_add_provider_dialog():
    """Show a dialog for adding a new provider"""
    return gr.Textbox(visible=True, label="Provider Name")


def add_new_provider(provider_name: str, provider_type: str, api_key: str, base_url: str):
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


def create_simplified_gradio_app() -> gr.Blocks:
    """Create a simplified single-page Gradio interface"""
    
    # Initialize providers
    initialize_llm_providers()
    
    with gr.Blocks(
        title="AI Assistant",
        theme=gr.themes.Soft(),
        css="""
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .provider-section {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .response-section {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 16px;
        }
        """,
    ) as app:
        gr.Markdown("# AI Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")
                
                with gr.Group(elem_classes=["provider-section"]):
                    gr.Markdown("### Select Provider")
                    
                    # Provider dropdown
                    provider_dropdown = gr.Dropdown(
                        label="Provider",
                        choices=get_providers_list(),
                        value=None,
                        interactive=True,
                    )
                    
                    # Model dropdown (initially hidden)
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=[],
                        value=None,
                        visible=False,
                        interactive=True,
                    )
                    
                    # Add provider section (initially hidden)
                    with gr.Group(visible=False) as add_provider_group:
                        gr.Markdown("### Add New Provider")
                        
                        new_provider_name = gr.Textbox(label="Provider Name")
                        new_provider_type = gr.Dropdown(
                            label="Provider Type",
                            choices=["openai_compatible", "ollama"],
                            value="openai_compatible",
                        )
                        new_provider_api_key = gr.Textbox(label="API Key", type="password")
                        new_provider_base_url = gr.Textbox(
                            label="Base URL (optional)",
                            placeholder="https://openrouter.ai/api/v1"
                        )
                        
                        add_provider_btn = gr.Button("Add Provider", variant="primary")
                        add_provider_status = gr.Textbox(label="Status", interactive=False)
                
                # Parameters
                with gr.Group():
                    gr.Markdown("### Parameters")
                    
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                    )
                    
                    max_tokens = gr.Number(
                        label="Max Tokens (0 for unlimited)",
                        value=0,
                        minimum=0,
                        maximum=4000,
                        precision=0,
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("## Query")
                
                # Query input
                query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your message here...",
                    lines=5,
                )
                
                submit_btn = gr.Button("Submit Query", variant="primary")
                
                # Response output
                response_output = gr.Textbox(
                    label="Response",
                    interactive=False,
                    lines=15,
                    placeholder="Response will appear here...",
                    elem_classes=["response-section"],
                )
        
        # Event handlers
        
        # Update model dropdown when provider is selected
        def on_provider_change(provider_name):
            if provider_name == "Add Provider":
                return (
                    gr.update(visible=False),  # Hide model dropdown
                    gr.update(visible=True),   # Show add provider group
                )
            else:
                models = get_models_for_provider(provider_name)
                return (
                    gr.update(
                        choices=models,
                        value=models[0] if models else None,
                        visible=True,
                    ),
                    gr.update(visible=False),  # Hide add provider group
                )
        
        provider_dropdown.change(
            on_provider_change,
            inputs=[provider_dropdown],
            outputs=[model_dropdown, add_provider_group],
        )
        
        # Add provider functionality
        add_provider_btn.click(
            add_new_provider,
            inputs=[
                new_provider_name,
                new_provider_type,
                new_provider_api_key,
                new_provider_base_url,
            ],
            outputs=[add_provider_status],
        )
        
        # Query submission with loading state
        def submit_query_with_loading(
            message,
            provider_name,
            model_name,
            temperature,
            max_tokens,
        ):
            """Submit query with loading state"""
            import asyncio
            
            if not message or not message.strip():
                return "Please enter a message to test"
            
            if not provider_name or provider_name == "Add Provider":
                return "Please select a provider"
            
            if not model_name:
                return "Please select a model"
            
            # Show loading state
            yield "⏳ Processing your query..."
            
            try:
                # Execute the actual query
                import nest_asyncio
                
                nest_asyncio.apply()
                
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, use run_in_executor
                    import concurrent.futures
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = loop.run_in_executor(
                            executor,
                            lambda: asyncio.run(
                                execute_query_function(
                                    message,
                                    provider_name,
                                    model_name,
                                    temperature,
                                    max_tokens,
                                )
                            ),
                        ).result()
                except RuntimeError:
                    # No running loop, we can use asyncio.run
                    result = asyncio.run(
                        execute_query_function(
                            message,
                            provider_name,
                            model_name,
                            temperature,
                            max_tokens,
                        )
                    )
                
                yield result
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Query submission error: {error_msg}")
                yield error_msg
        
        submit_btn.click(
            submit_query_with_loading,
            inputs=[
                query_input,
                provider_dropdown,
                model_dropdown,
                temperature_slider,
                max_tokens,
            ],
            outputs=[response_output],
        )
        
        # Refresh providers when add provider is successful
        def refresh_providers_after_add(status_message):
            if status_message.startswith("Success:"):
                return gr.update(choices=get_providers_list())
            return gr.update()
        
        add_provider_status.change(
            refresh_providers_after_add,
            inputs=[add_provider_status],
            outputs=[provider_dropdown],
        )
    
    return app


# Keep the original function for backward compatibility
def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio interface (backward compatibility)"""
    return create_simplified_gradio_app()


def mount_gradio_app(fastapi_app, gradio_app: gr.Blocks, path: str = "/gradio"):
    """Mount the Gradio app to a FastAPI application"""
    import gradio as gr
    
    # Create a FastAPI app wrapper for Gradio
    gradio_app_fastapi = gr.mount_gradio_app(fastapi_app, gradio_app, path=path)
    return gradio_app_fastapi