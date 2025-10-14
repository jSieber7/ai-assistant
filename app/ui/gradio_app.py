"""
Gradio interface for the AI Assistant application.

This module provides a web-based UI for configuring settings and testing queries.
"""

import gradio as gr
import httpx
from typing import List, Optional
from ..core.config import settings, get_available_models
from ..core.tools import tool_registry
from ..core.llm_providers import provider_registry


def get_models_list() -> List[str]:
    """Get list of available models for the dropdown"""
    try:
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we can't use run_until_complete
                # Return default model for now
                return [settings.default_model]
            else:
                # If loop exists but not running, use it
                models = loop.run_until_complete(get_available_models())
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                models = loop.run_until_complete(get_available_models())
            finally:
                loop.close()

        return [
            (
                f"{model.provider.value}:{model.name}"
                if model.provider.value not in ["openrouter", "openai_compatible"]
                else model.name
            )
            for model in models
        ]
    except Exception:
        return [settings.default_model]


def get_providers_info() -> str:
    """Get information about available providers"""
    try:
        providers = provider_registry.list_providers()
        providers_info = []
        for provider in providers:
            status = "✓ Configured" if provider.is_configured else "✗ Not configured"
            health = "✓ Healthy" if provider.is_healthy() else "✗ Unhealthy"
            is_default = (
                " (Default)"
                if provider.provider_type == provider_registry._default_provider
                else ""
            )
            providers_info.append(f"{provider.name}{is_default}: {status}, {health}")
        return (
            "\n".join(providers_info) if providers_info else "No providers configured"
        )
    except Exception:
        return "Unable to fetch provider information"


def get_tools_info() -> str:
    """Get information about available tools"""
    try:
        tools = tool_registry.list_tools(enabled_only=True)
        if not tools:
            return "No tools available"

        tools_info = []
        for tool in tools:
            status = "✓ Enabled" if tool.enabled else "✗ Disabled"
            tools_info.append(f"{tool.name}: {tool.description} ({status})")
        return "\n".join(tools_info)
    except Exception:
        return "Unable to fetch tools information"


async def test_query(
    message: str,
    model: str,
    temperature: float,
    max_tokens: int,
    use_agent_system: bool,
    agent_name: Optional[str] = None,
) -> str:
    """Test a query against the AI assistant"""
    try:
        # Prepare the request payload
        payload = {
            "messages": [{"role": "user", "content": message}],
            "model": model if model else settings.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens > 0 else None,
            "stream": False,
        }

        if use_agent_system and agent_name:
            payload["agent_name"] = agent_name

        # Make the request to the FastAPI endpoint
        base_url = f"http://{settings.host}:{settings.port}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Add agent info if available
            if "agent_name" in result and result["agent_name"]:
                content = f"[Agent: {result['agent_name']}]\n\n{content}"

            # Add tool results if available
            if "tool_results" in result and result["tool_results"]:
                content += "\n\n--- Tool Results ---\n"
                for tool_result in result["tool_results"]:
                    tool_name = tool_result["tool_name"]
                    success = "✓" if tool_result["success"] else "✗"
                    content += (
                        f"\n{success} {tool_name}: {tool_result.get('data', 'No data')}"
                    )
                    if tool_result.get("error"):
                        content += f"\nError: {tool_result['error']}"

            return content
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error testing query: {str(e)}"


def update_settings(
    tool_system_enabled: bool,
    agent_system_enabled: bool,
    preferred_provider: str,
    enable_fallback: bool,
    debug_mode: bool,
) -> str:
    """Update application settings"""
    # Note: This is a placeholder function. In a real implementation,
    # you would update the configuration file or environment variables
    # and potentially restart the application with new settings.

    return "Settings updated! Note: This is a demo function. In production, you would need to implement persistent configuration storage."


def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="AI Assistant Interface",
        theme=gr.themes.Soft(),
        css="""
        .system-info {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
        }
        """,
    ) as app:

        gr.Markdown("# AI Assistant Configuration & Testing Interface")

        with gr.Tabs():
            # System Information Tab
            with gr.TabItem("System Information"):
                gr.Markdown("## Current Configuration Status")

                with gr.Row():
                    with gr.Column():
                        models_info = gr.Textbox(
                            label="Available Models",
                            value=lambda: "\n".join(get_models_list()),
                            interactive=False,
                            lines=5,
                        )

                    with gr.Column():
                        providers_info = gr.Textbox(
                            label="Provider Status",
                            value=get_providers_info,
                            interactive=False,
                            lines=5,
                        )

                tools_info = gr.Textbox(
                    label="Available Tools",
                    value=get_tools_info,
                    interactive=False,
                    lines=5,
                )

                refresh_btn = gr.Button("Refresh Information")
                refresh_btn.click(
                    lambda: [
                        "\n".join(get_models_list()),
                        get_providers_info(),
                        get_tools_info(),
                    ],
                    outputs=[models_info, providers_info, tools_info],
                )

            # Settings Configuration Tab
            with gr.TabItem("Settings Configuration"):
                gr.Markdown("## Application Settings")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### System Configuration")
                        tool_system_enabled = gr.Checkbox(
                            label="Enable Tool System",
                            value=settings.tool_system_enabled,
                        )
                        agent_system_enabled = gr.Checkbox(
                            label="Enable Agent System",
                            value=settings.agent_system_enabled,
                        )
                        debug_mode = gr.Checkbox(
                            label="Debug Mode", value=settings.debug
                        )

                    with gr.Column():
                        gr.Markdown("### Provider Configuration")
                        preferred_provider = gr.Dropdown(
                            label="Preferred Provider",
                            choices=["openai_compatible", "ollama", "auto"],
                            value=settings.preferred_provider,
                        )
                        enable_fallback = gr.Checkbox(
                            label="Enable Provider Fallback",
                            value=settings.enable_fallback,
                        )

                update_settings_btn = gr.Button("Update Settings", variant="primary")
                settings_status = gr.Textbox(label="Status", interactive=False, lines=2)

                update_settings_btn.click(
                    update_settings,
                    inputs=[
                        tool_system_enabled,
                        agent_system_enabled,
                        preferred_provider,
                        enable_fallback,
                        debug_mode,
                    ],
                    outputs=[settings_status],
                )

            # Query Testing Tab
            with gr.TabItem("Query Testing"):
                gr.Markdown("## Test AI Assistant Queries")

                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Your Query",
                            placeholder="Enter your message here...",
                            lines=3,
                        )

                        with gr.Row():
                            with gr.Column():
                                model_dropdown = gr.Dropdown(
                                    label="Model",
                                    choices=get_models_list(),
                                    value=settings.default_model,
                                )
                            with gr.Column():
                                temperature_slider = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                )

                        with gr.Row():
                            max_tokens = gr.Number(
                                label="Max Tokens (0 for unlimited)",
                                value=0,
                                minimum=0,
                                maximum=4000,
                                precision=0,
                            )

                            use_agent_system = gr.Checkbox(
                                label="Use Agent System",
                                value=settings.agent_system_enabled,
                            )

                        if settings.agent_system_enabled:
                            agent_name = gr.Textbox(
                                label="Agent Name (optional)",
                                placeholder="Leave empty for default agent",
                            )

                        submit_btn = gr.Button("Submit Query", variant="primary")

                    with gr.Column(scale=3):
                        response_output = gr.Textbox(
                            label="Response", interactive=False, lines=15
                        )

                submit_btn.click(
                    # Wrapper to handle async function
                    lambda *args, **kwargs: gr.run_sync(test_query(*args, **kwargs)),
                    inputs=[
                        query_input,
                        model_dropdown,
                        temperature_slider,
                        max_tokens,
                        use_agent_system,
                        agent_name if settings.agent_system_enabled else gr.State(None),
                    ],
                    outputs=[response_output],
                )

    return app


def mount_gradio_app(fastapi_app, gradio_app: gr.Blocks, path: str = "/gradio"):
    """Mount the Gradio app to a FastAPI application"""
    import gradio as gr

    # Create a FastAPI app wrapper for Gradio
    gradio_app_fastapi = gr.mount_gradio_app(fastapi_app, gradio_app, path=path)
    return gradio_app_fastapi
