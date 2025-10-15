"""
Gradio interface for the AI Assistant application.

This module provides a web-based UI for configuring settings and testing queries.
"""

import gradio as gr
import httpx
import asyncio
import logging
import json
from typing import List, Optional, Tuple
from ..core.config import settings, get_available_models, initialize_llm_providers
from ..core.tools import tool_registry
from ..core.llm_providers import provider_registry

# Set up logging
logger = logging.getLogger(__name__)


def get_models_list() -> List[str]:
    """Get list of available models for the dropdown"""
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()

        # Get models synchronously with proper error handling
        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, need to run in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, get_available_models())
                    models = future.result(timeout=10)
            except RuntimeError:
                # No running loop, we can use asyncio.run
                models = asyncio.run(get_available_models(), timeout=10)
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return [settings.default_model]

        if not models:
            logger.warning("No models available, returning default model")
            return [settings.default_model]

        # Format model names for display
        formatted_models = []
        for model in models:
            if model.provider.value not in ["openrouter", "openai_compatible"]:
                formatted_models.append(f"{model.provider.value}:{model.name}")
            else:
                formatted_models.append(model.name)

        logger.info(f"Retrieved {len(formatted_models)} models")
        return formatted_models

    except Exception as e:
        logger.error(f"Unexpected error getting models list: {str(e)}", exc_info=True)
        return [settings.default_model]


def get_providers_info() -> str:
    """Get information about available providers"""
    try:
        # Initialize providers if not already done
        if not provider_registry.list_providers():
            initialize_llm_providers()

        providers = provider_registry.list_providers()
        if not providers:
            return "No providers configured"

        providers_info = []
        for provider in providers:
            try:
                status = (
                    "✓ Configured" if provider.is_configured else "✗ Not configured"
                )

                # Check health safely
                try:
                    health = "✓ Healthy" if provider.is_healthy() else "✗ Unhealthy"
                except Exception as health_error:
                    logger.warning(
                        f"Health check failed for {provider.name}: {health_error}"
                    )
                    health = "? Unknown"

                # Check if default provider
                is_default = ""
                try:
                    if provider.provider_type == provider_registry._default_provider:
                        is_default = " (Default)"
                except Exception:
                    pass

                providers_info.append(
                    f"{provider.name}{is_default}: {status}, {health}"
                )
            except Exception as provider_error:
                logger.error(
                    f"Error processing provider {provider.name}: {provider_error}"
                )
                providers_info.append(f"{provider.name}: ✗ Error processing provider")

        return "\n".join(providers_info)

    except Exception as e:
        logger.error(f"Error getting provider information: {str(e)}", exc_info=True)
        return f"Unable to fetch provider information: {str(e)}"


def get_tools_info() -> str:
    """Get information about available tools"""
    try:
        # Initialize tools if not already done
        if not tool_registry.list_tools():
            try:
                from ..core.tools.examples import initialize_default_tools

                initialize_default_tools()
                logger.info("Initialized default tools")
            except Exception as init_error:
                logger.error(f"Failed to initialize default tools: {init_error}")
                return f"Failed to initialize tools: {str(init_error)}"

        tools = tool_registry.list_tools(enabled_only=True)
        if not tools:
            return "No tools available (tools may be disabled)"

        tools_info = []
        for tool in tools:
            try:
                status = "✓ Enabled" if tool.enabled else "✗ Disabled"
                description = getattr(tool, "description", "No description available")
                tools_info.append(f"{tool.name}: {description} ({status})")
            except Exception as tool_error:
                logger.error(
                    f"Error processing tool {getattr(tool, 'name', 'unknown')}: {tool_error}"
                )
                tools_info.append(f"Error processing tool: {str(tool_error)}")

        return "\n".join(tools_info)

    except Exception as e:
        logger.error(f"Error getting tools information: {str(e)}", exc_info=True)
        return f"Unable to fetch tools information: {str(e)}"


def get_agents_list() -> List[str]:
    """Get list of available agents for the dropdown"""
    try:
        from ..core.agents.registry import agent_registry

        # Check if agent system is enabled
        if not settings.agent_system_enabled:
            logger.info("Agent system is disabled")
            return ["default (disabled)"]

        # Initialize agent system if not already done
        if not agent_registry.list_agents():
            try:
                from ..core.config import initialize_agent_system

                initialize_agent_system()
                logger.info("Initialized agent system")
            except Exception as init_error:
                logger.error(f"Failed to initialize agent system: {init_error}")
                return ["default (initialization failed)"]

        agents = agent_registry.list_agents()
        if not agents:
            logger.warning("No agents found after initialization")
            return ["default (no agents)"]

        agent_names = [agent.name for agent in agents]
        logger.info(f"Found {len(agent_names)} agents: {agent_names}")
        return agent_names

    except Exception as e:
        logger.error(f"Error getting agents list: {str(e)}", exc_info=True)
        return ["default (error)"]


def get_agents_info() -> str:
    """Get information about available agents"""
    try:
        from ..core.agents.registry import agent_registry

        # Check if agent system is enabled
        if not settings.agent_system_enabled:
            return "Agent system is disabled in settings"

        # Initialize agent system if not already done
        if not agent_registry.list_agents():
            try:
                from ..core.config import initialize_agent_system

                initialize_agent_system()
            except Exception as init_error:
                logger.error(f"Failed to initialize agent system: {init_error}")
                return f"Failed to initialize agent system: {str(init_error)}"

        agents = agent_registry.list_agents()
        if not agents:
            return "No agents available (check agent system configuration)"

        agents_info = []
        for agent in agents:
            try:
                is_default = " (Default)" if getattr(agent, "is_default", False) else ""
                description = getattr(agent, "description", "No description available")
                agents_info.append(f"{agent.name}{is_default}: {description}")
            except Exception as agent_error:
                logger.error(
                    f"Error processing agent {getattr(agent, 'name', 'unknown')}: {agent_error}"
                )
                agents_info.append(f"Error processing agent: {str(agent_error)}")

        return "\n".join(agents_info)

    except Exception as e:
        logger.error(f"Error getting agents information: {str(e)}", exc_info=True)
        return f"Unable to fetch agents information: {str(e)}"


def initialize_gradio_components() -> Tuple[bool, str]:
    """Initialize all components needed for the Gradio interface"""
    initialization_status = []

    try:
        # Initialize LLM providers
        try:
            if not provider_registry.list_providers():
                initialize_llm_providers()
                initialization_status.append("✓ LLM providers initialized")
            else:
                initialization_status.append("✓ LLM providers already initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {str(e)}")
            initialization_status.append(
                f"✗ Failed to initialize LLM providers: {str(e)}"
            )

        # Initialize tools
        try:
            if not tool_registry.list_tools():
                from ..core.tools.examples import initialize_default_tools

                initialize_default_tools()
                initialization_status.append("✓ Default tools initialized")
            else:
                tools_count = len(tool_registry.list_tools())
                initialization_status.append(
                    f"✓ Tools already initialized ({tools_count} tools)"
                )
        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            initialization_status.append(f"✗ Failed to initialize tools: {str(e)}")

        # Initialize agents if enabled
        if settings.agent_system_enabled:
            try:
                from ..core.agents.registry import agent_registry

                if not agent_registry.list_agents():
                    from ..core.config import initialize_agent_system

                    initialize_agent_system()
                    initialization_status.append("✓ Agent system initialized")
                else:
                    agents_count = len(agent_registry.list_agents())
                    initialization_status.append(
                        f"✓ Agents already initialized ({agents_count} agents)"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize agent system: {str(e)}")
                initialization_status.append(
                    f"✗ Failed to initialize agent system: {str(e)}"
                )
        else:
            initialization_status.append("- Agent system disabled in settings")

        success = not any("✗" in status for status in initialization_status)
        status_message = "\n".join(initialization_status)

        logger.info(
            f"Gradio components initialization: {'SUCCESS' if success else 'PARTIAL'}\n{status_message}"
        )
        return success, status_message

    except Exception as e:
        error_msg = f"Critical error during initialization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


async def test_query(
    message: str,
    model: str,
    temperature: float,
    max_tokens: int,
    use_agent_system: bool,
    agent_dropdown: str = "default",
    agent_name: Optional[str] = None,
) -> str:
    """Test a query against the AI assistant"""
    if not message or not message.strip():
        return "Error: Please enter a message to test"

    try:
        # Validate inputs
        if not model:
            model = settings.default_model

        # Prepare the request payload
        payload = {
            "messages": [{"role": "user", "content": message.strip()}],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens > 0 else None,
            "stream": False,
        }

        # Handle agent system
        if use_agent_system and settings.agent_system_enabled:
            # Use custom agent name if provided, otherwise use dropdown selection
            selected_agent = (
                agent_name if agent_name and agent_name.strip() else agent_dropdown
            )

            # Validate agent selection
            if (
                selected_agent
                and selected_agent != "default"
                and not selected_agent.endswith("disabled")
            ):
                payload["agent_name"] = selected_agent
                logger.info(f"Using agent: {selected_agent}")
            elif selected_agent and selected_agent.endswith("disabled"):
                return "Error: Agent system is disabled or not properly initialized"

        # Make the request to the FastAPI endpoint
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
                formatted_response.append(f"**Model:** {result.get('model', model)}")

                # Add agent info if available
                if "agent_name" in result and result["agent_name"]:
                    formatted_response.append(f"**Agent:** {result['agent_name']}")

                formatted_response.append("\n--- Response ---\n")
                formatted_response.append(content)

                # Add tool results if available
                if "tool_results" in result and result["tool_results"]:
                    formatted_response.append("\n\n--- Tool Results ---")
                    for tool_result in result["tool_results"]:
                        tool_name = tool_result.get("tool_name", "Unknown tool")
                        success = "✓" if tool_result.get("success", False) else "✗"
                        formatted_response.append(f"\n{success} **{tool_name}:**")

                        if tool_result.get("data"):
                            formatted_response.append(f"  {tool_result['data']}")

                        if tool_result.get("error"):
                            formatted_response.append(
                                f"  Error: {tool_result['error']}"
                            )

                # Add usage info if available
                if "usage" in result:
                    usage = result["usage"]
                    formatted_response.append("\n\n--- Usage Information ---")
                    if "prompt_tokens" in usage:
                        formatted_response.append(
                            f"Prompt tokens: {usage['prompt_tokens']}"
                        )
                    if "completion_tokens" in usage:
                        formatted_response.append(
                            f"Completion tokens: {usage['completion_tokens']}"
                        )
                    if "total_tokens" in usage:
                        formatted_response.append(
                            f"Total tokens: {usage['total_tokens']}"
                        )

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
                error_msg += f": {response.text[:200]}"

            return f"Error: {error_msg}"

    except httpx.TimeoutException:
        return "Error: Request timed out. The server took too long to respond."
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {settings.host}:{settings.port}. Please ensure the server is running."
    except Exception as e:
        logger.error(f"Error testing query: {str(e)}", exc_info=True)
        return f"Error testing query: {str(e)}"


def update_settings(
    tool_system_enabled: bool,
    agent_system_enabled: bool,
    preferred_provider: str,
    enable_fallback: bool,
    debug_mode: bool,
) -> str:
    """Update application settings"""
    try:
        from pathlib import Path

        logger.info("Updating application settings...")

        # Track what changed
        changes = []

        # Check for changes and update the settings object
        if settings.tool_system_enabled != tool_system_enabled:
            changes.append(
                f"Tool system: {settings.tool_system_enabled} → {tool_system_enabled}"
            )
            settings.tool_system_enabled = tool_system_enabled

        if settings.agent_system_enabled != agent_system_enabled:
            changes.append(
                f"Agent system: {settings.agent_system_enabled} → {agent_system_enabled}"
            )
            settings.agent_system_enabled = agent_system_enabled

        if settings.preferred_provider != preferred_provider:
            changes.append(
                f"Preferred provider: {settings.preferred_provider} → {preferred_provider}"
            )
            settings.preferred_provider = preferred_provider

        if settings.enable_fallback != enable_fallback:
            changes.append(
                f"Provider fallback: {settings.enable_fallback} → {enable_fallback}"
            )
            settings.enable_fallback = enable_fallback

        if settings.debug != debug_mode:
            changes.append(f"Debug mode: {settings.debug} → {debug_mode}")
            settings.debug = debug_mode

        # Create or update .env file for persistence
        env_file = Path(".env")
        env_content = []

        # Read existing .env file if it exists
        if env_file.exists():
            with open(env_file, "r") as f:
                env_content = f.readlines()

        # Update or add the settings
        settings_map = {
            "TOOL_SYSTEM_ENABLED": str(tool_system_enabled),
            "AGENT_SYSTEM_ENABLED": str(agent_system_enabled),
            "PREFERRED_PROVIDER": preferred_provider,
            "ENABLE_FALLBACK": str(enable_fallback),
            "DEBUG": str(debug_mode),
        }

        # Update existing lines or add new ones
        updated_lines = []
        settings_updated = set()

        for line in env_content:
            line = line.strip()
            if not line or line.startswith("#"):
                updated_lines.append(line + "\n")
                continue

            key, value = line.split("=", 1)
            if key in settings_map:
                updated_lines.append(f"{key}={settings_map[key]}\n")
                settings_updated.add(key)
            else:
                updated_lines.append(line + "\n")

        # Add any missing settings
        for key, value in settings_map.items():
            if key not in settings_updated:
                updated_lines.append(f"{key}={value}\n")

        # Write back to .env file
        with open(env_file, "w") as f:
            f.writelines(updated_lines)

        logger.info(f"Settings saved to {env_file}")

        # Apply runtime changes
        reinit_results = []

        # Reinitialize providers if provider settings changed
        provider_changed = any("provider" in change.lower() for change in changes)
        if provider_changed:
            try:
                initialize_llm_providers()
                reinit_results.append("✓ LLM providers reinitialized")
            except Exception as e:
                logger.error(f"Failed to reinitialize providers: {str(e)}")
                reinit_results.append(f"✗ Provider reinitialization failed: {str(e)}")

        # Initialize agent system if newly enabled
        if agent_system_enabled:
            try:
                from ..core.agents.registry import agent_registry
                
                if not agent_registry.list_agents():
                    from ..core.config import initialize_agent_system

                    initialize_agent_system()
                    reinit_results.append("✓ Agent system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize agent system: {str(e)}")
                reinit_results.append(f"✗ Agent system initialization failed: {str(e)}")

        # Initialize tools if newly enabled
        if tool_system_enabled and not tool_registry.list_tools():
            try:
                from ..core.tools.examples import initialize_default_tools

                initialize_default_tools()
                reinit_results.append("✓ Tool system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize tool system: {str(e)}")
                reinit_results.append(f"✗ Tool system initialization failed: {str(e)}")

        # Format response message
        if changes:
            change_summary = "\n".join(f"  • {change}" for change in changes)
            result_msg = (
                f"Settings updated successfully!\n\nChanges applied:\n{change_summary}"
            )

            if reinit_results:
                result_msg += "\n\nRuntime updates:\n" + "\n".join(
                    f"  {result}" for result in reinit_results
                )

            result_msg += f"\n\nSettings have been saved to {env_file}"
            result_msg += "\n\n⚠️ Note: Some changes may require a server restart to take full effect."
        else:
            result_msg = "No settings changes detected."

        logger.info("Settings update completed successfully")
        return result_msg

    except Exception as e:
        error_msg = f"Failed to update settings: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio interface"""

    # Initialize all components and get status
    init_success, init_status = initialize_gradio_components()

    # Log initialization status
    if init_success:
        logger.info("Gradio app initialized successfully")
    else:
        logger.warning(f"Gradio app initialized with issues: {init_status}")

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
        .loading {
            opacity: 0.7;
        }
        .error {
            color: red;
            background-color: #ffe6e6;
            padding: 10px;
            border-radius: 5px;
        }
        .success {
            color: green;
            background-color: #e6ffe6;
            padding: 10px;
            border-radius: 5px;
        }
        """,
    ) as app:

        gr.Markdown("# AI Assistant Configuration & Testing Interface")

        # Show initialization status if there were issues
        if not init_success:
            gr.Markdown(
                f"⚠️ **Initialization Warning**\n\n{init_status}", elem_classes=["error"]
            )

        with gr.Tabs():
            # System Information Tab
            with gr.TabItem("System Information"):
                gr.Markdown("## Current Configuration Status")

                # Add initialization status display
                gr.Textbox(
                    label="Initialization Status",
                    value=init_status,
                    interactive=False,
                    lines=3,
                    elem_classes=["success" if init_success else "error"],
                )

                with gr.Row():
                    with gr.Column():
                        models_info = gr.Textbox(
                            label="Available Models",
                            value="Loading...",
                            interactive=False,
                            lines=5,
                        )
                        models_loading = gr.Markdown(
                            "⏳ Loading models...", visible=False
                        )

                    with gr.Column():
                        providers_info = gr.Textbox(
                            label="Provider Status",
                            value="Loading...",
                            interactive=False,
                            lines=5,
                        )
                        providers_loading = gr.Markdown(
                            "⏳ Loading providers...", visible=False
                        )

                tools_info = gr.Textbox(
                    label="Available Tools",
                    value="Loading...",
                    interactive=False,
                    lines=5,
                )
                tools_loading = gr.Markdown("⏳ Loading tools...", visible=False)

                with gr.Row():
                    with gr.Column():
                        agents_info = gr.Textbox(
                            label="Available Agents",
                            value="Loading...",
                            interactive=False,
                            lines=5,
                        )
                        agents_loading = gr.Markdown(
                            "⏳ Loading agents...", visible=False
                        )

                refresh_btn = gr.Button("🔄 Refresh Information", variant="primary")

                # Define refresh function with loading states
                def refresh_all_info():
                    """Refresh all system information with proper error handling"""
                    results = []

                    # Refresh models
                    try:
                        models_loading.update(visible=True)
                        models = "\n".join(get_models_list())
                        results.append(models)
                    except Exception as e:
                        logger.error(f"Error refreshing models: {e}")
                        results.append(f"Error loading models: {str(e)}")
                    finally:
                        models_loading.update(visible=False)

                    # Refresh providers
                    try:
                        providers_loading.update(visible=True)
                        providers = get_providers_info()
                        results.append(providers)
                    except Exception as e:
                        logger.error(f"Error refreshing providers: {e}")
                        results.append(f"Error loading providers: {str(e)}")
                    finally:
                        providers_loading.update(visible=False)

                    # Refresh tools
                    try:
                        tools_loading.update(visible=True)
                        tools = get_tools_info()
                        results.append(tools)
                    except Exception as e:
                        logger.error(f"Error refreshing tools: {e}")
                        results.append(f"Error loading tools: {str(e)}")
                    finally:
                        tools_loading.update(visible=False)

                    # Refresh agents
                    try:
                        agents_loading.update(visible=True)
                        agents = get_agents_info()
                        results.append(agents)
                    except Exception as e:
                        logger.error(f"Error refreshing agents: {e}")
                        results.append(f"Error loading agents: {str(e)}")
                    finally:
                        agents_loading.update(visible=False)

                    return results

                refresh_btn.click(
                    refresh_all_info,
                    outputs=[models_info, providers_info, tools_info, agents_info],
                )

                # Initial load
                app.load(
                    refresh_all_info,
                    outputs=[models_info, providers_info, tools_info, agents_info],
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
                                refresh_models_btn = gr.Button(
                                    "🔄", size="sm", variant="secondary"
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

                        with gr.Row():
                            agent_dropdown = gr.Dropdown(
                                label="Agent",
                                choices=get_agents_list(),
                                value="default",
                                visible=settings.agent_system_enabled,
                            )
                            refresh_agents_btn = gr.Button(
                                "🔄",
                                size="sm",
                                variant="secondary",
                                visible=settings.agent_system_enabled,
                            )

                        agent_name = gr.Textbox(
                            label="Custom Agent Name (optional)",
                            placeholder="Leave empty to use selected agent",
                            visible=False,
                        )

                        with gr.Row():
                            submit_btn = gr.Button("🚀 Submit Query", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                    with gr.Column(scale=3):
                        response_output = gr.Textbox(
                            label="Response",
                            interactive=False,
                            lines=15,
                            placeholder="Response will appear here...",
                        )
                        query_status = gr.Markdown(
                            "Ready to submit queries", visible=False
                        )

                # Query submission with loading state
                def submit_query_with_loading(
                    message,
                    model,
                    temperature,
                    max_tokens,
                    use_agents,
                    agent_dropdown_value,
                    agent_custom_name,
                ):
                    """Submit query with loading state"""
                    if not message or not message.strip():
                        return (
                            "Please enter a message to test",
                            "⚠️ Please enter a message",
                        )

                    # Show loading state
                    yield "⏳ Processing your query...", "🔄 Processing query..."

                    try:
                        # Execute the actual query
                        result = test_query(
                            message,
                            model,
                            temperature,
                            max_tokens,
                            use_agents,
                            agent_dropdown_value,
                            agent_custom_name,
                        )

                        # Determine status based on result
                        if result.startswith("Error:"):
                            status = "❌ Query failed"
                        else:
                            status = "✅ Query completed successfully"

                        yield result, status
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        logger.error(f"Query submission error: {error_msg}")
                        yield error_msg, "❌ Unexpected error occurred"

                # Connect query submission
                submit_btn.click(
                    submit_query_with_loading,
                    inputs=[
                        query_input,
                        model_dropdown,
                        temperature_slider,
                        max_tokens,
                        use_agent_system,
                        agent_dropdown,
                        agent_name,
                    ],
                    outputs=[response_output, query_status],
                )

                # Clear button functionality
                clear_btn.click(
                    lambda: ("", "", "Ready to submit queries", False),
                    outputs=[query_input, response_output, query_status],
                )

                # Refresh models button
                refresh_models_btn.click(
                    lambda: gr.update(choices=get_models_list()),
                    outputs=[model_dropdown],
                )

                # Refresh agents button
                refresh_agents_btn.click(
                    lambda: gr.update(choices=get_agents_list()),
                    outputs=[agent_dropdown],
                )

                # Toggle agent controls visibility based on agent system checkbox
                def toggle_agent_controls(enabled):
                    """Toggle visibility of agent-related controls"""
                    return (
                        gr.update(visible=enabled),  # agent_dropdown
                        gr.update(visible=enabled),  # agent_name
                        gr.update(visible=enabled),  # refresh_agents_btn
                    )

                use_agent_system.change(
                    toggle_agent_controls,
                    inputs=[use_agent_system],
                    outputs=[agent_dropdown, agent_name, refresh_agents_btn],
                )

                # Add examples for quick testing
                gr.Markdown("### Example Queries")
                with gr.Row():
                    example_1 = gr.Button("Hello, introduce yourself!")
                    example_2 = gr.Button("What can you help me with?")
                    example_3 = gr.Button("Explain quantum computing simply")

                def set_example_query(example_text):
                    """Set an example query in the input"""
                    return example_text

                example_1.click(
                    lambda: set_example_query("Hello, introduce yourself!"),
                    outputs=[query_input],
                )
                example_2.click(
                    lambda: set_example_query("What can you help me with?"),
                    outputs=[query_input],
                )
                example_3.click(
                    lambda: set_example_query("Explain quantum computing simply"),
                    outputs=[query_input],
                )

    return app


def mount_gradio_app(fastapi_app, gradio_app: gr.Blocks, path: str = "/gradio"):
    """Mount the Gradio app to a FastAPI application"""
    import gradio as gr

    # Create a FastAPI app wrapper for Gradio
    gradio_app_fastapi = gr.mount_gradio_app(fastapi_app, gradio_app, path=path)
    return gradio_app_fastapi
