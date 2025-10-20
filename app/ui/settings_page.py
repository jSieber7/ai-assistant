"""
Gradio settings page for secure configuration management.

This module provides a comprehensive UI for configuring all application settings
including API keys, provider settings, and system configuration.
"""

import gradio as gr
import logging
from typing import Dict, Any, Optional
from ..core.config import initialize_llm_providers
from ..core.secure_settings import secure_settings

logger = logging.getLogger(__name__)


def get_current_settings() -> Dict[str, Any]:
    """Get current settings from secure storage."""
    try:
        return secure_settings.get_all_settings()
    except Exception as e:
        logger.error(f"Failed to get current settings: {e}")
        return {}


def validate_api_key(provider: str, api_key: str) -> str:
    """Validate an API key and return status message."""
    if not api_key or not api_key.strip():
        return "⚠️ Please enter an API key"

    try:
        is_valid = secure_settings.validate_api_key(provider, api_key.strip())
        if is_valid:
            return "✅ API key is valid"
        else:
            return "❌ API key is invalid or could not be verified"
    except Exception as e:
        logger.error(f"Error validating API key for {provider}: {e}")
        return f"❌ Error validating API key: {str(e)}"


def update_llm_provider_settings(
    openai_enabled: bool,
    openai_api_key: str,
    openai_base_url: str,
    openai_default_model: str,
    openai_provider_name: str,
    openai_timeout: int,
    openai_max_retries: int,
    ollama_enabled: bool,
    ollama_base_url: str,
    ollama_default_model: str,
    ollama_timeout: int,
    ollama_max_retries: int,
    ollama_temperature: float,
    ollama_max_tokens: Optional[int],
    ollama_streaming: bool,
) -> str:
    """Update LLM provider settings."""
    try:
        # Update OpenAI-compatible settings
        openai_config = {
            "enabled": openai_enabled,
            "api_key": openai_api_key.strip() if openai_api_key else "",
            "base_url": openai_base_url.strip()
            if openai_base_url
            else "https://openrouter.ai/api/v1",
            "default_model": openai_default_model.strip()
            if openai_default_model
            else "anthropic/claude-3.5-sonnet",
            "provider_name": openai_provider_name.strip()
            if openai_provider_name
            else "",
            "timeout": openai_timeout,
            "max_retries": openai_max_retries,
        }

        # Update Ollama settings
        ollama_config = {
            "enabled": ollama_enabled,
            "base_url": ollama_base_url.strip()
            if ollama_base_url
            else "http://localhost:11434",
            "default_model": ollama_default_model.strip()
            if ollama_default_model
            else "llama2",
            "timeout": ollama_timeout,
            "max_retries": ollama_max_retries,
            "temperature": ollama_temperature,
            "max_tokens": ollama_max_tokens
            if ollama_max_tokens and ollama_max_tokens > 0
            else None,
            "streaming": ollama_streaming,
        }

        # Save to secure storage
        secure_settings.set_category(
            "llm_providers",
            {"openai_compatible": openai_config, "ollama": ollama_config},
        )

        # Reinitialize providers
        initialize_llm_providers()

        return "✅ LLM provider settings updated successfully"

    except Exception as e:
        logger.error(f"Failed to update LLM provider settings: {e}")
        return f"❌ Failed to update settings: {str(e)}"


def update_external_service_settings(
    firecrawl_enabled: bool,
    firecrawl_docker_url: str,
    firecrawl_bull_auth_key: str,
    firecrawl_scraping_enabled: bool,
    firecrawl_max_concurrent_scrapes: int,
    firecrawl_scrape_timeout: int,
    jina_enabled: bool,
    jina_api_key: str,
    jina_url: str,
    jina_model: str,
    jina_timeout: int,
    jina_cache_ttl: int,
    jina_max_retries: int,
    searxng_secret_key: str,
    searxng_url: str,
) -> str:
    """Update external service settings."""
    try:
        # Update Firecrawl settings
        firecrawl_config = {
            "enabled": firecrawl_enabled,
            "deployment_mode": "docker",
            "docker_url": firecrawl_docker_url.strip()
            if firecrawl_docker_url
            else "http://firecrawl-api:3002",
            "bull_auth_key": firecrawl_bull_auth_key.strip()
            if firecrawl_bull_auth_key
            else "",
            "scraping_enabled": firecrawl_scraping_enabled,
            "max_concurrent_scrapes": firecrawl_max_concurrent_scrapes,
            "scrape_timeout": firecrawl_scrape_timeout,
        }

        # Update Jina Reranker settings
        jina_config = {
            "enabled": jina_enabled,
            "api_key": jina_api_key.strip() if jina_api_key else "",
            "url": jina_url.strip() if jina_url else "http://jina-reranker:8080",
            "model": jina_model.strip()
            if jina_model
            else "jina-reranker-v2-base-multilingual",
            "timeout": jina_timeout,
            "cache_ttl": jina_cache_ttl,
            "max_retries": jina_max_retries,
        }

        # Update SearXNG settings
        searxng_config = {
            "secret_key": searxng_secret_key.strip() if searxng_secret_key else "",
            "url": searxng_url.strip() if searxng_url else "http://searxng:8080",
        }

        # Save to secure storage
        secure_settings.set_category(
            "external_services",
            {
                "firecrawl": firecrawl_config,
                "jina_reranker": jina_config,
                "searxng": searxng_config,
            },
        )

        return "✅ External service settings updated successfully"

    except Exception as e:
        logger.error(f"Failed to update external service settings: {e}")
        return f"❌ Failed to update settings: {str(e)}"


def update_system_settings(
    tool_system_enabled: bool,
    agent_system_enabled: bool,
    multi_writer_enabled: bool,
    preferred_provider: str,
    enable_fallback: bool,
    debug_mode: bool,
    host: str,
    port: int,
    environment: str,
    secret_key: str,
) -> str:
    """Update system settings."""
    try:
        system_config = {
            "tool_system_enabled": tool_system_enabled,
            "agent_system_enabled": agent_system_enabled,
            "multi_writer_enabled": multi_writer_enabled,
            "preferred_provider": preferred_provider,
            "enable_fallback": enable_fallback,
            "debug": debug_mode,
            "host": host.strip() if host else "127.0.0.1",
            "port": port,
            "environment": environment.strip() if environment else "development",
            "secret_key": secret_key.strip() if secret_key else "",
        }

        # Save to secure storage
        secure_settings.set_category("system_config", system_config)

        return "✅ System settings updated successfully"

    except Exception as e:
        logger.error(f"Failed to update system settings: {e}")
        return f"❌ Failed to update settings: {str(e)}"


def update_multi_writer_settings(
    mongodb_connection_string: str,
    mongodb_database_name: str,
) -> str:
    """Update multi-writer settings."""
    try:
        multi_writer_config = {
            "mongodb_connection_string": mongodb_connection_string.strip()
            if mongodb_connection_string
            else "mongodb://localhost:27017",
            "mongodb_database_name": mongodb_database_name.strip()
            if mongodb_database_name
            else "multi_writer_system",
        }

        # Save to secure storage
        secure_settings.set_category("multi_writer", multi_writer_config)

        return "✅ Multi-writer settings updated successfully"

    except Exception as e:
        logger.error(f"Failed to update multi-writer settings: {e}")
        return f"❌ Failed to update settings: {str(e)}"


def export_settings(include_secrets: bool = False) -> str:
    """Export settings to JSON."""
    try:
        return secure_settings.export_settings(include_secrets=include_secrets)
    except Exception as e:
        logger.error(f"Failed to export settings: {e}")
        return f"Error exporting settings: {str(e)}"


def import_settings(settings_json: str) -> str:
    """Import settings from JSON."""
    try:
        if not settings_json or not settings_json.strip():
            return "⚠️ Please provide valid JSON settings"

        secure_settings.import_settings(settings_json.strip(), merge=True)
        return "✅ Settings imported successfully"
    except Exception as e:
        logger.error(f"Failed to import settings: {e}")
        return f"❌ Failed to import settings: {str(e)}"


def create_settings_page() -> gr.Blocks:
    """Create the comprehensive settings page."""

    # Get current settings for form initialization
    current_settings = get_current_settings()

    with gr.Blocks(
        title="AI Assistant Settings",
        theme=gr.themes.Soft(),
        css="""
        .settings-section {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .api-key-input {
            font-family: monospace;
        }
        .success {
            color: green;
            font-weight: bold;
        }
        .error {
            color: red;
            font-weight: bold;
        }
        .warning {
            color: orange;
            font-weight: bold;
        }
        """,
    ) as app:
        gr.Markdown("# 🔐 AI Assistant Settings")
        gr.Markdown(
            "Configure your AI assistant settings securely. API keys and sensitive data are encrypted and stored locally."
        )

        with gr.Tabs():
            # LLM Providers Tab
            with gr.TabItem("🤖 LLM Providers"):
                gr.Markdown("## Configure Language Model Providers")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### OpenAI-Compatible Provider")
                        with gr.Group(elem_classes=["settings-section"]):
                            openai_enabled = gr.Checkbox(
                                label="Enable OpenAI-Compatible Provider",
                                value=current_settings.get("llm_providers", {})
                                .get("openai_compatible", {})
                                .get("enabled", True),
                            )
                            openai_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="Enter your API key...",
                                value="",  # Don't show existing keys
                                elem_classes=["api-key-input"],
                            )
                            openai_base_url = gr.Textbox(
                                label="Base URL",
                                value=current_settings.get("llm_providers", {})
                                .get("openai_compatible", {})
                                .get("base_url", "https://openrouter.ai/api/v1"),
                            )
                            openai_default_model = gr.Textbox(
                                label="Default Model",
                                value=current_settings.get("llm_providers", {})
                                .get("openai_compatible", {})
                                .get("default_model", "anthropic/claude-3.5-sonnet"),
                            )
                            openai_provider_name = gr.Textbox(
                                label="Provider Name (optional)",
                                placeholder="e.g., OpenRouter, Anthropic",
                                value=current_settings.get("llm_providers", {})
                                .get("openai_compatible", {})
                                .get("provider_name", ""),
                            )
                            with gr.Row():
                                openai_timeout = gr.Number(
                                    label="Timeout (seconds)",
                                    value=current_settings.get("llm_providers", {})
                                    .get("openai_compatible", {})
                                    .get("timeout", 30),
                                    minimum=1,
                                    maximum=300,
                                )
                                openai_max_retries = gr.Number(
                                    label="Max Retries",
                                    value=current_settings.get("llm_providers", {})
                                    .get("openai_compatible", {})
                                    .get("max_retries", 3),
                                    minimum=0,
                                    maximum=10,
                                )

                            with gr.Row():
                                validate_openai_btn = gr.Button(
                                    "🔍 Validate API Key", size="sm"
                                )
                                openai_validation_status = gr.Textbox(
                                    label="Validation Status",
                                    interactive=False,
                                    elem_classes=["success"],
                                )

                    with gr.Column(scale=1):
                        gr.Markdown("### Ollama (Local Models)")
                        with gr.Group(elem_classes=["settings-section"]):
                            ollama_enabled = gr.Checkbox(
                                label="Enable Ollama",
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("enabled", True),
                            )
                            ollama_base_url = gr.Textbox(
                                label="Base URL",
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("base_url", "http://localhost:11434"),
                            )
                            ollama_default_model = gr.Textbox(
                                label="Default Model",
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("default_model", "llama2"),
                            )
                            with gr.Row():
                                ollama_timeout = gr.Number(
                                    label="Timeout (seconds)",
                                    value=current_settings.get("llm_providers", {})
                                    .get("ollama", {})
                                    .get("timeout", 30),
                                    minimum=1,
                                    maximum=300,
                                )
                                ollama_max_retries = gr.Number(
                                    label="Max Retries",
                                    value=current_settings.get("llm_providers", {})
                                    .get("ollama", {})
                                    .get("max_retries", 3),
                                    minimum=0,
                                    maximum=10,
                                )
                            ollama_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("temperature", 0.7),
                            )
                            ollama_max_tokens = gr.Number(
                                label="Max Tokens (0 for unlimited)",
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("max_tokens", 0)
                                or 0,
                                minimum=0,
                                maximum=8000,
                            )
                            ollama_streaming = gr.Checkbox(
                                label="Enable Streaming",
                                value=current_settings.get("llm_providers", {})
                                .get("ollama", {})
                                .get("streaming", True),
                            )

                update_llm_btn = gr.Button(
                    "💾 Save LLM Provider Settings", variant="primary"
                )
                llm_status = gr.Textbox(
                    label="Status", interactive=False, elem_classes=["success"]
                )

            # External Services Tab
            with gr.TabItem("🔌 External Services"):
                gr.Markdown("## Configure External Services")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Firecrawl (Web Scraping)")
                        with gr.Group(elem_classes=["settings-section"]):
                            firecrawl_enabled = gr.Checkbox(
                                label="Enable Firecrawl",
                                value=current_settings.get("external_services", {})
                                .get("firecrawl", {})
                                .get("enabled", False),
                            )
                            firecrawl_docker_url = gr.Textbox(
                                label="Docker URL",
                                value=current_settings.get("external_services", {})
                                .get("firecrawl", {})
                                .get("docker_url", "http://firecrawl-api:3002"),
                            )
                            firecrawl_bull_auth_key = gr.Textbox(
                                label="Bull Auth Key",
                                type="password",
                                placeholder="Enter auth key...",
                                value="",
                                elem_classes=["api-key-input"],
                            )
                            firecrawl_scraping_enabled = gr.Checkbox(
                                label="Enable Web Scraping",
                                value=current_settings.get("external_services", {})
                                .get("firecrawl", {})
                                .get("scraping_enabled", True),
                            )
                            with gr.Row():
                                firecrawl_max_concurrent_scrapes = gr.Number(
                                    label="Max Concurrent Scrapes",
                                    value=current_settings.get("external_services", {})
                                    .get("firecrawl", {})
                                    .get("max_concurrent_scrapes", 5),
                                    minimum=1,
                                    maximum=20,
                                )
                                firecrawl_scrape_timeout = gr.Number(
                                    label="Scrape Timeout (seconds)",
                                    value=current_settings.get("external_services", {})
                                    .get("firecrawl", {})
                                    .get("scrape_timeout", 60),
                                    minimum=10,
                                    maximum=300,
                                )

                    with gr.Column():
                        gr.Markdown("### Jina AI Reranker")
                        with gr.Group(elem_classes=["settings-section"]):
                            jina_enabled = gr.Checkbox(
                                label="Enable Jina Reranker",
                                value=current_settings.get("external_services", {})
                                .get("jina_reranker", {})
                                .get("enabled", False),
                            )
                            jina_api_key = gr.Textbox(
                                label="Jina API Key",
                                type="password",
                                placeholder="Enter your Jina API key...",
                                value="",
                                elem_classes=["api-key-input"],
                            )
                            jina_url = gr.Textbox(
                                label="Reranker URL",
                                value=current_settings.get("external_services", {})
                                .get("jina_reranker", {})
                                .get("url", "http://jina-reranker:8080"),
                            )
                            jina_model = gr.Textbox(
                                label="Model",
                                value=current_settings.get("external_services", {})
                                .get("jina_reranker", {})
                                .get("model", "jina-reranker-v2-base-multilingual"),
                            )
                            with gr.Row():
                                jina_timeout = gr.Number(
                                    label="Timeout (seconds)",
                                    value=current_settings.get("external_services", {})
                                    .get("jina_reranker", {})
                                    .get("timeout", 30),
                                    minimum=1,
                                    maximum=300,
                                )
                                jina_cache_ttl = gr.Number(
                                    label="Cache TTL (seconds)",
                                    value=current_settings.get("external_services", {})
                                    .get("jina_reranker", {})
                                    .get("cache_ttl", 3600),
                                    minimum=0,
                                    maximum=86400,
                                )
                            jina_max_retries = gr.Number(
                                label="Max Retries",
                                value=current_settings.get("external_services", {})
                                .get("jina_reranker", {})
                                .get("max_retries", 3),
                                minimum=0,
                                maximum=10,
                            )

                            with gr.Row():
                                validate_jina_btn = gr.Button(
                                    "🔍 Validate API Key", size="sm"
                                )
                                jina_validation_status = gr.Textbox(
                                    label="Validation Status",
                                    interactive=False,
                                    elem_classes=["success"],
                                )

                gr.Markdown("### SearXNG (Search Engine)")
                with gr.Group(elem_classes=["settings-section"]):
                    with gr.Row():
                        searxng_secret_key = gr.Textbox(
                            label="SearXNG Secret Key",
                            type="password",
                            placeholder="Enter SearXNG secret key...",
                            value="",
                            elem_classes=["api-key-input"],
                        )
                        searxng_url = gr.Textbox(
                            label="SearXNG URL",
                            value=current_settings.get("external_services", {})
                            .get("searxng", {})
                            .get("url", "http://searxng:8080"),
                        )

                update_services_btn = gr.Button(
                    "💾 Save External Service Settings", variant="primary"
                )
                services_status = gr.Textbox(
                    label="Status", interactive=False, elem_classes=["success"]
                )

            # System Configuration Tab
            with gr.TabItem("⚙️ System Configuration"):
                gr.Markdown("## System Settings")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Feature Toggles")
                        with gr.Group(elem_classes=["settings-section"]):
                            tool_system_enabled = gr.Checkbox(
                                label="Enable Tool System",
                                value=current_settings.get("system_config", {}).get(
                                    "tool_system_enabled", True
                                ),
                            )
                            agent_system_enabled = gr.Checkbox(
                                label="Enable Agent System",
                                value=current_settings.get("system_config", {}).get(
                                    "agent_system_enabled", True
                                ),
                            )
                            multi_writer_enabled = gr.Checkbox(
                                label="Enable Multi-Writer System",
                                value=current_settings.get("system_config", {}).get(
                                    "multi_writer_enabled", False
                                ),
                            )
                            enable_fallback = gr.Checkbox(
                                label="Enable Provider Fallback",
                                value=current_settings.get("system_config", {}).get(
                                    "enable_fallback", True
                                ),
                            )
                            debug_mode = gr.Checkbox(
                                label="Debug Mode",
                                value=current_settings.get("system_config", {}).get(
                                    "debug", True
                                ),
                            )

                    with gr.Column():
                        gr.Markdown("### Server Configuration")
                        with gr.Group(elem_classes=["settings-section"]):
                            preferred_provider = gr.Dropdown(
                                label="Preferred Provider",
                                choices=["openai_compatible", "ollama", "auto"],
                                value=current_settings.get("system_config", {}).get(
                                    "preferred_provider", "openai_compatible"
                                ),
                            )
                            environment = gr.Dropdown(
                                label="Environment",
                                choices=["development", "production", "testing"],
                                value=current_settings.get("system_config", {}).get(
                                    "environment", "development"
                                ),
                            )
                            with gr.Row():
                                host = gr.Textbox(
                                    label="Host",
                                    value=current_settings.get("system_config", {}).get(
                                        "host", "127.0.0.1"
                                    ),
                                )
                                port = gr.Number(
                                    label="Port",
                                    value=current_settings.get("system_config", {}).get(
                                        "port", 8000
                                    ),
                                    minimum=1,
                                    maximum=65535,
                                )
                            secret_key = gr.Textbox(
                                label="Secret Key",
                                type="password",
                                placeholder="Enter secret key...",
                                value="",
                                elem_classes=["api-key-input"],
                            )

                update_system_btn = gr.Button(
                    "💾 Save System Settings", variant="primary"
                )
                system_status = gr.Textbox(
                    label="Status", interactive=False, elem_classes=["success"]
                )

            # Multi-Writer Tab
            with gr.TabItem("📝 Multi-Writer System"):
                gr.Markdown("## Multi-Writer Configuration")
                gr.Markdown("⚠️ **Note**: Enable/disable the Multi-Writer System in the **System Configuration** tab.")

                with gr.Group(elem_classes=["settings-section"]):
                    mongodb_connection_string = gr.Textbox(
                        label="MongoDB Connection String",
                        value=current_settings.get("multi_writer", {}).get(
                            "mongodb_connection_string", "mongodb://localhost:27017"
                        ),
                    )
                    mongodb_database_name = gr.Textbox(
                        label="Database Name",
                        value=current_settings.get("multi_writer", {}).get(
                            "mongodb_database_name", "multi_writer_system"
                        ),
                    )

                update_multi_writer_btn = gr.Button(
                    "💾 Save Multi-Writer Settings", variant="primary"
                )
                multi_writer_status = gr.Textbox(
                    label="Status", interactive=False, elem_classes=["success"]
                )

            # Import/Export Tab
            with gr.TabItem("📥 Import/Export"):
                gr.Markdown("## Import and Export Settings")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Export Settings")
                        include_secrets = gr.Checkbox(
                            label="Include API Keys and Secrets",
                            value=False,
                            info="⚠️ Only enable this if you're exporting for your own backup",
                        )
                        export_btn = gr.Button(
                            "📤 Export Settings", variant="secondary"
                        )
                        export_output = gr.Code(
                            label="Exported Settings (JSON)",
                            language="json",
                            interactive=True,
                            lines=10,
                        )

                    with gr.Column():
                        gr.Markdown("### Import Settings")
                        import_input = gr.Code(
                            label="Settings JSON", language="json", lines=10
                        )
                        import_btn = gr.Button(
                            "📥 Import Settings", variant="secondary"
                        )
                        import_status = gr.Textbox(
                            label="Import Status",
                            interactive=False,
                            elem_classes=["success"],
                        )

        # Event handlers
        validate_openai_btn.click(
            validate_api_key,
            inputs=[
                gr.Textbox(value="openai_compatible", visible=False),
                openai_api_key,
            ],
            outputs=[openai_validation_status],
        )

        validate_jina_btn.click(
            validate_api_key,
            inputs=[gr.Textbox(value="jina_reranker", visible=False), jina_api_key],
            outputs=[jina_validation_status],
        )

        update_llm_btn.click(
            update_llm_provider_settings,
            inputs=[
                openai_enabled,
                openai_api_key,
                openai_base_url,
                openai_default_model,
                openai_provider_name,
                openai_timeout,
                openai_max_retries,
                ollama_enabled,
                ollama_base_url,
                ollama_default_model,
                ollama_timeout,
                ollama_max_retries,
                ollama_temperature,
                ollama_max_tokens,
                ollama_streaming,
            ],
            outputs=[llm_status],
        )

        update_services_btn.click(
            update_external_service_settings,
            inputs=[
                firecrawl_enabled,
                firecrawl_docker_url,
                firecrawl_bull_auth_key,
                firecrawl_scraping_enabled,
                firecrawl_max_concurrent_scrapes,
                firecrawl_scrape_timeout,
                jina_enabled,
                jina_api_key,
                jina_url,
                jina_model,
                jina_timeout,
                jina_cache_ttl,
                jina_max_retries,
                searxng_secret_key,
                searxng_url,
            ],
            outputs=[services_status],
        )

        update_system_btn.click(
            update_system_settings,
            inputs=[
                tool_system_enabled,
                agent_system_enabled,
                multi_writer_enabled,
                preferred_provider,
                enable_fallback,
                debug_mode,
                host,
                port,
                environment,
                secret_key,
            ],
            outputs=[system_status],
        )

        update_multi_writer_btn.click(
            update_multi_writer_settings,
            inputs=[
                mongodb_connection_string,
                mongodb_database_name,
            ],
            outputs=[multi_writer_status],
        )

        export_btn.click(
            export_settings, inputs=[include_secrets], outputs=[export_output]
        )

        import_btn.click(
            import_settings, inputs=[import_input], outputs=[import_status]
        )

    return app
