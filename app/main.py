# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import (
    settings,
    initialize_agent_system,
    initialize_llm_providers,
    initialize_firecrawl_system,
    initialize_playwright_system,
    initialize_visual_system,
)
from .core.tools import tool_registry
from .core.agents.management.registry import agent_registry
from .core.monitoring.middleware import MonitoringMiddleware
from .core.multi_writer_config import (
    initialize_multi_writer_system,
    is_multi_writer_enabled,
)
from .api.routes import router
from .api.tool_routes import router as tool_router
from .api.agent_routes import router as agent_router
from .api.monitoring_routes import router as monitoring_router
from .api.conversation_routes import router as conversation_router
from .api import ui_routes
from app import __version__


# Create FastAPI app
app = FastAPI(
    title="LangChain Agent Hub",
    description="Multi-agent system with FastAPI interface for an OpenAI Compatible API",
    version=__version__,
)

# Add CORS middleware for OpenWebUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Mount SearxNG static files
try:
    app.mount(
        "/static",
        StaticFiles(directory="/usr/local/searxng/searx/static", html=True),
        name="searxng-static",
    )
except Exception as e:
    print(f"Warning: Failed to mount SearxNG static files: {str(e)}")

# Initialize LLM providers
try:
    initialize_llm_providers()
    print("LLM providers initialized (mock responses will be used if no API keys configured)")
except Exception as e:
    print(f"Warning: Failed to initialize LLM providers: {str(e)}")
    print("Continuing without LLM providers - mock responses will be used")

# Initialize agent system
if settings.agent_system_enabled:
    try:
        # Try async initialization first
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                asyncio.create_task(initialize_agent_system_async())
                print("Agent system initialization scheduled (async)")
            else:
                # We're not in an async context, run it
                loop.run_until_complete(initialize_agent_system_async())
                print("Agent system initialized successfully (async)")
        except RuntimeError:
            # No event loop, fall back to sync initialization
            initialize_agent_system()
            print("Agent system initialized successfully (sync)")
    except Exception as e:
        print(f"Warning: Failed to initialize agent system: {str(e)}")
        print("Agent system will be disabled until properly configured")

# Initialize Deep Agents system
if settings.deep_agents_enabled:
    try:
        from .core.deep_agents import deep_agent_manager
        # Try async initialization first
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                asyncio.create_task(deep_agent_manager.initialize())
                print("Deep Agents system initialization scheduled (async)")
            else:
                # We're not in an async context, run it
                loop.run_until_complete(deep_agent_manager.initialize())
                print("Deep Agents system initialized successfully (async)")
        except RuntimeError:
            # No event loop, this should not happen in FastAPI startup
            # but we handle it gracefully
            print("Warning: Deep Agents requires an async context to initialize.")
            print("Deep Agents system will be disabled until properly configured")
    except Exception as e:
        print(f"Warning: Failed to initialize Deep Agents system: {str(e)}")
        print("Deep Agents system will be disabled until properly configured")

# Initialize Firecrawl system
if settings.firecrawl_settings.enabled:
    try:
        initialize_firecrawl_system()
        print("Firecrawl system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Firecrawl system: {str(e)}")
        print("Firecrawl system will be disabled until properly configured")

# Initialize Playwright system
if settings.playwright_settings.enabled:
    try:
        initialize_playwright_system()
        if settings.environment == "development":
            print("Development mode: Playwright system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Playwright system: {str(e)}")
        if settings.environment == "development":
            print("Development mode: Playwright system will be disabled until properly configured")
        else:
            print("Playwright system will be disabled until properly configured")

# Initialize Custom Reranker system
if settings.custom_reranker_enabled:
    try:
        from .core.tools.content.custom_reranker_tool import CustomRerankerTool

        custom_reranker_tool = CustomRerankerTool()
        tool_registry.register(custom_reranker_tool, category="reranking")
        print("Custom Reranker tool registered successfully")
        print("Custom Reranker system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Custom Reranker system: {str(e)}")
        print("Custom Reranker system will be disabled until properly configured")

# Initialize Ollama Reranker system
if settings.ollama_reranker_enabled:
    try:
        from .core.tools.content.ollama_reranker_tool import OllamaRerankerTool

        ollama_reranker_tool = OllamaRerankerTool()
        tool_registry.register(ollama_reranker_tool, category="reranking")
        print("Ollama Reranker tool registered successfully")
        print("Ollama Reranker system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Ollama Reranker system: {str(e)}")
        print("Ollama Reranker system will be disabled until properly configured")

# Initialize Jina Reranker system (legacy, kept for backward compatibility)
if settings.jina_reranker_enabled:
    try:
        from .core.tools.content.jina_reranker_tool import JinaRerankerTool

        jina_reranker_tool = JinaRerankerTool()
        tool_registry.register(jina_reranker_tool, category="reranking")
        print("Jina Reranker tool registered successfully")
        print("Jina Reranker system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Jina Reranker system: {str(e)}")
        print("Jina Reranker system will be disabled until properly configured")

# Initialize visual system
if settings.visual_system_enabled:
    try:
        initialize_visual_system()
        print("Visual system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize visual system: {str(e)}")
        print("Visual system will be disabled until properly configured")

# Initialize multi-writer system (optional)
multi_writer_config = None
if is_multi_writer_enabled():
    try:
        multi_writer_config = initialize_multi_writer_system()
        print("Multi-writer system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize multi-writer system: {str(e)}")
        print("Multi-writer system will be disabled until properly configured")

# Include API routes
app.include_router(router)
app.include_router(tool_router)
app.include_router(agent_router)
app.include_router(monitoring_router)
app.include_router(conversation_router, prefix="/api/v1", tags=["conversations"])
app.include_router(ui_routes.router)

# Include multi-writer routes if enabled
if is_multi_writer_enabled():
    from .api.multi_writer_routes import router as multi_writer_router

    app.include_router(multi_writer_router)

# Chainlit has been removed from this application
# A React-based UI will be implemented in the future

@app.get("/")
async def root():
    tool_registry_stats = tool_registry.get_registry_stats()
    agent_registry_stats = (
        agent_registry.get_registry_stats() if settings.agent_system_enabled else {}
    )

    response = {
        "message": "AI Assistant Tool System is running!",
        "version": __version__,
        "status": "ready",
        "tool_system": {
            "enabled": settings.tool_system_enabled,
            "tools_registered": tool_registry_stats["total_tools"],
            "tools_enabled": tool_registry_stats["enabled_tools"],
            "categories": tool_registry_stats["categories"],
        },
        "agent_system": {
            "enabled": settings.agent_system_enabled,
            "agents_registered": agent_registry_stats.get("total_agents", 0),
            "agents_active": agent_registry_stats.get("active_agents", 0),
            "default_agent": agent_registry_stats.get("default_agent", "none"),
            "categories": agent_registry_stats.get("categories", []),
        },
    }

    # Add multi-writer system info if enabled
    if is_multi_writer_enabled():
        response["multi_writer_system"] = {
            "enabled": True,
            "config": multi_writer_config if multi_writer_config else {},
            "api_prefix": settings.multi_writer_settings.api_prefix,
        }
    else:
        response["multi_writer_system"] = {
            "enabled": False,
            "message": "Multi-writer system is disabled",
        }

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
