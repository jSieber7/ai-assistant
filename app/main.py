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
from chainlit.utils import mount_chainlit
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
    allow_origins=["*"],  # Configure appropriately for production
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
        initialize_agent_system()
        print("Agent system initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize agent system: {str(e)}")
        print("Agent system will be disabled until properly configured")

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

# Initialize Jina Reranker system
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

# Include multi-writer routes if enabled
if is_multi_writer_enabled():
    from .api.multi_writer_routes import router as multi_writer_router

    app.include_router(multi_writer_router)

# Mount Chainlit as a subapplication
try:
    # Mount Chainlit at /chat path
    mount_chainlit(app=app, target="chainlit_app.py", path="/chat")
    print("Chainlit app mounted successfully at /chat")
except Exception as e:
    print(f"Warning: Failed to mount Chainlit app: {str(e)}")
    print("Chainlit interface will not be available")

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
        "chainlit_interface": f"http://{settings.host}:{settings.port}/chat",
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
