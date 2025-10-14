# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .api.tool_routes import router as tool_router
from .api.agent_routes import router as agent_router
from .api.monitoring_routes import router as monitoring_router
from .core.config import settings, initialize_agent_system, initialize_llm_providers
from .core.tools import tool_registry
from .core.agents.registry import agent_registry
from .core.monitoring.middleware import MonitoringMiddleware
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

# Initialize LLM providers
initialize_llm_providers()

# Initialize agent system
if settings.agent_system_enabled:
    initialize_agent_system()

# Include API routes
app.include_router(router)
app.include_router(tool_router)
app.include_router(agent_router)
app.include_router(monitoring_router)


@app.get("/")
async def root():
    tool_registry_stats = tool_registry.get_registry_stats()
    agent_registry_stats = (
        agent_registry.get_registry_stats() if settings.agent_system_enabled else {}
    )

    return {
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
