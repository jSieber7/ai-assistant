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
from .api.agent_designer_routes import router as agent_designer_router
from .api import ui_routes
from app import __version__

# LangChain integration imports
from .core.langchain.integration import integration_layer
from .core.langchain.agent_manager import agent_manager
from .core.langchain.llm_manager import llm_manager
from .core.langchain.tool_registry import tool_registry as langchain_tool_registry
from .core.langchain.memory_manager import memory_manager

import asyncio
import logging

logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="LangChain Agent Hub",
    description="Multi-agent system with FastAPI interface for an OpenAI Compatible API",
    version=__version__,
)

# Add CORS middleware for OpenWebUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://frontend.localhost",  # Allow frontend domain in Docker
        "http://app.localhost",       # Allow direct backend access
        "http://localhost:3000",      # Allow local development
        "http://127.0.0.1:3000",      # Allow local development
    ],  # Configure appropriately for production
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


async def initialize_systems():
    """Initialize all systems with LangChain integration"""
    try:
        # Initialize LangChain integration layer first
        await integration_layer.initialize()
        logger.info("LangChain integration layer initialized")
        
        # Get integration mode and feature flags
        integration_mode = integration_layer.get_integration_mode()
        feature_flags = integration_layer.get_feature_flags()
        
        logger.info(f"Integration mode: {integration_mode.value}")
        logger.info(f"Feature flags: {feature_flags}")
        
        # Initialize legacy systems if in legacy or hybrid mode
        if integration_mode.value in ["legacy", "hybrid"]:
            # Initialize LLM providers (legacy)
            if not feature_flags.get("use_langchain_llm", False):
                try:
                    initialize_llm_providers()
                    print("Legacy LLM providers initialized (mock responses will be used if no API keys configured)")
                except Exception as e:
                    print(f"Warning: Failed to initialize legacy LLM providers: {str(e)}")
                    print("Continuing without legacy LLM providers - mock responses will be used")
        
        # Initialize agent system (legacy or LangChain)
        if settings.agent_system_enabled:
            try:
                if feature_flags.get("use_langchain_agents", False):
                    # LangChain agents are already initialized by integration layer
                    print("LangGraph Agent Manager initialized")
                else:
                    # Use legacy agent system
                    # Try async initialization first
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're in an async context, create a task
                            asyncio.create_task(initialize_agent_system_async())
                            print("Legacy agent system initialization scheduled (async)")
                        else:
                            # We're not in an async context, run it
                            loop.run_until_complete(initialize_agent_system_async())
                            print("Legacy agent system initialized successfully (async)")
                    except RuntimeError:
                        # No event loop, fall back to sync initialization
                        initialize_agent_system()
                        print("Legacy agent system initialized successfully (sync)")
            except Exception as e:
                print(f"Warning: Failed to initialize agent system: {str(e)}")
                print("Agent system will be disabled until properly configured")

        # Initialize Deep Agents system (legacy)
        if settings.deep_agents_enabled:
            try:
                from .core.deep_agents import deep_agent_manager
                # Try async initialization first
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

        # Initialize Firecrawl system (legacy)
        if settings.firecrawl_settings.enabled:
            try:
                initialize_firecrawl_system()
                print("Firecrawl system initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Firecrawl system: {str(e)}")
                print("Firecrawl system will be disabled until properly configured")

        # Initialize Playwright system (legacy)
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

        # Initialize Custom Reranker system (legacy)
        if settings.custom_reranker_enabled:
            try:
                from .core.tools.content.custom_reranker_tool import CustomRerankerTool

                custom_reranker_tool = CustomRerankerTool()
                
                # Register with appropriate registry
                if feature_flags.get("use_langchain_tools", False):
                    await langchain_tool_registry.register_custom_tool(custom_reranker_tool, category="reranking")
                else:
                    tool_registry.register(custom_reranker_tool, category="reranking")
                    
                print("Custom Reranker tool registered successfully")
                print("Custom Reranker system initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Custom Reranker system: {str(e)}")
                print("Custom Reranker system will be disabled until properly configured")

        # Initialize Ollama Reranker system (legacy)
        if settings.ollama_reranker_enabled:
            try:
                from .core.tools.content.ollama_reranker_tool import OllamaRerankerTool

                ollama_reranker_tool = OllamaRerankerTool()
                
                # Register with appropriate registry
                if feature_flags.get("use_langchain_tools", False):
                    await langchain_tool_registry.register_custom_tool(ollama_reranker_tool, category="reranking")
                else:
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
                
                # Register with appropriate registry
                if feature_flags.get("use_langchain_tools", False):
                    await langchain_tool_registry.register_custom_tool(jina_reranker_tool, category="reranking")
                else:
                    tool_registry.register(jina_reranker_tool, category="reranking")
                    
                print("Jina Reranker tool registered successfully")
                print("Jina Reranker system initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Jina Reranker system: {str(e)}")
                print("Jina Reranker system will be disabled until properly configured")

        # Initialize visual system (legacy)
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
        
        # Initialize specialized LangChain agents
        try:
            from .core.agents.specialized_langchain import (
                SummarizeAgent,
                WebdriverAgent,
                ScraperAgent,
                SearchQueryAgent,
                ChainOfThoughtAgent,
                CreativeStoryAgent,
                ToolSelectionAgent,
                SemanticUnderstandingAgent,
                FactCheckerAgent,
            )
            
            # Register specialized agents with LangGraph agent manager
            specialized_agents = [
                SummarizeAgent(),
                WebdriverAgent(),
                ScraperAgent(),
                SearchQueryAgent(),
                ChainOfThoughtAgent(),
                CreativeStoryAgent(),
                ToolSelectionAgent(),
                SemanticUnderstandingAgent(),
                FactCheckerAgent(),
            ]
            
            for agent in specialized_agents:
                # Create agent config for registration
                from .core.langchain.agent_manager import AgentConfig, AgentType
                config = AgentConfig(
                    name=agent.name,
                    agent_type=AgentType.CONVERSATIONAL,  # Default type, can be refined
                    llm_model="gpt-4",  # Default model, can be configured
                    system_prompt=agent.description,
                    tools=[],  # Tools will be assigned based on agent type
                )
                
                await agent_manager.register_agent(config)
                print(f"Registered specialized agent: {agent.name}")
                
            print("Specialized LangChain agents initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize specialized LangChain agents: {str(e)}")
            print("Specialized agents will be disabled until properly configured")
        
        return multi_writer_config
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {str(e)}")
        print(f"Error during system initialization: {str(e)}")
        return None


# Initialize systems on startup
@app.on_event("startup")
async def startup_event():
    """Initialize all systems on startup"""
    await initialize_systems()


# Include API routes
app.include_router(router)
app.include_router(tool_router)
app.include_router(agent_router)
app.include_router(monitoring_router)
app.include_router(conversation_router, prefix="/api/v1", tags=["conversations"])
app.include_router(agent_designer_router)
app.include_router(ui_routes.router)

# Include multi-writer routes if enabled
if is_multi_writer_enabled():
    from .api.multi_writer_routes import router as multi_writer_router
    app.include_router(multi_writer_router)

# Chainlit has been removed from this application
# A React-based UI will be implemented in future

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns a 200 OK status if application is running.
    """
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with system information"""
    # Get integration mode and feature flags
    integration_mode = integration_layer.get_integration_mode()
    feature_flags = integration_layer.get_feature_flags()
    
    # Get registry stats based on active systems
    if feature_flags.get("use_langchain_tools", False):
        tool_registry_stats = langchain_tool_registry.get_registry_stats()
    else:
        tool_registry_stats = tool_registry.get_registry_stats()
    
    if feature_flags.get("use_langchain_agents", False):
        agent_registry_stats = agent_manager.get_registry_stats()
    else:
        agent_registry_stats = (
            agent_registry.get_registry_stats() if settings.agent_system_enabled else {}
        )
    
    # Get LangChain component stats
    langchain_stats = {}
    if feature_flags.get("use_langchain_llm", False):
        try:
            models = await llm_manager.list_models()
            langchain_stats["llm_manager"] = {
                "available_models": len(models),
                "providers": list(set(m.provider.value for m in models)),
            }
        except Exception as e:
            langchain_stats["llm_manager"] = {"error": str(e)}
    
    if feature_flags.get("use_langchain_tools", False):
        langchain_stats["tool_registry"] = tool_registry_stats
    
    if feature_flags.get("use_langchain_agents", False):
        langchain_stats["agent_manager"] = agent_registry_stats
    
    if feature_flags.get("use_langchain_memory", False):
        try:
            memory_stats = await memory_manager.get_memory_stats()
            langchain_stats["memory_manager"] = memory_stats
        except Exception as e:
            langchain_stats["memory_manager"] = {"error": str(e)}
    
    response = {
        "message": "AI Assistant Tool System is running!",
        "version": __version__,
        "status": "ready",
        "integration": {
            "mode": integration_mode.value,
            "feature_flags": feature_flags,
            "stats": langchain_stats,
        },
        "tool_system": {
            "enabled": settings.tool_system_enabled,
            "tools_registered": tool_registry_stats["total_tools"],
            "tools_enabled": tool_registry_stats["enabled_tools"],
            "categories": tool_registry_stats["categories"],
            "using_langchain": feature_flags.get("use_langchain_tools", False),
        },
        "agent_system": {
            "enabled": settings.agent_system_enabled,
            "agents_registered": agent_registry_stats.get("total_agents", 0),
            "agents_active": agent_registry_stats.get("active_agents", 0),
            "default_agent": agent_registry_stats.get("default_agent", "none"),
            "categories": agent_registry_stats.get("categories", []),
            "using_langchain": feature_flags.get("use_langchain_agents", False),
        },
    }
    
    # Add multi-writer system info if enabled
    if is_multi_writer_enabled():
        try:
            multi_writer_config = initialize_multi_writer_system()
            response["multi_writer_system"] = {
                "enabled": True,
                "config": multi_writer_config if multi_writer_config else {},
                "api_prefix": settings.multi_writer_settings.api_prefix,
            }
        except Exception as e:
            response["multi_writer_system"] = {
                "enabled": True,
                "error": str(e),
            }
    else:
        response["multi_writer_system"] = {
            "enabled": False,
            "message": "Multi-writer system is disabled",
        }
    
    return response


@app.get("/langchain/health")
async def langchain_health_check():
    """
    LangChain components health check endpoint.
    Returns detailed health status of LangChain integration components.
    """
    return await integration_layer.health_check()


@app.get("/langchain/mode")
async def get_integration_mode():
    """
    Get current integration mode and feature flags.
    """
    return {
        "mode": integration_layer.get_integration_mode().value,
        "feature_flags": integration_layer.get_feature_flags(),
        "migration_stats": integration_layer.get_migration_stats(),
    }


@app.post("/langchain/mode")
async def set_integration_mode(mode: str):
    """
    Set integration mode.
    
    Args:
        mode: Integration mode (legacy, langchain, hybrid, migration)
    """
    from .core.langchain.integration import IntegrationMode
    
    try:
        integration_mode = IntegrationMode(mode)
        success = integration_layer.set_integration_mode(integration_mode)
        
        if success:
            return {
                "success": True,
                "mode": integration_mode.value,
                "message": f"Integration mode set to {integration_mode.value}",
            }
        else:
            return {
                "success": False,
                "error": "Failed to set integration mode",
            }
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid integration mode: {mode}",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)