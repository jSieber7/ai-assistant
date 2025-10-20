"""
Visual Agent System Initialization

This module provides initialization functions for the Visual LMM agent system,
including visual tools registration and agent setup.
"""

import logging
from typing import Dict, Any

from . import get_llm, settings
from ..tools.execution.registry import tool_registry
from ..agents.management.registry import agent_registry
from ..visual_llm_provider import (
    visual_provider_registry,
    OpenAIVisionProvider,
    OllamaVisionProvider,
)
from ..tools.visual.image_processor import ImageProcessorTool
from ..tools.visual.visual_analyzer import VisualAnalyzerTool
from ..tools.visual.visual_browser import VisualBrowserTool
from ..agents.specialized.visual_agent import VisualAgent

logger = logging.getLogger(__name__)


def initialize_visual_providers() -> Dict[str, Any]:
    """
    Initialize visual LMM providers based on available configuration
    
    Returns:
        Dictionary with initialization status and provider information
    """
    providers_info = {}
    
    # Initialize OpenAI Vision provider if API key is available
    if settings.openai_api_key:
        try:
            openai_vision_provider = OpenAIVisionProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or "https://api.openai.com/v1"
            )
            visual_provider_registry.register_provider(openai_vision_provider)
            providers_info["openai_vision"] = {
                "status": "initialized",
                "name": openai_vision_provider.name,
                "models": ["gpt-4-vision-preview", "gpt-4o"],
            }
            logger.info("OpenAI Vision provider initialized")
        except Exception as e:
            providers_info["openai_vision"] = {
                "status": "failed",
                "error": str(e),
            }
            logger.error(f"Failed to initialize OpenAI Vision provider: {str(e)}")
    else:
        providers_info["openai_vision"] = {
            "status": "skipped",
            "reason": "OpenAI API key not configured",
        }
    
    # Initialize Ollama Vision provider if enabled
    if settings.ollama_settings.enabled:
        try:
            ollama_vision_provider = OllamaVisionProvider(
                base_url=settings.ollama_settings.base_url
            )
            visual_provider_registry.register_provider(ollama_vision_provider)
            providers_info["ollama_vision"] = {
                "status": "initialized",
                "name": ollama_vision_provider.name,
                "base_url": settings.ollama_settings.base_url,
            }
            logger.info("Ollama Vision provider initialized")
        except Exception as e:
            providers_info["ollama_vision"] = {
                "status": "failed",
                "error": str(e),
            }
            logger.error(f"Failed to initialize Ollama Vision provider: {str(e)}")
    else:
        providers_info["ollama_vision"] = {
            "status": "skipped",
            "reason": "Ollama not enabled",
        }
    
    # Set default visual provider
    configured_providers = visual_provider_registry.list_configured_providers()
    if configured_providers:
        # Prefer OpenAI Vision if available, otherwise use first configured
        for provider in configured_providers:
            if provider.provider_type.value == "openai_vision":
                visual_provider_registry.set_default_provider(provider.provider_type)
                logger.info(f"Set {provider.name} as default visual provider")
                break
        else:
            # Use first available provider
            default_provider = configured_providers[0]
            visual_provider_registry.set_default_provider(default_provider.provider_type)
            logger.info(f"Set {default_provider.name} as default visual provider")
    else:
        logger.warning("No visual providers configured - visual analysis features will not work")
    
    return providers_info


def initialize_visual_tools() -> Dict[str, Any]:
    """
    Initialize and register visual tools
    
    Returns:
        Dictionary with initialization status and tool information
    """
    tools_info = {}
    
    try:
        # Initialize Image Processor Tool
        image_processor = ImageProcessorTool()
        tool_registry.register(image_processor, category="visual")
        tools_info["image_processor"] = {
            "status": "registered",
            "name": image_processor.name,
            "description": image_processor.description,
        }
        logger.info("Image Processor tool registered")
        
    except Exception as e:
        tools_info["image_processor"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to register Image Processor tool: {str(e)}")
    
    try:
        # Initialize Visual Analyzer Tool
        default_visual_model = "openai_vision:gpt-4-vision-preview"
        visual_analyzer = VisualAnalyzerTool(default_model=default_visual_model)
        tool_registry.register(visual_analyzer, category="visual")
        tools_info["visual_analyzer"] = {
            "status": "registered",
            "name": visual_analyzer.name,
            "description": visual_analyzer.description,
            "default_model": default_visual_model,
        }
        logger.info("Visual Analyzer tool registered")
        
    except Exception as e:
        tools_info["visual_analyzer"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to register Visual Analyzer tool: {str(e)}")
    
    try:
        # Initialize Visual Browser Tool
        visual_browser = VisualBrowserTool(
            visual_model="openai_vision:gpt-4-vision-preview",
            headless=True,
            browser_type="chromium",
        )
        tool_registry.register(visual_browser, category="visual")
        tools_info["visual_browser"] = {
            "status": "registered",
            "name": visual_browser.name,
            "description": visual_browser.description,
        }
        logger.info("Visual Browser tool registered")
        
    except Exception as e:
        tools_info["visual_browser"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to register Visual Browser tool: {str(e)}")
    
    return tools_info


def initialize_visual_agent() -> Dict[str, Any]:
    """
    Initialize and register the Visual Agent
    
    Returns:
        Dictionary with initialization status and agent information
    """
    agent_info = {}
    
    try:
        # Get LLM instance
        llm = get_llm()
        if not llm:
            raise ValueError("LLM not available for Visual Agent")
        
        # Create Visual Agent
        visual_agent = VisualAgent(
            llm=llm,
            tool_registry=tool_registry,
            max_iterations=3,
            default_visual_model="openai_vision:gpt-4-vision-preview",
            enable_browser_control=True,
        )
        
        # Register the agent
        agent_registry.register(visual_agent, category="visual")
        
        agent_info = {
            "status": "registered",
            "name": visual_agent.name,
            "description": visual_agent.description,
            "capabilities": [
                "Image analysis",
                "OCR and text extraction",
                "Object detection",
                "Web page visual analysis",
                "Browser control with visual understanding",
                "Multi-image comparison",
            ],
        }
        
        logger.info("Visual Agent registered successfully")
        
    except Exception as e:
        agent_info = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to register Visual Agent: {str(e)}")
    
    return agent_info


def initialize_visual_system() -> Dict[str, Any]:
    """
    Initialize the complete Visual LMM system
    
    Returns:
        Dictionary with initialization status for all components
    """
    logger.info("Initializing Visual LMM system...")
    
    initialization_result = {
        "visual_system": {
            "status": "initializing",
            "components": {},
        }
    }
    
    # Initialize visual providers
    try:
        providers_info = initialize_visual_providers()
        initialization_result["visual_system"]["components"]["providers"] = providers_info
        
        # Check if at least one provider is configured
        configured_providers = [
            name for name, info in providers_info.items()
            if info.get("status") == "initialized"
        ]
        
        if not configured_providers:
            logger.warning("No visual providers configured - visual features will be limited")
            initialization_result["visual_system"]["status"] = "degraded"
        else:
            logger.info(f"Visual providers initialized: {configured_providers}")
            
    except Exception as e:
        initialization_result["visual_system"]["components"]["providers"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to initialize visual providers: {str(e)}")
    
    # Initialize visual tools
    try:
        tools_info = initialize_visual_tools()
        initialization_result["visual_system"]["components"]["tools"] = tools_info
        
        # Check if tools were registered successfully
        registered_tools = [
            name for name, info in tools_info.items()
            if info.get("status") == "registered"
        ]
        
        if not registered_tools:
            logger.warning("No visual tools registered - visual features will be limited")
            initialization_result["visual_system"]["status"] = "degraded"
        else:
            logger.info(f"Visual tools registered: {registered_tools}")
            
    except Exception as e:
        initialization_result["visual_system"]["components"]["tools"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to initialize visual tools: {str(e)}")
    
    # Initialize visual agent
    try:
        agent_info = initialize_visual_agent()
        initialization_result["visual_system"]["components"]["agent"] = agent_info
        
        if agent_info.get("status") == "registered":
            logger.info("Visual Agent system initialization completed successfully")
            if initialization_result["visual_system"]["status"] != "degraded":
                initialization_result["visual_system"]["status"] = "success"
        else:
            logger.warning("Visual Agent registration failed")
            initialization_result["visual_system"]["status"] = "degraded"
            
    except Exception as e:
        initialization_result["visual_system"]["components"]["agent"] = {
            "status": "failed",
            "error": str(e),
        }
        logger.error(f"Failed to initialize Visual Agent: {str(e)}")
        initialization_result["visual_system"]["status"] = "failed"
    
    return initialization_result


async def health_check_visual_system() -> Dict[str, Any]:
    """
    Perform health check on the Visual LMM system
    
    Returns:
        Dictionary with health check results
    """
    health_result = {
        "visual_system": {
            "status": "unknown",
            "components": {},
        }
    }
    
    # Check visual providers
    try:
        await visual_provider_registry.health_check_all()
        configured_providers = visual_provider_registry.list_configured_providers()
        
        if configured_providers:
            provider_health = {
                "status": "healthy",
                "configured_providers": len(configured_providers),
                "providers": [],
            }
            
            for provider in configured_providers:
                provider_health["providers"].append({
                    "name": provider.name,
                    "healthy": provider.is_healthy(),
                    "configured": provider.is_configured(),
                })
            
            health_result["visual_system"]["components"]["providers"] = provider_health
        else:
            health_result["visual_system"]["components"]["providers"] = {
                "status": "unhealthy",
                "reason": "No visual providers configured",
            }
            
    except Exception as e:
        health_result["visual_system"]["components"]["providers"] = {
            "status": "error",
            "error": str(e),
        }
    
    # Check visual tools
    try:
        visual_tools = tool_registry.get_tools_by_category("visual")
        if visual_tools:
            health_result["visual_system"]["components"]["tools"] = {
                "status": "healthy",
                "registered_tools": len(visual_tools),
                "tools": [tool.name for tool in visual_tools],
            }
        else:
            health_result["visual_system"]["components"]["tools"] = {
                "status": "unhealthy",
                "reason": "No visual tools registered",
            }
            
    except Exception as e:
        health_result["visual_system"]["components"]["tools"] = {
            "status": "error",
            "error": str(e),
        }
    
    # Check visual agent
    try:
        visual_agents = agent_registry.get_agents_by_category("visual")
        if visual_agents:
            health_result["visual_system"]["components"]["agent"] = {
                "status": "healthy",
                "registered_agents": len(visual_agents),
                "agents": [agent.name for agent in visual_agents],
            }
        else:
            health_result["visual_system"]["components"]["agent"] = {
                "status": "unhealthy",
                "reason": "No visual agents registered",
            }
            
    except Exception as e:
        health_result["visual_system"]["components"]["agent"] = {
            "status": "error",
            "error": str(e),
        }
    
    # Determine overall status
    component_statuses = [
        component.get("status", "unknown")
        for component in health_result["visual_system"]["components"].values()
    ]
    
    if all(status == "healthy" for status in component_statuses):
        health_result["visual_system"]["status"] = "healthy"
    elif any(status == "error" for status in component_statuses):
        health_result["visual_system"]["status"] = "error"
    elif any(status == "unhealthy" for status in component_statuses):
        health_result["visual_system"]["status"] = "degraded"
    else:
        health_result["visual_system"]["status"] = "unknown"
    
    return health_result