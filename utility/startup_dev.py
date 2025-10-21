#!/usr/bin/env python3
"""
Development startup script for AI Assistant.

This script ensures the app can start healthy in development mode
without requiring API keys. It performs basic checks and starts
the application with appropriate configuration.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def check_environment():
    """Check if the environment is properly set up."""
    env = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Checking {env} environment...")
    
    # Check for required environment variables
    required_vars = ["SECRET_KEY", "SEARXNG_SECRET_KEY", "POSTGRES_PASSWORD"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {missing_vars}")
        logger.warning("Setting default values for startup...")
        
        # Set default values for missing required variables
        defaults = {
            "SECRET_KEY": "default-secret-key-change-in-production",
            "SEARXNG_SECRET_KEY": "default-searxng-secret-key",
            "POSTGRES_PASSWORD": "default-postgres-password"
        }
        
        for var in missing_vars:
            if var in defaults:
                os.environ[var] = defaults[var]
                logger.info(f"Set default value for {var}")
    
    # Check for optional API keys (not required for startup)
    api_keys = {
        "OPENAI_COMPATIBLE_API_KEY": os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "JINA_API_KEY": os.getenv("JINA_API_KEY"),
    }
    
    configured_keys = [key for key, value in api_keys.items() if value]
    
    if configured_keys:
        logger.info(f"Configured API keys: {configured_keys}")
    else:
        logger.info("No API keys configured - app will use mock responses")
    
    return True

async def initialize_basic_systems():
    """Initialize basic systems that don't require API keys."""
    logger.info("Initializing basic systems...")
    
    try:
        # Import and initialize basic configuration
        from core.config import settings
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        
        # Initialize tool registry (doesn't require API keys)
        from core.tools.execution.registry import tool_registry
        tool_stats = tool_registry.get_registry_stats()
        logger.info(f"Tool registry initialized: {tool_stats['total_tools']} tools registered")
        
        # Initialize monitoring (doesn't require API keys)
        from core.monitoring.health import health_monitor
        logger.info("Health monitoring initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize basic systems: {e}")
        return False

async def startup_application():
    """Main startup function."""
    env = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Starting AI Assistant in {env} mode...")
    
    # Check environment
    if not await check_environment():
        sys.exit(1)
    
    # Initialize basic systems
    if not await initialize_basic_systems():
        sys.exit(1)
    
    logger.info("✅ Startup checks passed")
    logger.info("🚀 Application is ready to start")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(startup_application())
        if result:
            logger.info("Startup completed successfully")
            sys.exit(0)
        else:
            logger.error("Startup failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Startup interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        sys.exit(1)