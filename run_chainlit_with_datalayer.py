#!/usr/bin/env python3
"""
Chainlit runner with data layer integration.

This script starts the Chainlit application with the PostgreSQL/Supabase
data layer enabled for chat lifecycle management and persistence.
"""

import os
import sys
import asyncio
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test the database connection before starting Chainlit."""
    try:
        from app.ui.chainlit_data_layer import get_data_layer
        
        logger.info("Testing database connection for Chainlit data layer...")
        data_layer = get_data_layer()
        await data_layer.connect()
        
        # Test a simple query
        result = await data_layer.get_user("test_user")
        logger.info("Database connection test successful!")
        
        # Close the connection
        await data_layer.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def main():
    """Main function to run Chainlit with data layer."""
    logger.info("Starting Chainlit with PostgreSQL/Supabase data layer...")
    
    # Check environment variables
    database_url = os.getenv("CHAINLIT_DATABASE_URL")
    if not database_url:
        logger.error("CHAINLIT_DATABASE_URL environment variable not set!")
        logger.error("Please set this variable to your PostgreSQL/Supabase connection string.")
        sys.exit(1)
    
    logger.info(f"Using database URL: {database_url}")
    
    # Test database connection
    try:
        connection_ok = asyncio.run(test_database_connection())
        if not connection_ok:
            logger.error("Database connection test failed. Please check your configuration.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error testing database connection: {e}")
        logger.error("Please ensure your database is running and accessible.")
        sys.exit(1)
    
    # Import and run Chainlit
    try:
        from app.ui.chainlit_app import create_chainlit_app
        
        # Initialize the app
        create_chainlit_app()
        
        # Run Chainlit
        import chainlit as cl
        
        # Set the host and port
        host = os.getenv("CHAINLIT_HOST", "0.0.0.0")
        port = int(os.getenv("CHAINLIT_PORT", "8000"))
        
        logger.info(f"Starting Chainlit on {host}:{port}")
        logger.info("Chat history and lifecycle management enabled!")
        logger.info("Open your browser and navigate to the Chainlit interface.")
        
        # Run the Chainlit app
        cl.run(host=host, port=port)
        
    except Exception as e:
        logger.error(f"Error starting Chainlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()