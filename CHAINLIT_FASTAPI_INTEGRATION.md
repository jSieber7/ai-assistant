# Chainlit FastAPI Integration

## Overview
This document describes the changes made to move Chainlit from a separate service to a FastAPI subapplication to resolve persistent routing errors.

## Changes Made

### 1. Modified app/main.py
- Added import for `mount_chainlit` from `chainlit.utils`
- Added code to mount Chainlit as a subapplication at the `/chat` path using the correct target file (`chainlit_app.py`)
- Added error handling for the mounting process

### 2. Updated docker-compose.yml
- Removed the separate `chainlit` service
- Updated Traefik routing rules for the `ai-assistant` service to handle all paths (including `/chat`)
- Updated Traefik routing rules for the `ai-assistant-dev` service to handle all paths (except `/search`)
- Removed all Chainlit-specific labels and configurations

### 3. Simplified config/docker/middlewares.yml
- Removed Chainlit-specific middlewares:
  - `chainlit-stripprefix`
  - `chainlit-websocket`
  - `chainlit-assets`
  - `chainlit-assets-prefix`

## Benefits of This Approach

1. **Simpler Architecture**: Single service instead of two, with internal routing handled by FastAPI
2. **No Complex Traefik Rules**: Eliminated the need for complex Traefik routing rules
3. **Consistent Path Handling**: All paths are handled consistently by FastAPI
4. **Easier Maintenance**: Fewer services to manage and configure
5. **No Asset Path Issues**: FastAPI handles all asset paths correctly

## How to Test

### Using Docker Compose

1. **For Development**:
   ```bash
   docker compose --profile dev up
   ```

2. **For Production**:
   ```bash
   docker compose up
   ```

### Access Points

After starting the services, you can access:
- FastAPI main application: `http://localhost:8000/`
- Chainlit interface: `http://localhost:8000/chat`
- SearXNG search: `http://localhost:8000/search`

### Verification

1. Check that the main FastAPI application is accessible at `http://localhost:8000/`
2. Check that the Chainlit interface is accessible at `http://localhost:8000/chat`
3. Verify that all Chainlit functionality works properly (chatting, provider selection, etc.)

## Troubleshooting

If you encounter issues:

1. Check the logs of the `ai-assistant` or `ai-assistant-dev` container for any errors related to mounting Chainlit
2. Ensure that the Chainlit app file (`app/ui/chainlit_app.py`) is accessible and properly formatted
3. Verify that all required dependencies are installed in the Docker container

## Notes

- The FastAPI application now handles all routing internally, so there's no need for complex Traefik rules
- WebSocket connections used by Chainlit are now handled by FastAPI
- Asset paths are managed by FastAPI, eliminating the previous asset path issues
- The changes are backward compatible and shouldn't affect existing functionality