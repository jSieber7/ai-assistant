# app/api/ui_routes.py
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

# Create a new router specifically for UI-related endpoints
router = APIRouter(prefix="/api/ui", tags=["ui"])

# Pydantic models for data validation
class UIState(BaseModel):
    selectedProvider: Optional[str] = None
    selectedModel: Optional[str] = None
    apiHost: Optional[str] = None
    isApiServing: bool = False

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

# IMPORTANT: This is a placeholder for a real state management system.
# In a production app, replace this with a database, Redis, or server-side sessions.
UI_STATE_STORE = {}

@router.get("/state", response_model=UIState)
async def get_ui_state(request: Request):
    """
    Provides the current UI configuration for the frontend.
    The session ID should be retrieved securely from a cookie or token.
    """
    # Example of getting a session ID from a cookie
    session_id = request.cookies.get("session_id", "default_user")
    
    if session_id not in UI_STATE_STORE:
        # Return default state if nothing is set for the user
        return UIState(isApiServing=False)
    
    return UI_STATE_STORE[session_id]

@router.post("/select-model")
async def select_model(request: ModelSelectionRequest, http_request: Request):
    """
    Updates the selected model and provider for the user's session.
    """
    session_id = http_request.cookies.get("session_id", "default_user")

    if session_id not in UI_STATE_STORE:
        UI_STATE_STORE[session_id] = UIState()
    
    # Update the state in your store
    UI_STATE_STORE[session_id].selectedProvider = request.provider
    UI_STATE_STORE[session_id].selectedModel = request.model
    UI_STATE_STORE[session_id].isApiServing = True # Assume serving if a model is selected

    return {"success": True}