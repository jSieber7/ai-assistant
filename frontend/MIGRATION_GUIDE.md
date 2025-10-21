# Migration Guide: Python SSR to React CSR Components

This guide outlines the migration of Python-based Server-side Rendering (SSR) UI components to React/JSX Client-side Rendering (CSR) components for your Chainlit application.

## Overview of Changes

The original Python components (`app/ui/top_bar_component.py` and `app/ui/status_bar_component.py`) rendered HTML strings on the server and sent them to the client. This has been replaced with a single, reusable React component:

1.  **`NavigationHeader.jsx`**: A flexible component that can act as either a sticky top navigation bar or a non-sticky status bar, controlled by a `variant` prop.

This component is self-contained, receives data via props, and signals events via callback props, which is the standard React pattern.

## React Component

### `NavigationHeader.jsx`

Located at [`frontend/components/NavigationHeader.jsx`](frontend/components/NavigationHeader.jsx).

This component consolidates the functionality of the original `TopBar` and `StatusBar` into a single, flexible interface.

**Props:**
-   `variant` (string): The style variant. Use `'top'` for a sticky top bar or `'status'` for a non-sticky status bar. Default is `'top'`.
-   `selectedProvider` (string, optional): The currently selected provider name.
-   `selectedModel` (string, optional): The currently selected model name.
-   `apiHost` (string, optional): The host address for the API (e.g., `localhost:8000`). The component will construct the full URL.
-   `isApiServing` (boolean): `true` if the API is active, `false` otherwise.
-   `onAgentClick` (function, optional): Callback for the "Agents" button click.
-   `onModelClick` (function, optional): Callback for when the model info area is clicked.

---

## Required Backend API Endpoints

The React components are "dumb" and require data to be fetched from the backend. The original Python components had direct access to server-side state (e.g., `cl.user_session`). To replicate this functionality in a CSR architecture, your backend must provide the following API endpoints.

### 1. Get Current UI State

This endpoint is **missing** and must be created. It should return the current configuration for the user's session, which the frontend will use to populate the components.

**Endpoint:** `GET /api/ui/state`

**Response:**
```json
{
  "selectedProvider": "openai_compatible",
  "selectedModel": "gpt-4",
  "apiHost": "localhost:8000",
  "isApiServing": true
}
```

### 2. Update Model Selection

This endpoint is **missing** and must be created. It allows the frontend to update the selected model and provider for the session.

**Endpoint:** `POST /api/ui/select-model`

**Request Body:**
```json
{
  "provider": "openai_compatible",
  "model": "gpt-4"
}
```

**Response:**
```json
{
  "success": true
}
```

### 3. Existing Endpoints Used by Components

The following endpoints already exist in your application and can be used to build more dynamic UIs, like the `LLMProviderDropdown` component.

-   `GET /v1/providers`: Lists all available LLM providers and their status. ([`app/api/routes.py:331`](app/api/routes.py:331))
-   `GET /v1/providers/{provider_type}/models`: Lists available models for a specific provider. ([`app/api/routes.py:363`](app/api/routes.py:363))
-   `GET /api/v1/agents/`: Lists available agents, used for the agent management functionality. ([`app/api/agent_routes.py:155`](app/api/agent_routes.py:155))

### Implementation Example (FastAPI)

For a clean and scalable implementation, we recommend creating a new router file for your UI endpoints. This separates UI concerns from your core API logic.

**1. Create `app/api/ui_routes.py`**

Create a new file with the following content. This code includes the two required endpoints and uses Pydantic for data validation.

```python
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
```

**2. Include the New Router in `app/main.py`**

To make these endpoints available, you must import and include the new router in your main application file.

```python
# In app/main.py

# ... (other imports)
from app.api import ui_routes  # Add this import

# ... (app creation)
app.include_router(ui_routes.router) # Add this line to include the UI routes

# ... (rest of your main.py file)
```

**Note on Session Management:** The `UI_STATE_STORE` in the example above is an in-memory dictionary for demonstration purposes only. It will not work in a multi-process environment and will reset every time the server restarts. You must replace it with a persistent session management solution like a database, Redis, or a dedicated session middleware library.

### Production-Ready Session Management (Hybrid Strategy)

For a robust and scalable application, we recommend a hybrid strategy that leverages both Supabase (PostgreSQL) and Redis for different types of data.

*   **Supabase (PostgreSQL)**: For **persistent user settings**. These are settings a user wants to be the same every time they log in (e.g., preferred model, provider).
*   **Redis**: For **volatile session state**. This is temporary data that only needs to exist for the current session (e.g., `isApiServing` status, current conversation ID).

This approach gives you the durability of a database for long-term preferences and the high-speed performance of an in-memory cache for temporary session data.

#### Step 1: Configure Your FastAPI App

First, ensure your FastAPI application can connect to both services. You'll need to install the necessary clients:

```bash
pip install asyncpg redis
```

Then, in your settings or configuration, you'd have the connection strings (ideally loaded from `.env`):

```python
# In app/core/config.py or similar
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "postgresql+asyncpg://...")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
```

#### Step 2: Set Up Persistent User Settings in Supabase

In your Supabase SQL editor or via a migration, create a table to store user preferences.

**SQL for `user_settings` table:**

```sql
-- This table links to your users table (adjust the foreign key as needed)
CREATE TABLE IF NOT EXISTS user_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL, -- Supabase auth user ID
  selected_provider TEXT,
  selected_model TEXT,
  api_host TEXT DEFAULT 'localhost:8000',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create an index for fast lookups
CREATE INDEX IF NOT EXISTS idx_user_settings_user_id ON user_settings(user_id);
```

#### Step 3: Implement the API Logic with the Hybrid Approach

Now, let's rewrite the `ui_routes.py` to use both Supabase and Redis. This example assumes you have a way to get the authenticated user's ID (e.g., from a JWT token passed by the frontend).

```python
# app/api/ui_routes.py
import asyncpg
import redis.asyncio as redis
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import json
import os

# Assume you have a function to get the current user from a token
# from app.core.auth import get_current_user

router = APIRouter(prefix="/api/ui", tags=["ui"])

# --- Pydantic Models ---
class UIState(BaseModel):
    selectedProvider: Optional[str] = None
    selectedModel: Optional[str] = None
    apiHost: Optional[str] = None
    isApiServing: bool = False

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

# --- Database and Cache Connections ---
# These should be initialized once at startup, not per request
redis_client: redis.Redis = None
db_pool: asyncpg.Pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(os.getenv("SUPABASE_DB_URL"))
    return db_pool

async def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    return redis_client

# --- API Endpoints ---
@router.get("/state", response_model=UIState)
async def get_ui_state(request: Request):
    """
    Provides the current UI configuration by merging persistent (Supabase)
    and volatile (Redis) state.
    """
    # In a real app, get user_id from authentication token
    # user_id = await get_current_user(request)
    user_id = request.cookies.get("session_id", "default_user") # Placeholder

    # 1. Fetch persistent settings from Supabase
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            "SELECT selected_provider, selected_model, api_host FROM user_settings WHERE user_id = $1",
            user_id
        )
    
    persistent_settings = {
        "selectedProvider": record['selected_provider'] if record else None,
        "selectedModel": record['selected_model'] if record else None,
        "apiHost": record['api_host'] if record else 'localhost:8000',
    }

    # 2. Fetch volatile state from Redis
    client = await get_redis_client()
    volatile_state_json = await client.get(f"ui_state:{user_id}")
    volatile_state = json.loads(volatile_state_json) if volatile_state_json else {"isApiServing": False}

    # 3. Merge and return
    final_state = {**persistent_settings, **volatile_state}
    return UIState(**final_state)

@router.post("/select-model")
async def select_model(request: ModelSelectionRequest, http_request: Request):
    """
    Updates the selected model. Persists to Supabase and updates Redis.
    """
    # user_id = await get_current_user(http_request)
    user_id = http_request.cookies.get("session_id", "default_user") # Placeholder

    # 1. Update persistent settings in Supabase
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        # Upsert the user's preference
        await connection.execute(
            """
            INSERT INTO user_settings (user_id, selected_provider, selected_model, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id)
            DO UPDATE SET
                selected_provider = EXCLUDED.selected_provider,
                selected_model = EXCLUDED.selected_model,
                updated_at = NOW();
            """,
            user_id, request.provider, request.model
        )

    # 2. Update volatile state in Redis
    client = await get_redis_client()
    await client.setex(
        f"ui_state:{user_id}",
        3600, # Set TTL to 1 hour (3600 seconds)
        json.dumps({"isApiServing": True})
    )
    
    return {"success": True}
```

#### What About the Chainlit Data Layer?

The Chainlit data layer is great for persisting things that are native to Chainlit, like conversations, message steps, and user feedback. It abstracts away the database connection for you.

However, for our custom UI state (`selectedProvider`, etc.), it's better to interact with the database directly. It gives you more control, is more explicit, and avoids trying to force your data into a schema that might not be a good fit. Think of the Chainlit data layer as for *Chainlit's data*, and your direct database connections as for *your app's data*.

---

## Frontend Integration Example

Here is an example of a parent React component, `AppShell.jsx`, that fetches data and manages the state for the `NavigationHeader` component.

```jsx
// frontend/components/AppShell.jsx
import React, { useState, useEffect } from 'react';
import NavigationHeader from './NavigationHeader';

const AppShell = () => {
  const [uiState, setUiState] = useState({
    selectedProvider: null,
    selectedModel: null,
    apiHost: 'localhost:8000',
    isApiServing: false,
  });
  const [isLoading, setIsLoading] = useState(true);

  // Fetch initial UI state from the backend
  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch('/api/ui/state');
        const state = await response.json();
        setUiState(state);
      } catch (error) {
        console.error("Failed to fetch UI state:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchState();
  }, []);

  // Handle agent button click
  const handleAgentClick = () => {
    // This could open a modal, navigate to a new page, or trigger an action
    alert("Agent management functionality would go here.");
  };

  // Handle model info click (e.g., to open a model selection dialog)
  const handleModelClick = () => {
    alert("Model selection dialog would go here.");
  };

  if (isLoading) {
    return <div>Loading UI...</div>;
  }

  return (
    <div>
      {/* Use the 'top' variant to replace Chainlit's default top bar */}
      {/* <NavigationHeader
        variant="top"
        selectedProvider={uiState.selectedProvider}
        selectedModel={uiState.selectedModel}
        apiHost={uiState.apiHost}
        isApiServing={uiState.isApiServing}
        onAgentClick={handleAgentClick}
        onModelClick={handleModelClick}
      /> */}

      {/* Use the 'status' variant to work alongside Chainlit's default top bar */}
      <NavigationHeader
        variant="status"
        selectedProvider={uiState.selectedProvider}
        selectedModel={uiState.selectedModel}
        apiHost={uiState.apiHost}
        isApiServing={uiState.isApiServing}
        onAgentClick={handleAgentClick}
        onModelClick={handleModelClick}
      />

      {/* Main application content would go here */}
      <main style={{ padding: '20px' }}>
        <h1>Welcome to the AI Assistant</h1>
        <p>The rest of your application UI goes here.</p>
      </main>
    </div>
  );
};

export default AppShell;
```

---

## Summary of Actions Required

1.  **Backend (Python/FastAPI):**
    -   Implement the `GET /api/ui/state` endpoint to provide the current session's UI configuration.
    -   Implement the `POST /api/ui/select-model` endpoint to allow the frontend to update the model selection.

2.  **Frontend (React):**
    -   Set up a React application (e.g., using Vite or Create React App) if you don't have one.
    -   Create a parent component (like the `AppShell.jsx` example) to manage state and data fetching.
    -   Integrate the `NavigationHeader.jsx` component into your application's layout, choosing the appropriate `variant`.
    -   Implement the logic for the `onAgentClick` and `onModelClick` callbacks as needed by your application.

By following these steps, you will successfully migrate your UI components from a server-side rendering model to a modern, interactive client-side rendering architecture using React, with a simplified and consolidated codebase.