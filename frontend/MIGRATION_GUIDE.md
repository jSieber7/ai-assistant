# Migration Guide: Python SSR to React CSR Components

This guide outlines the migration of Python-based Server-side Rendering (SSR) UI components to React/JSX Client-side Rendering (CSR) components for your Chainlit application.

## Overview of Changes

The original Python components (`app/ui/top_bar_component.py` and `app/ui/status_bar_component.py`) rendered HTML strings on the server and sent them to the client. This has been replaced with two reusable React components:

1.  **`TopBar.jsx`**: A sticky top navigation bar.
2.  **`StatusBar.jsx`**: A non-sticky status bar designed to appear below Chainlit's default top bar.

These components are now self-contained, receive data via props, and signal events via callback props, which is the standard React pattern.

## React Components

### `TopBar.jsx`

Located at [`frontend/components/TopBar.jsx`](frontend/components/TopBar.jsx).

**Props:**
-   `selectedProvider` (string, optional): The currently selected provider name.
-   `selectedModel` (string, optional): The currently selected model name.
-   `outputApiEndpoint` (string, optional): The full URL for the OpenAI-compatible API.
-   `isApiServing` (boolean): `true` if the API is active, `false` otherwise.
-   `onAgentClick` (function, optional): Callback for the "Agents" button click.
-   `onModelClick` (function, optional): Callback for when the model info area is clicked.

### `StatusBar.jsx`

Located at [`frontend/components/StatusBar.jsx`](frontend/components/StatusBar.jsx).

**Props:**
-   `selectedProvider` (string, optional): The currently selected provider name.
-   `selectedModel` (string, optional): The currently selected model name.
-   `apiHost` (string, optional): The host address for the API (e.g., `localhost:8000`).
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

---

## Frontend Integration Example

Here is an example of a parent React component, `AppShell.jsx`, that fetches data and manages the state for the `TopBar` and `StatusBar` components.

```jsx
// frontend/components/AppShell.jsx
import React, { useState, useEffect } from 'react';
import TopBar from './TopBar';
import StatusBar from './StatusBar';

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
      {/* The TopBar can be used if you want to replace Chainlit's default top bar */}
      {/* <TopBar
        selectedProvider={uiState.selectedProvider}
        selectedModel={uiState.selectedModel}
        outputApiEndpoint={`http://${uiState.apiHost}/v1`}
        isApiServing={uiState.isApiServing}
        onAgentClick={handleAgentClick}
        onModelClick={handleModelClick}
      /> */}

      {/* The StatusBar is designed to work alongside Chainlit's default top bar */}
      <StatusBar
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
    -   Integrate the `TopBar.jsx` and/or `StatusBar.jsx` components into your application's layout.
    -   Implement the logic for the `onAgentClick` and `onModelClick` callbacks as needed by your application.

By following these steps, you will successfully migrate your UI components from a server-side rendering model to a modern, interactive client-side rendering architecture using React.