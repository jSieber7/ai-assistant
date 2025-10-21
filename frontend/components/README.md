# Frontend Components

This directory contains reusable React components for Chainlit applications.

## Components

### 1. LLM Provider Dropdown Component

A dynamic, nested dropdown menu component for selecting LLM providers and models in Chainlit applications.

#### Features

- **Provider Selection**: Lists all available LLM providers
- **Model Selection**: Nested menu showing models for each provider
- **Search Functionality**: Search models by name or description
- **Add Provider**: Button to add new providers
- **Responsive Design**: Works well on different screen sizes
- **Accessibility**: Proper ARIA attributes and keyboard navigation

#### Files

- `LLMProviderDropdown.jsx` - The main dropdown component
- `../examples/LLMProviderDropdownExample.jsx` - Example implementation with Chainlit integration

#### Usage

##### Basic Usage

```jsx
import React, { useState } from 'react';
import LLMProviderDropdown from './components/LLMProviderDropdown';

const App = () => {
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  
  // Fetch providers from your Chainlit backend
  const [providers, setProviders] = useState([
    {
      id: 'openai_compatible',
      name: 'OpenAI Compatible',
      display_name: 'OpenAI Compatible',
      models: [
        { id: 'gpt-4', name: 'GPT-4', description: 'OpenAI\'s GPT-4 model' },
        // ... more models
      ]
    },
    // ... more providers
  ]);

  const handleModelSelect = (provider, model) => {
    setSelectedProvider(provider.id);
    setSelectedModel(model.id);
    // Send selection to Chainlit backend
  };

  return (
    <LLMProviderDropdown
      providers={providers}
      onModelSelect={handleModelSelect}
      selectedProvider={selectedProvider}
      selectedModel={selectedModel}
    />
  );
};
```

##### Integration with Chainlit

To integrate this component with Chainlit, you'll need to:

1. **Create API Endpoints** in your Chainlit backend:

```python
# In your Chainlit app
from chainlit.server import app
from app.core.llm_providers import provider_registry

@app.get("/api/providers")
async def get_providers():
    """Get all configured providers with their models"""
    providers = []
    for provider in provider_registry.list_configured_providers():
        models = await provider.list_models()
        providers.append({
            "id": provider.provider_type.value,
            "name": provider.name,
            "display_name": provider.name,
            "models": [
                {
                    "id": model.name,
                    "name": model.display_name,
                    "description": model.description,
                    "context_length": model.context_length,
                    "provider": model.provider.value
                }
                for model in models
            ]
        })
    return {"providers": providers}

@app.post("/api/select-model")
async def select_model(data: dict):
    """Update the selected model for the current session"""
    provider_id = data.get("provider")
    model_id = data.get("model")
    # Update session state with selected model
    # ... implementation depends on your Chainlit setup
    return {"success": True}
```

2. **Fetch Data in React**:

```jsx
const fetchProviders = async () => {
  try {
    const response = await fetch('/api/providers');
    const data = await response.json();
    setProviders(data.providers);
  } catch (error) {
    console.error('Failed to fetch providers:', error);
  }
};

useEffect(() => {
  fetchProviders();
}, []);
```

3. **Handle Model Selection**:

```jsx
const handleModelSelect = async (provider, model) => {
  setSelectedProvider(provider.id);
  setSelectedModel(model.id);
  
  try {
    await fetch('/api/select-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: provider.id,
        model: model.id
      })
    });
    console.log('Model selection updated');
  } catch (error) {
    console.error('Failed to update model selection:', error);
  }
};
```

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `providers` | Array | `[]` | Array of provider objects with their models |
| `onProviderSelect` | Function | `null` | Callback when a provider is selected |
| `onModelSelect` | Function | `null` | Callback when a model is selected |
| `onAddProvider` | Function | `null` | Callback when "Add Provider" is clicked |
| `selectedProvider` | String | `null` | ID of currently selected provider |
| `selectedModel` | String | `null` | ID of currently selected model |

#### Data Structure

##### Provider Object

```javascript
{
  id: 'openai_compatible',           // Unique identifier
  name: 'OpenAI Compatible',         // Internal name
  display_name: 'OpenAI Compatible', // Display name
  models: [...]                      // Array of model objects
}
```

##### Model Object

```javascript
{
  id: 'gpt-4',                       // Unique identifier
  name: 'GPT-4',                     // Display name
  provider: 'openai_compatible',     // Provider ID
  description: 'OpenAI\'s GPT-4 model', // Optional description
  context_length: 8192,              // Optional context length
  supports_streaming: true,          // Optional streaming support
  supports_tools: true               // Optional tool support
}
```

### 2. Tools Toggle Dropdown Component

A dynamic dropdown menu component for enabling/disabling tools organized by categories in Chainlit applications.

#### Features

- **Tool Categories**: Lists tools organized by categories
- **Toggle Controls**: Individual toggle switches for each tool
- **Category Toggle**: Enable/disable all tools in a category at once
- **Search Functionality**: Search tools by name, description, or keywords
- **Visual Indicators**: Clear visual feedback for enabled/disabled state
- **Summary Display**: Shows count of enabled tools

#### Files

- `ToolsToggleDropdown.jsx` - The main tools toggle component
- `../examples/ToolsToggleDropdownExample.jsx` - Example implementation with Chainlit integration

#### Usage

##### Basic Usage

```jsx
import React, { useState } from 'react';
import ToolsToggleDropdown from './components/ToolsToggleDropdown';

const App = () => {
  const [enabledTools, setEnabledTools] = useState(new Set(['calculator', 'time']));
  
  // Fetch tools from your Chainlit backend
  const [tools, setTools] = useState([
    {
      id: 'utility',
      name: 'Utility Tools',
      description: 'General purpose utility tools',
      tools: [
        {
          id: 'calculator',
          name: 'Calculator',
          description: 'Perform mathematical calculations',
          keywords: ['calculate', 'math', 'equation'],
          enabled: true
        },
        // ... more tools
      ]
    },
    // ... more categories
  ]);

  const handleToolToggle = (toolId, isEnabled) => {
    const newEnabledTools = new Set(enabledTools);
    if (isEnabled) {
      newEnabledTools.add(toolId);
    } else {
      newEnabledTools.delete(toolId);
    }
    setEnabledTools(newEnabledTools);
    // Send update to Chainlit backend
  };

  const handleCategoryToggle = (categoryId, toolIds, isEnabled) => {
    const newEnabledTools = new Set(enabledTools);
    if (isEnabled) {
      toolIds.forEach(id => newEnabledTools.add(id));
    } else {
      toolIds.forEach(id => newEnabledTools.delete(id));
    }
    setEnabledTools(newEnabledTools);
    // Send update to Chainlit backend
  };

  return (
    <ToolsToggleDropdown
      tools={tools}
      onToolToggle={handleToolToggle}
      onCategoryToggle={handleCategoryToggle}
      enabledTools={enabledTools}
    />
  );
};
```

##### Integration with Chainlit

To integrate this component with Chainlit, you'll need to:

1. **Create API Endpoints** in your Chainlit backend:

```python
# In your Chainlit app
from chainlit.server import app
from app.core.tools.execution.registry import tool_registry

@app.get("/api/tools")
async def get_tools():
    """Get all available tools organized by categories"""
    categories = {}
    for tool in tool_registry.list_tools():
        category = tool.categories[0] if tool.categories else "general"
        if category not in categories:
            categories[category] = {
                "id": category,
                "name": category.title(),
                "description": f"{category.title()} tools",
                "tools": []
            }
        
        categories[category]["tools"].append({
            "id": tool.name,
            "name": tool.name.replace("_", " ").title(),
            "description": tool.description,
            "keywords": tool.keywords,
            "enabled": tool.enabled
        })
    
    return {"tools": list(categories.values())}

@app.post("/api/tools/toggle")
async def toggle_tool(data: dict):
    """Toggle a specific tool on/off"""
    tool_id = data.get("toolId")
    enabled = data.get("enabled")
    
    # Update tool state in registry
    if enabled:
        tool_registry.enable_tool(tool_id)
    else:
        tool_registry.disable_tool(tool_id)
    
    return {"success": True}

@app.post("/api/tools/category-toggle")
async def toggle_category(data: dict):
    """Toggle all tools in a category"""
    category_id = data.get("categoryId")
    tool_ids = data.get("toolIds", [])
    enabled = data.get("enabled")
    
    # Update all tools in the category
    for tool_id in tool_ids:
        if enabled:
            tool_registry.enable_tool(tool_id)
        else:
            tool_registry.disable_tool(tool_id)
    
    return {"success": True}
```

2. **Fetch Data in React**:

```jsx
const fetchTools = async () => {
  try {
    const response = await fetch('/api/tools');
    const data = await response.json();
    setTools(data.tools);
    
    // Initialize enabled tools state
    const enabled = new Set();
    data.tools.forEach(category => {
      category.tools.forEach(tool => {
        if (tool.enabled) {
          enabled.add(tool.id);
        }
      });
    });
    setEnabledTools(enabled);
  } catch (error) {
    console.error('Failed to fetch tools:', error);
  }
};

useEffect(() => {
  fetchTools();
}, []);
```

3. **Handle Tool Toggle**:

```jsx
const handleToolToggle = async (toolId, isEnabled) => {
  const newEnabledTools = new Set(enabledTools);
  if (isEnabled) {
    newEnabledTools.add(toolId);
  } else {
    newEnabledTools.delete(toolId);
  }
  setEnabledTools(newEnabledTools);
  
  try {
    await fetch('/api/tools/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        toolId: toolId,
        enabled: isEnabled
      })
    });
    console.log(`Tool ${toolId} ${isEnabled ? 'enabled' : 'disabled'}`);
  } catch (error) {
    console.error('Failed to toggle tool:', error);
  }
};
```

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `tools` | Array | `[]` | Array of category objects with their tools |
| `onToolToggle` | Function | `null` | Callback when a tool is toggled |
| `onCategoryToggle` | Function | `null` | Callback when a category is toggled |
| `enabledTools` | Set | `new Set()` | Set of currently enabled tool IDs |

#### Data Structure

##### Category Object

```javascript
{
  id: 'utility',                     // Unique identifier
  name: 'Utility Tools',             // Display name
  description: 'General purpose...', // Description
  tools: [...]                       // Array of tool objects
}
```

##### Tool Object

```javascript
{
  id: 'calculator',                  // Unique identifier
  name: 'Calculator',                // Display name
  description: 'Perform calculations', // Description
  keywords: ['calculate', 'math'],   // Keywords for search
  enabled: true                      // Current enabled state
}
```

## Styling

Both components use inline styles with CSS-in-JS for easy customization. You can modify the styles directly in the components or override them with CSS classes.

### Custom CSS Example

```css
.llm-dropdown-button {
  background-color: #your-color;
  border-color: #your-border-color;
}

.llm-dropdown-menu {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.tools-toggle-switch {
  /* Custom toggle switch styling */
}
```

## Accessibility

Both components include several accessibility features:

- `aria-expanded` attribute for the dropdown state
- `aria-haspopup="menu"` for the dropdown button
- Keyboard navigation support
- Semantic HTML structure
- Focus management
- Proper ARIA labels for toggle switches

## Browser Support

The components support all modern browsers including:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Dependencies

- React (with hooks support)
- No additional dependencies required

## License

These components are part of the Chainlit integration project.