# Gradio Interface for AI Assistant (DEPRECATED)

⚠️ **DEPRECATED**: The Gradio interface has been removed from the codebase in favor of the Chainlit interface. Please see [Chainlit Interface](chainlit-interface.md) for the current user interface.

This document described the Gradio web interface that previously provided a user-friendly way to configure settings and test queries for the AI Assistant system.

## Overview

**Note**: This interface is no longer available. The following information is kept for historical reference only.

The Gradio interface was a web-based UI that allowed users to:

1. View system information and status
2. Configure application settings
3. Test queries against the AI Assistant with different models and parameters

## Accessing the Interface

The Gradio interface was previously accessible at:

```
http://localhost:8000/gradio
```

But this endpoint is no longer available.

(Replace `localhost:8000` with your actual host and port if different)

## Interface Sections

### 1. System Information Tab

This tab provides an overview of the current system configuration and status:

#### Available Models
- Lists all available models from configured providers
- Shows model names in the format `provider:model` when applicable

#### Provider Status
- Shows the status of each configured LLM provider
- Indicates whether providers are configured and healthy
- Marks the default provider

#### Available Tools
- Lists all registered tools in the system
- Shows tool descriptions and enabled status

#### Refresh Button
- Click to refresh all system information

### 2. Settings Configuration Tab

This tab allows you to configure various application settings:

#### System Configuration
- **Enable Tool System**: Toggle the tool system on/off
- **Enable Agent System**: Toggle the agent system on/off
- **Debug Mode**: Enable/disable debug mode for more verbose logging

#### Provider Configuration
- **Preferred Provider**: Select your preferred LLM provider
  - Options: `openai_compatible`, `ollama`, `auto`
- **Enable Provider Fallback**: Allow falling back to other providers if the preferred one fails

#### Update Settings Button
- Click to apply the new settings
- Note: This is a demo function. In production, you would need to implement persistent configuration storage.

### 3. Query Testing Tab

This tab allows you to test queries against the AI Assistant:

#### Query Input
- Enter your message or question in the text box

#### Model Selection
- Choose which model to use for the query
- Dropdown is populated with available models from the system

#### Parameters
- **Temperature**: Control the randomness of the response (0.0 to 2.0)
- **Max Tokens**: Set the maximum number of tokens in the response (0 for unlimited)
- **Use Agent System**: Toggle whether to use the agent system for processing
- **Agent Name**: Specify which agent to use (optional, leave empty for default)

#### Submit Query Button
- Click to send your query to the AI Assistant
- The response will appear in the output box below

#### Response Output
- Displays the AI Assistant's response
- Shows which agent was used (if applicable)
- Displays tool execution results if tools were used

## Integration with FastAPI

The Gradio interface is integrated with the FastAPI application and uses the same endpoints:

- Query testing uses the `/v1/chat/completions` endpoint
- System information is fetched from the existing API endpoints
- Settings updates would need to be implemented with persistent storage in production

## Technical Details

### File Structure
```
app/
├── ui/
│   ├── __init__.py
│   └── gradio_app.py
└── main.py (updated to include Gradio)
```

### Dependencies
- `gradio>=4.44.0` added to project dependencies

### Mounting
The Gradio app is mounted to the FastAPI application at the `/gradio` path in `app/main.py`:

```python
gradio_app = create_gradio_app()
gradio_app.mount(app, path="/gradio")
```

## Future Enhancements

Potential improvements for the Gradio interface:

1. **Persistent Settings**: Implement actual settings persistence to a configuration file or database
2. **Conversation History**: Add a chat history feature to maintain conversation context
3. **Tool Management**: Add UI for enabling/disabling individual tools
4. **Real-time Monitoring**: Add real-time metrics and monitoring displays
5. **File Upload**: Add support for file uploads in queries
6. **Export/Import**: Allow exporting and importing configurations
7. **Authentication**: Add authentication to secure the interface
8. **Themes**: Allow users to customize the interface appearance

## Troubleshooting

### Common Issues

1. **Gradio Interface Not Loading**
   - Ensure the application is running
   - Check that you're accessing the correct URL: `http://localhost:8000/gradio`
   - Verify the port is not blocked by firewall

2. **Query Testing Fails**
   - Check that at least one LLM provider is configured
   - Verify the API keys are set correctly
   - Check the system information tab for provider status

3. **Settings Not Persisting**
   - This is expected behavior as settings persistence is not implemented
   - Settings will reset when the application restarts

4. **Models Not Showing**
   - Ensure providers are configured correctly
   - Check API keys and network connectivity
   - Try refreshing the system information

### Getting Help

If you encounter issues with the Gradio interface:

1. Check the application logs for error messages
2. Verify your configuration in the `.env` file
3. Ensure all required dependencies are installed
4. Check the FastAPI documentation for API endpoint details