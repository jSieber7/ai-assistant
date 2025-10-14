# Tool Management API

This section describes the API endpoints for managing tools in the AI Assistant System.

## List Tools

Retrieve a list of all available tools.

```http
GET /api/tools
```

**Response:**
```json
{
  "tools": [
    {
      "name": "search_tool",
      "description": "Search the web using SearXNG",
      "parameters": {
        "query": {
          "type": "string",
          "description": "Search query"
        }
      }
    }
  ]
}
```

## Get Tool Details

Get detailed information about a specific tool.

```http
GET /api/tools/{tool_name}
```

**Parameters:**
- `tool_name`: Name of the tool

**Response:**
```json
{
  "name": "search_tool",
  "description": "Search the web using SearXNG",
  "parameters": {
    "query": {
      "type": "string",
      "description": "Search query",
      "required": true
    }
  },
  "examples": [
    {
      "input": {"query": "Python programming"},
      "output": "Search results for Python programming"
    }
  ]
}
```

## Execute Tool

Execute a tool with provided parameters.

```http
POST /api/tools/{tool_name}/execute
```

**Parameters:**
- `tool_name`: Name of the tool

**Request Body:**
```json
{
  "parameters": {
    "query": "Python programming"
  }
}
```

**Response:**
```json
{
  "result": "Search results for Python programming",
  "execution_time": 1.23,
  "cached": false
}
```

## Register Custom Tool

Register a new custom tool.

```http
POST /api/tools/register
```

**Request Body:**
```json
{
  "name": "custom_tool",
  "description": "Description of the custom tool",
  "code": "def custom_tool(param1): return f'Result: {param1}'",
  "parameters": {
    "param1": {
      "type": "string",
      "description": "First parameter"
    }
  }
}
```

**Response:**
```json
{
  "message": "Tool registered successfully",
  "tool_name": "custom_tool"
}
```

## Update Tool

Update an existing tool.

```http
PUT /api/tools/{tool_name}
```

**Parameters:**
- `tool_name`: Name of the tool

**Request Body:**
```json
{
  "description": "Updated description",
  "code": "def updated_tool(param1): return f'Updated: {param1}'"
}
```

## Delete Tool

Delete a tool from the system.

```http
DELETE /api/tools/{tool_name}
```

**Parameters:**
- `tool_name`: Name of the tool

**Response:**
```json
{
  "message": "Tool deleted successfully"
}
```

## Tool Categories

Tools can be organized into categories for better management.

### List Categories

```http
GET /api/tools/categories
```

### Get Tools by Category

```http
GET /api/tools/categories/{category_name}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Tool not found
- `500 Internal Server Error`: Execution error

**Error Response Example:**
```json
{
  "error": "Tool not found",
  "message": "The specified tool does not exist"
}