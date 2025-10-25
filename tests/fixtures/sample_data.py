"""
Sample test data for AI Assistant application.

Contains commonly used test data structures and responses.
"""

import time
from typing import Dict, Any, List


# Sample chat messages
SAMPLE_CHAT_MESSAGES = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
    {"role": "user", "content": "What tools are available?"},
    {"role": "assistant", "content": "I have access to various tools including search, calculation, and more."}
]

# Sample chat requests
SAMPLE_CHAT_REQUEST = {
    "messages": [
        {"role": "user", "content": "Hello, can you respond with just the word SUCCESS?"}
    ],
    "model": "deepseek/deepseek-v3.1-terminus",
    "stream": False,
    "temperature": 0.7,
    "max_tokens": 1000
}

SAMPLE_CHAT_REQUEST_WITH_TOOLS = {
    "messages": [
        {"role": "user", "content": "What's the weather like today in New York?"}
    ],
    "model": "deepseek/deepseek-v3.1-terminus",
    "stream": False,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
}

SAMPLE_STREAMING_CHAT_REQUEST = {
    "messages": [
        {"role": "user", "content": "Tell me a story"}
    ],
    "model": "deepseek/deepseek-v3.1-terminus",
    "stream": True,
    "temperature": 0.8,
    "max_tokens": 2000
}

# Sample chat responses
SAMPLE_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": "deepseek/deepseek-v3.1-terminus",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "SUCCESS"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25
    }
}

SAMPLE_STREAMING_RESPONSE_CHUNKS = [
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek/deepseek-v3.1-terminus",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Once"
                }
            }
        ]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek/deepseek-v3.1-terminus",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": " upon"
                }
            }
        ]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek/deepseek-v3.1-terminus",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": " a"
                }
            }
        ]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek/deepseek-v3.1-terminus",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": " time"
                }
            }
        ]
    },
    {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek/deepseek-v3.1-terminus",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
]

# Sample models
SAMPLE_MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": "deepseek/deepseek-v3.1-terminus",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "deepseek"
        },
        {
            "id": "openai/gpt-4",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        },
        {
            "id": "ollama/llama2",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama"
        }
    ]
}

# Sample agents
SAMPLE_AGENTS = [
    {
        "name": "research_agent",
        "description": "Agent for research tasks",
        "type": "workflow",
        "enabled": True,
        "tools": ["search", "scrape", "analyze"]
    },
    {
        "name": "calculation_agent",
        "description": "Agent for mathematical calculations",
        "type": "react",
        "enabled": True,
        "tools": ["calculator", "converter"]
    },
    {
        "name": "creative_agent",
        "description": "Agent for creative tasks",
        "type": "workflow",
        "enabled": False,
        "tools": ["generate", "format"]
    }
]

# Sample tools
SAMPLE_TOOLS = [
    {
        "name": "search_tool",
        "description": "Search the web for information",
        "category": "search",
        "enabled": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator_tool",
        "description": "Perform mathematical calculations",
        "category": "utility",
        "enabled": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "weather_tool",
        "description": "Get weather information",
        "category": "external",
        "enabled": False,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location to get weather for"
                }
            },
            "required": ["location"]
        }
    }
]

# Sample conversations
SAMPLE_CONVERSATIONS = [
    {
        "conversation_id": "conv_123",
        "title": "Weather Inquiry",
        "agent_name": "research_agent",
        "created_at": "2023-01-01T10:00:00",
        "updated_at": "2023-01-01T10:05:00",
        "message_count": 4,
        "messages": [
            {"role": "user", "content": "What's the weather like?", "timestamp": "2023-01-01T10:00:00"},
            {"role": "assistant", "content": "I'll check the weather for you.", "timestamp": "2023-01-01T10:01:00"},
            {"role": "tool", "content": "Weather in New York: 72°F, sunny", "timestamp": "2023-01-01T10:02:00"},
            {"role": "assistant", "content": "The weather in New York is 72°F and sunny.", "timestamp": "2023-01-01T10:03:00"}
        ]
    },
    {
        "conversation_id": "conv_456",
        "title": "Math Problem",
        "agent_name": "calculation_agent",
        "created_at": "2023-01-01T11:00:00",
        "updated_at": "2023-01-01T11:02:00",
        "message_count": 2,
        "messages": [
            {"role": "user", "content": "Calculate 15 + 27", "timestamp": "2023-01-01T11:00:00"},
            {"role": "assistant", "content": "15 + 27 = 42", "timestamp": "2023-01-01T11:01:00"}
        ]
    }
]

# Sample metrics
SAMPLE_METRICS = {
    "llm": {
        "request_count": 100,
        "success_rate": 0.95,
        "avg_duration": 1.5,
        "error_count": 5,
        "total_tokens": 50000
    },
    "tools": {
        "execution_count": 50,
        "success_rate": 0.90,
        "avg_duration": 0.8,
        "error_count": 5,
        "most_used": "search_tool"
    },
    "agents": {
        "active_agents": 3,
        "total_conversations": 25,
        "avg_response_time": 2.1,
        "most_active": "research_agent"
    },
    "system": {
        "uptime": 99.9,
        "memory_usage": 65.2,
        "cpu_usage": 45.8,
        "disk_usage": 78.3
    }
}

# Sample health checks
SAMPLE_HEALTH_CHECK = {
    "status": "healthy",
    "timestamp": "2023-01-01T12:00:00",
    "version": "1.0.0",
    "components": {
        "api": {"status": "healthy", "response_time": 0.1},
        "database": {"status": "healthy", "connection_pool": "8/10"},
        "cache": {"status": "healthy", "hit_rate": 0.85},
        "llm_providers": {"status": "healthy", "available": 3},
        "agents": {"status": "healthy", "active": 2}
    }
}

SAMPLE_LANGCHAIN_HEALTH_CHECK = {
    "overall_status": "healthy",
    "timestamp": "2023-01-01T12:00:00",
    "components": {
        "llm_manager": {"status": "healthy", "models_loaded": 5},
        "tool_registry": {"status": "healthy", "tools_registered": 12},
        "agent_manager": {"status": "healthy", "agents_loaded": 3},
        "memory_manager": {"status": "healthy", "conversations_active": 8},
        "monitoring": {"status": "healthy", "metrics_collected": True}
    }
}

# Sample configurations
SAMPLE_CONFIG = {
    "environment": "testing",
    "debug": True,
    "host": "127.0.0.1",
    "port": 8000,
    "database": {
        "url": "postgresql://test:test@localhost/test_db",
        "pool_size": 10,
        "max_overflow": 20
    },
    "redis": {
        "url": "redis://localhost:6379/0",
        "max_connections": 10
    },
    "llm_providers": {
        "openrouter": {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1"
        },
        "ollama": {
            "enabled": True,
            "base_url": "http://localhost:11434"
        }
    },
    "langchain": {
        "integration_mode": "langchain",
        "llm_manager_enabled": True,
        "tool_registry_enabled": True,
        "agent_manager_enabled": True,
        "memory_workflow_enabled": True
    }
}

# Sample error responses
SAMPLE_ERROR_RESPONSES = {
    "invalid_request": {
        "error": {
            "message": "Invalid request format",
            "type": "invalid_request_error",
            "code": "invalid_request"
        },
        "status_code": 400
    },
    "authentication_error": {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        },
        "status_code": 401
    },
    "rate_limit_error": {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        },
        "status_code": 429
    },
    "server_error": {
        "error": {
            "message": "Internal server error",
            "type": "server_error",
            "code": "internal_error"
        },
        "status_code": 500
    }
}

# Sample image data
SAMPLE_IMAGE_DATA = {
    "png_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "jpeg_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A",
    "image_url": "https://example.com/image.jpg",
    "image_description": "A simple test image"
}

# Sample visual LLM responses
SAMPLE_VISUAL_RESPONSES = {
    "image_analysis": {
        "description": "The image shows a simple red square on a white background",
        "objects": ["square"],
        "colors": ["red", "white"],
        "confidence": 0.95
    },
    "ocr_result": {
        "text": "Hello World",
        "confidence": 0.87,
        "bounding_boxes": [
            {"x": 10, "y": 10, "width": 100, "height": 20, "text": "Hello"},
            {"x": 10, "y": 40, "width": 80, "height": 20, "text": "World"}
        ]
    }
}

# Sample multi-writer data
SAMPLE_MULTI_WRITER_REQUEST = {
    "prompt": "Write a blog post about artificial intelligence",
    "writers": ["technical_writer", "creative_writer"],
    "checkers": ["grammar_checker", "fact_checker"],
    "format": "markdown",
    "tone": "professional",
    "length": "medium"
}

SAMPLE_MULTI_WRITER_RESPONSE = {
    "workflow_id": "workflow_123",
    "status": "completed",
    "content": {
        "title": "The Future of Artificial Intelligence",
        "body": "# The Future of Artificial Intelligence\n\nAI is transforming...",
        "format": "markdown"
    },
    "check_results": [
        {
            "checker": "grammar_checker",
            "status": "passed",
            "score": 95,
            "issues": []
        },
        {
            "checker": "fact_checker",
            "status": "passed",
            "score": 88,
            "issues": []
        }
    ],
    "execution_time": 15.2
}