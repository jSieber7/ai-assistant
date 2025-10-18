# File Structure Reorganization

This document outlines the comprehensive reorganization of the AI Assistant project's file structure to improve maintainability, clarity, and developer experience.

## Overview

The reorganization focuses on:
1. Categorizing agents and tools by functionality
2. Centralizing configuration files
3. Restructuring tests to mirror the application structure
4. Creating a more logical and scalable project layout

## Agents Directory Reorganization

The `app/core/agents/` directory has been reorganized into the following subdirectories:

### Base/Foundation
- `app/core/agents/base/` - Contains base agent classes and interfaces
  - `base.py` - Base agent classes (BaseAgent, AgentResult, etc.)

### Content Creation
- `app/core/agents/content/` - Contains agents for content generation
  - `writer_agent.py` - AI writer agent for different content types
  - `multi_content_orchestrator.py` - Orchestrates multiple content creation workflows
  - `content_processor.py` - Processes web content for multi-writer system

### Validation
- `app/core/agents/validation/` - Contains agents for content validation and checking
  - `checker_agent.py` - AI checker agent for content validation
  - `master_checker.py` - Advanced master checker with comprehensive assessment
  - `collaborative_checker.py` - Orchestrates collaborative checking between checkers

### Specialized
- `app/core/agents/specialized/` - Contains specialized tool agents
  - `tool_agent.py` - Agent that can call tools
  - `firecrawl_agent.py` - Agent specialized in web scraping using Firecrawl
  - `deep_search_agent.py` - Agent for deep search and RAG operations

### Collaboration
- `app/core/agents/collaboration/` - Contains agents for collaboration and coordination
  - `debate_system.py` - Orchestrates structured debates between agents
  - `dynamic_selector.py` - Intelligently selects agents based on task analysis

### Enhancement
- `app/core/agents/enhancement/` - Contains agents for learning and personality
  - `learning_system.py` - Learning and adaptation system
  - `personality_system.py` - Manages personality profiles for agents
  - `context_sharing.py` - Enhanced context sharing system

### Integration
- `app/core/agents/integration/` - Contains integration with external frameworks
  - `langchain_integration.py` - Integration with LangChain ecosystem

### Management
- `app/core/agents/management/` - Contains agent management and registry
  - `registry.py` - Central registry for managing agents

### Utilities
- `app/core/agents/utilities/` - Contains utility classes and strategies
  - `strategies.py` - Tool selection strategies

## Tools Directory Reorganization

The `app/core/tools/` directory has been reorganized into the following subdirectories:

### Base/Foundation
- `app/core/tools/base/` - Contains base tool classes and configuration
  - `base.py` - Base tool classes (BaseTool, ToolResult, etc.)
  - `config.py` - Tool system configuration

### Web
- `app/core/tools/web/` - Contains web-related tools
  - `firecrawl_tool.py` - Web scraping using Firecrawl
  - `playwright_tool.py` - Browser automation using Playwright
  - `searxng_tool.py` - Web search using SearXNG
  - `standalone_scraper.py` - Standalone web scraping functionality

### Content
- `app/core/tools/content/` - Contains content processing tools
  - `jina_reranker_tool.py` - Document reranking using Jina AI

### Utilities
- `app/core/tools/utilities/` - Contains utility and example tools
  - `examples.py` - Example tools (Calculator, Time, Echo)

### Integration
- `app/core/tools/integration/` - Contains integration with external frameworks
  - `langchain_integration.py` - Integration with LangChain ecosystem

### Execution
- `app/core/tools/execution/` - Contains execution and coordination tools
  - `dynamic_executor.py` - Dynamic tool execution
  - `registry.py` - Central registry for managing tools

## Configuration Centralization

Configuration files have been centralized in a new `config/` directory:

- `config/docker/` - Docker-related configurations
  - `middlewares.yml` - Docker middleware configurations
  - `traefik-prod.yml` - Production Traefik configuration
  - `traefik.yml` - Development Traefik configuration

- `config/monitoring/` - Monitoring configurations
  - `prometheus.yml` - Prometheus configuration

- `config/searxng/` - SearXNG configurations
  - `limiter.toml` - Rate limiting configuration
  - `plugins.yml` - Plugin configuration
  - `settings.yml` - SearXNG settings

- `config/jina-reranker/` - Jina Reranker configurations
  - `app.py` - Reranker application
  - `config.yml` - Reranker configuration
  - `Dockerfile` - Reranker Dockerfile
  - `requirements.txt` - Reranker dependencies

- `config/mongodb/` - MongoDB configurations
  - `init-mongo.js` - MongoDB initialization script

## Test Structure Reorganization

Tests have been reorganized to mirror the application structure while maintaining the unit/integration/system separation:

### Unit Tests
- `tests/unit/app/core/agents/` - Unit tests for agent components
- `tests/unit/app/core/tools/` - Unit tests for tool components
- `tests/unit/app/core/caching/` - Unit tests for caching components
- `tests/unit/app/core/monitoring/` - Unit tests for monitoring components
- `tests/unit/app/core/services/` - Unit tests for service components
- `tests/unit/app/api/` - Unit tests for API endpoints

### Integration Tests
- `tests/integration/app/core/agents/` - Integration tests for agent systems
- `tests/integration/app/core/tools/` - Integration tests for tool systems
- `tests/integration/app/core/services/` - Integration tests for service systems
- `tests/integration/app/` - Integration tests for application components

### System Tests
- `tests/system/app/` - System tests for the entire application

## Import Statement Updates

Due to the reorganization, import statements throughout the codebase need to be updated. Here are some common patterns:

### Agents
Old import:
```python
from app.core.agents.base import BaseAgent
from app.core.agents.tool_agent import ToolAgent
from app.core.agents.registry import AgentRegistry
```

New import:
```python
from app.core.agents.base.base import BaseAgent
from app.core.agents.specialized.tool_agent import ToolAgent
from app.core.agents.management.registry import AgentRegistry
```

### Tools
Old import:
```python
from app.core.tools.base import BaseTool
from app.core.tools.firecrawl_tool import FirecrawlTool
from app.core.tools.registry import ToolRegistry
```

New import:
```python
from app.core.tools.base.base import BaseTool
from app.core.tools.web.firecrawl_tool import FirecrawlTool
from app.core.tools.execution.registry import ToolRegistry
```

## Benefits of Reorganization

1. **Improved Maintainability**: Related code is grouped together, making it easier to find and modify
2. **Better Scalability**: New components can be added to appropriate categories
3. **Clearer Structure**: The purpose of each directory is immediately apparent
4. **Easier Navigation**: Developers can quickly locate relevant code
5. **Reduced Cognitive Load**: Developers don't need to remember where specific files are located

## Migration Checklist

- [ ] Update import statements in all Python files
- [ ] Update documentation references to file paths
- [ ] Update CI/CD pipeline configurations
- [ ] Update Dockerfile references if needed
- [ ] Update any external tool configurations
- [ ] Run tests to ensure everything works correctly
- [ ] Update development documentation

## Notes

- The reorganization maintains backward compatibility at the package level through updated `__init__.py` files
- All existing functionality is preserved, only the file locations have changed
- The test structure now mirrors the application structure for easier navigation
- Configuration files are centralized for better organization and maintenance