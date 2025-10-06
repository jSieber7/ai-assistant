# LLM Tool-System Foundation Plan

## Executive Summary

This document outlines the architecture and implementation plan for building an extensible LLM tool-system foundation using LangChain. The system will transform the current basic LLM API into a powerful tool-calling platform that can be extended with various capabilities like web search, RAG, calculators, and custom tools.

## Current State Analysis

### Strengths
- ✅ FastAPI foundation with OpenAI-compatible API
- ✅ LangChain integration with OpenRouter support
- ✅ Comprehensive testing infrastructure
- ✅ Well-documented architecture plans
- ✅ Configuration management system

### Gaps Identified
- ❌ No tool calling capabilities in current API
- ❌ Missing tool registry and management system
- ❌ No agent system for tool orchestration
- ❌ Basic LLM calls without tool integration
- ❌ No example tools implemented

## Architecture Design

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │◄──►│   Tool Agent     │◄──►│   Tool Registry │
│ (OpenAI-compat) │    │  (Orchestrator)  │    │  (Management)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Request/Resp  │    │   Tool Selection │    │   Tool Storage │
│   Formatting    │    │   & Execution    │    │   & Discovery   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Tool Interface Layer
- **BaseTool Abstract Class**: Standard interface for all tools
- **ToolResult Class**: Standardized tool execution results
- **ToolError Hierarchy**: Comprehensive error handling

#### 2. Tool Registry System
- **ToolRegistry Class**: Central tool management and discovery
- **Tool Registration**: Dynamic tool loading and registration
- **Tool Discovery**: Intelligent tool selection based on context

#### 3. Agent System
- **ToolAgent Class**: Orchestrates LLM-tool interactions
- **Tool Selection**: Context-aware tool recommendation
- **Execution Flow**: Manages tool calling sequence

#### 4. API Integration Layer
- **Enhanced Chat Endpoint**: Tool-enabled chat completions
- **Tool Management API**: Tool registration and configuration
- **Monitoring Endpoints**: Tool usage metrics and health

## Implementation Phases

### Phase 1: Core Foundation (Week 1)
- Implement BaseTool interface and abstract classes
- Create ToolRegistry with registration mechanisms
- Build basic ToolAgent with simple tool calling
- Integrate with existing FastAPI endpoints

### Phase 2: Advanced Features (Week 2)
- Implement sophisticated tool selection strategies
- Add error handling and fallback mechanisms
- Create configuration management for tools
- Implement basic example tools

### Phase 3: Production Ready (Week 3)
- Add monitoring and metrics collection
- Implement performance optimizations (caching, etc.)
- Create comprehensive test suite
- Develop tool development guidelines

## Detailed Component Design

### BaseTool Interface
```python
class BaseTool(ABC):
    """Standard interface for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def description(self) -> str: ...
    
    @property
    def parameters(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...
    
    def should_use(self, query: str) -> bool: ...
```

### ToolRegistry Design
```python
class ToolRegistry:
    """Central tool management system"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: BaseTool, category: str = "general"):
        """Register a tool with optional categorization"""
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
    
    def find_relevant_tools(self, query: str, context: Dict[str, Any]) -> List[BaseTool]:
        """Find tools relevant to query and context"""
```

### ToolAgent Design
```python
class ToolAgent:
    """Orchestrates LLM-tool interactions"""
    
    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.conversation_memory = ConversationMemory()
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> AgentResponse:
        """Process message with tool calling capabilities"""
        
        # 1. Analyze if tools are needed
        tool_calls = await self._select_tools(message, context)
        
        # 2. Execute tools and gather results
        tool_results = await self._execute_tools(tool_calls)
        
        # 3. Generate final response with tool insights
        final_response = await self._generate_response(message, tool_results, context)
        
        return final_response
```

## Integration with Existing API

### Enhanced Chat Completions Endpoint
```python
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Enhanced chat endpoint with tool calling support"""
    
    # Initialize tool agent with current conversation context
    agent = ToolAgent(llm=get_llm(request.model), tool_registry=global_tool_registry)
    
    # Process message with tool capabilities
    response = await agent.process_message(
        message=request.messages[-1].content,
        context={"conversation_history": request.messages[:-1]}
    )
    
    return format_agent_response(response, request.model)
```

## Example Tools to Implement

### 1. Calculator Tool
- Basic mathematical operations
- Unit conversions
- Scientific calculations

### 2. Time and Date Tool
- Current time in different timezones
- Date calculations
- Timezone conversions

### 3. Web Search Tool (SearX Integration)
- Web search capabilities
- News and current events
- Information retrieval

### 4. Knowledge Base Tool (RAG Foundation)
- Document search
- Internal knowledge retrieval
- Citation generation

## Configuration Management

### Environment Variables
```bash
# Tool System Configuration
TOOL_CALLING_ENABLED=true
MAX_TOOLS_PER_QUERY=3
TOOL_TIMEOUT_SECONDS=30
TOOL_CACHE_ENABLED=true
TOOL_CACHE_TTL=300

# Individual Tool Configuration
CALCULATOR_TOOL_ENABLED=true
WEB_SEARCH_TOOL_ENABLED=false  # Enable when SearX is configured
RAG_TOOL_ENABLED=false         # Enable when RAG system is ready
```

## Testing Strategy

### Unit Tests
- Tool interface compliance
- Tool registry functionality
- Agent tool selection logic
- Error handling scenarios

### Integration Tests
- End-to-end tool calling workflows
- API endpoint integration
- Tool configuration management
- Performance testing

### Example Test Cases
```python
async def test_tool_calling_workflow():
    """Test complete tool calling workflow"""
    agent = ToolAgent(llm, tool_registry)
    response = await agent.process_message("What's 15 * 25?")
    assert "375" in response.content
    assert response.tools_used == ["calculator"]
```

## Performance Considerations

### Caching Strategy
- Tool result caching with TTL
- Tool selection caching for similar queries
- LLM response caching when appropriate

### Optimization Techniques
- Parallel tool execution when possible
- Lazy tool loading for infrequently used tools
- Connection pooling for external tool APIs

## Security Considerations

### Input Validation
- Validate all tool parameters
- Sanitize tool inputs and outputs
- Implement rate limiting for tools

### Access Control
- Tool-level permissions
- User-based tool access control
- Audit logging for tool usage

## Monitoring and Metrics

### Key Metrics to Track
- Tool usage frequency and success rates
- Tool execution times and performance
- Error rates and types
- User satisfaction with tool-enhanced responses

### Health Monitoring
- Tool availability checks
- External service connectivity
- Performance degradation detection

## Implementation Timeline

### Week 1: Core Foundation
- Day 1-2: Implement BaseTool and ToolRegistry
- Day 3-4: Build ToolAgent with basic tool calling
- Day 5: Integrate with API and create example tools

### Week 2: Advanced Features
- Day 6-7: Implement sophisticated tool selection
- Day 8-9: Add error handling and configuration
- Day 10: Create comprehensive tests

### Week 3: Production Ready
- Day 11-12: Add monitoring and optimizations
- Day 13-14: Documentation and guidelines
- Day 15: Final testing and deployment

## Success Criteria

### Technical Success
- ✅ Tools can be registered and discovered dynamically
- ✅ Agent successfully selects and executes appropriate tools
- ✅ API maintains OpenAI compatibility while adding tool features
- ✅ System handles tool errors gracefully

### User Experience Success
- ✅ Users get more accurate and comprehensive responses
- ✅ Tool usage is transparent and helpful
- ✅ System performance remains acceptable
- ✅ Error messages are clear and actionable

## Risk Mitigation

### Technical Risks
- **Performance Impact**: Implement caching and optimizations
- **Tool Conflicts**: Design clear tool selection priorities
- **API Compatibility**: Maintain strict OpenAI API compliance

### Operational Risks
- **Tool Reliability**: Implement fallback mechanisms
- **External Dependencies**: Design for service outages
- **Security Vulnerabilities**: Comprehensive input validation

## Next Steps

1. **Review this architecture plan** with stakeholders
2. **Begin Phase 1 implementation** with core tool system
3. **Iterate based on feedback** and testing results
4. **Progress through phases** according to timeline

This foundation will enable the extensible tool system you envisioned, supporting capabilities like SearX, RAG, LLM checkers, stock analysis, and any custom tools you need to develop.