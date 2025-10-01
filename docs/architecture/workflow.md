# Agent Workflow

This document describes the workflow and decision-making process of the AI agent, including how it processes requests, decides when to use tools, and generates responses.

## Workflow Overview

The agent follows a structured workflow to handle user requests:

```
User Request
     ↓
Message Processing
     ↓
Intent Analysis
     ↓
Tool Selection (if needed)
     ↓
Tool Execution (if applicable)
     ↓
Context Augmentation
     ↓
Response Generation
     ↓
Response Formatting
     ↓
User Response
```

## Detailed Workflow Steps

### 1. Message Processing

**Input**: User message in OpenAI-compatible format

**Processing**:
```python
# Convert OpenAI messages to LangChain format
def process_messages(messages: List[OpenAIMessage]) -> List[LangChainMessage]:
    langchain_messages = []
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
    return langchain_messages
```

**Output**: LangChain-compatible message sequence

### 2. Intent Analysis

The agent analyzes the user's intent to determine if tools are needed:

**Decision Factors**:
- **Query specificity**: Specific questions often need tools
- **Temporal relevance**: Current events require web search
- **Domain knowledge**: Specialized topics may need RAG
- **Conversation context**: Previous tool usage patterns

**Example Decision Logic**:
```python
def needs_tools(messages: List[LangChainMessage]) -> bool:
    last_message = messages[-1].content.lower()
    
    # Patterns that indicate tool need
    tool_patterns = [
        "current", "recent", "latest",
        "search", "find", "look up",
        "what's new", "update on",
        "how to", "tutorial", "guide"
    ]
    
    return any(pattern in last_message for pattern in tool_patterns)
```

### 3. Tool Selection

If tools are needed, the agent selects the appropriate tool:

**Tool Selection Criteria**:
- **Relevance**: How well the tool matches the query
- **Capability**: What information the tool can provide
- **Performance**: Tool response time and reliability
- **Cost**: Resource usage considerations

**Available Tools**:<br>
- **Web Search (SearX)**: For current information, news, real-time data
- **RAG System**: For document-based knowledge, specific content
- **Calculator**: For mathematical computations (planned)
- **Code Execution**: For code-related queries (planned)

### 4. Tool Execution

**Sequential Execution**:
```python
async def execute_tools(query: str, selected_tools: List[Tool]) -> Dict[str, Any]:
    results = {}
    for tool in selected_tools:
        try:
            results[tool.name] = await tool.execute(query)
        except ToolError as e:
            logger.warning(f"Tool {tool.name} failed: {e}")
            results[tool.name] = None
    return results
```

**Parallel Execution** (for independent tools):
```python
async def execute_tools_parallel(query: str, tools: List[Tool]) -> Dict[str, Any]:
    tasks = {tool.name: tool.execute(query) for tool in tools}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return dict(zip(tasks.keys(), results))
```

### 5. Context Augmentation

The agent combines tool results with the original conversation:

**Context Building**:
```python
def build_augmented_context(original_messages: List[Message], tool_results: Dict) -> str:
    context = "Conversation history:\n"
    for msg in original_messages:
        context += f"{msg.role}: {msg.content}\n"
    
    context += "\nTool results:\n"
    for tool_name, result in tool_results.items():
        if result:
            context += f"{tool_name}: {result}\n"
    
    return context
```

### 6. Response Generation

The LLM generates a response using the augmented context:

**Prompt Construction**:
```python
def build_response_prompt(user_query: str, context: str) -> str:
    return f"""
You are a helpful AI assistant with access to various tools.

Context information:
{context}

User question: {user_query}

Please provide a helpful response based on the available information.
If you used tools, mention the sources appropriately.
"""
```

### 7. Response Formatting

The response is formatted according to OpenAI's specification:

**Formatting**:
```python
def format_openai_response(content: str, model: str) -> OpenAIRresponse:
    return {
        "id": f"chatcmpl-{generate_id()}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
    }
```

## Streaming Workflow

For streaming responses, the workflow is similar but with incremental delivery:

### Streaming Steps
1. **Initial processing**: Same as non-streaming
2. **Tool execution**: Tools run before streaming starts
3. **Incremental generation**: LLM generates response in chunks
4. **Real-time delivery**: Chunks are sent as they're generated

### Streaming Implementation
```python
async def stream_response(messages: List[Message], tools: List[Tool]):
    # Process messages and execute tools
    processed_messages = process_messages(messages)
    tool_results = await execute_tools_if_needed(processed_messages)
    context = build_context(processed_messages, tool_results)
    
    # Stream LLM response
    async for chunk in llm.stream(context):
        yield format_streaming_chunk(chunk)
```

## Error Handling Workflow

### Tool Failure Handling
```python
async def handle_tool_failure(tool_name: str, error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return f"The {tool_name} tool timed out. Please try again."
    elif isinstance(error, ConnectionError):
        return f"The {tool_name} service is currently unavailable."
    else:
        return f"An error occurred with the {tool_name} tool."
```

### Fallback Strategies
1. **Tool-specific fallbacks**: Use alternative tools
2. **Cached results**: Return recently cached data
3. **LLM knowledge**: Rely on the model's training data
4. **Error transparency**: Inform the user about limitations

## Performance Optimization

### Caching Strategy
- **Tool results**: Cache frequent queries for 5 minutes
- **Embeddings**: Cache computed embeddings
- **LLM responses**: Cache identical prompts (with caution)

### Parallel Execution
- Independent tools run concurrently
- Batch processing for multiple queries
- Connection pooling for external APIs

## Monitoring and Logging

### Key Metrics
- **Response time**: End-to-end processing time
- **Tool usage**: Which tools are used and how often
- **Error rates**: Tool failure rates and types
- **User satisfaction**: Implicit feedback from usage patterns

### Logging Structure
```python
{
    "request_id": "unique-id",
    "user_query": "original query",
    "tools_used": ["tool1", "tool2"],
    "tool_results": {"tool1": "summary"},
    "response_time": 2.5,
    "error": null
}
```

## Example Workflow Scenarios

### Scenario 1: Simple Question
```
User: "What is the capital of France?"

Workflow:
1. Message processing → Convert to LangChain format
2. Intent analysis → No tools needed (common knowledge)
3. Response generation → Use LLM knowledge
4. Response formatting → Return answer

Result: "The capital of France is Paris."
```

### Scenario 2: Current Information
```
User: "What are the latest developments in AI?"

Workflow:
1. Message processing → Convert to LangChain format
2. Intent analysis → Tools needed (current information)
3. Tool selection → Web search (SearX)
4. Tool execution → Search for recent AI news
5. Context augmentation → Combine search results with query
6. Response generation → Generate informed response
7. Response formatting → Return with citations

Result: "Based on recent news, the latest developments include..."
```

### Scenario 3: Document-Based Query
```
User: "What does our project documentation say about security?"

Workflow:
1. Message processing → Convert to LangChain format
2. Intent analysis → Tools needed (specific documents)
3. Tool selection → RAG system
4. Tool execution → Search project documentation
5. Context augmentation → Combine relevant document sections
6. Response generation → Generate security overview
7. Response formatting → Return with document references

Result: "According to our documentation, security measures include..."
```

## Customization Points

### Tool Selection Logic
Override the default tool selection algorithm for specific use cases.

### Response Formatting
Customize how responses are formatted for different clients.

### Error Handling
Implement domain-specific error handling strategies.

### Caching Strategy
Adjust caching parameters based on data freshness requirements.

## Related Documentation

- [System Architecture Overview](overview.md)
- [Tool System Design](tools.md)
- [API Endpoints Reference](../api/endpoints.md)
- [Development Setup Guide](../development/setup.md)

This workflow provides a flexible yet structured approach to handling user requests, ensuring that the AI assistant can effectively leverage tools when needed while maintaining fast response times for simple queries.