from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import asyncio
import uuid

# Import directly from the config module to avoid circular imports
from app.core.config import get_llm, settings
from ..core.agents.management.registry import agent_registry

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    agent_name: Optional[str] = None  # Specify which agent to use
    conversation_id: Optional[str] = None  # For conversation context
    context: Optional[Dict[str, Any]] = None  # Additional context


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[dict]
    agent_name: Optional[str] = None  # Indicate which agent was used
    tool_results: Optional[List[dict]] = None  # If tools were used


@router.get("/v1/models")
async def list_models():
    """OpenAI Compatible API compatible models endpoint"""
    from app.core.config import get_available_models

    try:
        models = get_available_models()

        # Convert to OpenAI-compatible format
        model_data = []
        for model in models:
            model_data.append(
                {
                    "id": (
                        f"{model.provider.value}:{model.name}"
                        if model.provider.value
                        not in ["openrouter", "openai_compatible"]
                        else model.name
                    ),
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": model.provider.value,
                    "permission": [],
                    "root": model.name,
                    "parent": None,
                    "description": model.description,
                    "context_length": model.context_length,
                    "supports_streaming": model.supports_streaming,
                    "supports_tools": model.supports_tools,
                }
            )

        return {
            "object": "list",
            "data": model_data,
        }
    except Exception:
        # Fallback to basic response if model listing fails
        return {
            "object": "list",
            "data": [
                {
                    "id": "langchain-agent-hub",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "langchain-agent-hub",
                    "permission": [],
                    "root": "langchain-agent-hub",
                    "parent": None,
                }
            ],
        }


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI Compatible API compatible chat endpoint with agent system integration"""

    try:
        # Check if agent system should be used
        use_agent_system = settings.agent_system_enabled and (
            request.agent_name is not None or settings.default_agent is not None
        )

        if use_agent_system:
            # Use agent system for processing
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                raise HTTPException(
                    status_code=400, detail="No user messages found in request"
                )

            last_user_message = user_messages[-1].content

            # Process message with agent system
            result = await agent_registry.process_message(
                message=last_user_message,
                agent_name=request.agent_name,
                conversation_id=request.conversation_id,
                context=request.context or {},
            )

            # Format tool results for response
            tool_results = None
            if result.tool_results:
                tool_results = [
                    {
                        "tool_name": tr.tool_name,
                        "success": tr.success,
                        "execution_time": tr.execution_time,
                        "data": tr.data,
                        "error": tr.error,
                        "metadata": tr.metadata,
                    }
                    for tr in result.tool_results
                ]

            # Generate response ID
            response_id = f"chatcmpl-{uuid.uuid4()}"

            return ChatResponse(
                id=response_id,
                model=request.model or "agent-system",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.response},
                        "finish_reason": "stop",
                    }
                ],
                agent_name=result.agent_name,
                tool_results=tool_results,
            )
        else:
            # Use direct LLM approach (original behavior)
            # Convert messages to LangChain format
            langchain_messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = []
            for msg in request.messages:
                if msg.role == "user":
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    langchain_messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    langchain_messages.append(SystemMessage(content=msg.content))

            # Get LLM and generate response
            llm = await get_llm(
                request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                streaming=request.stream,
            )

            if request.stream:
                return await _stream_response(
                    langchain_messages, llm, request.model or "langchain-agent-hub"
                )
            else:
                response = await llm.ainvoke(langchain_messages)

                return ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    model=request.model or "langchain-agent-hub",
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.content,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except ConnectionError:
        # Handle connection errors
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later.",
        )
    except Exception as e:
        # Log the full error for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error in chat_completions: {str(e)}", exc_info=True)

        # Check if it's an OpenRouter API error and preserve the message
        if "OpenRouter API error" in str(e):
            raise HTTPException(
                status_code=500, detail=f"OpenRouter API error: {str(e)}"
            )

        # Return a generic error message to the user
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )


async def _stream_response(messages, llm, model_name):
    """Handle streaming responses for OpenAI Compatible API"""
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            # Check if LLM supports streaming
            if hasattr(llm, "astream") and getattr(llm, "streaming", False):
                # Use native streaming if available
                stream_id = f"chatcmpl-{uuid.uuid4()}"

                # Get the async iterator
                stream = llm.astream(messages)

                # Check if it's already an async iterator or a coroutine
                if asyncio.iscoroutine(stream):
                    stream = await stream

                async for chunk in stream:
                    if hasattr(chunk, "content") and chunk.content:
                        chunk_data = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                # Final chunk
                final_chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                # Fallback to simulated streaming if LLM doesn't support streaming
                response = await llm.ainvoke(messages)
                stream_id = f"chatcmpl-{uuid.uuid4()}"

                # Split response into chunks for streaming simulation
                words = response.content.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

                # Final chunk
                final_chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            try:
                error_chunk = {"error": {"message": str(e), "type": "error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"
            except Exception:
                # Fallback if JSON serialization fails
                yield f'data: {{"error": {{"message": "{str(e).replace('"', '\\"')}", "type": "error"}}}}\n\n'

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/health")
async def health_check():
    """Simple health check endpoint that always returns healthy regardless of API key configuration"""
    return {
        "status": "healthy",
        "service": "langchain-agent-hub",
        "environment": settings.environment,
        "message": "Application is running (mock responses will be used if no API keys configured)",
        "api_keys_configured": {
            "openai_compatible": bool(settings.openai_settings.api_key),
            "openrouter": bool(settings.openrouter_api_key),
            "jina_reranker": bool(settings.jina_reranker_api_key),
        }
    }


@router.get("/v1/providers")
async def list_providers():
    """List available LLM providers and their status"""
    from ..core.llm_providers import provider_registry

    try:
        providers_data = []
        for provider in provider_registry.list_providers():
            providers_data.append(
                {
                    "name": provider.name,
                    "type": provider.provider_type.value,
                    "configured": provider.is_configured,
                    "healthy": provider.is_healthy(),
                    "default": provider.provider_type
                    == provider_registry._default_provider,
                }
            )

        return {
            "object": "list",
            "data": providers_data,
            "default_provider": (
                provider_registry._default_provider.value
                if provider_registry._default_provider
                else None
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/providers/{provider_type}/models")
async def list_provider_models(provider_type: str):
    """List models for a specific provider"""
    from ..core.llm_providers import provider_registry, ProviderType

    try:
        # Validate provider type
        try:
            provider_enum = ProviderType(provider_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid provider type: {provider_type}"
            )

        provider = provider_registry.get_provider(provider_enum)
        if not provider:
            raise HTTPException(
                status_code=404, detail=f"Provider {provider_type} not found"
            )

        if not provider.is_configured:
            raise HTTPException(
                status_code=400, detail=f"Provider {provider_type} not configured"
            )

        models = await provider.list_models()

        # Convert to OpenAI-compatible format
        model_data = []
        for model in models:
            model_data.append(
                {
                    "id": model.name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": provider_type,
                    "permission": [],
                    "root": model.name,
                    "parent": None,
                    "description": model.description,
                    "context_length": model.context_length,
                    "supports_streaming": model.supports_streaming,
                    "supports_tools": model.supports_tools,
                }
            )

        return {
            "object": "list",
            "data": model_data,
            "provider": provider_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/providers/health-check")
async def health_check_providers():
    """Perform health check on all providers"""
    from ..core.llm_providers import provider_registry

    try:
        await provider_registry.health_check_all()

        providers_data = []
        for provider in provider_registry.list_providers():
            providers_data.append(
                {
                    "name": provider.name,
                    "type": provider.provider_type.value,
                    "configured": provider.is_configured,
                    "healthy": provider.is_healthy(),
                }
            )

        return {
            "status": "completed",
            "providers": providers_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
