from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import asyncio
import uuid
from ..core.config import get_llm, settings
from ..core.agents.registry import agent_registry

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
            llm = get_llm(request.model)

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(messages, llm, model_name):
    """Handle streaming responses for OpenAI Compatible API"""
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            # For now, simulate streaming with the non-streaming response
            # We'll implement proper streaming in Phase 2
            response = await llm.ainvoke(messages)

            # Generate a unique ID for this streaming session
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
            error_chunk = {"error": {"message": str(e), "type": "error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "langchain-agent-hub"}
