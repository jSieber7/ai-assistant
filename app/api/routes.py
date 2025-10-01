from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import asyncio
from ..core.config import get_llm

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


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[dict]


@router.get("/v1/models")
async def list_models():
    """OpenWebUI compatible models endpoint"""
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
    """OpenWebUI compatible chat endpoint"""

    try:
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
                id="chatcmpl-" + "123456789",  # Generate proper ID later
                model=request.model or "langchain-agent-hub",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.content},
                        "finish_reason": "stop",
                    }
                ],
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(messages, llm, model_name):
    """Handle streaming responses for OpenWebUI"""
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            # For now, simulate streaming with the non-streaming response
            # We'll implement proper streaming in Phase 2
            response = await llm.ainvoke(messages)

            # Split response into chunks for streaming simulation
            words = response.content.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": "chatcmpl-123456789",
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
                "id": "chatcmpl-123456789",
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
