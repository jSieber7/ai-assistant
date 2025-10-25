"""
Mock API responses for testing AI Assistant application.

Contains common mock responses for external services and APIs.
"""

import time
from typing import Dict, Any, List
from unittest.mock import Mock


class MockLLMResponse:
    """Mock LLM response for testing"""
    
    @staticmethod
    def chat_completion(content: str, model: str = "test-model") -> Dict[str, Any]:
        """Create a mock chat completion response"""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
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
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": len(content.split()) * 2,
                "total_tokens": 20 + len(content.split()) * 2
            }
        }
    
    @staticmethod
    def streaming_chunk(content: str, model: str = "test-model", is_final: bool = False) -> Dict[str, Any]:
        """Create a mock streaming response chunk"""
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content
                    }
                }
            ]
        }
        
        if is_final:
            chunk["choices"][0]["finish_reason"] = "stop"
        
        return chunk


class MockToolResponse:
    """Mock tool response for testing"""
    
    @staticmethod
    def success(tool_name: str, result: Any, execution_time: float = 0.5) -> Dict[str, Any]:
        """Create a mock successful tool response"""
        return {
            "success": True,
            "tool_name": tool_name,
            "result": result,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
    
    @staticmethod
    def error(tool_name: str, error_message: str, execution_time: float = 0.5) -> Dict[str, Any]:
        """Create a mock error tool response"""
        return {
            "success": False,
            "tool_name": tool_name,
            "error": error_message,
            "execution_time": execution_time,
            "timestamp": time.time()
        }


class MockAgentResponse:
    """Mock agent response for testing"""
    
    @staticmethod
    def success(agent_name: str, result: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create a mock successful agent response"""
        if messages is None:
            messages = [
                {"role": "user", "content": "Test input"},
                {"role": "assistant", "content": result}
            ]
        
        return {
            "success": True,
            "agent_name": agent_name,
            "result": result,
            "messages": messages,
            "execution_time": 1.5,
            "timestamp": time.time()
        }
    
    @staticmethod
    def error(agent_name: str, error_message: str, execution_time: float = 1.5) -> Dict[str, Any]:
        """Create a mock error agent response"""
        return {
            "success": False,
            "agent_name": agent_name,
            "error": error_message,
            "execution_time": execution_time,
            "timestamp": time.time()
        }


class MockAPIResponse:
    """Mock API response for testing"""
    
    @staticmethod
    def success(data: Any, status_code: int = 200) -> Mock:
        """Create a mock successful API response"""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        response.headers = {"content-type": "application/json"}
        response.raise_for_status.return_value = None
        return response
    
    @staticmethod
    def error(error_message: str, status_code: int = 400, error_type: str = "error") -> Mock:
        """Create a mock error API response"""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = {
            "error": {
                "message": error_message,
                "type": error_type
            }
        }
        response.headers = {"content-type": "application/json"}
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return response


class MockDatabaseResponse:
    """Mock database response for testing"""
    
    @staticmethod
    def query_result(data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create a mock query result"""
        return data or []
    
    @staticmethod
    def insert_result(record_id: int = 1) -> Dict[str, Any]:
        """Create a mock insert result"""
        return {"id": record_id}
    
    @staticmethod
    def update_result(rows_affected: int = 1) -> Dict[str, Any]:
        """Create a mock update result"""
        return {"rows_affected": rows_affected}
    
    @staticmethod
    def delete_result(rows_affected: int = 1) -> Dict[str, Any]:
        """Create a mock delete result"""
        return {"rows_affected": rows_affected}


class MockCacheResponse:
    """Mock cache response for testing"""
    
    @staticmethod
    def hit(value: Any) -> Any:
        """Create a mock cache hit"""
        return value
    
    @staticmethod
    def miss() -> None:
        """Create a mock cache miss"""
        return None
    
    @staticmethod
    def success() -> bool:
        """Create a mock successful cache operation"""
        return True
    
    @staticmethod
    def failure() -> bool:
        """Create a mock failed cache operation"""
        return False


class MockProviderResponse:
    """Mock provider response for testing"""
    
    @staticmethod
    def models_list(provider_type: str, models: List[str] = None) -> List[Dict[str, Any]]:
        """Create a mock models list response"""
        if models is None:
            models = {
                "openrouter": ["deepseek/deepseek-v3.1-terminus", "anthropic/claude-3-opus"],
                "openai": ["gpt-4", "gpt-3.5-turbo"],
                "ollama": ["llama2", "codellama"]
            }.get(provider_type, ["test-model"])
        
        return [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": provider_type
            }
            for model in models
        ]
    
    @staticmethod
    def health_check(status: bool = True) -> Dict[str, Any]:
        """Create a mock health check response"""
        return {
            "status": "healthy" if status else "unhealthy",
            "timestamp": time.time(),
            "response_time": 0.1 if status else 5.0
        }


class MockVisualLLMResponse:
    """Mock visual LLM response for testing"""
    
    @staticmethod
    def image_analysis(description: str, confidence: float = 0.95) -> Dict[str, Any]:
        """Create a mock image analysis response"""
        return {
            "description": description,
            "confidence": confidence,
            "objects": ["object1", "object2"],
            "colors": ["red", "blue", "green"],
            "timestamp": time.time()
        }
    
    @staticmethod
    def ocr_result(text: str, confidence: float = 0.87) -> Dict[str, Any]:
        """Create a mock OCR result"""
        return {
            "text": text,
            "confidence": confidence,
            "bounding_boxes": [
                {
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 20,
                    "text": "extracted"
                }
            ],
            "timestamp": time.time()
        }


class MockMonitoringResponse:
    """Mock monitoring response for testing"""
    
    @staticmethod
    def metrics(component_type: str, metrics_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create mock metrics response"""
        if metrics_data is None:
            metrics_data = {
                "request_count": 100,
                "success_rate": 0.95,
                "avg_duration": 1.5,
                "error_count": 5
            }
        
        return {
            "component_type": component_type,
            "component_name": f"test_{component_type}",
            "metrics": metrics_data,
            "timestamp": time.time()
        }
    
    @staticmethod
    def health_check(components: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create mock health check response"""
        if components is None:
            components = {
                "api": {"status": "healthy", "response_time": 0.1},
                "database": {"status": "healthy", "connection_pool": "8/10"},
                "cache": {"status": "healthy", "hit_rate": 0.85},
                "llm_providers": {"status": "healthy", "available": 3}
            }
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": components
        }


class MockLangChainResponse:
    """Mock LangChain response for testing"""
    
    @staticmethod
    def integration_health(components: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create mock LangChain integration health response"""
        if components is None:
            components = {
                "llm_manager": {"status": "healthy", "models_loaded": 5},
                "tool_registry": {"status": "healthy", "tools_registered": 12},
                "agent_manager": {"status": "healthy", "agents_loaded": 3},
                "memory_manager": {"status": "healthy", "conversations_active": 8},
                "monitoring": {"status": "healthy", "metrics_collected": True}
            }
        
        return {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": components
        }
    
    @staticmethod
    def agent_execution(result: str, tool_calls: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create mock LangChain agent execution response"""
        messages = [
            {"role": "user", "content": "Test input"},
            {"role": "assistant", "content": result}
        ]
        
        if tool_calls:
            for tool_call in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"Tool {tool_call['name']} executed successfully"
                })
        
        return {
            "result": result,
            "messages": messages,
            "tool_calls": tool_calls or [],
            "execution_time": 2.0,
            "timestamp": time.time()
        }


class MockMultiWriterResponse:
    """Mock multi-writer response for testing"""
    
    @staticmethod
    def workflow_creation(workflow_id: str, status: str = "created") -> Dict[str, Any]:
        """Create mock workflow creation response"""
        return {
            "workflow_id": workflow_id,
            "status": status,
            "created_at": time.time(),
            "estimated_completion": time.time() + 300  # 5 minutes
        }
    
    @staticmethod
    def workflow_status(workflow_id: str, status: str, progress: float = 0.5) -> Dict[str, Any]:
        """Create mock workflow status response"""
        return {
            "workflow_id": workflow_id,
            "status": status,
            "progress": progress,
            "updated_at": time.time(),
            "current_step": "generating_content" if progress < 0.8 else "checking_content"
        }
    
    @staticmethod
    def workflow_completion(workflow_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock workflow completion response"""
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "content": content,
            "check_results": [
                {
                    "checker": "grammar_checker",
                    "status": "passed",
                    "score": 95,
                    "issues": []
                }
            ],
            "execution_time": 15.2,
            "completed_at": time.time()
        }


# Factory for creating mock responses
class MockResponseFactory:
    """Factory for creating various mock responses"""
    
    @staticmethod
    def create_llm_response(content: str, model: str = "test-model") -> Dict[str, Any]:
        """Create a mock LLM response"""
        return MockLLMResponse.chat_completion(content, model)
    
    @staticmethod
    def create_tool_response(tool_name: str, result: Any, success: bool = True) -> Dict[str, Any]:
        """Create a mock tool response"""
        if success:
            return MockToolResponse.success(tool_name, result)
        else:
            return MockToolResponse.error(tool_name, str(result))
    
    @staticmethod
    def create_agent_response(agent_name: str, result: str, success: bool = True) -> Dict[str, Any]:
        """Create a mock agent response"""
        if success:
            return MockAgentResponse.success(agent_name, result)
        else:
            return MockAgentResponse.error(agent_name, result)
    
    @staticmethod
    def create_api_response(data: Any, success: bool = True, status_code: int = 200) -> Mock:
        """Create a mock API response"""
        if success:
            return MockAPIResponse.success(data, status_code)
        else:
            return MockAPIResponse.error(str(data), status_code)
    
    @staticmethod
    def create_health_response(status: bool = True, components: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a mock health check response"""
        return MockMonitoringResponse.health_check(components)