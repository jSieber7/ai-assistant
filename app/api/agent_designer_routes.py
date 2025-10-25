"""
Agent Designer API routes for the AI Assistant.

This module provides FastAPI routes for the LLM Agent Designer feature,
allowing users to create custom agents using LLM assistance.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import logging
import os
import json
import asyncio
from datetime import datetime

from ..core.config import settings
from ..core.agents.management.registry import agent_registry
from ..core.tools.execution.registry import tool_registry
from ..core.secure_settings import secure_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent-designer", tags=["agent-designer"])

# Directory to store custom agents
CUSTOM_AGENTS_DIR = "app/core/agents/custom"

class AgentDesignRequest(BaseModel):
    """Request for creating a new agent"""
    
    name: str = Field(..., description="Name of the agent to create")
    description: str = Field(..., description="Description of what the agent should do")
    requirements: str = Field(..., description="Detailed requirements for the agent")
    category: str = Field(default="custom", description="Category for the agent")
    tools_needed: List[str] = Field(default_factory=list, description="List of tools the agent should use")
    model_preference: Optional[str] = Field(None, description="Preferred LLM model for the agent")
    additional_context: Optional[str] = Field(None, description="Additional context or constraints")

class AgentDesignResponse(BaseModel):
    """Response from agent creation"""
    
    success: bool
    agent_id: str
    agent_name: str
    agent_code: Optional[str] = None
    message: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = {}

class AgentValidationRequest(BaseModel):
    """Request for validating agent code"""
    
    agent_code: str = Field(..., description="The agent code to validate")
    agent_name: str = Field(..., description="Name of the agent")

class AgentValidationResponse(BaseModel):
    """Response from agent validation"""
    
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class SavedAgent(BaseModel):
    """Information about a saved custom agent"""
    
    id: str
    name: str
    description: str
    category: str
    created_at: datetime
    file_path: str
    is_active: bool
    metadata: Dict[str, Any] = {}

def ensure_custom_agents_directory():
    """Ensure the custom agents directory exists"""
    os.makedirs(CUSTOM_AGENTS_DIR, exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(CUSTOM_AGENTS_DIR, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Custom agents created by users"""\n')

async def generate_agent_code(request: AgentDesignRequest) -> str:
    """
    Generate agent code using the LLM based on the request
    
    This function would integrate with the LLM to generate appropriate agent code.
    For now, it returns a template that would be filled by the LLM.
    """
    # Read the agent creation prompt template
    prompt_template_path = "notes/llm-agent-designer/agent_creation_prompt.md"
    
    try:
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt template not found at {prompt_template_path}")
        prompt_template = "Create a Python agent class that inherits from ToolCallingAgent"
    
    # In a real implementation, this would call the LLM with the prompt
    # For now, we'll create a basic template
    agent_code = f'''"""
Custom agent: {request.name}

Generated based on user requirements:
{request.requirements}
"""

from app.core.agents.base.base import ToolCallingAgent, AgentResult
from app.core.tools.base.base import ToolResult
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class {request.name.replace(" ", "").replace("-", "_")}Agent(ToolCallingAgent):
    """Custom agent: {request.description}"""
    
    def __init__(self, tool_registry):
        super().__init__(tool_registry)
        self.agent_config = {{
            "name": "{request.name}",
            "description": "{request.description}",
            "category": "{request.category}",
            "tools_needed": {request.tools_needed},
            "model_preference": "{request.model_preference or 'default'}"
        }}
    
    @property
    def name(self) -> str:
        return "{request.name}"
    
    @property
    def description(self) -> str:
        return "{request.description}"
    
    async def decide_tool_usage(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decide which tools to use based on the message and context
        
        Args:
            message: User message to analyze
            context: Additional context information
            
        Returns:
            Dictionary with tool usage decision
        """
        # TODO: Implement tool decision logic based on requirements
        return {{
            "use_tools": False,
            "tool_calls": [],
            "reasoning": "Tool decision logic to be implemented based on: {request.requirements}"
        }}
    
    async def generate_response(self, message: str, tool_results: List[ToolResult] = None, context: Dict[str, Any] = None) -> str:
        """
        Generate a response based on the message and tool results
        
        Args:
            message: Original user message
            tool_results: Results from tool executions
            context: Additional context information
            
        Returns:
            Generated response text
        """
        # TODO: Implement response generation logic
        return f"Response generation to be implemented for: {{message}}"
    
    async def _process_message_impl(self, message: str, conversation_id: Optional[str] = None, context: Dict[str, Any] = None) -> AgentResult:
        """
        Main message processing implementation
        
        Args:
            message: User message to process
            conversation_id: Optional conversation ID
            context: Additional context information
            
        Returns:
            AgentResult with response and any tool results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Decide if tools are needed
            tool_decision = await self.decide_tool_usage(message, context)
            
            tool_results = []
            if tool_decision.get("use_tools", False):
                for tool_call in tool_decision.get("tool_calls", []):
                    result = await self.execute_tool(
                        tool_call["tool_name"], 
                        **tool_call.get("parameters", {{}})
                    )
                    tool_results.append(result)
            
            # Generate response
            response = await self.generate_response(message, tool_results, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResult(
                success=True,
                response=response,
                tool_results=tool_results,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=conversation_id,
                metadata={{
                    "tool_decision": tool_decision,
                    "agent_config": self.agent_config
                }}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error in {{self.name}} agent: {{str(e)}}")
            
            return AgentResult(
                success=False,
                response=f"An error occurred: {{str(e)}}",
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=conversation_id,
                error=str(e)
            )
'''
    
    return agent_code

async def validate_agent_code(agent_code: str, agent_name: str) -> AgentValidationResponse:
    """
    Validate the generated agent code
    
    Args:
        agent_code: The agent code to validate
        agent_name: Name of the agent
        
    Returns:
        AgentValidationResponse with validation results
    """
    errors = []
    warnings = []
    suggestions = []
    
    try:
        # Basic syntax check
        compile(agent_code, f'<{agent_name}>', 'exec')
        
        # Check for required methods
        required_methods = ['name', 'description', 'decide_tool_usage', 'generate_response', '_process_message_impl']
        for method in required_methods:
            if f'def {method}' not in agent_code:
                errors.append(f"Missing required method: {method}")
        
        # Check for proper inheritance
        if 'ToolCallingAgent' not in agent_code:
            errors.append("Agent must inherit from ToolCallingAgent")
        
        # Check for imports
        if 'from app.core.agents.base.base import' not in agent_code:
            warnings.append("Missing import for base agent classes")
        
        # Suggestions
        if 'TODO:' in agent_code:
            suggestions.append("Agent contains TODO items that should be implemented")
        
        if not errors and not warnings:
            suggestions.append("Agent code looks good! Consider adding comprehensive error handling.")
            
    except SyntaxError as e:
        errors.append(f"Syntax error: {str(e)}")
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
    
    return AgentValidationResponse(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions
    )

async def save_agent_to_file(agent_code: str, agent_name: str, request: AgentDesignRequest) -> str:
    """
    Save the agent code to a file in the custom agents directory
    
    Args:
        agent_code: The agent code to save
        agent_name: Name of the agent
        request: Original request for metadata
        
    Returns:
        Path to the saved file
    """
    ensure_custom_agents_directory()
    
    # Create a safe filename
    safe_name = agent_name.replace(" ", "_").replace("-", "_").lower()
    filename = f"{safe_name}_agent.py"
    file_path = os.path.join(CUSTOM_AGENTS_DIR, filename)
    
    # Add metadata header
    metadata = {
        "name": agent_name,
        "description": request.description,
        "category": request.category,
        "created_at": datetime.now().isoformat(),
        "requirements": request.requirements,
        "tools_needed": request.tools_needed,
        "model_preference": request.model_preference,
        "agent_id": str(uuid.uuid4())
    }
    
    header = f'''"""
Custom Agent: {agent_name}

Metadata:
{json.dumps(metadata, indent=2)}
"""

'''
    
    full_code = header + agent_code
    
    # Write to file
    with open(file_path, 'w') as f:
        f.write(full_code)
    
    return file_path

@router.post("/create", response_model=AgentDesignResponse)
async def create_agent(request: AgentDesignRequest):
    """
    Create a new agent using LLM assistance
    
    This endpoint generates agent code based on the user's requirements,
    validates it, and saves it to the custom agents directory.
    """
    try:
        # Generate agent code
        agent_code = await generate_agent_code(request)
        
        # Validate the generated code
        validation = await validate_agent_code(agent_code, request.name)
        
        if not validation.is_valid:
            return AgentDesignResponse(
                success=False,
                agent_id="",
                agent_name=request.name,
                message=f"Generated code has validation errors: {', '.join(validation.errors)}",
                metadata={"validation": validation.dict()}
            )
        
        # Save the agent to file
        file_path = await save_agent_to_file(agent_code, request.name, request)
        
        # Generate agent ID
        agent_id = str(uuid.uuid4())
        
        return AgentDesignResponse(
            success=True,
            agent_id=agent_id,
            agent_name=request.name,
            agent_code=agent_code,
            message=f"Agent '{request.name}' created successfully",
            file_path=file_path,
            metadata={
                "validation": validation.dict(),
                "category": request.category,
                "tools_needed": request.tools_needed
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate", response_model=AgentValidationResponse)
async def validate_agent(request: AgentValidationRequest):
    """
    Validate agent code without saving it
    
    This endpoint allows users to validate agent code before saving it.
    """
    try:
        validation = await validate_agent_code(request.agent_code, request.agent_name)
        return validation
    except Exception as e:
        logger.error(f"Error validating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/saved-agents", response_model=List[SavedAgent])
async def list_saved_agents():
    """
    List all saved custom agents
    
    Returns a list of all custom agents that have been created and saved.
    """
    try:
        ensure_custom_agents_directory()
        
        agents = []
        for filename in os.listdir(CUSTOM_AGENTS_DIR):
            if filename.endswith('_agent.py') and filename != '__init__.py':
                file_path = os.path.join(CUSTOM_AGENTS_DIR, filename)
                
                try:
                    # Read metadata from file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Extract metadata (basic parsing)
                    metadata = {}
                    if 'Metadata:' in content:
                        metadata_start = content.find('{')
                        metadata_end = content.rfind('}') + 1
                        if metadata_start > 0 and metadata_end > metadata_start:
                            metadata_str = content[metadata_start:metadata_end]
                            try:
                                metadata = json.loads(metadata_str)
                            except json.JSONDecodeError:
                                pass
                    
                    # Get file creation time
                    stat = os.stat(file_path)
                    created_at = datetime.fromtimestamp(stat.st_ctime)
                    
                    agents.append(SavedAgent(
                        id=metadata.get("agent_id", str(uuid.uuid4())),
                        name=metadata.get("name", filename.replace('_agent.py', '')),
                        description=metadata.get("description", "No description available"),
                        category=metadata.get("category", "custom"),
                        created_at=created_at,
                        file_path=file_path,
                        is_active=False,  # TODO: Check if agent is registered
                        metadata=metadata
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error reading agent file {filename}: {str(e)}")
                    continue
        
        return sorted(agents, key=lambda x: x.created_at, reverse=True)
        
    except Exception as e:
        logger.error(f"Error listing saved agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/saved-agents/{agent_id}")
async def get_saved_agent(agent_id: str):
    """
    Get details of a specific saved agent
    
    Returns the full code and metadata for a specific custom agent.
    """
    try:
        ensure_custom_agents_directory()
        
        # Find the agent file
        agent_file = None
        for filename in os.listdir(CUSTOM_AGENTS_DIR):
            if filename.endswith('_agent.py') and filename != '__init__.py':
                file_path = os.path.join(CUSTOM_AGENTS_DIR, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check if this is the right agent by checking metadata
                    if f'"{agent_id}"' in content:
                        agent_file = file_path
                        break
                        
                except Exception:
                    continue
        
        if not agent_file:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        # Read and return the agent code
        with open(agent_file, 'r') as f:
            agent_code = f.read()
        
        return {
            "agent_id": agent_id,
            "agent_code": agent_code,
            "file_path": agent_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting saved agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/saved-agents/{agent_id}")
async def delete_saved_agent(agent_id: str):
    """
    Delete a saved custom agent
    
    Removes the agent file from the custom agents directory.
    """
    try:
        ensure_custom_agents_directory()
        
        # Find and delete the agent file
        agent_file = None
        for filename in os.listdir(CUSTOM_AGENTS_DIR):
            if filename.endswith('_agent.py') and filename != '__init__.py':
                file_path = os.path.join(CUSTOM_AGENTS_DIR, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check if this is the right agent by checking metadata
                    if f'"{agent_id}"' in content:
                        agent_file = file_path
                        break
                        
                except Exception:
                    continue
        
        if not agent_file:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        # Delete the file
        os.remove(agent_file)
        
        return {"message": f"Agent with ID {agent_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/saved-agents/{agent_id}/activate")
async def activate_saved_agent(agent_id: str):
    """
    Activate a saved custom agent
    
    This would register the agent with the agent registry.
    Note: This is a placeholder implementation.
    """
    try:
        # Check if LangChain integration is enabled for agents
        use_langchain_agents = secure_settings.get_setting("langchain_integration", "use_langchain_agents", False)
        
        if use_langchain_agents:
            # TODO: Implement agent activation logic for LangGraph agents
            # This would involve:
            # 1. Loading the agent class from the file
            # 2. Registering it with the LangGraph agent manager
            # 3. Making it available for use
            
            return {"message": f"LangGraph agent activation not yet implemented for ID {agent_id}"}
        else:
            # TODO: Implement agent activation logic for legacy agents
            # This would involve:
            # 1. Loading the agent class from the file
            # 2. Registering it with the agent_registry
            # 3. Making it available for use
            
            return {"message": f"Legacy agent activation not yet implemented for ID {agent_id}"}
        
    except Exception as e:
        logger.error(f"Error activating saved agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/available")
async def get_available_tools_for_designer():
    """
    Get list of available tools that can be used in custom agents
    
    Returns a list of tools that custom agents can use.
    """
    try:
        # Check if LangChain integration is enabled for tools
        use_langchain_tools = secure_settings.get_setting("langchain_integration", "use_langchain_tools", False)
        
        if use_langchain_tools:
            # Use LangChain tool registry
            from ..core.langchain.integration import langchain_integration
            
            # Initialize the integration layer if needed
            if not langchain_integration.is_initialized:
                await langchain_integration.initialize()
            
            # Get tools from LangChain integration
            tools = await langchain_integration.list_tools()
            
            tools_info = []
            for tool_name, tool_info in tools.items():
                tools_info.append({
                    "name": tool_name,
                    "description": tool_info.get("description", ""),
                    "categories": tool_info.get("categories", []),
                    "keywords": tool_info.get("keywords", []),
                    "enabled": tool_info.get("enabled", True)
                })
            
            return {
                "tools": tools_info,
                "total_count": len(tools_info),
                "system": "langchain"
            }
        else:
            # Use legacy tool registry
            available_tools = tool_registry.list_tools(enabled_only=True)
            
            tools_info = []
            for tool in available_tools:
                tools_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "categories": tool.categories,
                    "keywords": tool.keywords,
                    "enabled": tool.enabled
                })
            
            return {
                "tools": tools_info,
                "total_count": len(tools_info),
                "system": "legacy"
            }
        
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))