import axios from 'axios';

// API base URL - adjust this to match your backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://my-stack-app:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for API responses
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  agent_name?: string;
  tool_results?: Array<{
    tool_name: string;
    success: boolean;
    execution_time: number;
    data: any;
    error?: string;
    metadata?: Record<string, any>;
  }>;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  agent_name?: string;
  conversation_id?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  id: string;
  object: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  agent_name?: string;
  tool_results?: Array<{
    tool_name: string;
    success: boolean;
    execution_time: number;
    data: any;
    error?: string;
    metadata?: Record<string, any>;
  }>;
  conversation_id?: string;
}

// Conversation types
export interface Conversation {
  id: string;
  title?: string;
  user_id?: string;
  model_id?: string;
  agent_name?: string;
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

export interface ConversationCreate {
  title?: string;
  model_id?: string;
  agent_name?: string;
  metadata?: Record<string, any>;
}

export interface MessageCreate {
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, any>;
}

export interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  permission: any[];
  root: string;
  parent: string | null;
  description?: string;
  context_length?: number;
  supports_streaming?: boolean;
  supports_tools?: boolean;
}

export interface ModelsResponse {
  object: string;
  data: Model[];
}

export interface Provider {
  name: string;
  type: string;
  configured: boolean;
  healthy: boolean;
  default: boolean;
}

export interface ProvidersResponse {
  object: string;
  data: Provider[];
  default_provider?: string;
}

export interface Agent {
  name: string;
  description: string;
  version: string;
  state: string;
  usage_count: number;
  last_used?: number;
  categories: string[];
}

export interface AgentsResponse {
  agents: Agent[];
  total_count: number;
}

export interface AddProviderRequest {
  name: string;
  type: string;
  api_key?: string;
  api_base?: string;
  model_list?: string[];
  [key: string]: any; // Allow additional provider-specific fields
}

export interface AddProviderResponse {
  success: boolean;
  message: string;
  provider?: Provider;
}

// Tool-related types
export interface Tool {
  name: string;
  description: string;
  categories: string[];
  enabled: boolean;
}

export interface ToolInfo {
  name: string;
  description: string;
  version: string;
  author: string;
  categories: string[];
  keywords: string[];
  parameters: Record<string, Record<string, any>>;
  enabled: boolean;
  timeout: number;
}

export interface ToolExecutionRequest {
  tool_name: string;
  parameters: Record<string, any>;
}

export interface ToolExecutionResponse {
  success: boolean;
  data: any;
  error?: string;
  tool_name: string;
  execution_time: number;
  metadata: Record<string, any>;
}

export interface ToolsListResponse {
  tools: Tool[];
  total: number;
  enabled_only: boolean;
}

export interface ToolsByCategoryResponse {
  category: string;
  tools: Tool[];
  total: number;
}

export interface ToolRegistryStats {
  total_tools: number;
  enabled_tools: number;
  disabled_tools: number;
  categories: Record<string, number>;
}

// API functions
export const apiService = {
  // Health check
  async healthCheck(): Promise<any> {
    const response = await api.get('/health');
    return response.data;
  },

  // Chat completion
  async chatCompletion(request: ChatRequest): Promise<ChatResponse> {
    const response = await api.post('/v1/chat/completions', request);
    return response.data;
  },

  // Agent chat completion
  async agentChatCompletion(request: ChatRequest): Promise<ChatResponse> {
    const response = await api.post('/api/v1/agents/chat/completions', request);
    return response.data;
  },

  // List models
  async listModels(): Promise<ModelsResponse> {
    const response = await api.get('/v1/models');
    return response.data;
  },

  // List providers
  async listProviders(): Promise<ProvidersResponse> {
    const response = await api.get('/v1/providers');
    return response.data;
  },

  // List agents
  async listAgents(): Promise<AgentsResponse> {
    const response = await api.get('/api/v1/agents/');
    return response.data;
  },

  // Get agent info
  async getAgentInfo(agentName: string): Promise<any> {
    const response = await api.get(`/api/v1/agents/${agentName}`);
    return response.data;
  },

  // Activate agent
  async activateAgent(agentName: string): Promise<any> {
    const response = await api.post(`/api/v1/agents/${agentName}/activate`);
    return response.data;
  },

  // Deactivate agent
  async deactivateAgent(agentName: string): Promise<any> {
    const response = await api.post(`/api/v1/agents/${agentName}/deactivate`);
    return response.data;
  },

  // Set default agent
  async setDefaultAgent(agentName: string): Promise<any> {
    const response = await api.post(`/api/v1/agents/${agentName}/set-default`);
    return response.data;
  },

  // Get registry info
  async getRegistryInfo(): Promise<any> {
    const response = await api.get('/api/v1/agents/registry/info');
    return response.data;
  },

  // Get UI state
  async getUIState(): Promise<any> {
    const response = await api.get('/api/ui/state');
    return response.data;
  },

  // Select model
  async selectModel(provider: string, model: string): Promise<any> {
    const response = await api.post('/api/ui/select-model', {
      provider,
      model,
    });
    return response.data;
  },

  // Add provider
  async addProvider(request: AddProviderRequest): Promise<AddProviderResponse> {
    const response = await api.post('/v1/providers', request);
    return response.data;
  },

  // Conversation management endpoints
  
  // Create a new conversation
  async createConversation(request: ConversationCreate): Promise<Conversation> {
    const response = await api.post('/api/v1/conversations', request);
    return response.data;
  },

  // List conversations
  async listConversations(limit = 50, offset = 0): Promise<Conversation[]> {
    const response = await api.get('/api/v1/conversations', {
      params: { limit, offset }
    });
    return response.data;
  },

  // Get a conversation with all its messages
  async getConversation(conversationId: string): Promise<ConversationWithMessages> {
    const response = await api.get(`/api/v1/conversations/${conversationId}`);
    return response.data;
  },

  // Add a message to a conversation
  async addMessage(request: MessageCreate): Promise<Message> {
    const response = await api.post(`/api/v1/conversations/${request.conversation_id}/messages`, request);
    return response.data;
  },

  // Delete a conversation
  async deleteConversation(conversationId: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/v1/conversations/${conversationId}`);
    return response.data;
  },

  // Update conversation metadata
  async updateConversation(
    conversationId: string,
    title?: string,
    metadata?: Record<string, any>
  ): Promise<Conversation> {
    const response = await api.put(`/api/v1/conversations/${conversationId}`, {
      title,
      metadata
    });
    return response.data;
  },

  // Tool-related endpoints
  
  // List all available tools
  async listTools(enabledOnly = true): Promise<ToolsListResponse> {
    const response = await api.get('/v1/tools', {
      params: { enabled_only: enabledOnly }
    });
    return response.data;
  },

  // Get detailed information about a specific tool
  async getToolInfo(toolName: string): Promise<ToolInfo> {
    const response = await api.get(`/v1/tools/${toolName}`);
    return response.data;
  },

  // Execute a tool with given parameters
  async executeTool(request: ToolExecutionRequest): Promise<ToolExecutionResponse> {
    const response = await api.post('/v1/tools/execute', request);
    return response.data;
  },

  // Get tool registry statistics
  async getRegistryStats(): Promise<ToolRegistryStats> {
    const response = await api.get('/v1/tools/registry/stats');
    return response.data;
  },

  // Enable a specific tool
  async enableTool(toolName: string): Promise<{ status: string; tool: string }> {
    const response = await api.post(`/v1/tools/${toolName}/enable`);
    return response.data;
  },

  // Disable a specific tool
  async disableTool(toolName: string): Promise<{ status: string; tool: string }> {
    const response = await api.post(`/v1/tools/${toolName}/disable`);
    return response.data;
  },

  // Get all tools in a specific category
  async getToolsByCategory(category: string): Promise<ToolsByCategoryResponse> {
    const response = await api.get(`/v1/tools/categories/${category}`);
    return response.data;
  },
};

export default apiService;