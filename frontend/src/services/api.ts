import axios from 'axios';

// API base URL - adjust this to match your backend
const API_BASE_URL = 'http://localhost:8000';

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
};

export default apiService;