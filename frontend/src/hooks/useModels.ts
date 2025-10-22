import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';
import type { Model, Provider, Agent } from '../services/api';

export interface ModelsState {
  models: Model[];
  providers: Provider[];
  agents: Agent[];
  selectedProvider: string | null;
  selectedModel: string | null;
  selectedAgent: string | null;
  isLoading: boolean;
  error: string | null;
}

export const useModels = () => {
  const [state, setState] = useState<ModelsState>({
    models: [],
    providers: [],
    agents: [],
    selectedProvider: null,
    selectedModel: null,
    selectedAgent: null,
    isLoading: false,
    error: null,
  });

  // Function to load models
  const loadModels = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const modelsResponse = await apiService.listModels();
      setState(prev => ({
        ...prev,
        models: modelsResponse.data,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load models';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to load providers
  const loadProviders = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const providersResponse = await apiService.listProviders();
      setState(prev => ({
        ...prev,
        providers: providersResponse.data,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load providers';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to load agents
  const loadAgents = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const agentsResponse = await apiService.listAgents();
      setState(prev => ({
        ...prev,
        agents: agentsResponse.agents,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load agents';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to select a model and provider
  const selectModel = useCallback(async (provider: string, model: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.selectModel(provider, model);
      setState(prev => ({
        ...prev,
        selectedProvider: provider,
        selectedModel: model,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to select model';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to select an agent
  const selectAgent = useCallback((agentName: string) => {
    setState(prev => ({
      ...prev,
      selectedAgent: agentName,
    }));
  }, []);

  // Function to activate an agent
  const activateAgent = useCallback(async (agentName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.activateAgent(agentName);
      // Reload agents to get updated state
      await loadAgents();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to activate agent';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadAgents]);

  // Function to deactivate an agent
  const deactivateAgent = useCallback(async (agentName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.deactivateAgent(agentName);
      // Reload agents to get updated state
      await loadAgents();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to deactivate agent';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadAgents]);

  // Function to set default agent
  const setDefaultAgent = useCallback(async (agentName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.setDefaultAgent(agentName);
      // Reload agents to get updated state
      await loadAgents();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to set default agent';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadAgents]);

  // Function to add a provider
  const addProvider = useCallback(async (providerData: {
    name: string;
    type: string;
    api_key?: string;
    api_base?: string;
    model_list?: string[];
    [key: string]: any;
  }) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const response = await apiService.addProvider(providerData);
      
      if (response.success) {
        // Reload providers to get updated list
        await loadProviders();
        return { success: true, message: response.message };
      } else {
        const errorMessage = response.message || 'Failed to add provider';
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
        return { success: false, message: errorMessage };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add provider';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
      return { success: false, message: errorMessage };
    }
  }, [loadProviders]);

  // Function to set error
  const setError = useCallback((error: string | null) => {
    setState(prev => ({
      ...prev,
      error,
    }));
  }, []);

  // Function to load all data
  const loadAll = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await Promise.all([
        loadModels(),
        loadProviders(),
        loadAgents(),
      ]);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load data';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadModels, loadProviders, loadAgents]);

  // Load data on mount
  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return {
    ...state,
    loadModels,
    loadProviders,
    loadAgents,
    selectModel,
    selectAgent,
    activateAgent,
    deactivateAgent,
    setDefaultAgent,
    setError,
    loadAll,
    addProvider,
  };
};