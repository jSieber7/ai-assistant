import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';
import type { Tool, ToolInfo, ToolRegistryStats, ToolsListResponse, ToolsByCategoryResponse } from '../services/api';

export interface ToolsState {
  tools: Tool[];
  selectedTool: ToolInfo | null;
  registryStats: ToolRegistryStats | null;
  categories: string[];
  toolsByCategory: Record<string, Tool[]>;
  isLoading: boolean;
  error: string | null;
}

export const useTools = () => {
  const [state, setState] = useState<ToolsState>({
    tools: [],
    selectedTool: null,
    registryStats: null,
    categories: [],
    toolsByCategory: {},
    isLoading: false,
    error: null,
  });

  // Function to load all tools
  const loadTools = useCallback(async (enabledOnly = true) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const toolsResponse = await apiService.listTools(enabledOnly);
      const categories = [...new Set(toolsResponse.tools.flatMap(tool => tool.categories))];
      
      setState(prev => ({
        ...prev,
        tools: toolsResponse.tools,
        categories,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load tools';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to load detailed information about a specific tool
  const loadToolInfo = useCallback(async (toolName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const toolInfo = await apiService.getToolInfo(toolName);
      setState(prev => ({
        ...prev,
        selectedTool: toolInfo,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load tool information';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to load tool registry statistics
  const loadRegistryStats = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const stats = await apiService.getRegistryStats();
      setState(prev => ({
        ...prev,
        registryStats: stats,
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load registry statistics';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to load tools by category
  const loadToolsByCategory = useCallback(async (category: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const response = await apiService.getToolsByCategory(category);
      setState(prev => ({
        ...prev,
        toolsByCategory: {
          ...prev.toolsByCategory,
          [category]: response.tools,
        },
        isLoading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load tools by category';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  // Function to enable a tool
  const enableTool = useCallback(async (toolName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.enableTool(toolName);
      // Reload tools to get updated state
      await loadTools();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to enable tool';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadTools]);

  // Function to disable a tool
  const disableTool = useCallback(async (toolName: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.disableTool(toolName);
      // Reload tools to get updated state
      await loadTools();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to disable tool';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadTools]);

  // Function to execute a tool
  const executeTool = useCallback(async (toolName: string, parameters: Record<string, any>) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const result = await apiService.executeTool({
        tool_name: toolName,
        parameters,
      });
      setState(prev => ({
        ...prev,
        isLoading: false,
      }));
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to execute tool';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  // Function to clear selected tool
  const clearSelectedTool = useCallback(() => {
    setState(prev => ({
      ...prev,
      selectedTool: null,
    }));
  }, []);

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
        loadTools(),
        loadRegistryStats(),
      ]);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load data';
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [loadTools, loadRegistryStats]);

  // Load data on mount
  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return {
    ...state,
    loadTools,
    loadToolInfo,
    loadRegistryStats,
    loadToolsByCategory,
    enableTool,
    disableTool,
    executeTool,
    clearSelectedTool,
    setError,
    loadAll,
  };
};