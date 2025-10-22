import { useState, useEffect, useCallback } from 'react';
import { apiService, type Conversation, type ConversationWithMessages } from '../services/api';

export interface ConversationsState {
  conversations: Conversation[];
  currentConversation: ConversationWithMessages | null;
  isLoading: boolean;
  error: string | null;
}

export const useConversations = () => {
  const [state, setState] = useState<ConversationsState>({
    conversations: [],
    currentConversation: null,
    isLoading: false,
    error: null,
  });

  // Load conversations from the server
  const loadConversations = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const conversations = await apiService.listConversations();
      setState(prev => ({
        ...prev,
        conversations,
        isLoading: false,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load conversations',
      }));
    }
  }, []);

  // Create a new conversation
  const createConversation = useCallback(async (title?: string, modelId?: string, agentName?: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const conversation = await apiService.createConversation({
        title,
        model_id: modelId,
        agent_name: agentName,
      });
      
      setState(prev => ({
        ...prev,
        conversations: [conversation, ...prev.conversations],
        currentConversation: null, // Clear the current conversation
        isLoading: false,
      }));
      
      return conversation;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to create conversation',
      }));
      throw error;
    }
  }, []);

  // Load a specific conversation with its messages
  const loadConversation = useCallback(async (conversationId: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const conversation = await apiService.getConversation(conversationId);
      setState(prev => ({
        ...prev,
        currentConversation: conversation,
        isLoading: false,
      }));
      
      return conversation;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load conversation',
      }));
      throw error;
    }
  }, []);

  // Delete a conversation
  const deleteConversation = useCallback(async (conversationId: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      await apiService.deleteConversation(conversationId);
      
      setState(prev => {
        const newConversations = prev.conversations.filter(c => c.id !== conversationId);
        const newCurrentConversation = prev.currentConversation?.id === conversationId 
          ? null 
          : prev.currentConversation;
        
        return {
          ...prev,
          conversations: newConversations,
          currentConversation: newCurrentConversation,
          isLoading: false,
        };
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to delete conversation',
      }));
      throw error;
    }
  }, []);

  // Update conversation metadata
  const updateConversation = useCallback(async (
    conversationId: string, 
    title?: string, 
    metadata?: Record<string, any>
  ) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const updatedConversation = await apiService.updateConversation(conversationId, title, metadata);
      
      setState(prev => {
        const newConversations = prev.conversations.map(c => 
          c.id === conversationId ? updatedConversation : c
        );
        
        const newCurrentConversation = prev.currentConversation?.id === conversationId
          ? { ...prev.currentConversation, ...updatedConversation }
          : prev.currentConversation;
        
        return {
          ...prev,
          conversations: newConversations,
          currentConversation: newCurrentConversation,
          isLoading: false,
        };
      });
      
      return updatedConversation;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to update conversation',
      }));
      throw error;
    }
  }, []);

  // Clear the current conversation
  const clearCurrentConversation = useCallback(() => {
    setState(prev => ({ ...prev, currentConversation: null }));
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Load conversations on initial mount
  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  return {
    ...state,
    loadConversations,
    createConversation,
    loadConversation,
    deleteConversation,
    updateConversation,
    clearCurrentConversation,
    clearError,
  };
};