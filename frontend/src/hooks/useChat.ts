import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService } from '../services/api';
import type { ChatMessage, ChatRequest } from '../services/api';

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  conversationId: string | null;
}

export interface UseChatOptions {
  agentName?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  conversationId?: string;
  onMessage?: (message: ChatMessage) => void;
}

export const useChat = (options: UseChatOptions = {}) => {
  const [chatState, setChatState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    error: null,
    conversationId: options.conversationId || null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Function to send a message
  const sendMessage = useCallback(async (content: string) => {
    // Add user message to state
    const userMessage: ChatMessage = { role: 'user', content };
    setChatState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
      error: null,
    }));

    try {
      // Create request
      const request: ChatRequest = {
        messages: [...chatState.messages, userMessage],
        model: options.model,
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        agent_name: options.agentName,
        conversation_id: chatState.conversationId || options.conversationId,
      };

      // Create abort controller for this request
      abortControllerRef.current = new AbortController();

      let response;
      // Use agent chat if agent name is provided
      if (options.agentName) {
        response = await apiService.agentChatCompletion(request);
      } else {
        response = await apiService.chatCompletion(request);
      }

      // Extract assistant message
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.choices[0]?.message?.content || '',
      };

      // Update state with assistant response
      setChatState(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        isLoading: false,
        conversationId: response.conversation_id || prev.conversationId,
      }));

      // Call callback if provided
      if (options.onMessage) {
        options.onMessage(assistantMessage);
      }

      return assistantMessage;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      setChatState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
      throw error;
    }
  }, [chatState.messages, chatState.conversationId, options]);

  // Function to stop the current request
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setChatState(prev => ({
        ...prev,
        isLoading: false,
      }));
    }
  }, []);

  // Function to clear the chat
  const clearChat = useCallback(() => {
    setChatState({
      messages: [],
      isLoading: false,
      error: null,
      conversationId: null,
    });
  }, []);

  // Function to set messages (useful for loading chat history)
  const setMessages = useCallback((messages: ChatMessage[]) => {
    setChatState(prev => ({
      ...prev,
      messages,
    }));
  }, []);

  // Function to set conversation ID
  const setConversationId = useCallback((id: string | null) => {
    setChatState(prev => ({
      ...prev,
      conversationId: id,
    }));
  }, []);

  // Function to set error
  const setError = useCallback((error: string | null) => {
    setChatState(prev => ({
      ...prev,
      error,
    }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    ...chatState,
    sendMessage,
    stopGeneration,
    clearChat,
    setMessages,
    setConversationId,
    setError,
  };
};