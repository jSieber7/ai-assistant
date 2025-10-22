import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

export interface BackendConnectionState {
  isConnected: boolean;
  isChecking: boolean;
  error: string | null;
  lastChecked: Date | null;
}

export const useBackendConnection = (checkInterval: number = 30000) => {
  const [state, setState] = useState<BackendConnectionState>({
    isConnected: false,
    isChecking: false,
    error: null,
    lastChecked: null,
  });

  const checkConnection = useCallback(async () => {
    setState(prev => ({ ...prev, isChecking: true, error: null }));
    
    try {
      await apiService.healthCheck();
      setState(prev => ({
        ...prev,
        isConnected: true,
        isChecking: false,
        error: null,
        lastChecked: new Date(),
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect to backend';
      setState(prev => ({
        ...prev,
        isConnected: false,
        isChecking: false,
        error: errorMessage,
        lastChecked: new Date(),
      }));
    }
  }, []);

  // Check connection on mount
  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  // Set up periodic connection checks
  useEffect(() => {
    if (checkInterval <= 0) return;
    
    const interval = setInterval(checkConnection, checkInterval);
    return () => clearInterval(interval);
  }, [checkConnection, checkInterval]);

  return {
    ...state,
    checkConnection,
  };
};