import { useState, useEffect, useCallback, useRef } from 'react';
import { apiService } from '../services/api';

export interface BackendConnectionState {
  isConnected: boolean;
  isChecking: boolean;
  error: string | null;
  lastChecked: Date | null;
  isStartupMode: boolean;
  startupAttempts: number;
  hasTimedOut: boolean;
}

export interface UseBackendConnectionOptions {
  checkInterval?: number;
  startupMode?: boolean;
  startupCheckInterval?: number;
  startupTimeout?: number;
  maxStartupAttempts?: number;
}

export const useBackendConnection = (options: UseBackendConnectionOptions = {}) => {
  const {
    checkInterval = 30000,
    startupMode = false,
    startupCheckInterval = 2000,
    startupTimeout = 60000,
    maxStartupAttempts = 30,
  } = options;

  const [state, setState] = useState<BackendConnectionState>({
    isConnected: false,
    isChecking: false,
    error: null,
    lastChecked: null,
    isStartupMode: startupMode,
    startupAttempts: 0,
    hasTimedOut: false,
  });

  const startupTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startupAttemptsRef = useRef(0);

  const checkConnection = useCallback(async () => {
    setState(prev => ({
      ...prev,
      isChecking: true,
      error: prev.isStartupMode ? null : prev.error // Clear error only in startup mode
    }));
    
    try {
      await apiService.healthCheck();
      setState(prev => ({
        ...prev,
        isConnected: true,
        isChecking: false,
        error: null,
        lastChecked: new Date(),
        isStartupMode: false, // Exit startup mode on successful connection
        startupAttempts: 0,
      }));
      
      // Clear startup timeout if connection is successful
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
        startupTimeoutRef.current = null;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect to backend';
      const newAttempts = startupAttemptsRef.current + 1;
      startupAttemptsRef.current = newAttempts;
      
      setState(prev => ({
        ...prev,
        isConnected: false,
        isChecking: false,
        error: errorMessage,
        lastChecked: new Date(),
        startupAttempts: newAttempts,
        hasTimedOut: prev.isStartupMode && (newAttempts >= maxStartupAttempts),
      }));
    }
  }, [maxStartupAttempts]);

  // Set up startup mode with timeout
  useEffect(() => {
    if (!startupMode) return;

    // Set up timeout for startup mode
    startupTimeoutRef.current = setTimeout(() => {
      setState(prev => ({
        ...prev,
        hasTimedOut: true,
        isStartupMode: false,
      }));
    }, startupTimeout);

    return () => {
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
        startupTimeoutRef.current = null;
      }
    };
  }, [startupMode, startupTimeout]);

  // Check connection on mount
  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  // Set up periodic connection checks
  // Set up periodic connection checks
  useEffect(() => {
    const interval = state.isStartupMode ? startupCheckInterval : checkInterval;
    if (interval <= 0) return;
    
    const intervalId = setInterval(checkConnection, interval);
    return () => clearInterval(intervalId);
  }, [checkConnection, state.isStartupMode, startupCheckInterval, checkInterval]);
  return {
    ...state,
    checkConnection,
  };
};