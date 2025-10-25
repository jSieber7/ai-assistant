import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card';
import { Cpu, RefreshCw, AlertTriangle } from 'lucide-react';

interface BackendLoadingScreenProps {
  isChecking: boolean;
  error: string | null;
  attempts: number;
  hasTimedOut: boolean;
  onRetry: () => void;
}

const BackendLoadingScreen: React.FC<BackendLoadingScreenProps> = ({
  isChecking,
  error,
  attempts,
  hasTimedOut,
  onRetry,
}) => {
  const getLoadingMessage = () => {
    if (hasTimedOut) {
      return "Backend is taking longer than expected to start...";
    }
    if (attempts === 0) {
      return "Connecting to backend...";
    }
    if (attempts < 5) {
      return "Waiting for backend to start...";
    }
    if (attempts < 10) {
      return "Backend is still starting up...";
    }
    return "Continuing to wait for backend...";
  };

  const getProgressPercentage = () => {
    // Cap progress at 95% to avoid showing 100% when not actually connected
    return Math.min(95, Math.floor((attempts / 30) * 100));
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader className="text-center pb-4">
          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
            <Cpu className="w-9 h-9 text-blue-600 dark:text-blue-400" />
          </div>
          <h1 className="text-2xl font-semibold">AI Assistant</h1>
          <CardDescription className="text-base">
            {getLoadingMessage()}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress bar */}
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${getProgressPercentage()}%` }}
            ></div>
          </div>
          
          {/* Status information */}
          <div className="text-center space-y-2">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Attempt {attempts} of 30
            </p>
            
            {isChecking && (
              <div className="flex items-center justify-center gap-2 text-sm text-blue-600 dark:text-blue-400">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Checking connection...</span>
              </div>
            )}
            
            {error && !hasTimedOut && (
              <div className="flex items-center justify-center gap-2 text-sm text-orange-600 dark:text-orange-400">
                <AlertTriangle className="w-4 h-4" />
                <span>Connection failed, retrying...</span>
              </div>
            )}
            
            {hasTimedOut && (
              <div className="space-y-3">
                <div className="flex items-center justify-center gap-2 text-sm text-red-600 dark:text-red-400">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Connection timeout</span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  The backend is taking longer than expected to start. This might be normal if it's the first time starting.
                </p>
                <Button 
                  onClick={onRetry}
                  variant="outline"
                  className="w-full"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retry Connection
                </Button>
              </div>
            )}
          </div>
          
          {/* Help text */}
          <div className="text-xs text-gray-500 dark:text-gray-400 text-center space-y-1">
            <p>If this problem persists, check that:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>The backend service is running</li>
              <li>Network connectivity is available</li>
              <li>No firewall is blocking the connection</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default BackendLoadingScreen;