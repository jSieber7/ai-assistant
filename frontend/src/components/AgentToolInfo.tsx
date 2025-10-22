import React, { useState } from 'react';
import { Bot, Wrench, ChevronDown, ChevronUp, Clock, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChatMessage } from '@/services/api';

interface AgentToolInfoProps {
  message: ChatMessage;
}

const AgentToolInfo: React.FC<AgentToolInfoProps> = ({ message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Don't render anything if there's no agent or tool information
  if (!message.agent_name && (!message.tool_results || message.tool_results.length === 0)) {
    return null;
  }

  const formatExecutionTime = (timeMs: number) => {
    if (timeMs < 1000) {
      return `${timeMs}ms`;
    } else {
      return `${(timeMs / 1000).toFixed(2)}s`;
    }
  };

  return (
    <div className="mt-2">
      <div 
        className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer hover:text-gray-800 dark:hover:text-gray-200"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {message.agent_name && (
          <Badge variant="secondary" className="flex items-center gap-1">
            <Bot className="w-3 h-3" />
            {message.agent_name}
          </Badge>
        )}
        
        {message.tool_results && message.tool_results.length > 0 && (
          <Badge variant="outline" className="flex items-center gap-1">
            <Wrench className="w-3 h-3" />
            {message.tool_results.length} tool{message.tool_results.length > 1 ? 's' : ''}
          </Badge>
        )}
        
        {isExpanded ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </div>
      
      {isExpanded && (
        <Card className="mt-2 bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700">
          <CardContent className="p-3">
            {message.agent_name && (
              <div className="mb-3">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Agent</h4>
                <div className="flex items-center gap-2">
                  <Bot className="w-4 h-4 text-blue-500" />
                  <span className="text-sm">{message.agent_name}</span>
                </div>
              </div>
            )}
            
            {message.tool_results && message.tool_results.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Tools Used</h4>
                <div className="space-y-2">
                  {message.tool_results.map((tool, index) => (
                    <div key={index} className="flex items-start gap-2 p-2 bg-white dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-600">
                      <div className="flex-shrink-0 mt-0.5">
                        {tool.success ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Wrench className="w-3 h-3 text-gray-500" />
                          <span className="text-sm font-medium">{tool.tool_name}</span>
                          <div className="flex items-center gap-1 text-xs text-gray-500">
                            <Clock className="w-3 h-3" />
                            {formatExecutionTime(tool.execution_time)}
                          </div>
                        </div>
                        
                        {tool.error && (
                          <div className="text-xs text-red-600 dark:text-red-400 mb-1">
                            Error: {tool.error}
                          </div>
                        )}
                        
                        {tool.data && typeof tool.data === 'string' && (
                          <div className="text-xs text-gray-600 dark:text-gray-400 truncate">
                            {tool.data}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AgentToolInfo;