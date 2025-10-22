import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Bot,
  Power,
  PowerOff,
  Star,
  Settings,
  RefreshCw,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react';
import type { Agent } from '../services/api';
import { showToast } from '../lib/toast';

interface AgentManagerProps {
  agents: Agent[];
  isLoading: boolean;
  error: string | null;
  onActivateAgent: (agentName: string) => Promise<void>;
  onDeactivateAgent: (agentName: string) => Promise<void>;
  onSetDefaultAgent: (agentName: string) => Promise<void>;
  onRefreshAgents: () => Promise<void>;
}

const AgentManager: React.FC<AgentManagerProps> = ({
  agents,
  isLoading,
  error,
  onActivateAgent,
  onDeactivateAgent,
  onSetDefaultAgent,
  onRefreshAgents
}) => {
  const [activeTab, setActiveTab] = useState('all');
  const [processingAgents, setProcessingAgents] = useState<Set<string>>(new Set());

  // Show error notifications when errors occur
  useEffect(() => {
    if (error) {
      showToast.error(error);
    }
  }, [error]);

  // Filter agents based on active tab
  const filteredAgents = agents.filter(agent => {
    if (activeTab === 'all') return true;
    if (activeTab === 'active') return agent.state === 'active';
    if (activeTab === 'inactive') return agent.state !== 'active';
    return true;
  });

  // Handle agent activation/deactivation
  const handleToggleAgent = async (agentName: string, currentState: string) => {
    if (processingAgents.has(agentName)) return;
    
    setProcessingAgents(prev => new Set(prev).add(agentName));
    
    try {
      if (currentState === 'active') {
        await onDeactivateAgent(agentName);
      } else {
        await onActivateAgent(agentName);
      }
    } finally {
      setProcessingAgents(prev => {
        const newSet = new Set(prev);
        newSet.delete(agentName);
        return newSet;
      });
    }
  };

  // Handle setting default agent
  const handleSetDefault = async (agentName: string) => {
    if (processingAgents.has(agentName)) return;
    
    setProcessingAgents(prev => new Set(prev).add(agentName));
    
    try {
      await onSetDefaultAgent(agentName);
    } finally {
      setProcessingAgents(prev => {
        const newSet = new Set(prev);
        newSet.delete(agentName);
        return newSet;
      });
    }
  };

  // Get state icon and color
  const getStateIcon = (state: string) => {
    switch (state) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'inactive':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  // Get state badge variant
  const getStateBadgeVariant = (state: string) => {
    switch (state) {
      case 'active':
        return 'default';
      case 'inactive':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Agent Management</h1>
          <p className="text-muted-foreground">
            Manage and configure your AI agents
          </p>
        </div>
        <Button
          variant="outline"
          onClick={onRefreshAgents}
          disabled={isLoading}
        >
          <RefreshCw className={`h-5 w-5 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="all">All Agents ({agents.length})</TabsTrigger>
          <TabsTrigger value="active">
            Active ({agents.filter(a => a.state === 'active').length})
          </TabsTrigger>
          <TabsTrigger value="inactive">
            Inactive ({agents.filter(a => a.state !== 'active').length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          {filteredAgents.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Bot className="h-14 w-14 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No agents found</h3>
                <p className="text-muted-foreground text-center">
                  {activeTab === 'active' && 'No active agents found. Activate some agents to get started.'}
                  {activeTab === 'inactive' && 'No inactive agents found. All agents are currently active.'}
                  {activeTab === 'all' && 'No agents configured. Add agents to get started.'}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredAgents.map((agent) => (
                <Card key={agent.name} className="relative">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <Bot className="h-6 w-6 text-primary" />
                        <CardTitle className="text-lg">{agent.name}</CardTitle>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="scale-110">
                          {getStateIcon(agent.state)}
                        </div>
                        <Badge variant={getStateBadgeVariant(agent.state)}>
                          {agent.state}
                        </Badge>
                      </div>
                    </div>
                    <CardDescription className="line-clamp-2">
                      {agent.description}
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-1">
                      {agent.categories.map((category) => (
                        <Badge key={category} variant="outline" className="text-xs">
                          {category}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="text-sm text-muted-foreground space-y-1">
                      <div>Version: {agent.version}</div>
                      <div>Usage: {agent.usage_count} times</div>
                      {agent.last_used && (
                        <div>Last used: {new Date(agent.last_used * 1000).toLocaleDateString()}</div>
                      )}
                    </div>
                    
                    <div className="flex gap-2 pt-2">
                      <Button
                        variant={agent.state === 'active' ? 'destructive' : 'default'}
                        size="sm"
                        onClick={() => handleToggleAgent(agent.name, agent.state)}
                        disabled={processingAgents.has(agent.name)}
                        className="flex-1"
                      >
                        {processingAgents.has(agent.name) ? (
                          <RefreshCw className="h-5 w-5 animate-spin mr-2" />
                        ) : agent.state === 'active' ? (
                          <PowerOff className="h-5 w-5 mr-2" />
                        ) : (
                          <Power className="h-5 w-5 mr-2" />
                        )}
                        {agent.state === 'active' ? 'Deactivate' : 'Activate'}
                      </Button>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleSetDefault(agent.name)}
                        disabled={processingAgents.has(agent.name)}
                        title="Set as default agent"
                      >
                        {processingAgents.has(agent.name) ? (
                          <RefreshCw className="h-5 w-5 animate-spin" />
                        ) : (
                          <Star className="h-5 w-5" />
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AgentManager;