import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Checkbox } from '@/components/ui/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { ThemeToggle } from './components/ThemeToggle';
import { Plus, MessageSquare, Cpu, User, Send, Paperclip, Settings, ChevronDown, Users, Plug, SquarePen, PanelLeft, RefreshCw, Bot, Wrench } from 'lucide-react';
import Sidebar from './components/Sidebar';
import SettingsModal from './components/SettingsModal';
import AgentManager from './components/AgentManager';
import AddProviderModal from './components/AddProviderModal';
import ToolAnalysis from './components/ToolAnalysis';
import AgentToolInfo from './components/AgentToolInfo';
import BackendLoadingScreen from './components/BackendLoadingScreen';
import { useChat } from './hooks/useChat';
import { useModels } from './hooks/useModels';
import { useBackendConnection } from './hooks/useBackendConnection';
import { useConversations } from './hooks/useConversations';
import type { ChatMessage, Model, Agent } from './services/api';
import { showToast } from './lib/toast';

// Simple inline components first
const SimpleSidebar = () => (
  <div className="w-64 h-screen bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 p-4">
    <div className="flex items-center gap-2 mb-6">
      <Avatar className="h-8 w-8">
        <AvatarImage src="" alt="AI" />
        <AvatarFallback className="bg-blue-600 text-white">AI</AvatarFallback>
      </Avatar>
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">AI Assistant</h2>
    </div>
    
    <Button className="w-full mb-4" variant="default">
      <SquarePen className="w-4 h-4 mr-2" />
      New Chat
    </Button>
    
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Chat History</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 italic">No chats yet</p>
    </div>
  </div>
);


const SimpleHeader = ({
  selectedModel,
  onSelectModel,
  models,
  isLoadingModels,
  onAddProvider,
  selectedAgents,
  onSelectAgents,
  agents,
  connectionState,
  onRefreshConnection,
  currentView,
  onChangeView
}: {
  selectedModel: string;
  onSelectModel: (model: string) => void;
  models: Model[];
  isLoadingModels: boolean;
  onAddProvider: () => void;
  selectedAgents: string[];
  onSelectAgents: (agents: string[]) => void;
  agents: Agent[];
  connectionState: {
    isConnected: boolean;
    isChecking: boolean;
    error: string | null;
    lastChecked: Date | null;
  };
  onRefreshConnection: () => void;
  currentView: 'chat' | 'agents' | 'tools';
  onChangeView: (view: 'chat' | 'agents' | 'tools') => void;
}) => {
  const handleAgentToggle = (agentName: string) => {
    if (selectedAgents.includes(agentName)) {
      onSelectAgents(selectedAgents.filter(a => a !== agentName));
    } else {
      onSelectAgents([...selectedAgents, agentName]);
    }
  };

  const getSelectedAgentsDisplay = () => {
    if (selectedAgents.length === 0) return 'Select agents';
    if (selectedAgents.length === 1) return selectedAgents[0];
    if (selectedAgents.length === agents.length) return 'All agents selected';
    return `${selectedAgents.length} agents selected`;
  };

  return (
    <div className="h-16 bg-background border-b border-white/20 px-6 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-3">
        <Button
          variant={currentView === 'chat' ? 'default' : 'outline'}
          onClick={() => onChangeView('chat')}
          className="flex items-center gap-2"
        >
          <MessageSquare className="h-5 w-5" />
          <span className="hidden sm:inline">Chat</span>
        </Button>
        
        <Button
          variant={currentView === 'agents' ? 'default' : 'outline'}
          onClick={() => onChangeView('agents')}
          className="flex items-center gap-2"
        >
          <Bot className="h-5 w-5" />
          <span className="hidden sm:inline">Agents</span>
        </Button>
        
        <Button
          variant={currentView === 'tools' ? 'default' : 'outline'}
          onClick={() => onChangeView('tools')}
          className="flex items-center gap-2"
        >
          <Wrench className="h-5 w-5" />
          <span className="hidden sm:inline">Tools</span>
        </Button>
        
        {currentView === 'chat' && (
          <>
            <Select value={selectedModel} onValueChange={onSelectModel} disabled={isLoadingModels}>
              <SelectTrigger className="w-48 md:w-40 sm:w-36 border border-input bg-background hover:bg-accent hover:text-accent-foreground text-foreground">
                <Plug className="h-5 w-5 mr-2" />
                <SelectValue placeholder={isLoadingModels ? 'Loading...' : 'Select a model'} className="hidden md:inline" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.id} {model.supports_tools && '🔧'}
                  </SelectItem>
                ))}
                <SelectItem value="add-model-action">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Model
                </SelectItem>
              </SelectContent>
            </Select>
            
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-48 md:w-40 sm:w-36 justify-between">
                  <Users className="h-5 w-5 mr-2" />
                  <span className="hidden md:inline truncate max-w-[100px]">{getSelectedAgentsDisplay()}</span>
                  <ChevronDown className="h-5 w-5" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-48 p-2" align="start">
                {agents.map((agent) => (
                  <div key={agent.name} className="flex items-center gap-2 p-2 hover:bg-accent rounded cursor-pointer">
                    <Checkbox
                      checked={selectedAgents.includes(agent.name)}
                      onCheckedChange={() => handleAgentToggle(agent.name)}
                    />
                    <label className="text-sm cursor-pointer flex-1" onClick={() => handleAgentToggle(agent.name)}>
                      {agent.name}
                    </label>
                  </div>
                ))}
              </PopoverContent>
            </Popover>
          </>
        )}
      </div>
    
      <div className="flex items-center gap-4">
        <div className="hidden lg:flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
          <span className={`inline-block w-2 h-2 rounded-full ${connectionState.isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          <span>{connectionState.isConnected ? 'Connected' : 'Disconnected'}</span>
          <button
            onClick={onRefreshConnection}
            disabled={connectionState.isChecking}
            className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
            title="Refresh connection"
          >
            <RefreshCw className={`w-4 h-4 ${connectionState.isChecking ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <ThemeToggle />
      </div>
    </div>
  );
};

const SimpleMessageArea = ({
  messages,
  isConnected,
  models,
  onAddProvider
}: {
  messages: ChatMessage[];
  isConnected: boolean;
  models: Model[];
  onAddProvider: () => void;
}) => (
  <div className="flex-1 overflow-y-auto p-6">
    {messages.length === 0 ? (
      <div className="h-full flex items-center justify-center">
        <Card className="w-full max-w-md shadow-lg border-0 bg-gray-50 dark:bg-gray-900">
          <CardHeader className="text-center pb-4">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="w-9 h-9 text-blue-600 dark:text-blue-400" />
            </div>
            <CardTitle className="text-2xl">Welcome to AI Assistant</CardTitle>
            <CardDescription className="flex items-center justify-center gap-2">
              <span className={`inline-block w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
              {isConnected ? 'Connected to backend' : 'Not connected to backend'}
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pt-0">
            {models.length === 0 ? (
              <div className="space-y-4">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Add an LLM provider to start chatting.
                </p>
                <Button variant="outline" className="w-full" onClick={onAddProvider}>
                  Add Provider
                </Button>
              </div>
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Select a model from the dropdown above to start chatting with the AI.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    ) : (
      <div className="space-y-4 max-w-4xl mx-auto">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl ${
              msg.role === 'user' ? 'order-2' : 'order-1'
            }`}>
              <div className="flex items-start gap-3 mb-1">
                {msg.role === 'assistant' && (
                  <Avatar className="h-8 w-8">
                    <AvatarImage src="" alt="AI" />
                    <AvatarFallback className="bg-blue-600 text-white">
                      <Cpu className="w-5 h-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
                <Card className={`${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}>
                  <CardContent className="p-3">
                    <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                    {msg.role === 'assistant' && <AgentToolInfo message={msg} />}
                  </CardContent>
                </Card>
                {msg.role === 'user' && (
                  <Avatar className="h-8 w-8">
                    <AvatarImage src="" alt="User" />
                    <AvatarFallback className="bg-gray-500 text-white">
                      <User className="w-5 h-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

const SimpleInputArea = ({
  onSendMessage,
  disabled,
  isLoading,
  temperature,
  maxTokens,
  onTemperatureChange,
  onMaxTokensChange,
  onSettingsOpen
}: {
  onSendMessage: (msg: string) => void;
  disabled: boolean;
  isLoading: boolean;
  temperature: number;
  maxTokens: number;
  onTemperatureChange: (value: number[]) => void;
  onMaxTokensChange: (value: number[]) => void;
  onSettingsOpen: () => void;
}) => {
  const [input, setInput] = useState('');
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleFileUpload = () => {
    // TODO: Implement file upload functionality
    console.log('File upload not implemented yet');
  };

  return (
    <div className="p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={disabled ? "Please select a model first" : "Type your message..."}
                disabled={disabled || isLoading}
                className="w-full px-4 py-3 bg-white dark:bg-gray-900 border-0 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed min-h-[48px] max-h-[200px]"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '200px', backgroundColor: 'inherit' }}
              />
            </div>
          </div>
          <div className="flex justify-between items-center mt-3">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleFileUpload}
                disabled={disabled || isLoading}
                className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Paperclip className="h-5 w-5" />
              </button>
              <button
                type="button"
                onClick={onSettingsOpen}
                disabled={disabled || isLoading}
                className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Settings className="h-5 w-5" />
              </button>
            </div>
            <button
              type="submit"
              disabled={!input.trim() || disabled || isLoading}
              className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
        {disabled && (
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400 text-center">
            Select a model from the dropdown to enable messaging
          </p>
        )}
      </form>
    </div>
  );
};

const App: React.FC = () => {
  // Settings state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAddProviderOpen, setIsAddProviderOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(0);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [currentView, setCurrentView] = useState<'chat' | 'agents' | 'tools'>('chat');
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [isSidebarOffScreen, setIsSidebarOffScreen] = useState(false);

  // Handle sidebar animation state based on currentView
  useEffect(() => {
    if (currentView === 'agents' || currentView === 'tools') {
      // Start closing animation
      setIsSidebarOffScreen(true);
      const timer = setTimeout(() => {
        setIsSidebarVisible(false);
      }, 300); // Match the CSS transition duration
      return () => clearTimeout(timer);
    } else if (currentView === 'chat') {
      // Start opening animation
      if (!isSidebarVisible) {
        setIsSidebarVisible(true);
        setIsSidebarOffScreen(true); // Start off-screen
        const timer = setTimeout(() => {
          setIsSidebarOffScreen(false); // Animate on-screen
        }, 20); // Small delay to ensure render
        return () => clearTimeout(timer);
      }
    }
  }, [currentView]); // Only depend on currentView
  
  // Use the hooks for state management
  const modelsState = useModels();
  const conversationsState = useConversations();
  
  const chatState = useChat({
    model: modelsState.selectedModel || undefined,
    agentName: modelsState.selectedAgent || undefined,
    temperature,
    maxTokens,
    conversationId: conversationsState.currentConversation?.id,
  });
  
  // Use the backend connection hook with startup mode
  const connectionState = useBackendConnection({
    startupMode: true,
    startupCheckInterval: 2000,
    startupTimeout: 60000,
    maxStartupAttempts: 30,
  });

  // Show error notifications when errors occur
  useEffect(() => {
    if (modelsState.error) {
      showToast.error(modelsState.error);
    }
  }, [modelsState.error]);

  useEffect(() => {
    if (chatState.error) {
      showToast.error(chatState.error);
    }
  }, [chatState.error]);

  useEffect(() => {
    if (connectionState.error) {
      showToast.error(connectionState.error);
    }
  }, [connectionState.error]);

  // Show loading screen during startup or when backend is not available
  if (connectionState.isStartupMode || (!connectionState.isConnected && !connectionState.hasTimedOut)) {
    return (
      <BackendLoadingScreen
        isChecking={connectionState.isChecking}
        error={connectionState.error}
        attempts={connectionState.startupAttempts}
        hasTimedOut={connectionState.hasTimedOut}
        onRetry={connectionState.checkConnection}
      />
    );
  }

  // Show error screen if backend connection has timed out
  if (connectionState.hasTimedOut && !connectionState.isConnected) {
    return (
      <BackendLoadingScreen
        isChecking={connectionState.isChecking}
        error={connectionState.error}
        attempts={connectionState.startupAttempts}
        hasTimedOut={connectionState.hasTimedOut}
        onRetry={connectionState.checkConnection}
      />
    );
  }

  return (
    <div className="h-screen bg-background flex">
      <Sidebar
        chatHistory={conversationsState.conversations.map(conv => ({
          id: conv.id,
          title: conv.title || 'Untitled Conversation',
          lastMessage: conv.message_count && conv.message_count > 0
            ? 'Click to view messages'
            : 'No messages yet',
          timestamp: new Date(conv.updated_at),
          messages: conversationsState.currentConversation?.id === conv.id
            ? conversationsState.currentConversation.messages.map(msg => ({
                role: msg.role,
                content: msg.content,
              }))
            : []
        }))}
        currentChatId={conversationsState.currentConversation?.id || null}
        onNewChat={async () => {
          try {
            const newConversation = await conversationsState.createConversation(
              undefined,
              modelsState.selectedModel || undefined,
              modelsState.selectedAgent || undefined
            );
            // Clear the current chat state
            chatState.clearChat();
            chatState.setConversationId(newConversation.id);
          } catch (error) {
            console.error('Failed to create new conversation:', error);
          }
        }}
        onSelectChat={async (chatId) => {
          try {
            await conversationsState.loadConversation(chatId);
            // Update the chat state with the conversation messages
            if (conversationsState.currentConversation) {
              const messages = conversationsState.currentConversation.messages.map(msg => ({
                role: msg.role,
                content: msg.content,
              }));
              chatState.setMessages(messages);
              chatState.setConversationId(chatId);
            }
          } catch (error) {
            console.error('Failed to load conversation:', error);
          }
        }}
        onDeleteChat={async (chatId) => {
          try {
            await conversationsState.deleteConversation(chatId);
            // If the deleted conversation was the current one, clear the chat state
            if (conversationsState.currentConversation?.id === chatId) {
              chatState.clearChat();
              conversationsState.clearCurrentConversation();
            }
          } catch (error) {
            console.error('Failed to delete conversation:', error);
          }
        }}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        isOffScreen={isSidebarOffScreen}
      />
      
      <div className={`flex-1 flex flex-col transition-all duration-300 ${
        isSidebarCollapsed ? 'ml-12' : (currentView === 'agents' || currentView === 'tools') ? 'ml-0' : 'ml-80'
      }`}>
        <SimpleHeader
          selectedModel={modelsState.selectedModel || ''}
          onSelectModel={(model) => {
            if (model === 'add-model-action') {
              setIsAddProviderOpen(true);
            } else {
              modelsState.selectModel(modelsState.selectedProvider || '', model);
            }
          }}
          models={modelsState.models}
          isLoadingModels={modelsState.isLoading}
          onAddProvider={() => setIsAddProviderOpen(true)}
          selectedAgents={modelsState.selectedAgent ? [modelsState.selectedAgent] : []}
          onSelectAgents={() => {}} // Placeholder for now
          agents={modelsState.agents}
          connectionState={connectionState}
          onRefreshConnection={connectionState.checkConnection}
          currentView={currentView}
          onChangeView={setCurrentView}
        />
        
        {currentView === 'chat' ? (
          <>
            <SimpleMessageArea
              messages={chatState.messages}
              isConnected={connectionState.isConnected}
              models={modelsState.models}
              onAddProvider={() => setIsAddProviderOpen(true)}
            />
            <SimpleInputArea
              onSendMessage={chatState.sendMessage}
              disabled={!modelsState.selectedModel}
              isLoading={chatState.isLoading}
              temperature={temperature}
              maxTokens={maxTokens}
              onTemperatureChange={(value) => setTemperature(value[0])}
              onMaxTokensChange={(value) => setMaxTokens(value[0])}
              onSettingsOpen={() => setIsSettingsOpen(true)}
            />
          </>
        ) : currentView === 'tools' ? (
          <ToolAnalysis />
        ) : (
          <AgentManager
            agents={modelsState.agents}
            isLoading={modelsState.isLoading}
            error={modelsState.error}
            onActivateAgent={modelsState.activateAgent}
            onDeactivateAgent={modelsState.deactivateAgent}
            onSetDefaultAgent={modelsState.setDefaultAgent}
            onRefreshAgents={modelsState.loadAgents}
          />
        )}
        
        <SettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          temperature={temperature}
          maxTokens={maxTokens}
          onTemperatureChange={(value) => setTemperature(value[0])}
          onMaxTokensChange={(value) => setMaxTokens(value[0])}
          onReset={() => {
            setTemperature(0.7);
            setMaxTokens(0);
          }}
          onSave={() => setIsSettingsOpen(false)}
        />
        
        <AddProviderModal
          isOpen={isAddProviderOpen}
          onClose={() => setIsAddProviderOpen(false)}
          onAddProvider={modelsState.addProvider}
          isLoading={modelsState.isLoading}
        />
      </div>
    </div>
  );
};

export default App;