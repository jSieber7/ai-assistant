import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Checkbox } from '@/components/ui/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Plus, MessageSquare, Cpu, User, Send, Paperclip, Settings, ChevronDown, Users, Plug, SquarePen, PanelLeft } from 'lucide-react';
import Sidebar from './components/Sidebar';
import SettingsModal from './components/SettingsModal';

// Define types locally to avoid import issues
interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  permission: any[];
  root: string;
  parent: string | null;
  description?: string;
  context_length?: number;
  supports_streaming?: boolean;
  supports_tools?: boolean;
}

// Simple API service inline to avoid import issues
const apiService = {
  async listModels(): Promise<{ data: Model[] }> {
    const response = await fetch('http://localhost:8000/v1/models');
    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }
    return response.json();
  },

  async chatCompletion(request: {
    messages: ChatMessage[];
    model?: string;
  }): Promise<any> {
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    
    return response.json();
  },
};

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

// Define Agent type for the SimpleHeader
interface Agent {
  name: string;
  description: string;
  state: 'active' | 'inactive';
  usage_count: number;
}

const SimpleHeader = ({
  selectedModel,
  onSelectModel,
  models,
  isLoadingModels,
  selectedAgents,
  onSelectAgents,
  agents
}: {
  selectedModel: string;
  onSelectModel: (model: string) => void;
  models: Model[];
  isLoadingModels: boolean;
  selectedAgents: string[];
  onSelectAgents: (agents: string[]) => void;
  agents: Agent[];
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
        <Select value={selectedModel} onValueChange={onSelectModel} disabled={isLoadingModels}>
          <SelectTrigger className="w-48 md:w-40 sm:w-36">
            <Plug className="h-4 w-4 mr-2" />
            <SelectValue placeholder={isLoadingModels ? 'Loading...' : 'Select a model'} className="hidden md:inline" />
          </SelectTrigger>
          <SelectContent>
            {models.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                {model.id} {model.supports_tools && '🔧'}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" className="w-48 md:w-40 sm:w-36 justify-between">
              <Users className="h-4 w-4 mr-2" />
              <span className="hidden md:inline truncate max-w-[100px]">{getSelectedAgentsDisplay()}</span>
              <ChevronDown className="h-4 w-4" />
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
      </div>
    
      <div className="flex items-center gap-4">
        <div className="hidden lg:block text-sm text-gray-500 dark:text-gray-400">
          Hosting API: <span className="font-mono">http://localhost:8000/v1</span>
        </div>
        <ThemeToggle />
      </div>
    </div>
  );
};

const SimpleMessageArea = ({
  messages,
  isConnected,
  models
}: {
  messages: ChatMessage[];
  isConnected: boolean;
  models: Model[];
}) => (
  <div className="flex-1 overflow-y-auto p-6">
    {messages.length === 0 ? (
      <div className="h-full flex items-center justify-center">
        <Card className="w-full max-w-md shadow-lg border-0 bg-white dark:bg-gray-800">
          <CardHeader className="text-center pb-4">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="w-8 h-8 text-blue-600 dark:text-blue-400" />
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
                <Button variant="outline" className="w-full">
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
                      <Cpu className="w-4 h-4" />
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
                  </CardContent>
                </Card>
                {msg.role === 'user' && (
                  <Avatar className="h-8 w-8">
                    <AvatarImage src="" alt="User" />
                    <AvatarFallback className="bg-gray-500 text-white">
                      <User className="w-4 h-4" />
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
                <Paperclip className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={onSettingsOpen}
                disabled={disabled || isLoading}
                className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Settings className="h-4 w-4" />
              </button>
            </div>
            <button
              type="submit"
              disabled={!input.trim() || disabled || isLoading}
              className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <Send className="w-4 h-4" />
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  
  // Settings state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(0);
  
  // Mock agents data
  const [agents] = useState<Agent[]>([
    { name: 'General Assistant', description: 'A general purpose AI assistant', state: 'active', usage_count: 15 },
    { name: 'Code Helper', description: 'Specialized in programming and code review', state: 'active', usage_count: 8 },
    { name: 'Research Analyst', description: 'Helps with research and data analysis', state: 'inactive', usage_count: 3 },
  ]);
  
  // Initialize with all agents selected
  const [selectedAgents, setSelectedAgents] = useState<string[]>(agents.map(agent => agent.name));

  // Load models from backend on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelsResponse = await apiService.listModels();
        setModels(modelsResponse.data);
        setIsConnected(true);
      } catch (error) {
        console.error('Failed to load models:', error);
        setIsConnected(false);
        // Set some default models if backend is not available
        setModels([
          { id: 'gpt-3.5-turbo', object: 'model', created: 0, owned_by: 'openai', permission: [], root: '', parent: null },
          { id: 'gpt-4', object: 'model', created: 0, owned_by: 'openai', permission: [], root: '', parent: null },
          { id: 'claude-3', object: 'model', created: 0, owned_by: 'anthropic', permission: [], root: '', parent: null },
        ]);
      } finally {
        setIsLoadingModels(false);
      }
    };

    loadModels();
  }, []);

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: ChatMessage = { role: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    
    // Set loading state
    setIsLoading(true);
    
    try {
      // Try to connect to the actual backend
      const response = await apiService.chatCompletion({
        messages: [{ role: 'user', content }],
        model: selectedModel || 'gpt-3.5-turbo',
      });
      
      const assistantMessage: ChatMessage = { 
        role: 'assistant', 
        content: response.choices[0]?.message?.content || 'No response received'
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error connecting to backend:', error);
      // Simulate assistant response if backend is not available
      setTimeout(() => {
        const assistantMessage: ChatMessage = { 
          role: 'assistant', 
          content: `You said: "${content}". This is a simulated response. The backend is not available.` 
        };
        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      }, 1000);
      return;
    }
    
    setIsLoading(false);
  };

  return (
    <div className="h-screen bg-background flex">
      <Sidebar
        chatHistory={[]}
        currentChatId={null}
        onNewChat={() => {}}
        onSelectChat={() => {}}
        onDeleteChat={() => {}}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />
      
      <div className={`flex-1 flex flex-col transition-all duration-300 ${
        isSidebarCollapsed ? 'ml-12' : 'ml-80'
      }`}>
        <SimpleHeader
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          models={models}
          isLoadingModels={isLoadingModels}
          selectedAgents={selectedAgents}
          onSelectAgents={setSelectedAgents}
          agents={agents}
        />
        <SimpleMessageArea messages={messages} isConnected={isConnected} models={models} />
        <SimpleInputArea
          onSendMessage={handleSendMessage}
          disabled={!selectedModel}
          isLoading={isLoading}
          temperature={temperature}
          maxTokens={maxTokens}
          onTemperatureChange={(value) => setTemperature(value[0])}
          onMaxTokensChange={(value) => setMaxTokens(value[0])}
          onSettingsOpen={() => setIsSettingsOpen(true)}
        />
        
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
      </div>
    </div>
  );
};

export default App;