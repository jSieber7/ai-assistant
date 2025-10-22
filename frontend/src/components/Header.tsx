import React, { useState } from 'react';
import { Settings, ChevronDown, Sun, Moon } from 'lucide-react';
import { Model, Provider, Agent } from '../services/api';

interface HeaderProps {
  selectedModel: string | null;
  selectedProvider: string | null;
  selectedAgent: string | null;
  models: Model[];
  providers: Provider[];
  agents: Agent[];
  onSelectModel: (provider: string, model: string) => void;
  onSelectAgent: (agentName: string) => void;
  onToggleDarkMode: () => void;
  isDarkMode: boolean;
}

const Header: React.FC<HeaderProps> = ({
  selectedModel,
  selectedProvider,
  selectedAgent,
  models,
  providers,
  agents,
  onSelectModel,
  onSelectAgent,
  onToggleDarkMode,
  isDarkMode,
}) => {
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isAgentDropdownOpen, setIsAgentDropdownOpen] = useState(false);

  const handleModelSelect = (provider: string, model: string) => {
    onSelectModel(provider, model);
    setIsModelDropdownOpen(false);
  };

  const handleAgentSelect = (agentName: string) => {
    onSelectAgent(agentName);
    setIsAgentDropdownOpen(false);
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-4">
          {/* Model Selection */}
          <div className="relative">
            <button
              onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
              className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              <span className="text-sm font-medium">
                {selectedModel || 'Select Model'}
              </span>
              {selectedProvider && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  ({selectedProvider})
                </span>
              )}
              <ChevronDown className="h-4 w-4" />
            </button>

            {isModelDropdownOpen && (
              <div className="absolute top-full left-0 mt-2 w-80 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
                <div className="p-2">
                  {providers.map((provider) => (
                    <div key={provider.name} className="mb-2">
                      <div className="flex items-center gap-2 px-2 py-1 text-xs font-medium text-gray-500 dark:text-gray-400">
                        <span>{provider.name}</span>
                        <span className={`px-2 py-0.5 rounded-full text-xs ${
                          provider.healthy
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        }`}>
                          {provider.healthy ? 'Healthy' : 'Unhealthy'}
                        </span>
                      </div>
                      {models
                        .filter((model) => model.owned_by === provider.name)
                        .map((model) => (
                          <button
                            key={model.id}
                            onClick={() => handleModelSelect(provider.name, model.id)}
                            className={`w-full text-left px-4 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                              selectedModel === model.id && selectedProvider === provider.name
                                ? 'bg-blue-100 dark:bg-blue-900'
                                : ''
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="font-medium text-sm">{model.id}</div>
                                {model.description && (
                                  <div className="text-xs text-gray-500 dark:text-gray-400">
                                    {model.description}
                                  </div>
                                )}
                              </div>
                              <div className="flex gap-1">
                                {model.supports_streaming && (
                                  <span className="px-1.5 py-0.5 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 text-xs rounded">
                                    Stream
                                  </span>
                                )}
                                {model.supports_tools && (
                                  <span className="px-1.5 py-0.5 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 text-xs rounded">
                                    Tools
                                  </span>
                                )}
                              </div>
                            </div>
                          </button>
                        ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Agent Selection */}
          <div className="relative">
            <button
              onClick={() => setIsAgentDropdownOpen(!isAgentDropdownOpen)}
              className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              <span className="text-sm font-medium">
                {selectedAgent || 'Select Agent'}
              </span>
              <ChevronDown className="h-4 w-4" />
            </button>

            {isAgentDropdownOpen && (
              <div className="absolute top-full left-0 mt-2 w-80 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
                <div className="p-2">
                  {agents.map((agent) => (
                    <button
                      key={agent.name}
                      onClick={() => handleAgentSelect(agent.name)}
                      className={`w-full text-left px-4 py-3 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                        selectedAgent === agent.name
                          ? 'bg-blue-100 dark:bg-blue-900'
                          : ''
                      }`}
                    >
                      <div className="font-medium text-sm">{agent.name}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {agent.description}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`px-2 py-0.5 rounded-full text-xs ${
                          agent.state === 'active'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {agent.state}
                        </span>
                        <span className="text-xs text-gray-400">
                          Used {agent.usage_count} times
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Dark Mode Toggle */}
          <button
            onClick={onToggleDarkMode}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Toggle dark mode"
          >
            {isDarkMode ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </button>

          {/* Settings */}
          <button
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Settings"
          >
            <Settings className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Header;