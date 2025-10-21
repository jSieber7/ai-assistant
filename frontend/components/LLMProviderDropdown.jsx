import React, { useState, useEffect, useRef } from 'react';

/**
 * LLM Provider Dropdown Button Component
 * 
 * A dynamic, nested dropdown menu for selecting LLM providers and models.
 * Features:
 * - Lists all available LLM providers
 * - Nested menu showing models for each provider
 * - Search functionality for models
 * - Button to add new providers
 * 
 * @param {Object} props
 * @param {Array} props.providers - Array of provider objects
 * @param {Function} props.onProviderSelect - Callback when provider is selected
 * @param {Function} props.onModelSelect - Callback when model is selected
 * @param {Function} props.onAddProvider - Callback when add provider is clicked
 * @param {string} props.selectedProvider - Currently selected provider
 * @param {string} props.selectedModel - Currently selected model
 */
const LLMProviderDropdown = ({
  providers = [],
  onProviderSelect,
  onModelSelect,
  onAddProvider,
  selectedProvider = null,
  selectedModel = null
}) => {
  // State management
  const [isOpen, setIsOpen] = useState(false);
  const [expandedProviders, setExpandedProviders] = useState(new Set());
  const [searchModel, setSearchModel] = useState('');
  const [activeProvider, setActiveProvider] = useState(null);
  
  // Refs for dropdown and click outside detection
  const dropdownRef = useRef(null);
  const searchInputRef = useRef(null);

  // Mock data for demonstration (replace with actual data from API)
  const mockProviders = [
    {
      id: 'openai_compatible',
      name: 'OpenAI Compatible',
      display_name: 'OpenAI Compatible',
      models: [
        { id: 'gpt-4', name: 'GPT-4', provider: 'openai_compatible', description: 'OpenAI\'s GPT-4 model', context_length: 8192 },
        { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai_compatible', description: 'OpenAI\'s GPT-3.5 Turbo model', context_length: 4096 },
        { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'openai_compatible', description: 'Anthropic\'s Claude 3 Sonnet model', context_length: 200000 },
      ]
    },
    {
      id: 'ollama',
      name: 'Ollama',
      display_name: 'Ollama',
      models: [
        { id: 'llama2', name: 'Llama 2', provider: 'ollama', description: 'Meta\'s Llama 2 model', context_length: 4096 },
        { id: 'codellama', name: 'Code Llama', provider: 'ollama', description: 'Meta\'s Code Llama model', context_length: 16384 },
        { id: 'mistral', name: 'Mistral', provider: 'ollama', description: 'Mistral AI model', context_length: 8192 },
      ]
    },
    {
      id: 'openrouter',
      name: 'OpenRouter',
      display_name: 'OpenRouter',
      models: [
        { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'openrouter', description: 'Anthropic\'s Claude 3.5 Sonnet', context_length: 200000 },
        { id: 'openai/gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openrouter', description: 'OpenAI\'s GPT-4 Turbo model', context_length: 128000 },
        { id: 'google/gemini-pro', name: 'Gemini Pro', provider: 'openrouter', description: 'Google\'s Gemini Pro model', context_length: 32768 },
      ]
    }
  ];

  // Use provided providers or mock data
  const providersData = providers.length > 0 ? providers : mockProviders;

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Focus search input when provider is expanded
  useEffect(() => {
    if (activeProvider && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [activeProvider]);

  // Toggle dropdown
  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setSearchModel('');
      setActiveProvider(null);
    }
  };

  // Toggle provider expansion
  const toggleProvider = (providerId) => {
    const newExpanded = new Set(expandedProviders);
    if (newExpanded.has(providerId)) {
      newExpanded.delete(providerId);
      if (activeProvider === providerId) {
        setActiveProvider(null);
      }
    } else {
      newExpanded.add(providerId);
      setActiveProvider(providerId);
    }
    setExpandedProviders(newExpanded);
  };

  // Handle provider selection
  const handleProviderSelect = (provider) => {
    if (onProviderSelect) {
      onProviderSelect(provider);
    }
    toggleProvider(provider.id);
  };

  // Handle model selection
  const handleModelSelect = (provider, model) => {
    if (onModelSelect) {
      onModelSelect(provider, model);
    }
    setIsOpen(false);
  };

  // Handle add provider
  const handleAddProvider = () => {
    setIsOpen(false);
    if (onAddProvider) {
      onAddProvider();
    }
  };

  // Filter models based on search
  const getFilteredModels = (models) => {
    if (!searchModel.trim()) return models;
    
    const searchLower = searchModel.toLowerCase();
    return models.filter(model => 
      model.name.toLowerCase().includes(searchLower) ||
      model.description?.toLowerCase().includes(searchLower)
    );
  };

  // Get display text for button
  const getDisplayText = () => {
    if (selectedProvider && selectedModel) {
      const provider = providersData.find(p => p.id === selectedProvider);
      const model = provider?.models.find(m => m.id === selectedModel);
      return model ? `${provider.display_name}: ${model.name}` : 'Select Provider & Model';
    }
    return 'Select Provider & Model';
  };

  return (
    <div className="llm-provider-dropdown" ref={dropdownRef}>
      {/* Main button */}
      <button
        className="llm-dropdown-button"
        onClick={toggleDropdown}
        aria-expanded={isOpen}
        aria-haspopup="menu"
      >
        <span className="llm-dropdown-text">{getDisplayText()}</span>
        <svg
          className={`llm-dropdown-arrow ${isOpen ? 'open' : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M2 4L6 8L10 4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="llm-dropdown-menu">
          <div className="llm-dropdown-content">
            {/* Provider list */}
            <div className="llm-provider-list">
              {providersData.map((provider) => (
                <div key={provider.id} className="llm-provider-item">
                  {/* Provider header */}
                  <div
                    className={`llm-provider-header ${
                      selectedProvider === provider.id ? 'selected' : ''
                    }`}
                    onClick={() => handleProviderSelect(provider)}
                  >
                    <span className="llm-provider-name">{provider.display_name}</span>
                    <svg
                      className={`llm-expand-icon ${
                        expandedProviders.has(provider.id) ? 'expanded' : ''
                      }`}
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M4 5L6 7L8 5"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>

                  {/* Models list (expanded) */}
                  {expandedProviders.has(provider.id) && (
                    <div className="llm-models-container">
                      {/* Search input */}
                      <div className="llm-search-container">
                        <input
                          ref={searchInputRef}
                          type="text"
                          className="llm-search-input"
                          placeholder="Search models..."
                          value={searchModel}
                          onChange={(e) => setSearchModel(e.target.value)}
                        />
                        <svg
                          className="llm-search-icon"
                          width="14"
                          height="14"
                          viewBox="0 0 14 14"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M6.33333 10.6667C8.91885 10.6667 11.0167 8.56885 11.0167 5.98333C11.0167 3.39781 8.91885 1.3 6.33333 1.3C3.74781 1.3 1.65 3.39781 1.65 5.98333C1.65 8.56885 3.74781 10.6667 6.33333 10.6667Z"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M9.66667 9.66667L12.3333 12.3333"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </div>

                      {/* Filtered models */}
                      <div className="llm-models-list">
                        {getFilteredModels(provider.models).map((model) => (
                          <div
                            key={model.id}
                            className={`llm-model-item ${
                              selectedModel === model.id ? 'selected' : ''
                            }`}
                            onClick={() => handleModelSelect(provider, model)}
                          >
                            <div className="llm-model-name">{model.name}</div>
                            {model.description && (
                              <div className="llm-model-description">{model.description}</div>
                            )}
                            {model.context_length && (
                              <div className="llm-model-context">
                                Context: {model.context_length.toLocaleString()} tokens
                              </div>
                            )}
                          </div>
                        ))}
                        
                        {getFilteredModels(provider.models).length === 0 && (
                          <div className="llm-no-models">No models found</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Add provider button */}
            <div className="llm-dropdown-footer">
              <button
                className="llm-add-provider-button"
                onClick={handleAddProvider}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 14 14"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M7 1V13M1 7H13"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Add New Provider
              </button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .llm-provider-dropdown {
          position: relative;
          display: inline-block;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        .llm-dropdown-button {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 16px;
          background-color: #ffffff;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          color: #374151;
          min-width: 250px;
          transition: all 0.2s ease;
        }

        .llm-dropdown-button:hover {
          border-color: #9ca3af;
          box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }

        .llm-dropdown-button:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .llm-dropdown-text {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .llm-dropdown-arrow {
          transition: transform 0.2s ease;
          color: #6b7280;
        }

        .llm-dropdown-arrow.open {
          transform: rotate(180deg);
        }

        .llm-dropdown-menu {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          z-index: 1000;
          margin-top: 4px;
          background-color: #ffffff;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
          max-height: 400px;
          overflow: hidden;
        }

        .llm-dropdown-content {
          display: flex;
          flex-direction: column;
        }

        .llm-provider-list {
          max-height: 350px;
          overflow-y: auto;
        }

        .llm-provider-item {
          border-bottom: 1px solid #f3f4f6;
        }

        .llm-provider-item:last-child {
          border-bottom: none;
        }

        .llm-provider-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          cursor: pointer;
          transition: background-color 0.2s ease;
          font-weight: 500;
        }

        .llm-provider-header:hover {
          background-color: #f9fafb;
        }

        .llm-provider-header.selected {
          background-color: #eff6ff;
          color: #1d4ed8;
        }

        .llm-provider-name {
          font-size: 14px;
        }

        .llm-expand-icon {
          transition: transform 0.2s ease;
          color: #6b7280;
        }

        .llm-expand-icon.expanded {
          transform: rotate(180deg);
        }

        .llm-models-container {
          background-color: #f9fafb;
          border-top: 1px solid #e5e7eb;
        }

        .llm-search-container {
          position: relative;
          padding: 12px 16px;
          border-bottom: 1px solid #e5e7eb;
        }

        .llm-search-input {
          width: 100%;
          padding: 8px 12px 8px 32px;
          border: 1px solid #d1d5db;
          border-radius: 4px;
          font-size: 13px;
          background-color: #ffffff;
        }

        .llm-search-input:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .llm-search-icon {
          position: absolute;
          left: 28px;
          top: 50%;
          transform: translateY(-50%);
          color: #9ca3af;
          pointer-events: none;
        }

        .llm-models-list {
          max-height: 200px;
          overflow-y: auto;
        }

        .llm-model-item {
          padding: 10px 16px 10px 24px;
          cursor: pointer;
          transition: background-color 0.2s ease;
          border-bottom: 1px solid #f3f4f6;
        }

        .llm-model-item:last-child {
          border-bottom: none;
        }

        .llm-model-item:hover {
          background-color: #f3f4f6;
        }

        .llm-model-item.selected {
          background-color: #dbeafe;
          color: #1d4ed8;
        }

        .llm-model-name {
          font-size: 13px;
          font-weight: 500;
          margin-bottom: 2px;
        }

        .llm-model-description {
          font-size: 12px;
          color: #6b7280;
          margin-bottom: 2px;
        }

        .llm-model-context {
          font-size: 11px;
          color: #9ca3af;
        }

        .llm-no-models {
          padding: 16px;
          text-align: center;
          color: #6b7280;
          font-size: 13px;
        }

        .llm-dropdown-footer {
          padding: 8px;
          border-top: 1px solid #e5e7eb;
          background-color: #f9fafb;
        }

        .llm-add-provider-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          width: 100%;
          padding: 8px 12px;
          background-color: transparent;
          border: 1px dashed #d1d5db;
          border-radius: 4px;
          color: #6b7280;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .llm-add-provider-button:hover {
          border-color: #9ca3af;
          color: #374151;
          background-color: #f3f4f6;
        }

        /* Scrollbar styling */
        .llm-provider-list::-webkit-scrollbar,
        .llm-models-list::-webkit-scrollbar {
          width: 6px;
        }

        .llm-provider-list::-webkit-scrollbar-track,
        .llm-models-list::-webkit-scrollbar-track {
          background: #f1f1f1;
        }

        .llm-provider-list::-webkit-scrollbar-thumb,
        .llm-models-list::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 3px;
        }

        .llm-provider-list::-webkit-scrollbar-thumb:hover,
        .llm-models-list::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
      `}</style>
    </div>
  );
};

export default LLMProviderDropdown;