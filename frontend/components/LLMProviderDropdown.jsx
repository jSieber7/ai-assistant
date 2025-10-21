import React from 'react';

/**
 * LLM Provider Display Component
 *
 * A simple component that displays the currently selected LLM provider and model.
 * It is clickable to trigger a model selection callback.
 *
 * @param {Object} props
 * @param {string} props.selectedProvider - Currently selected provider
 * @param {string} props.selectedModel - Currently selected model
 * @param {Function} props.onModelClick - Callback when model info is clicked
 * @param {string} props.variant - The style variant: 'top' for a top bar, 'status' for a status bar. Default is 'top'.
 */
const LLMProviderDropdown = ({
  selectedProvider = null,
  selectedModel = null,
  onModelClick,
  variant = 'top'
}) => {
  // Format model display text
  const getModelDisplayText = () => {
    if (selectedProvider && selectedModel) {
      return `${selectedProvider}: ${selectedModel}`;
    }
    return 'No Model Selected';
  };

  return (
    <div
      className={`
        flex items-center gap-${variant === 'top' ? '2' : '1.5'}
        px-${variant === 'top' ? '3' : '2'} py-${variant === 'top' ? '1.5' : '1'}
        bg-white border border-gray-200 rounded-${variant === 'top' ? 'md' : 'sm'}
        cursor-pointer transition-all duration-200
        hover:bg-gray-50 hover:border-gray-300
      `}
      onClick={onModelClick}
      title="Click to change model"
    >
      <svg
        className="text-gray-500 flex-shrink-0"
        width={variant === 'top' ? '16' : '14'}
        height={variant === 'top' ? '16' : '14'}
        viewBox="0 0 16 16"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M8 2L3 5V11L8 14L13 11V5L8 2Z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M8 8V14M8 8L3 5M8 8L13 5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className={`
        ${variant === 'top' ? 'text-sm' : 'text-xs'}
        font-medium text-gray-700
      `}>
        {getModelDisplayText()}
      </span>
    </div>
  );
};

export default LLMProviderDropdown;