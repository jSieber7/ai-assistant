import React, { useState, useEffect } from 'react';

/**
 * Navigation Header Component for Chainlit App
 *
 * A flexible navigation header that can act as either a top bar or a status bar.
 * It displays:
 * - Current model and provider information
 * - Agent button for agent management
 * - Current hosting address for OpenAI-compatible API
 * - Indicator for API serving status
 *
 * @param {Object} props
 * @param {string} props.variant - The style variant: 'top' for a sticky top bar, 'status' for a non-sticky status bar. Default is 'top'.
 * @param {string} props.selectedProvider - Currently selected provider
 * @param {string} props.selectedModel - Currently selected model
 * @param {string} props.apiHost - API host address (e.g., 'localhost:8000'). For 'top' variant, '/v1' is appended.
 * @param {boolean} props.isApiServing - Whether the API is currently serving
 * @param {Function} props.onAgentClick - Callback when agent button is clicked
 * @param {Function} props.onModelClick - Callback when model info is clicked
 */
const NavigationHeader = ({
  variant = 'top',
  selectedProvider = null,
  selectedModel = null,
  apiHost = null,
  isApiServing = false,
  onAgentClick,
  onModelClick
}) => {
  const [apiStatus, setApiStatus] = useState('unknown');
  const [endpointAddress, setEndpointAddress] = useState('');

  // Update API status based on props
  useEffect(() => {
    if (isApiServing) {
      setApiStatus('serving');
    } else {
      setApiStatus('idle');
    }
  }, [isApiServing]);

  // Update endpoint address based on props or default
  useEffect(() => {
    const host = apiHost || `${window.location.hostname}:8000`;
    const endpoint = variant === 'top' ? `http://${host}/v1` : `http://${host}`;
    setEndpointAddress(endpoint);
  }, [apiHost, variant]);

  // Format model display text
  const getModelDisplayText = () => {
    if (selectedProvider && selectedModel) {
      return `${selectedProvider}: ${selectedModel}`;
    }
    return 'No Model Selected';
  };

  // Base styles for both variants
  const baseStyles = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    border-bottom: 1px solid #e5e7eb;
  `;

  // Variant-specific styles
  const variantStyles = variant === 'top' ? `
    padding: 12px 16px;
    background-color: #ffffff;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin: -16px -16px 16px -16px;
  ` : `
    padding: 8px 16px;
    background-color: #f8f9fa;
    font-size: 13px;
    margin: 0 -16px 16px -16px;
  `;

  return (
    <div className={`navigation-header navigation-header--${variant}`}>
      <div className="navigation-header-left">
        {/* Model Information */}
        <div 
          className="model-info"
          onClick={onModelClick}
          title="Click to change model"
        >
          <svg
            className="model-icon"
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
          <span className="model-text">{getModelDisplayText()}</span>
        </div>
      </div>

      <div className="navigation-header-center">
        {/* API Status and Host Address */}
        <div className="api-status">
          <div className={`status-indicator ${apiStatus}`}></div>
          <span className="api-text">
            {variant === 'top' ? 'OpenAI API:' : 'API:'}
            <span className="host-address">{endpointAddress}</span>
            {variant === 'status' && <span className="status-text">({apiStatus === 'serving' ? 'Serving' : 'Idle'})</span>}
          </span>
        </div>
      </div>

      <div className="navigation-header-right">
        {/* Agent Button */}
        <button
          className="agent-button"
          onClick={onAgentClick}
          title="Agent Management"
        >
          <svg
            width={variant === 'top' ? '16' : '14'}
            height={variant === 'top' ? '16' : '14'}
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 8C10.21 8 12 6.21 12 4C12 1.79 10.21 0 8 0C5.79 0 4 1.79 4 4C4 6.21 5.79 8 8 8Z"
              fill="currentColor"
            />
            <path
              d="M12.5 9H11.71C11.11 9.64 10.36 10 9.5 10H6.5C5.64 10 4.89 9.64 4.29 9H3.5C1.57 9 0 10.57 0 12.5V14C0 15.1 0.9 16 2 16H14C15.1 16 16 15.1 16 14V12.5C16 10.57 14.43 9 12.5 9Z"
              fill="currentColor"
            />
          </svg>
          <span className="agent-text">Agents</span>
        </button>
      </div>

      <style jsx>{`
        .navigation-header {
          ${baseStyles}
          ${variantStyles}
        }

        .navigation-header-left,
        .navigation-header-center,
        .navigation-header-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .model-info {
          display: flex;
          align-items: center;
          gap: ${variant === 'top' ? '8px' : '6px'};
          padding: ${variant === 'top' ? '6px 12px' : '4px 8px'};
          background-color: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: ${variant === 'top' ? '6px' : '4px'};
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .model-info:hover {
          background-color: #f3f4f6;
          border-color: #d1d5db;
        }

        .model-icon {
          color: #6b7280;
          flex-shrink: 0;
        }

        .model-text {
          font-size: ${variant === 'top' ? '14px' : '12px'};
          font-weight: 500;
          color: #374151;
        }

        .api-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-indicator {
          width: ${variant === 'top' ? '8px' : '7px'};
          height: ${variant === 'top' ? '8px' : '7px'};
          border-radius: 50%;
          background-color: #9ca3af;
          transition: background-color 0.3s ease;
        }

        .status-indicator.serving {
          background-color: #10b981;
          box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        .status-indicator.idle {
          background-color: #f59e0b;
        }

        .api-text {
          font-size: ${variant === 'top' ? '13px' : '12px'};
          color: #6b7280;
        }

        .host-address {
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
          font-weight: 500;
          color: #374151;
        }
        
        .status-text {
            font-style: italic;
        }

        .agent-button {
          display: flex;
          align-items: center;
          gap: ${variant === 'top' ? '6px' : '4px'};
          padding: ${variant === 'top' ? '8px 12px' : '6px 10px'};
          background-color: #3b82f6;
          color: white;
          border: none;
          border-radius: ${variant === 'top' ? '6px' : '4px'};
          cursor: pointer;
          font-size: ${variant === 'top' ? '14px' : '12px'};
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .agent-button:hover {
          background-color: #2563eb;
        }

        .agent-button:focus {
          outline: none;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .agent-text {
          font-size: ${variant === 'top' ? '14px' : '12px'};
        }

        @media (max-width: 768px) {
          .navigation-header {
            padding: ${variant === 'top' ? '10px 12px' : '6px 12px'};
          }

          .api-text {
            display: ${variant === 'top' ? 'none' : 'flex'};
          }

          .host-address {
            display: ${variant === 'top' ? 'none' : 'flex'};
          }
          
          .status-text {
            display: none;
          }

          .agent-text {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};

export default NavigationHeader;