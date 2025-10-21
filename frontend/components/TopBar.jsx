import React, { useState, useEffect } from 'react';

/**
 * Top Bar Component for Chainlit App
 *
 * A top navigation bar that displays:
 * - Current model and provider information
 * - Agent button for agent management
 * - Current hosting address for OpenAI-compatible API
 * - Green indicator when model is being served via API
 *
 * @param {Object} props
 * @param {string} props.selectedProvider - Currently selected provider
 * @param {string} props.selectedModel - Currently selected model
 * @param {string} props.outputApiEndpoint - Output API endpoint where users connect to use this app
 * @param {boolean} props.isApiServing - Whether the API is currently serving
 * @param {Function} props.onAgentClick - Callback when agent button is clicked
 * @param {Function} props.onModelClick - Callback when model info is clicked
 */
const TopBar = ({
  selectedProvider = null,
  selectedModel = null,
  outputApiEndpoint = null,
  isApiServing = false,
  onAgentClick,
  onModelClick
}) => {
  const [apiStatus, setApiStatus] = useState('unknown');
  const [endpointAddress, setEndpointAddress] = useState('http://localhost:8000/v1');

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
    if (outputApiEndpoint) {
      setEndpointAddress(outputApiEndpoint);
    } else {
      // Default to localhost:8000/v1 for development
      setEndpointAddress(`http://${window.location.hostname}:8000/v1`);
    }
  }, [outputApiEndpoint]);

  // Format model display text
  const getModelDisplayText = () => {
    if (selectedProvider && selectedModel) {
      return `${selectedProvider}: ${selectedModel}`;
    }
    return 'No Model Selected';
  };

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        {/* Model Information */}
        <div 
          className="model-info"
          onClick={onModelClick}
          title="Click to change model"
        >
          <svg
            className="model-icon"
            width="16"
            height="16"
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

      <div className="top-bar-center">
        {/* API Status and Host Address */}
        <div className="api-status">
          <div className={`status-indicator ${apiStatus}`}></div>
          <span className="api-text">
            OpenAI API:
            <span className="host-address">{endpointAddress}</span>
          </span>
        </div>
      </div>

      <div className="top-bar-right">
        {/* Agent Button */}
        <button
          className="agent-button"
          onClick={onAgentClick}
          title="Agent Management"
        >
          <svg
            width="16"
            height="16"
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
        .top-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          background-color: #ffffff;
          border-bottom: 1px solid #e5e7eb;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
          position: sticky;
          top: 0;
          z-index: 1000;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .top-bar-left,
        .top-bar-center,
        .top-bar-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .model-info {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          background-color: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 6px;
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
          font-size: 14px;
          font-weight: 500;
          color: #374151;
        }

        .api-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
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
          font-size: 13px;
          color: #6b7280;
        }

        .host-address {
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
          font-weight: 500;
          color: #374151;
        }

        .agent-button {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 12px;
          background-color: #3b82f6;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
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
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .top-bar {
            padding: 10px 12px;
          }

          .api-text {
            display: none;
          }

          .host-address {
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

export default TopBar;