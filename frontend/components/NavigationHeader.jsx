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

  return (
    <div className={`
      flex justify-between items-center
      font-sans border-b border-gray-200
      ${variant === 'top'
        ? 'px-4 py-3 bg-white sticky top-0 z-50 shadow-sm -mx-4 -mt-4 mb-4'
        : 'px-4 py-2 bg-gray-50 text-sm -mx-4 mb-4'
      }
    `}>
      <div className="flex items-center gap-3">
      </div>

      <div className="flex items-center gap-3">
        {/* API Status and Host Address */}
        <div className="flex items-center gap-2">
          <div className={`
            ${variant === 'top' ? 'w-2 h-2' : 'w-1.5 h-1.5'}
            rounded-full bg-gray-400 transition-colors duration-300
            ${apiStatus === 'serving' ? 'bg-green-500 shadow-[0_0_0_2px_rgba(16,185,129,0.2)]' : ''}
            ${apiStatus === 'idle' ? 'bg-amber-500' : ''}
          `}></div>
          <span className={`
            ${variant === 'top' ? 'text-xs' : 'text-xs'}
            text-gray-600
          `}>
            {variant === 'top' ? 'OpenAI API:' : 'API:'}
            <span className="font-medium text-gray-800 font-mono">{endpointAddress}</span>
            {variant === 'status' && <span className="italic">({apiStatus === 'serving' ? 'Serving' : 'Idle'})</span>}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
      </div>

      {/* Responsive styles */}
      <style jsx>{`
        @media (max-width: 768px) {
          .api-text {
            display: ${variant === 'top' ? 'none' : 'flex'};
          }

          .host-address {
            display: ${variant === 'top' ? 'none' : 'flex'};
          }
          
          .status-text {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};

export default NavigationHeader;