import React, { useState } from 'react';
import TopBar from './TopBar';

/**
 * Example usage of the TopBar component
 * 
 * This example shows how to integrate the TopBar component with state management
 * for the model selection, API status, and agent functionality.
 */
const TopBarExample = () => {
  // State management for the top bar
  const [selectedProvider, setSelectedProvider] = useState('openai_compatible');
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [apiHost, setApiHost] = useState('localhost:8000');
  const [isApiServing, setIsApiServing] = useState(true);

  // Handle agent button click
  const handleAgentClick = () => {
    // Implement agent management functionality
    alert('Agent management panel would open here');
  };

  // Handle model info click
  const handleModelClick = () => {
    // Implement model selection functionality
    alert('Model selection dialog would open here');
  };

  return (
    <div>
      <TopBar
        selectedProvider={selectedProvider}
        selectedModel={selectedModel}
        apiHost={apiHost}
        isApiServing={isApiServing}
        onAgentClick={handleAgentClick}
        onModelClick={handleModelClick}
      />
      
      {/* Main content would go here */}
      <div style={{ padding: '20px' }}>
        <h1>Chainlit App</h1>
        <p>This is where your main application content would appear.</p>
      </div>
    </div>
  );
};

export default TopBarExample;