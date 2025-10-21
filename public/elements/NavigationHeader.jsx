import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Users } from 'lucide-react';

export default function NavigationHeader() {
  const [selectedProvider, setSelectedProvider] = useState(props.selectedProvider || null);
  const [selectedModel, setSelectedModel] = useState(props.selectedModel || null);
  const [apiHost, setApiHost] = useState(props.apiHost || null);
  const [isApiServing, setIsApiServing] = useState(props.isApiServing || false);
  const [variant, setVariant] = useState(props.variant || 'top');
  const [endpointAddress, setEndpointAddress] = useState('');

  useEffect(() => {
    setSelectedProvider(props.selectedProvider || null);
    setSelectedModel(props.selectedModel || null);
    setApiHost(props.apiHost || null);
    setIsApiServing(props.isApiServing || false);
    setVariant(props.variant || 'top');
    
    // Update endpoint address based on props or default
    const host = apiHost || `${window.location.hostname}:8000`;
    const endpoint = variant === 'top' ? `http://${host}/v1` : `http://${host}`;
    setEndpointAddress(endpoint);
  }, [props, apiHost, variant]);

  // Format model display text
  const getModelDisplayText = () => {
    if (selectedProvider && selectedModel) {
      return `${selectedProvider}: ${selectedModel}`;
    }
    return 'No Model Selected';
  };

  // Handle agent button click
  const handleAgentClick = () => {
    // This would trigger a Chainlit action
    if (window.chainlit) {
      window.chainlit.sendMessage('/agent');
    }
    
    // Also call the callback if provided
    if (props.onAgentClick) {
      props.onAgentClick();
    }
  };

  // Handle model info click
  const handleModelClick = () => {
    // This would trigger a model selection action
    if (window.chainlit) {
      window.chainlit.sendMessage('/settings');
    }
    
    // Also call the callback if provided
    if (props.onModelClick) {
      props.onModelClick();
    }
  };

  // Determine styles based on variant
  const containerClasses = variant === 'top' 
    ? "fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-3 bg-background border-b border-border font-sans text-sm shadow-sm"
    : "flex justify-between items-center p-2 bg-secondary border-b border-border font-sans text-xs";
  
  const modelInfoClasses = variant === 'top'
    ? "flex items-center gap-2 px-3 py-2 bg-secondary border border-border rounded-md cursor-pointer hover:bg-muted transition-colors"
    : "flex items-center gap-2 px-2 py-1 bg-background border border-border rounded cursor-pointer hover:bg-muted transition-colors";
  
  const statusClasses = variant === 'top'
    ? "text-muted-foreground text-sm flex items-center"
    : "text-muted-foreground text-xs flex items-center";
  
  const statusIndicatorClasses = variant === 'top'
    ? "inline-block w-2 h-2 rounded-full mr-2"
    : "inline-block w-1.5 h-1.5 rounded-full mr-1";
  
  const agentButtonClasses = variant === 'top'
    ? "h-8 px-3 flex items-center gap-2 text-sm"
    : "h-6 px-2 flex items-center gap-1 text-xs";

  return (
    <div className={containerClasses}>
      <div className="flex items-center gap-3">
        <div className={modelInfoClasses} onClick={handleModelClick} title="Click to change model">
          <span className="text-xs">📊 Model: {getModelDisplayText()}</span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        <span className={statusClasses}>
          <span className={`${statusIndicatorClasses} ${isApiServing ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
          {variant === 'top' ? 'OpenAI API:' : 'API:'}
          <span className="font-mono font-medium">{endpointAddress}</span>
          {variant === 'status' && <span className="italic">({isApiServing ? 'Serving' : 'Idle'})</span>}
        </span>
      </div>
      
      <div className="flex items-center gap-2">
        <Button
          variant="default"
          size="sm"
          onClick={handleAgentClick}
          className={agentButtonClasses}
          title="Agent Management"
        >
          <Users className="h-4 w-4" />
          <span>Agents</span>
        </Button>
      </div>
    </div>
  );
}