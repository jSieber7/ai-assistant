import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Moon, Sun, MessageSquarePlus, Users } from 'lucide-react';

export default function TopBar() {
  const [selectedProvider, setSelectedProvider] = useState(props.selectedProvider || null);
  const [selectedModel, setSelectedModel] = useState(props.selectedModel || null);
  const [isApiServing, setIsApiServing] = useState(props.isApiServing || false);
  const [outputApiEndpoint, setOutputApiEndpoint] = useState(props.outputApiEndpoint || 'http://localhost:8000/v1');
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    setSelectedProvider(props.selectedProvider || null);
    setSelectedModel(props.selectedModel || null);
    setIsApiServing(props.isApiServing || false);
    setOutputApiEndpoint(props.outputApiEndpoint || 'http://localhost:8000/v1');
    
    // Check for dark mode preference
    const darkModePreference = localStorage.getItem('chainlit-theme') === 'dark' ||
                              (!localStorage.getItem('chainlit-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    setIsDarkMode(darkModePreference);
  }, [props]);

  // Format model display text
  const getModelDisplayText = () => {
    if (selectedProvider && selectedModel) {
      return `${selectedProvider}: ${selectedModel}`;
    }
    return 'No Model Selected';
  };

  // Handle theme toggle
  const handleThemeToggle = () => {
    const newTheme = !isDarkMode ? 'dark' : 'light';
    setIsDarkMode(!isDarkMode);
    localStorage.setItem('chainlit-theme', newTheme);
    
    // Apply theme to document
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  // Handle new chat button click
  const handleNewChat = () => {
    // This would trigger a new chat action
    if (window.chainlit) {
      window.chainlit.sendMessage('/reset');
    }
  };

  // Handle agent button click
  const handleAgentClick = () => {
    // This would trigger a Chainlit action
    if (window.chainlit) {
      window.chainlit.sendMessage('/agent');
    }
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-2 bg-background border-b border-border font-sans text-sm">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-2 py-1 bg-secondary border border-border rounded-md">
          <span className="text-xs">📊 Model: {getModelDisplayText()}</span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground text-xs flex items-center">
          <span className={`inline-block w-2 h-2 rounded-full mr-1 ${isApiServing ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
          OpenAI API: {outputApiEndpoint} ({isApiServing ? 'Serving' : 'Idle'})
        </span>
      </div>
      
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleThemeToggle}
          className="h-8 w-8 p-0"
          title="Toggle theme"
        >
          {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={handleNewChat}
          className="h-8 w-8 p-0"
          title="New chat"
        >
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
        
        <Button
          variant="default"
          size="sm"
          onClick={handleAgentClick}
          className="h-8 px-2 flex items-center gap-1"
          title="Agent Management"
        >
          <Users className="h-4 w-4" />
          <span className="text-xs">Agents</span>
        </Button>
      </div>
    </div>
  );
}