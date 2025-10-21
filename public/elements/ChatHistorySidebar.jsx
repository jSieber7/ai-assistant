import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, MessageSquarePlus } from 'lucide-react';

export default function ChatHistorySidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [chatHistory, setChatHistory] = useState(props.chatHistory || []);
  const [width, setWidth] = useState(props.width || 300);
  const [position, setPosition] = useState(props.position || 'left');
  
  // Mock chat history data if none provided
  const mockChatHistory = [
    {
      id: 1,
      title: "Discussion about AI models",
      timestamp: "2023-10-21 10:30 AM",
      lastMessage: "What are the differences between GPT-3 and GPT-4?",
      unread: 2
    },
    {
      id: 2,
      title: "React component help",
      timestamp: "2023-10-21 09:15 AM",
      lastMessage: "How can I optimize my React app performance?",
      unread: 0
    },
    {
      id: 3,
      title: "Python debugging session",
      timestamp: "2023-10-20 04:45 PM",
      lastMessage: "I'm getting a TypeError in my function",
      unread: 1
    },
    {
      id: 4,
      title: "Database design consultation",
      timestamp: "2023-10-20 02:30 PM",
      lastMessage: "What's the best way to normalize this schema?",
      unread: 0
    },
    {
      id: 5,
      title: "Code review feedback",
      timestamp: "2023-10-19 11:20 AM",
      lastMessage: "Thanks for the suggestions on my PR",
      unread: 0
    }
  ];
  
  // Update state when props change
  React.useEffect(() => {
    setChatHistory(props.chatHistory || []);
    setWidth(props.width || 300);
    setPosition(props.position || 'left');
  }, [props]);
  
  // Dispatch event when component mounts or when position/width changes
  React.useEffect(() => {
    window.dispatchEvent(new CustomEvent('sidebarStateChanged', {
      detail: {
        isCollapsed: isCollapsed,
        width: isCollapsed ? 48 : width,
        position: position
      }
    }));
  }, [isCollapsed, width, position]);
  
  // Use provided chat history or mock data
  const history = chatHistory.length > 0 ? chatHistory : mockChatHistory;
  
  const toggleSidebar = () => {
    const newCollapsedState = !isCollapsed;
    setIsCollapsed(newCollapsedState);
    
    // Dispatch custom event to notify about sidebar state change
    window.dispatchEvent(new CustomEvent('sidebarStateChanged', {
      detail: {
        isCollapsed: newCollapsedState,
        width: newCollapsedState ? 48 : width,
        position: position
      }
    }));
  };
  
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };
  
  const handleNewChat = () => {
    // This would trigger a new chat action
    if (window.chainlit) {
      window.chainlit.sendMessage('/reset');
    }
  };
  
  const handleChatClick = (chatId) => {
    // This would load the selected chat
    console.log(`Loading chat ${chatId}`);
  };

  return (
    <div 
      className={`fixed top-0 ${position === 'left' ? 'left-0' : 'right-0'} h-screen bg-secondary border-border transition-all duration-300 ease-in-out z-40 flex flex-col ${
        isCollapsed ? 'w-12' : ''
      }`}
      style={{ width: isCollapsed ? '48px' : `${width}px` }}
    >
      {/* Header with toggle button */}
      <div className="flex justify-between items-center p-4 border-b border-border">
        {!isCollapsed && (
          <h3 className="m-0 text-base font-semibold">
            Chat History
          </h3>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="p-1 h-8 w-8"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>
      
      {/* Chat history list */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto py-2">
          {history.length === 0 ? (
            <div className="p-5 text-center text-muted-foreground text-sm">
              No chat history yet
            </div>
          ) : (
            history.map(chat => (
              <div
                key={chat.id}
                className="px-4 py-3 border-b border-border cursor-pointer transition-colors duration-200 hover:bg-muted relative"
                onClick={() => handleChatClick(chat.id)}
              >
                {chat.unread > 0 && (
                  <div className="absolute top-3 right-4 bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">
                    {chat.unread}
                  </div>
                )}
                <div className="font-medium mb-1 text-sm">
                  {chat.title}
                </div>
                <div className="text-muted-foreground text-xs mb-1 truncate">
                  {chat.lastMessage}
                </div>
                <div className="text-muted-foreground/70 text-xs">
                  {formatDate(chat.timestamp)}
                </div>
              </div>
            ))
          )}
        </div>
      )}
      
      {/* Footer with new chat button */}
      {!isCollapsed && (
        <div className="p-2 border-t border-border">
          <Button
            variant="default"
            className="w-full justify-center gap-2"
            onClick={handleNewChat}
          >
            <MessageSquarePlus className="h-4 w-4" />
            New Chat
          </Button>
        </div>
      )}
    </div>
  );
}