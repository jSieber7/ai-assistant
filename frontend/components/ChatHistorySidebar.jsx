import React, { useState } from 'react';

const ChatHistorySidebar = ({ chatHistory = [], width = 300, position = 'left' }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
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
  
  // Use provided chat history or mock data
  const history = chatHistory.length > 0 ? chatHistory : mockChatHistory;
  
  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
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
  
  return (
    <div
      className={`
        flex flex-col relative z-50 h-screen bg-gray-50 transition-all duration-300
        ${position === 'left' ? 'border-r border-gray-200' : ''}
        ${position === 'right' ? 'border-l border-gray-200' : ''}
      `}
      style={{ width: isCollapsed ? '50px' : `${width}px` }}
    >
      {/* Header with toggle button */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        {!isCollapsed && (
          <h3 className="m-0 text-base font-semibold">
            Chat History
          </h3>
        )}
        <button
          className="flex items-center justify-center p-1.5 rounded cursor-pointer border-0 bg-transparent hover:bg-gray-100"
          onClick={toggleSidebar}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          )}
        </button>
      </div>
      
      {/* Chat history list */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto py-2.5">
          {history.length === 0 ? (
            <div className="p-5 text-center text-gray-500 text-sm">
              No chat history yet
            </div>
          ) : (
            history.map(chat => (
              <div
                key={chat.id}
                className="relative px-4 py-3 border-b border-gray-200 cursor-pointer transition-colors duration-200 hover:bg-gray-100"
              >
                {chat.unread > 0 && (
                  <div className="absolute top-3 right-4 flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-blue-500 rounded-full">
                    {chat.unread}
                  </div>
                )}
                <div className="font-medium mb-1 text-sm">
                  {chat.title}
                </div>
                <div className="text-gray-500 text-xs mb-1 truncate">
                  {chat.lastMessage}
                </div>
                <div className="text-gray-400 text-xs">
                  {formatDate(chat.timestamp)}
                </div>
              </div>
            ))
          )}
        </div>
      )}
      
      {/* Footer with new chat button */}
      {!isCollapsed && (
        <div className="p-2.5 border-t border-gray-200">
          <button className="w-full py-2.5 text-sm font-medium text-white bg-blue-500 border-0 rounded cursor-pointer hover:bg-blue-600">
            New Chat
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatHistorySidebar;