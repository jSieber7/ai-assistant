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
      className={`chat-history-sidebar ${position}`}
      style={{
        width: isCollapsed ? '50px' : `${width}px`,
        transition: 'width 0.3s ease',
        height: '100vh',
        backgroundColor: '#f8f9fa',
        borderRight: position === 'left' ? '1px solid #e9ecef' : 'none',
        borderLeft: position === 'right' ? '1px solid #e9ecef' : 'none',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        zIndex: 1000
      }}
    >
      {/* Header with toggle button */}
      <div 
        style={{
          padding: '15px',
          borderBottom: '1px solid #e9ecef',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        {!isCollapsed && (
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>
            Chat History
          </h3>
        )}
        <button
          onClick={toggleSidebar}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '5px',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
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
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
          {history.length === 0 ? (
            <div style={{ 
              padding: '20px', 
              textAlign: 'center', 
              color: '#6c757d',
              fontSize: '14px'
            }}>
              No chat history yet
            </div>
          ) : (
            history.map(chat => (
              <div
                key={chat.id}
                style={{
                  padding: '12px 15px',
                  borderBottom: '1px solid #e9ecef',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                  position: 'relative'
                }}
                className="chat-history-item"
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f1f3f4';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                {chat.unread > 0 && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '12px',
                      right: '15px',
                      backgroundColor: '#007bff',
                      color: 'white',
                      borderRadius: '50%',
                      width: '20px',
                      height: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '11px',
                      fontWeight: 'bold'
                    }}
                  >
                    {chat.unread}
                  </div>
                )}
                <div style={{ fontWeight: '500', marginBottom: '4px', fontSize: '14px' }}>
                  {chat.title}
                </div>
                <div style={{ 
                  color: '#6c757d', 
                  fontSize: '12px', 
                  marginBottom: '4px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {chat.lastMessage}
                </div>
                <div style={{ color: '#adb5bd', fontSize: '11px' }}>
                  {formatDate(chat.timestamp)}
                </div>
              </div>
            ))
          )}
        </div>
      )}
      
      {/* Footer with new chat button */}
      {!isCollapsed && (
        <div style={{ padding: '10px', borderTop: '1px solid #e9ecef' }}>
          <button
            style={{
              width: '100%',
              padding: '10px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: '500',
              fontSize: '14px'
            }}
          >
            New Chat
          </button>
        </div>
      )}
      
      <style jsx>{`
        .chat-history-item:hover {
          background-color: #f1f3f4;
        }
      `}</style>
    </div>
  );
};

export default ChatHistorySidebar;