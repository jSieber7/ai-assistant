import React, { useState, useEffect } from 'react';
import ChatHistorySidebar from './ChatHistorySidebar';

/**
 * Layout Wrapper Component for Chainlit App
 *
 * This component manages the layout with a sidebar and main content area.
 * It ensures that when the sidebar is expanded, the main content is pushed aside.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children - The main content to display
 * @param {number} props.sidebarWidth - Width of the sidebar when expanded
 * @param {string} props.sidebarPosition - Position of the sidebar ('left' or 'right')
 * @param {Array} props.chatHistory - Chat history data for the sidebar
 */
const LayoutWrapper = ({
  children,
  sidebarWidth = 300,
  sidebarPosition = 'left',
  chatHistory = []
}) => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [sidebarStyle, setSidebarStyle] = useState({});

  // Update sidebar style when collapsed state changes
  useEffect(() => {
    const width = isSidebarCollapsed ? 50 : sidebarWidth;
    setSidebarStyle({ width: `${width}px` });
    
    // Store sidebar state in window for other components to access
    window.sidebarState = {
      isCollapsed: isSidebarCollapsed,
      width: width,
      position: sidebarPosition
    };
    
    // Dispatch custom event to notify other components of sidebar state change
    window.dispatchEvent(new CustomEvent('sidebarStateChanged', {
      detail: { isCollapsed: isSidebarCollapsed, width, position: sidebarPosition }
    }));
  }, [isSidebarCollapsed, sidebarWidth, sidebarPosition]);

  // Calculate main content margin based on sidebar position and state
  const getMainContentStyle = () => {
    if (isSidebarCollapsed) {
      return sidebarPosition === 'left' 
        ? { marginLeft: '50px' } 
        : { marginRight: '50px' };
    } else {
      return sidebarPosition === 'left' 
        ? { marginLeft: `${sidebarWidth}px` } 
        : { marginRight: `${sidebarWidth}px` };
    }
  };

  const handleSidebarToggle = (collapsed) => {
    setIsSidebarCollapsed(collapsed);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div 
        className={`relative z-50 h-screen bg-gray-50 transition-all duration-300 ${
          sidebarPosition === 'left' ? 'border-r border-gray-200' : 'border-l border-gray-200'
        }`}
        style={sidebarStyle}
      >
        <ChatHistorySidebar
          chatHistory={chatHistory}
          width={sidebarWidth}
          position={sidebarPosition}
          onToggle={handleSidebarToggle}
        />
      </div>
      
      {/* Main Content */}
      <div 
        className="flex-1 overflow-hidden transition-all duration-300"
        style={getMainContentStyle()}
      >
        {children}
      </div>
    </div>
  );
};

export default LayoutWrapper;