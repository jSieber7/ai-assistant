import React, { useEffect } from 'react';

export default function SidebarLayoutAdjuster() {
  useEffect(() => {
    // Function to adjust the main content margin based on sidebar state
    const adjustMainContentMargin = (event) => {
      const { isCollapsed, width, position } = event.detail;
      
      // Find the main content area
      const mainContentElements = document.querySelectorAll('.overflow-y-auto');
      const chainlitAppContainer = document.querySelector('#chainlit-app');
      const chatContainer = document.querySelector('.flex.flex-col.h-full');
      
      // Calculate the margin value
      const marginValue = isCollapsed ? '48px' : `${width}px`;
      
      // Apply margin to main content elements
      mainContentElements.forEach(element => {
        if (position === 'left') {
          element.style.marginLeft = marginValue;
          element.style.width = `calc(100% - ${marginValue})`;
        } else {
          element.style.marginRight = marginValue;
          element.style.width = `calc(100% - ${marginValue})`;
        }
      });
      
      // Apply margin to chainlit app container if found
      if (chainlitAppContainer) {
        if (position === 'left') {
          chainlitAppContainer.style.marginLeft = marginValue;
          chainlitAppContainer.style.width = `calc(100% - ${marginValue})`;
        } else {
          chainlitAppContainer.style.marginRight = marginValue;
          chainlitAppContainer.style.width = `calc(100% - ${marginValue})`;
        }
      }
      
      // Apply margin to chat container if found
      if (chatContainer) {
        if (position === 'left') {
          chatContainer.style.marginLeft = marginValue;
          chatContainer.style.width = `calc(100% - ${marginValue})`;
        } else {
          chatContainer.style.marginRight = marginValue;
          chatContainer.style.width = `calc(100% - ${marginValue})`;
        }
      }
      
      // Also adjust any message containers
      const messageContainers = document.querySelectorAll('[class*="message"]');
      messageContainers.forEach(element => {
        if (position === 'left') {
          element.style.marginLeft = marginValue;
          element.style.width = `calc(100% - ${marginValue})`;
        } else {
          element.style.marginRight = marginValue;
          element.style.width = `calc(100% - ${marginValue})`;
        }
      });
    };
    
    // Listen for sidebar state changes
    window.addEventListener('sidebarStateChanged', adjustMainContentMargin);
    
    // Initial adjustment in case sidebar is already rendered
    setTimeout(() => {
      // Try to get initial sidebar state
      const sidebarElement = document.querySelector('[class*="fixed"][class*="h-screen"]');
      if (sidebarElement) {
        const width = sidebarElement.style.width || '300px';
        const isLeft = sidebarElement.classList.contains('left-0');
        const isCollapsed = width === '48px';
        
        adjustMainContentMargin({
          detail: {
            isCollapsed,
            width: parseInt(width),
            position: isLeft ? 'left' : 'right'
          }
        });
      }
    }, 500);
    
    // Clean up event listener
    return () => {
      window.removeEventListener('sidebarStateChanged', adjustMainContentMargin);
    };
  }, []);
  
  // This component doesn't render anything visible
  return null;
}