import React, { useState, useEffect, useRef } from 'react';

/**
 * Tools Toggle Dropdown Component
 * 
 * A dynamic dropdown menu for enabling/disabling tools organized by categories.
 * Features:
 * - Lists tools organized by categories
 * - Toggle switches to enable/disable individual tools
 * - Search functionality for tools
 * - Category-level enable/disable
 * - Visual indicators for tool status
 * 
 * @param {Object} props
 * @param {Array} props.tools - Array of tool objects organized by categories
 * @param {Function} props.onToolToggle - Callback when a tool is toggled
 * @param {Function} props.onCategoryToggle - Callback when a category is toggled
 * @param {Set} props.enabledTools - Set of currently enabled tool IDs
 */
const ToolsToggleDropdown = ({
  tools = [],
  onToolToggle,
  onCategoryToggle,
  enabledTools = new Set()
}) => {
  // State management
  const [isOpen, setIsOpen] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState(null);
  
  // Refs for dropdown and click outside detection
  const dropdownRef = useRef(null);
  const searchInputRef = useRef(null);

  // Mock data for demonstration (replace with actual data from API)
  const mockTools = [
    {
      id: 'web',
      name: 'Web Tools',
      description: 'Tools for web scraping and searching',
      tools: [
        {
          id: 'firecrawl_scrape',
          name: 'Firecrawl Scraper',
          description: 'Scrape web content using Firecrawl API',
          keywords: ['scrape', 'web', 'firecrawl', 'crawl', 'extract'],
          enabled: true
        },
        {
          id: 'searxng_search',
          name: 'SearXNG Search',
          description: 'Search the web using SearXNG privacy-respecting search engine',
          keywords: ['search', 'web', 'internet', 'google', 'find'],
          enabled: true
        },
        {
          id: 'playwright_tool',
          name: 'Playwright Browser',
          description: 'Automate browser interactions with Playwright',
          keywords: ['browser', 'automation', 'playwright', 'web'],
          enabled: false
        }
      ]
    },
    {
      id: 'utility',
      name: 'Utility Tools',
      description: 'General purpose utility tools',
      tools: [
        {
          id: 'calculator',
          name: 'Calculator',
          description: 'Perform mathematical calculations',
          keywords: ['calculate', 'math', 'equation', 'convert'],
          enabled: true
        },
        {
          id: 'time',
          name: 'Time & Date',
          description: 'Get current time and date information',
          keywords: ['time', 'date', 'now', 'current', 'timezone'],
          enabled: true
        },
        {
          id: 'echo',
          name: 'Echo',
          description: 'Echo back the input text (for testing)',
          keywords: ['echo', 'repeat', 'test'],
          enabled: false
        }
      ]
    },
    {
      id: 'content',
      name: 'Content Tools',
      description: 'Tools for content processing and analysis',
      tools: [
        {
          id: 'jina_reranker',
          name: 'Jina Reranker',
          description: 'Rerank and reorder search results',
          keywords: ['rerank', 'reorder', 'rank', 'search'],
          enabled: true
        }
      ]
    },
    {
      id: 'visual',
      name: 'Visual Tools',
      description: 'Tools for image and visual processing',
      tools: [
        {
          id: 'visual_analyzer',
          name: 'Visual Analyzer',
          description: 'Analyze images and visual content',
          keywords: ['image', 'visual', 'analyze', 'picture'],
          enabled: false
        },
        {
          id: 'visual_browser',
          name: 'Visual Browser',
          description: 'Browser with visual analysis capabilities',
          keywords: ['browser', 'visual', 'screenshot', 'web'],
          enabled: false
        }
      ]
    }
  ];

  // Use provided tools or mock data
  const toolsData = tools.length > 0 ? tools : mockTools;

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Focus search input when category is expanded
  useEffect(() => {
    if (activeCategory && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [activeCategory]);

  // Toggle dropdown
  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setSearchTerm('');
      setActiveCategory(null);
    }
  };

  // Toggle category expansion
  const toggleCategory = (categoryId) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(categoryId)) {
      newExpanded.delete(categoryId);
      if (activeCategory === categoryId) {
        setActiveCategory(null);
      }
    } else {
      newExpanded.add(categoryId);
      setActiveCategory(categoryId);
    }
    setExpandedCategories(newExpanded);
  };

  // Handle tool toggle
  const handleToolToggle = (toolId, categoryId) => {
    if (onToolToggle) {
      onToolToggle(toolId, !enabledTools.has(toolId));
    }
  };

  // Handle category toggle (enable/disable all tools in category)
  const handleCategoryToggle = (categoryId) => {
    const category = toolsData.find(c => c.id === categoryId);
    if (!category) return;

    const allToolsEnabled = category.tools.every(tool => enabledTools.has(tool.id));
    const newEnabledState = !allToolsEnabled;

    if (onCategoryToggle) {
      onCategoryToggle(categoryId, category.tools.map(t => t.id), newEnabledState);
    }
  };

  // Filter tools based on search
  const getFilteredTools = (tools) => {
    if (!searchTerm.trim()) return tools;
    
    const searchLower = searchTerm.toLowerCase();
    return tools.filter(tool => 
      tool.name.toLowerCase().includes(searchLower) ||
      tool.description?.toLowerCase().includes(searchLower) ||
      tool.keywords?.some(keyword => keyword.toLowerCase().includes(searchLower))
    );
  };

  // Get count of enabled tools
  const getEnabledCount = (tools) => {
    return tools.filter(tool => enabledTools.has(tool.id)).length;
  };

  // Get count of enabled tools for all categories
  const getTotalEnabledCount = () => {
    let count = 0;
    toolsData.forEach(category => {
      count += getEnabledCount(category.tools);
    });
    return count;
  };

  // Get total tools count
  const getTotalToolsCount = () => {
    return toolsData.reduce((total, category) => total + category.tools.length, 0);
  };

  return (
    <div className="tools-toggle-dropdown" ref={dropdownRef}>
      {/* Main button */}
      <button
        className="tools-dropdown-button"
        onClick={toggleDropdown}
        aria-expanded={isOpen}
        aria-haspopup="menu"
      >
        <span className="tools-dropdown-text">
          Tools ({getTotalEnabledCount()}/{getTotalToolsCount()})
        </span>
        <svg
          className={`tools-dropdown-arrow ${isOpen ? 'open' : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M2 4L6 8L10 4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="tools-dropdown-menu">
          <div className="tools-dropdown-content">
            {/* Search input */}
            <div className="tools-search-container">
              <input
                ref={searchInputRef}
                type="text"
                className="tools-search-input"
                placeholder="Search tools..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <svg
                className="tools-search-icon"
                width="14"
                height="14"
                viewBox="0 0 14 14"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6.33333 10.6667C8.91885 10.6667 11.0167 8.56885 11.0167 5.98333C11.0167 3.39781 8.91885 1.3 6.33333 1.3C3.74781 1.3 1.65 3.39781 1.65 5.98333C1.65 8.56885 3.74781 10.6667 6.33333 10.6667Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M9.66667 9.66667L12.3333 12.3333"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            {/* Categories list */}
            <div className="tools-categories-list">
              {toolsData.map((category) => {
                const filteredTools = getFilteredTools(category.tools);
                const enabledCount = getEnabledCount(filteredTools);
                const allToolsEnabled = filteredTools.length > 0 && 
                  filteredTools.every(tool => enabledTools.has(tool.id));
                
                // Skip categories with no matching tools when searching
                if (searchTerm && filteredTools.length === 0) return null;

                return (
                  <div key={category.id} className="tools-category-item">
                    {/* Category header */}
                    <div
                      className={`tools-category-header ${
                        expandedCategories.has(category.id) ? 'expanded' : ''
                      }`}
                      onClick={() => toggleCategory(category.id)}
                    >
                      <div className="tools-category-info">
                        <span className="tools-category-name">{category.name}</span>
                        <span className="tools-category-count">
                          {enabledCount}/{filteredTools.length}
                        </span>
                      </div>
                      <div className="tools-category-controls">
                        {/* Category toggle */}
                        <button
                          className="tools-category-toggle"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCategoryToggle(category.id);
                          }}
                          title={allToolsEnabled ? "Disable all tools" : "Enable all tools"}
                        >
                          {allToolsEnabled ? (
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                              <rect x="2" y="2" width="12" height="12" rx="2" fill="currentColor" />
                            </svg>
                          ) : (
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                              <rect x="2" y="2" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
                            </svg>
                          )}
                        </button>
                        
                        {/* Expand/collapse icon */}
                        <svg
                          className="tools-expand-icon"
                          width="12"
                          height="12"
                          viewBox="0 0 12 12"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M4 5L6 7L8 5"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </div>
                    </div>

                    {/* Tools list (expanded) */}
                    {expandedCategories.has(category.id) && (
                      <div className="tools-list-container">
                        {filteredTools.map((tool) => (
                          <div
                            key={tool.id}
                            className={`tools-tool-item ${
                              enabledTools.has(tool.id) ? 'enabled' : 'disabled'
                            }`}
                          >
                            <div className="tools-tool-info">
                              <div className="tools-tool-name">{tool.name}</div>
                              {tool.description && (
                                <div className="tools-tool-description">{tool.description}</div>
                              )}
                              {tool.keywords && (
                                <div className="tools-tool-keywords">
                                  {tool.keywords.slice(0, 3).map((keyword, index) => (
                                    <span key={index} className="tools-keyword-tag">
                                      {keyword}
                                    </span>
                                  ))}
                                  {tool.keywords.length > 3 && (
                                    <span className="tools-keyword-more">
                                      +{tool.keywords.length - 3}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                            
                            {/* Tool toggle switch */}
                            <button
                              className={`tools-toggle-switch ${
                                enabledTools.has(tool.id) ? 'enabled' : 'disabled'
                              }`}
                              onClick={() => handleToolToggle(tool.id, category.id)}
                              aria-label={`Toggle ${tool.name}`}
                            >
                              <span className="tools-toggle-slider"></span>
                            </button>
                          </div>
                        ))}
                        
                        {filteredTools.length === 0 && (
                          <div className="tools-no-tools">No tools found</div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Footer with summary */}
            <div className="tools-dropdown-footer">
              <div className="tools-summary">
                <span>{getTotalEnabledCount()} of {getTotalToolsCount()} tools enabled</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .tools-toggle-dropdown {
          position: relative;
          display: inline-block;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        .tools-dropdown-button {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 16px;
          background-color: #ffffff;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          color: #374151;
          min-width: 200px;
          transition: all 0.2s ease;
        }

        .tools-dropdown-button:hover {
          border-color: #9ca3af;
          box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }

        .tools-dropdown-button:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .tools-dropdown-text {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .tools-dropdown-arrow {
          transition: transform 0.2s ease;
          color: #6b7280;
        }

        .tools-dropdown-arrow.open {
          transform: rotate(180deg);
        }

        .tools-dropdown-menu {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          z-index: 1000;
          margin-top: 4px;
          background-color: #ffffff;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
          max-height: 400px;
          overflow: hidden;
          width: 350px;
        }

        .tools-dropdown-content {
          display: flex;
          flex-direction: column;
        }

        .tools-search-container {
          position: relative;
          padding: 12px;
          border-bottom: 1px solid #e5e7eb;
        }

        .tools-search-input {
          width: 100%;
          padding: 8px 12px 8px 32px;
          border: 1px solid #d1d5db;
          border-radius: 4px;
          font-size: 13px;
          background-color: #ffffff;
        }

        .tools-search-input:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .tools-search-icon {
          position: absolute;
          left: 28px;
          top: 50%;
          transform: translateY(-50%);
          color: #9ca3af;
          pointer-events: none;
        }

        .tools-categories-list {
          max-height: 300px;
          overflow-y: auto;
        }

        .tools-category-item {
          border-bottom: 1px solid #f3f4f6;
        }

        .tools-category-item:last-child {
          border-bottom: none;
        }

        .tools-category-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          cursor: pointer;
          transition: background-color 0.2s ease;
          background-color: #f9fafb;
        }

        .tools-category-header:hover {
          background-color: #f3f4f6;
        }

        .tools-category-header.expanded {
          background-color: #f3f4f6;
        }

        .tools-category-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .tools-category-name {
          font-size: 14px;
          font-weight: 500;
          color: #374151;
        }

        .tools-category-count {
          font-size: 12px;
          color: #6b7280;
        }

        .tools-category-controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .tools-category-toggle {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 20px;
          height: 20px;
          border: none;
          background: none;
          cursor: pointer;
          color: #6b7280;
          border-radius: 3px;
          transition: all 0.2s ease;
        }

        .tools-category-toggle:hover {
          background-color: #e5e7eb;
        }

        .tools-expand-icon {
          transition: transform 0.2s ease;
          color: #6b7280;
        }

        .tools-category-header.expanded .tools-expand-icon {
          transform: rotate(180deg);
        }

        .tools-list-container {
          background-color: #ffffff;
          border-top: 1px solid #e5e7eb;
        }

        .tools-tool-item {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          padding: 12px 16px;
          border-bottom: 1px solid #f3f4f6;
          transition: background-color 0.2s ease;
        }

        .tools-tool-item:last-child {
          border-bottom: none;
        }

        .tools-tool-item:hover {
          background-color: #f9fafb;
        }

        .tools-tool-item.disabled {
          opacity: 0.6;
        }

        .tools-tool-info {
          flex: 1;
          margin-right: 12px;
        }

        .tools-tool-name {
          font-size: 13px;
          font-weight: 500;
          margin-bottom: 2px;
          color: #374151;
        }

        .tools-tool-description {
          font-size: 12px;
          color: #6b7280;
          margin-bottom: 4px;
          line-height: 1.4;
        }

        .tools-tool-keywords {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }

        .tools-keyword-tag {
          font-size: 10px;
          padding: 2px 6px;
          background-color: #e5e7eb;
          color: #6b7280;
          border-radius: 3px;
        }

        .tools-keyword-more {
          font-size: 10px;
          padding: 2px 6px;
          background-color: #f3f4f6;
          color: #9ca3af;
          border-radius: 3px;
        }

        .tools-toggle-switch {
          position: relative;
          display: inline-block;
          width: 36px;
          height: 20px;
          border: none;
          background: none;
          cursor: pointer;
          padding: 0;
        }

        .tools-toggle-slider {
          position: absolute;
          cursor: pointer;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: #d1d5db;
          transition: 0.2s;
          border-radius: 20px;
        }

        .tools-toggle-slider:before {
          position: absolute;
          content: "";
          height: 16px;
          width: 16px;
          left: 2px;
          bottom: 2px;
          background-color: white;
          transition: 0.2s;
          border-radius: 50%;
        }

        .tools-toggle-switch.enabled .tools-toggle-slider {
          background-color: #3b82f6;
        }

        .tools-toggle-switch.enabled .tools-toggle-slider:before {
          transform: translateX(16px);
        }

        .tools-no-tools {
          padding: 16px;
          text-align: center;
          color: #6b7280;
          font-size: 13px;
        }

        .tools-dropdown-footer {
          padding: 12px 16px;
          border-top: 1px solid #e5e7eb;
          background-color: #f9fafb;
        }

        .tools-summary {
          font-size: 12px;
          color: #6b7280;
          text-align: center;
        }

        /* Scrollbar styling */
        .tools-categories-list::-webkit-scrollbar {
          width: 6px;
        }

        .tools-categories-list::-webkit-scrollbar-track {
          background: #f1f1f1;
        }

        .tools-categories-list::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 3px;
        }

        .tools-categories-list::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
      `}</style>
    </div>
  );
};

export default ToolsToggleDropdown;