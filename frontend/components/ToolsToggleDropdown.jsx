import React from 'react';

/**
 * Agents Button Component
 *
 * A simple button component for agent management.
 *
 * @param {Object} props
 * @param {Function} props.onAgentClick - Callback when agent button is clicked
 * @param {string} props.variant - The style variant: 'top' for a top bar, 'status' for a status bar. Default is 'top'.
 */
const ToolsToggleDropdown = ({
  onAgentClick,
  variant = 'top'
}) => {
  return (
    <button
      className={`
        flex items-center gap-${variant === 'top' ? '1.5' : '1'}
        px-${variant === 'top' ? '3' : '2.5'} py-${variant === 'top' ? '2' : '1.5'}
        bg-blue-500 text-white
        border-0 rounded-${variant === 'top' ? 'md' : 'sm'}
        cursor-pointer font-medium
        ${variant === 'top' ? 'text-sm' : 'text-xs'}
        transition-all duration-200
        hover:bg-blue-600
        focus:outline-none focus:ring-[3px] focus:ring-blue-100
      `}
      onClick={onAgentClick}
      title="Agent Management"
    >
      <svg
        width={variant === 'top' ? '16' : '14'}
        height={variant === 'top' ? '16' : '14'}
        viewBox="0 0 16 16"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M8 8C10.21 8 12 6.21 12 4C12 1.79 10.21 0 8 0C5.79 0 4 1.79 4 4C4 6.21 5.79 8 8 8Z"
          fill="currentColor"
        />
        <path
          d="M12.5 9H11.71C11.11 9.64 10.36 10 9.5 10H6.5C5.64 10 4.89 9.64 4.29 9H3.5C1.57 9 0 10.57 0 12.5V14C0 15.1 0.9 16 2 16H14C15.1 16 16 15.1 16 14V12.5C16 10.57 14.43 9 12.5 9Z"
          fill="currentColor"
        />
      </svg>
      <span className={variant === 'top' ? 'text-sm' : 'text-xs'}>Agents</span>
    </button>
  );
};

export default ToolsToggleDropdown;