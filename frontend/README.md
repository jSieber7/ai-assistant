# Chat History Sidebar Component

A collapsible sidebar component for displaying chat history in React applications, designed for integration with Chainlit.

## Features

- Collapsible sidebar with smooth animations
- Displays chat history with titles, last messages, and timestamps
- Shows unread message count badges
- Responsive design with customizable width
- Can be positioned on the left or right side
- Includes hover effects and interactive elements

## Files

- `components/ChatHistorySidebar.jsx` - The main sidebar component
- `examples/ChatHistorySidebarExample.jsx` - Example implementation with a full chat interface

## Usage

### Basic Usage

```jsx
import React from 'react';
import ChatHistorySidebar from './components/ChatHistorySidebar';

const App = () => {
  const chatHistory = [
    {
      id: 1,
      title: "Discussion about AI models",
      timestamp: "2023-10-21T10:30:00Z",
      lastMessage: "What are the differences between GPT-3 and GPT-4?",
      unread: 2
    },
    // ... more chat items
  ];

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <ChatHistorySidebar 
        chatHistory={chatHistory}
        width={300}
        position="left"
      />
      {/* Main content area */}
    </div>
  );
};
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `chatHistory` | Array | `[]` | Array of chat objects to display |
| `width` | Number | `300` | Width of the sidebar when expanded (in pixels) |
| `position` | String | `'left'` | Position of the sidebar ('left' or 'right') |
| `onChatSelect` | Function | `null` | Callback function when a chat is selected |
| `onNewChat` | Function | `null` | Callback function when "New Chat" button is clicked |

### Chat Object Structure

Each chat object should have the following structure:

```javascript
{
  id: Number,           // Unique identifier for the chat
  title: String,        // Chat title or subject
  timestamp: String,    // ISO 8601 timestamp
  lastMessage: String,  // Last message in the chat
  unread: Number        // Number of unread messages (optional)
}
```

## Integration with Chainlit

To integrate this component with Chainlit, you'll need to:

1. Set up a React frontend that communicates with your Chainlit backend
2. Fetch chat history from your Chainlit API
3. Handle chat selection and creation through API calls

```jsx
// Example of fetching chat history from Chainlit API
useEffect(() => {
  fetch('/api/chat-history')
    .then(res => res.json())
    .then(data => setChatHistory(data))
    .catch(error => console.error('Error fetching chat history:', error));
}, []);
```

## Customization

The component uses inline styles for easy customization. You can modify the styles directly in the component or override them with CSS classes.

## Dependencies

- React (with hooks support)
- No additional dependencies required

## Browser Support

The component supports all modern browsers including:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This component is part of the Chainlit integration project.