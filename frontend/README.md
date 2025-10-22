# AI Assistant React Frontend

A modern React frontend for the AI Assistant backend, built with TypeScript, Tailwind CSS, and Vite.

## Features

- **Modern UI**: Clean, responsive interface with dark mode support
- **Chat Interface**: Full-featured chat with message history
- **Model Selection**: Choose from multiple LLM providers and models
- **Agent System**: Select and use specialized AI agents
- **Real-time Updates**: Streaming responses and typing indicators
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Syntax Highlighting**: Code blocks with syntax highlighting
- **File Upload**: Support for file attachments (placeholder)
- **Voice Input**: Voice message recording (placeholder)

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- The AI Assistant backend running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The build artifacts will be stored in the `dist` directory.

## Project Structure

```
src/
├── components/         # React components
│   ├── Sidebar.tsx    # Chat history sidebar
│   ├── Header.tsx     # Top navigation bar
│   ├── MessageArea.tsx # Message display area
│   └── InputArea.tsx  # Message input area
├── hooks/             # Custom React hooks
│   ├── useChat.ts     # Chat state management
│   └── useModels.ts   # Model and agent management
├── services/          # API services
│   └── api.ts         # Backend API client
├── utils/             # Utility functions
├── App.tsx            # Main application component
├── main.tsx           # Application entry point
└── index.css          # Global styles
```

## Configuration

### API Endpoint

The API endpoint is configured in `src/services/api.ts`. By default, it connects to `http://localhost:8000`. To change the backend URL:

```typescript
const API_BASE_URL = 'http://your-backend-url:port';
```

### Model and Agent Selection

The frontend automatically fetches available models, providers, and agents from the backend. You can select them from the dropdown menus in the header.

### Dark Mode

Dark mode is controlled by a state variable in the App component and is persisted in the browser's local storage.

## Features in Detail

### Chat Interface

- **Message History**: All conversations are stored locally and can be accessed from the sidebar
- **Message Formatting**: Supports markdown formatting with syntax highlighting for code blocks
- **Typing Indicators**: Shows when the AI is generating a response
- **Error Handling**: Displays error messages for failed requests

### Model Selection

- **Provider Support**: Supports multiple LLM providers (OpenAI, OpenRouter, etc.)
- **Model Information**: Displays model capabilities (streaming, tools support)
- **Health Status**: Shows provider health status

### Agent System

- **Agent Selection**: Choose from available specialized agents
- **Agent Information**: View agent descriptions and usage statistics
- **State Management**: Track agent status and usage

### Responsive Design

- **Mobile Support**: Fully responsive design that works on all screen sizes
- **Collapsible Sidebar**: Sidebar can be collapsed for more screen space
- **Touch-friendly**: All interactive elements are optimized for touch devices

## Development

### Adding New Features

1. Create new components in the `src/components/` directory
2. Add custom hooks in `src/hooks/` if needed
3. Update the API service in `src/services/api.ts` for new endpoints
4. Import and use components in `App.tsx`

### Styling

The project uses Tailwind CSS for styling. You can customize the theme in `tailwind.config.js`.

### TypeScript

The project is fully typed with TypeScript. Type definitions are included in the API service.

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Ensure the backend is running on the correct port
2. **CORS Issues**: Make sure the backend allows requests from the frontend URL
3. **Build Errors**: Check that all dependencies are installed correctly

### Debug Mode

To enable debug mode, set the environment variable:

```bash
VITE_DEBUG=true npm run dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.