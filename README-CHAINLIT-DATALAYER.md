# Chainlit Data Layer Integration

This document describes the implementation of Chainlit's data layer with PostgreSQL/Supabase for chat lifecycle management and persistence.

## Overview

The Chainlit data layer integration provides:
- **Chat Persistence**: All chat sessions are automatically saved to PostgreSQL/Supabase
- **Chat History**: Users can view and resume previous chat sessions
- **Chat Lifecycle Management**: Full support for chat start, resume, and end events
- **Sidebar Navigation**: Interactive sidebar with chat history and navigation options
- **User Management**: Basic user tracking and session management

## Architecture

### Components

1. **PostgreSQL Data Layer** (`app/ui/chainlit_data_layer.py`)
   - Custom implementation of Chainlit's `BaseDataLayer`
   - Handles all database operations for users, threads, steps, elements, and feedbacks
   - Uses asyncpg for efficient asynchronous database operations

2. **Database Schema** (`config/postgres/chainlit-schema.sql`)
   - Complete schema for Chainlit's data persistence
   - Includes tables for users, threads, steps, elements, and feedbacks
   - Optimized with proper indexes and constraints

3. **Enhanced Chainlit App** (`app/ui/chainlit_app.py`)
   - Integrated with data layer for automatic persistence
   - Added chat lifecycle event handlers
   - Implemented sidebar with chat history
   - Added new commands and actions for chat management

## Setup Instructions

### 1. Database Setup

The database schema is automatically created when your Supabase/PostgreSQL container starts. The schema is included in `config/postgres/init-supabase.sql`.

### 2. Environment Configuration

Set the following environment variable:

```bash
export CHAINLIT_DATABASE_URL=postgresql://postgres:your-password@supabase-db:5432/postgres
```

### 3. Running the Application

#### Option 1: Using the Custom Runner

```bash
python run_chainlit_with_datalayer.py
```

This script will:
- Test the database connection
- Initialize the data layer
- Start Chainlit with persistence enabled

#### Option 2: Direct Chainlit Execution

```bash
chainlit run app/ui/chainlit_app.py
```

Make sure the `CHAINLIT_DATABASE_URL` environment variable is set before running.

## Features

### Chat Lifecycle Management

The implementation supports all Chainlit lifecycle events:

- **`@cl.on_chat_start`**: Initializes new chat sessions, creates user records
- **`@cl.on_chat_resume`**: Restores previous chat sessions with full context
- **`@cl.on_chat_end`**: Handles cleanup when chat sessions end

### Chat History Sidebar

The sidebar displays:
- List of previous chat sessions
- Chat names and creation dates
- Message previews
- Resume buttons for each chat

### Commands

The following commands are available:

- `/settings` - Show current configuration
- `/reset` - Reset configuration and select new provider
- `/help` - Show help message
- `/history` - Refresh chat history sidebar
- `/new` - Start a new chat session

### Action Buttons

Additional action buttons have been added:
- 🧪 Test Provider - Test connection to selected provider
- ➕ Add New Provider - Add a new LLM provider
- 🔄 Refresh Providers - Refresh the provider list
- 📜 Show Chat History - Display chat history in sidebar
- 💬 New Chat - Start a new chat session

## Database Schema

### Tables

1. **users**
   - Stores user information and metadata
   - Indexed by identifier for quick lookups

2. **threads**
   - Represents chat sessions
   - Linked to users with proper foreign key constraints
   - Stores chat names, tags, and metadata

3. **steps**
   - Individual messages and actions within chats
   - Supports hierarchical structure with parent-child relationships
   - Stores message content, type, and metadata

4. **elements**
   - Files, images, and other media shared in chats
   - Linked to threads and specific steps
   - Stores file metadata and references

5. **feedbacks**
   - User feedback on chat responses
   - Supports thumbs up/down and comments
   - Linked to specific steps and threads

### Security

The schema includes:
- Row Level Security (RLS) policies for Supabase
- Proper user isolation - users can only access their own data
- Secure permissions for different user roles

## Usage Examples

### Starting a New Chat

1. Open the Chainlit interface
2. Configure your LLM provider and model
3. Start chatting - the session is automatically saved

### Resuming a Previous Chat

1. Click on the "📜 Show Chat History" button or use `/history` command
2. Select a previous chat from the sidebar
3. Click the "Resume" button
4. The chat will be restored with full context

### Managing Chat History

The sidebar automatically updates when:
- A new chat is started
- A message is sent
- A chat is resumed
- The history command is used

## Troubleshooting

### Database Connection Issues

If you encounter database connection errors:

1. Check that your Supabase/PostgreSQL container is running
2. Verify the `CHAINLIT_DATABASE_URL` environment variable
3. Ensure the database schema has been created
4. Check database permissions

### Chat History Not Showing

If chat history doesn't appear in the sidebar:

1. Verify the data layer is properly initialized
2. Check browser console for JavaScript errors
3. Ensure the user session is properly set
4. Try refreshing with the `/history` command

### Performance Issues

For better performance with large chat histories:

1. The sidebar limits to the last 10 chats
2. Database queries are optimized with indexes
3. Consider implementing pagination for extensive histories

## Development Notes

### Extending the Implementation

To add new features:

1. Modify the `PostgreSQLDataLayer` class for new database operations
2. Update the Chainlit app with new event handlers or actions
3. Add new commands or UI elements as needed

### Custom User Authentication

The current implementation uses a default user for demo purposes. To integrate with your authentication system:

1. Modify the `on_chat_start` handler to get the authenticated user
2. Update the user creation logic to use real user data
3. Adjust RLS policies if using Supabase

### Backup and Recovery

Regular backups of your chat data are recommended:

1. Use Supabase's built-in backup features
2. Export the chainlit schema tables periodically
3. Consider implementing archiving for old chats

## Future Enhancements

Potential improvements to consider:

1. **Chat Search**: Implement full-text search across chat history
2. **Chat Sharing**: Allow users to share specific chat sessions
3. **Chat Export**: Add functionality to export chats in various formats
4. **Analytics**: Implement usage analytics and insights
5. **Multi-user Support**: Enhanced collaboration features
6. **File Attachments**: Better support for file sharing and management