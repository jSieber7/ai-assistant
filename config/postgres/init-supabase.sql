-- =============================================================================
-- Essential PostgreSQL for LLM Agent System
-- =============================================================================

-- Essential extensions for modern apps
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";      -- Encryption/hashing

-- =============================================================================
-- Supabase Schemas and Roles
-- =============================================================================
-- Create schemas required by Supabase
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS storage;
CREATE SCHEMA IF NOT EXISTS graphql_public;
CREATE SCHEMA IF NOT EXISTS public;

-- Create roles required by Supabase
CREATE ROLE anon NOINHERIT LOGIN PASSWORD 'dev-password';
CREATE ROLE service_role NOINHERIT LOGIN PASSWORD 'dev-password';
CREATE ROLE authenticator NOINHERIT LOGIN PASSWORD 'dev-password';
CREATE ROLE authenticated NOINHERIT LOGIN PASSWORD 'dev-password';

-- Grant permissions for Supabase roles
GRANT USAGE ON SCHEMA auth TO anon, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA auth TO anon, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA auth TO anon, service_role;

GRANT USAGE ON SCHEMA storage TO anon, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA storage TO anon, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA storage TO anon, service_role;

GRANT USAGE ON SCHEMA public TO anon, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, service_role;

-- The authenticator role is used by PostgREST
GRANT USAGE ON SCHEMA auth TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA auth TO authenticator;
GRANT ALL ON ALL SEQUENCES IN SCHEMA auth TO authenticator;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA auth TO authenticator;

GRANT USAGE ON SCHEMA storage TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA storage TO authenticator;
GRANT ALL ON ALL SEQUENCES IN SCHEMA storage TO authenticator;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA storage TO authenticator;

GRANT USAGE ON SCHEMA public TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticator;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticator;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO authenticator;

-- Also grant permissions on multi_writer schema
GRANT USAGE ON SCHEMA multi_writer TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA multi_writer TO authenticator;
GRANT ALL ON ALL SEQUENCES IN SCHEMA multi_writer TO authenticator;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA multi_writer TO authenticator;

-- =============================================================================
-- Application-specific Schemas and Roles
-- =============================================================================
-- Create schemas for your agent system
CREATE SCHEMA IF NOT EXISTS firecrawl;
CREATE SCHEMA IF NOT EXISTS app_data;
CREATE SCHEMA IF NOT EXISTS agent_memory;
CREATE SCHEMA IF NOT EXISTS multi_writer;

-- Create basic role for your app
CREATE ROLE app_user NOLOGIN NOINHERIT;

-- Grant permissions
GRANT USAGE ON SCHEMA firecrawl TO app_user;
GRANT ALL ON ALL TABLES IN SCHEMA firecrawl TO app_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA firecrawl TO app_user;

GRANT USAGE ON SCHEMA app_data TO app_user;
GRANT ALL ON ALL TABLES IN SCHEMA app_data TO app_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA app_data TO app_user;

GRANT USAGE ON SCHEMA agent_memory TO app_user;
GRANT ALL ON ALL TABLES IN SCHEMA agent_memory TO app_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA agent_memory TO app_user;

-- Firecrawl table for web scraping results
CREATE TABLE IF NOT EXISTS firecrawl.crawls (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  url TEXT NOT NULL,
  content TEXT,
  title TEXT,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Chat conversations table for managing conversation sessions
CREATE TABLE IF NOT EXISTS agent_memory.chat_conversations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title TEXT,
  user_id TEXT, -- Optional user ID for multi-user support
  model_id TEXT, -- The model used for this conversation
  agent_name TEXT, -- The agent used for this conversation
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat messages table for storing individual messages
CREATE TABLE IF NOT EXISTS agent_memory.chat_messages (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  conversation_id UUID NOT NULL REFERENCES agent_memory.chat_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Keep the old conversations table for backward compatibility
CREATE TABLE IF NOT EXISTS agent_memory.conversations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id TEXT NOT NULL,
  user_message TEXT,
  agent_response TEXT,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- App data table for general storage
CREATE TABLE IF NOT EXISTS app_data.settings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  key TEXT UNIQUE NOT NULL,
  value JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS crawls_url_idx ON firecrawl.crawls(url);
CREATE INDEX IF NOT EXISTS crawls_status_idx ON firecrawl.crawls(status);
CREATE INDEX IF NOT EXISTS conversations_session_idx ON agent_memory.conversations(session_id);
CREATE INDEX IF NOT EXISTS conversations_created_idx ON agent_memory.conversations(created_at);
CREATE INDEX IF NOT EXISTS settings_key_idx ON app_data.settings(key);

-- Indexes for chat conversations
CREATE INDEX IF NOT EXISTS chat_conversations_user_id_idx ON agent_memory.chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS chat_conversations_created_at_idx ON agent_memory.chat_conversations(created_at);
CREATE INDEX IF NOT EXISTS chat_conversations_updated_at_idx ON agent_memory.chat_conversations(updated_at);

-- Indexes for chat messages
CREATE INDEX IF NOT EXISTS chat_messages_conversation_id_idx ON agent_memory.chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS chat_messages_created_at_idx ON agent_memory.chat_messages(created_at);

-- Multi-writer workflow table
CREATE TABLE IF NOT EXISTS multi_writer.workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id TEXT UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    sources JSONB DEFAULT '[]'::jsonb,
    style_guide JSONB DEFAULT '{}'::jsonb,
    template_name TEXT DEFAULT 'article.html.jinja',
    quality_threshold REAL DEFAULT 70.0,
    max_iterations INTEGER DEFAULT 2,
    status TEXT DEFAULT 'pending',
    stages JSONB DEFAULT '{}'::jsonb,
    final_output TEXT,
    errors TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Multi-writer content table
CREATE TABLE IF NOT EXISTS multi_writer.content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id TEXT NOT NULL,
    writer_id TEXT NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Multi-writer check results table
CREATE TABLE IF NOT EXISTS multi_writer.check_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id TEXT NOT NULL,
    checker_id TEXT NOT NULL,
    score REAL,
    feedback TEXT,
    passed BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS workflows_workflow_id_idx ON multi_writer.workflows(workflow_id);
CREATE INDEX IF NOT EXISTS workflows_status_idx ON multi_writer.workflows(status);
CREATE INDEX IF NOT EXISTS workflows_created_at_idx ON multi_writer.workflows(created_at);

CREATE INDEX IF NOT EXISTS content_workflow_id_idx ON multi_writer.content(workflow_id);
CREATE INDEX IF NOT EXISTS content_writer_id_idx ON multi_writer.content(writer_id);
CREATE INDEX IF NOT EXISTS content_created_at_idx ON multi_writer.content(created_at);

CREATE INDEX IF NOT EXISTS check_results_workflow_id_idx ON multi_writer.check_results(workflow_id);
CREATE INDEX IF NOT EXISTS check_results_checker_id_idx ON multi_writer.check_results(checker_id);
CREATE INDEX IF NOT EXISTS check_results_created_at_idx ON multi_writer.check_results(created_at);