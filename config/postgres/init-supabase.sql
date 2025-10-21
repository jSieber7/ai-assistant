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

GRANT USAGE ON SCHEMA storage TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA storage TO authenticator;

GRANT USAGE ON SCHEMA public TO authenticator;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO authenticator;


-- =============================================================================
-- Application-specific Schemas and Roles
-- =============================================================================
-- Create schemas for your agent system
CREATE SCHEMA IF NOT EXISTS firecrawl;
CREATE SCHEMA IF NOT EXISTS app_data;
CREATE SCHEMA IF NOT EXISTS agent_memory;

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

-- Agent memory table for conversation history
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

-- =============================================================================
-- Chainlit Schema for Chat Lifecycle Management
-- =============================================================================
-- This schema creates the necessary tables for Chainlit's data persistence
-- and chat lifecycle management

-- Create chainlit schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS chainlit;

-- =============================================================================
-- Chainlit Tables
-- =============================================================================

-- Users table
CREATE TABLE IF NOT EXISTS chainlit.users (
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Threads table (chat sessions)
CREATE TABLE IF NOT EXISTS chainlit.threads (
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[] DEFAULT '{}',
    "metadata" JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY ("userId") REFERENCES chainlit.users("id") ON DELETE CASCADE
);

-- Steps table (messages and actions)
CREATE TABLE IF NOT EXISTS chainlit.steps (
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL DEFAULT FALSE,
    "waitForAnswer" BOOLEAN DEFAULT FALSE,
    "isError" BOOLEAN DEFAULT FALSE,
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "tags" TEXT[] DEFAULT '{}',
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER DEFAULT 0,
    "defaultOpen" BOOLEAN DEFAULT FALSE,
    FOREIGN KEY ("threadId") REFERENCES chainlit.threads("id") ON DELETE CASCADE,
    FOREIGN KEY ("parentId") REFERENCES chainlit.steps("id") ON DELETE CASCADE
);

-- Elements table (files, images, etc.)
CREATE TABLE IF NOT EXISTS chainlit.elements (
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB DEFAULT '{}'::jsonb,
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY ("threadId") REFERENCES chainlit.threads("id") ON DELETE CASCADE,
    FOREIGN KEY ("forId") REFERENCES chainlit.steps("id") ON DELETE CASCADE
);

-- Feedbacks table (user feedback on messages)
CREATE TABLE IF NOT EXISTS chainlit.feedbacks (
    "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INTEGER NOT NULL CHECK ("value" >= -1 AND "value" <= 1),
    "comment" TEXT,
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY ("threadId") REFERENCES chainlit.threads("id") ON DELETE CASCADE,
    FOREIGN KEY ("forId") REFERENCES chainlit.steps("id") ON DELETE CASCADE
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS users_identifier_idx ON chainlit.users("identifier");
CREATE INDEX IF NOT EXISTS users_created_idx ON chainlit.users("createdAtTimestamp");

-- Threads indexes
CREATE INDEX IF NOT EXISTS threads_user_id_idx ON chainlit.threads("userId");
CREATE INDEX IF NOT EXISTS threads_user_identifier_idx ON chainlit.threads("userIdentifier");
CREATE INDEX IF NOT EXISTS threads_created_idx ON chainlit.threads("createdAtTimestamp");
CREATE INDEX IF NOT EXISTS threads_name_idx ON chainlit.threads("name");
CREATE INDEX IF NOT EXISTS threads_tags_idx ON chainlit.threads USING GIN("tags");

-- Steps indexes
CREATE INDEX IF NOT EXISTS steps_thread_id_idx ON chainlit.steps("threadId");
CREATE INDEX IF NOT EXISTS steps_parent_id_idx ON chainlit.steps("parentId");
CREATE INDEX IF NOT EXISTS steps_created_idx ON chainlit.steps("createdAtTimestamp");
CREATE INDEX IF NOT EXISTS steps_type_idx ON chainlit.steps("type");
CREATE INDEX IF NOT EXISTS steps_name_idx ON chainlit.steps("name");
CREATE INDEX IF NOT EXISTS steps_tags_idx ON chainlit.steps USING GIN("tags");

-- Elements indexes
CREATE INDEX IF NOT EXISTS elements_thread_id_idx ON chainlit.elements("threadId");
CREATE INDEX IF NOT EXISTS elements_for_id_idx ON chainlit.elements("forId");
CREATE INDEX IF NOT EXISTS elements_type_idx ON chainlit.elements("type");
CREATE INDEX IF NOT EXISTS elements_created_idx ON chainlit.elements("createdAtTimestamp");

-- Feedbacks indexes
CREATE INDEX IF NOT EXISTS feedbacks_thread_id_idx ON chainlit.feedbacks("threadId");
CREATE INDEX IF NOT EXISTS feedbacks_for_id_idx ON chainlit.feedbacks("forId");
CREATE INDEX IF NOT EXISTS feedbacks_created_idx ON chainlit.feedbacks("createdAtTimestamp");

-- =============================================================================
-- Permissions
-- =============================================================================

-- Grant permissions to authenticated users
GRANT USAGE ON SCHEMA chainlit TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA chainlit TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA chainlit TO authenticated;

-- Grant permissions to service role
GRANT USAGE ON SCHEMA chainlit TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA chainlit TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA chainlit TO service_role;

-- Grant read permissions to anon role (if needed for public access)
GRANT USAGE ON SCHEMA chainlit TO anon;
GRANT SELECT ON ALL TABLES IN SCHEMA chainlit TO anon;
