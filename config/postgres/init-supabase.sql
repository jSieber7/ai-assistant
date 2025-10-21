-- =============================================================================
-- Essential PostgreSQL for LLM Agent System
-- =============================================================================

-- Essential extensions for modern apps
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";      -- Encryption/hashing

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
