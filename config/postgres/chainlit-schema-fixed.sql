-- =============================================================================
-- Chainlit Data Layer Schema for PostgreSQL/Supabase
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
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "createdAt" TEXT NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    "createdAtTimestamp" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Threads table (chat sessions)
CREATE TABLE IF NOT EXISTS chainlit.threads (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

-- Grant permissions to service role
GRANT USAGE ON SCHEMA chainlit TO service_role;
GRANT ALL ON ALL TABLES IN SCHEMA chainlit TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA chainlit TO service_role;

-- Grant read permissions to anon role (if needed for public access)
GRANT USAGE ON SCHEMA chainlit TO anon;
GRANT SELECT ON ALL TABLES IN SCHEMA chainlit TO anon;

-- Grant permissions to authenticator
GRANT USAGE ON SCHEMA chainlit TO authenticator;
GRANT ALL ON ALL TABLES IN SCHEMA chainlit TO authenticator;
GRANT ALL ON ALL SEQUENCES IN SCHEMA chainlit TO authenticator;