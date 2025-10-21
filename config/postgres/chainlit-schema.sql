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

-- =============================================================================
-- Row Level Security (RLS) for Supabase
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE chainlit.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE chainlit.threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE chainlit.steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE chainlit.elements ENABLE ROW LEVEL SECURITY;
ALTER TABLE chainlit.feedbacks ENABLE ROW LEVEL SECURITY;

-- Users RLS policies
CREATE POLICY "Users can view own profile" ON chainlit.users
    FOR SELECT USING (auth.uid()::text = "identifier");

CREATE POLICY "Users can insert own profile" ON chainlit.users
    FOR INSERT WITH CHECK (auth.uid()::text = "identifier");

CREATE POLICY "Users can update own profile" ON chainlit.users
    FOR UPDATE USING (auth.uid()::text = "identifier");

-- Threads RLS policies
CREATE POLICY "Users can view own threads" ON chainlit.threads
    FOR SELECT USING (auth.uid()::text = "userIdentifier");

CREATE POLICY "Users can insert own threads" ON chainlit.threads
    FOR INSERT WITH CHECK (auth.uid()::text = "userIdentifier");

CREATE POLICY "Users can update own threads" ON chainlit.threads
    FOR UPDATE USING (auth.uid()::text = "userIdentifier");

CREATE POLICY "Users can delete own threads" ON chainlit.threads
    FOR DELETE USING (auth.uid()::text = "userIdentifier");

-- Steps RLS policies
CREATE POLICY "Users can view own thread steps" ON chainlit.steps
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.steps."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert steps in own threads" ON chainlit.steps
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.steps."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

CREATE POLICY "Users can update steps in own threads" ON chainlit.steps
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.steps."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

-- Elements RLS policies
CREATE POLICY "Users can view own thread elements" ON chainlit.elements
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.elements."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert elements in own threads" ON chainlit.elements
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.elements."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

-- Feedbacks RLS policies
CREATE POLICY "Users can view own thread feedbacks" ON chainlit.feedbacks
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.feedbacks."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert feedbacks in own threads" ON chainlit.feedbacks
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM chainlit.threads 
            WHERE "id" = chainlit.feedbacks."threadId" 
            AND "userIdentifier" = auth.uid()::text
        )
    );

-- =============================================================================
-- Functions and Triggers for Updated Timestamps
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION chainlit.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for threads (if we add updated_at column later)
-- CREATE TRIGGER update_threads_updated_at 
--     BEFORE UPDATE ON chainlit.threads 
--     FOR EACH ROW EXECUTE FUNCTION chainlit.update_updated_at_column();

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- View for user threads with message count
CREATE OR REPLACE VIEW chainlit.user_threads_summary AS
SELECT 
    t.id,
    t.name,
    t.createdAt,
    t.createdAtTimestamp,
    t.tags,
    t.metadata,
    COUNT(s.id) as message_count,
    MAX(s.createdAtTimestamp) as last_message_at
FROM chainlit.threads t
LEFT JOIN chainlit.steps s ON t.id = s."threadId"
WHERE t."userIdentifier" = auth.uid()::text
GROUP BY t.id, t.name, t.createdAt, t.createdAtTimestamp, t.tags, t.metadata
ORDER BY t.createdAtTimestamp DESC;

-- View for thread messages
CREATE OR REPLACE VIEW chainlit.thread_messages AS
SELECT 
    s.id,
    s."threadId",
    s.name,
    s.type,
    s.input,
    s.output,
    s.createdAt,
    s.createdAtTimestamp,
    s."parentId",
    s.tags,
    s.metadata
FROM chainlit.steps s
WHERE EXISTS (
    SELECT 1 FROM chainlit.threads t 
    WHERE t.id = s."threadId" 
    AND t."userIdentifier" = auth.uid()::text
)
ORDER BY s.createdAtTimestamp ASC;