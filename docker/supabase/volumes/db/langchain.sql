-- LangChain Integration Database Schema
-- This file creates all necessary tables for LangChain and LangGraph integration

-- Create LangChain schema
CREATE SCHEMA IF NOT EXISTS langchain;

-- Set permissions for the schema
GRANT ALL ON SCHEMA langchain TO postgres;
GRANT ALL ON SCHEMA langchain TO authenticator;
GRANT ALL ON SCHEMA langchain TO supabase_functions_admin;

-- ============================================================================
-- LangChain Memory Management Tables
-- ============================================================================

-- LangChain Conversations Table
CREATE TABLE IF NOT EXISTS langchain.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    title VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for conversations
CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON langchain.conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON langchain.conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON langchain.conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON langchain.conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_is_active ON langchain.conversations(is_active);

-- LangChain Chat Messages Table
CREATE TABLE IF NOT EXISTS langchain.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(255) NOT NULL,
    message_id VARCHAR(255) UNIQUE NOT NULL,
    message_type VARCHAR(50) NOT NULL, -- 'human', 'ai', 'system', 'tool'
    content TEXT NOT NULL,
    additional_kwargs JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    message_sequence INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    model_name VARCHAR(255),
    temperature FLOAT,
    max_tokens INTEGER
);

-- Create indexes for chat messages
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON langchain.chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_message_id ON langchain.chat_messages(message_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_type ON langchain.chat_messages(message_type);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON langchain.chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sequence ON langchain.chat_messages(conversation_id, message_sequence);

-- LangChain Memory Summaries Table
CREATE TABLE IF NOT EXISTS langchain.memory_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(255) NOT NULL,
    summary_id VARCHAR(255) UNIQUE NOT NULL,
    summary_type VARCHAR(50) NOT NULL, -- 'brief', 'concise', 'detailed', 'topical'
    content TEXT NOT NULL,
    message_range_start INTEGER,
    message_range_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    token_count INTEGER DEFAULT 0,
    model_name VARCHAR(255)
);

-- Create indexes for memory summaries
CREATE INDEX IF NOT EXISTS idx_memory_summaries_conversation_id ON langchain.memory_summaries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_summary_id ON langchain.memory_summaries(summary_id);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_type ON langchain.memory_summaries(summary_type);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_created_at ON langchain.memory_summaries(created_at);

-- ============================================================================
-- LangGraph Workflow Management Tables
-- ============================================================================

-- LangGraph Workflows Table
CREATE TABLE IF NOT EXISTS langchain.workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(255) UNIQUE NOT NULL,
    workflow_type VARCHAR(100) NOT NULL, -- 'agent', 'content', 'collaboration', 'validation', 'enhancement', 'memory'
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    user_id VARCHAR(255),
    session_id VARCHAR(255)
);

-- Create indexes for workflows
CREATE INDEX IF NOT EXISTS idx_workflows_workflow_id ON langchain.workflows(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflows_type ON langchain.workflows(workflow_type);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON langchain.workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON langchain.workflows(created_at);
CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON langchain.workflows(user_id);

-- LangGraph Workflow Steps Table
CREATE TABLE IF NOT EXISTS langchain.workflow_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) UNIQUE NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    step_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    error_message TEXT,
    parent_step_id VARCHAR(255),
    step_sequence INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for workflow steps
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON langchain.workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_step_id ON langchain.workflow_steps(step_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON langchain.workflow_steps(status);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_sequence ON langchain.workflow_steps(workflow_id, step_sequence);

-- LangGraph Checkpoints Table
CREATE TABLE IF NOT EXISTS langchain.checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_id VARCHAR(255) UNIQUE NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoints_checkpoint_id ON langchain.checkpoints(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow_id ON langchain.checkpoints(workflow_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON langchain.checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_expires_at ON langchain.checkpoints(expires_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_is_active ON langchain.checkpoints(is_active);

-- ============================================================================
-- LangChain Tool Execution Tables
-- ============================================================================

-- LangChain Tool Executions Table
CREATE TABLE IF NOT EXISTS langchain.tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    tool_type VARCHAR(100) NOT NULL,
    workflow_id VARCHAR(255),
    agent_session_id VARCHAR(255),
    input_parameters JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    user_id VARCHAR(255)
);

-- Create indexes for tool executions
CREATE INDEX IF NOT EXISTS idx_tool_executions_execution_id ON langchain.tool_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_tool_name ON langchain.tool_executions(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_executions_workflow_id ON langchain.tool_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_status ON langchain.tool_executions(status);
CREATE INDEX IF NOT EXISTS idx_tool_executions_created_at ON langchain.tool_executions(created_at);

-- LangChain Tool Results Table
CREATE TABLE IF NOT EXISTS langchain.tool_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255) NOT NULL,
    result_type VARCHAR(100) NOT NULL, -- 'success', 'error', 'partial'
    output_data JSONB NOT NULL DEFAULT '{}',
    raw_output TEXT,
    artifacts JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    token_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    FOREIGN KEY (execution_id) REFERENCES langchain.tool_executions(execution_id) ON DELETE CASCADE
);

-- Create indexes for tool results
CREATE INDEX IF NOT EXISTS idx_tool_results_execution_id ON langchain.tool_results(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_results_result_type ON langchain.tool_results(result_type);
CREATE INDEX IF NOT EXISTS idx_tool_results_created_at ON langchain.tool_results(created_at);

-- ============================================================================
-- LangChain Agent Management Tables
-- ============================================================================

-- LangChain Agent Sessions Table
CREATE TABLE IF NOT EXISTS langchain.agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    conversation_id VARCHAR(255),
    user_id VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for agent sessions
CREATE INDEX IF NOT EXISTS idx_agent_sessions_session_id ON langchain.agent_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_name ON langchain.agent_sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_conversation_id ON langchain.agent_sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id ON langchain.agent_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON langchain.agent_sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_last_activity ON langchain.agent_sessions(last_activity_at);

-- LangChain Agent Executions Table
CREATE TABLE IF NOT EXISTS langchain.agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    execution_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    error_message TEXT,
    token_count INTEGER DEFAULT 0,
    model_name VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES langchain.agent_sessions(session_id) ON DELETE CASCADE
);

-- Create indexes for agent executions
CREATE INDEX IF NOT EXISTS idx_agent_executions_execution_id ON langchain.agent_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_session_id ON langchain.agent_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_name ON langchain.agent_executions(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON langchain.agent_executions(status);
CREATE INDEX IF NOT EXISTS idx_agent_executions_created_at ON langchain.agent_executions(created_at);

-- ============================================================================
-- LangChain Integration Metrics Tables
-- ============================================================================

-- LangChain Metrics Table
CREATE TABLE IF NOT EXISTS langchain.metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL, -- 'counter', 'gauge', 'histogram', 'timer'
    value FLOAT NOT NULL,
    unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    component VARCHAR(255), -- 'llm', 'tools', 'agents', 'memory', 'workflows'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for metrics
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON langchain.metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_type ON langchain.metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_component ON langchain.metrics(component);
CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON langchain.metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_tags ON langchain.metrics USING GIN(tags);

-- LangChain Performance Logs Table
CREATE TABLE IF NOT EXISTS langchain.performance_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_level VARCHAR(20) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component VARCHAR(255) NOT NULL,
    operation VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    workflow_id VARCHAR(255),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance logs
CREATE INDEX IF NOT EXISTS idx_performance_logs_level ON langchain.performance_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_performance_logs_component ON langchain.performance_logs(component);
CREATE INDEX IF NOT EXISTS idx_performance_logs_operation ON langchain.performance_logs(operation);
CREATE INDEX IF NOT EXISTS idx_performance_logs_created_at ON langchain.performance_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_logs_user_id ON langchain.performance_logs(user_id);

-- ============================================================================
-- Triggers and Functions
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION langchain.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON langchain.conversations
    FOR EACH ROW EXECUTE FUNCTION langchain.update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON langchain.workflows
    FOR EACH ROW EXECUTE FUNCTION langchain.update_updated_at_column();

CREATE TRIGGER update_agent_sessions_updated_at BEFORE UPDATE ON langchain.agent_sessions
    FOR EACH ROW EXECUTE FUNCTION langchain.update_updated_at_column();

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE langchain.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.memory_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.workflow_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.tool_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.tool_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.agent_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.agent_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE langchain.performance_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies (allow authenticated users to access their own data)
CREATE POLICY "Users can view own conversations" ON langchain.conversations
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own conversations" ON langchain.conversations
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own conversations" ON langchain.conversations
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can view own chat messages" ON langchain.chat_messages
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own chat messages" ON langchain.chat_messages
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Similar policies for other tables...
CREATE POLICY "Users can view own workflows" ON langchain.workflows
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own workflows" ON langchain.workflows
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Conversation summary view
CREATE OR REPLACE VIEW langchain.conversation_summary AS
SELECT 
    c.conversation_id,
    c.title,
    c.user_id,
    c.created_at,
    c.updated_at,
    COUNT(cm.id) as message_count,
    MAX(cm.created_at) as last_message_at,
    COALESCE(SUM(cm.token_count), 0) as total_tokens
FROM langchain.conversations c
LEFT JOIN langchain.chat_messages cm ON c.conversation_id = cm.conversation_id
WHERE c.is_active = true
GROUP BY c.conversation_id, c.title, c.user_id, c.created_at, c.updated_at;

-- Workflow performance view
CREATE OR REPLACE VIEW langchain.workflow_performance AS
SELECT 
    w.workflow_id,
    w.workflow_type,
    w.status,
    w.created_at,
    w.completed_at,
    EXTRACT(EPOCH FROM (w.completed_at - w.started_at)) * 1000 as execution_time_ms,
    COUNT(ws.id) as step_count,
    COUNT(CASE WHEN ws.status = 'completed' THEN 1 END) as completed_steps,
    COUNT(CASE WHEN ws.status = 'failed' THEN 1 END) as failed_steps
FROM langchain.workflows w
LEFT JOIN langchain.workflow_steps ws ON w.workflow_id = ws.workflow_id
GROUP BY w.workflow_id, w.workflow_type, w.status, w.created_at, w.completed_at;

-- Tool usage statistics view
CREATE OR REPLACE VIEW langchain.tool_usage_stats AS
SELECT 
    te.tool_name,
    te.tool_type,
    COUNT(*) as execution_count,
    AVG(te.execution_time_ms) as avg_execution_time_ms,
    SUM(CASE WHEN te.status = 'completed' THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN te.status = 'failed' THEN 1 ELSE 0 END) as failure_count,
    ROUND(
        SUM(CASE WHEN te.status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) as success_rate_percent
FROM langchain.tool_executions te
GROUP BY te.tool_name, te.tool_type;

-- Grant access to views
GRANT SELECT ON langchain.conversation_summary TO authenticated;
GRANT SELECT ON langchain.workflow_performance TO authenticated;
GRANT SELECT ON langchain.tool_usage_stats TO authenticated;

-- ============================================================================
-- Initial Data and Configuration
-- ============================================================================

-- Insert default configuration
INSERT INTO langchain.metrics (metric_name, metric_type, value, component, tags)
VALUES 
    ('langchain_integration_initialized', 'counter', 1, 'system', '{"version": "1.0.0"}'),
    ('database_schema_created', 'counter', 1, 'database', '{"schema": "langchain"}')
ON CONFLICT DO NOTHING;

-- Create a function to clean up expired data
CREATE OR REPLACE FUNCTION langchain.cleanup_expired_data()
RETURNS void AS $$
BEGIN
    -- Clean up expired checkpoints
    DELETE FROM langchain.checkpoints 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    -- Clean up old performance logs (older than 30 days)
    DELETE FROM langchain.performance_logs 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Clean up old metrics (older than 90 days)
    DELETE FROM langchain.metrics 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    RAISE NOTICE 'Cleaned up expired LangChain data';
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to run cleanup (requires pg_cron extension)
-- This would be enabled if pg_cron is available
-- SELECT cron.schedule('langchain-cleanup', '0 2 * * *', 'SELECT langchain.cleanup_expired_data();');

-- ============================================================================
-- Comments and Documentation
-- ============================================================================

COMMENT ON SCHEMA langchain IS 'LangChain and LangGraph integration schema';
COMMENT ON TABLE langchain.conversations IS 'Stores conversation metadata for LangChain memory management';
COMMENT ON TABLE langchain.chat_messages IS 'Stores individual chat messages with full context';
COMMENT ON TABLE langchain.memory_summaries IS 'Stores conversation summaries for memory optimization';
COMMENT ON TABLE langchain.workflows IS 'Stores LangGraph workflow execution data';
COMMENT ON TABLE langchain.workflow_steps IS 'Stores individual workflow step execution data';
COMMENT ON TABLE langchain.checkpoints IS 'Stores LangGraph workflow checkpoints for state persistence';
COMMENT ON TABLE langchain.tool_executions IS 'Stores LangChain tool execution records';
COMMENT ON TABLE langchain.tool_results IS 'Stores results from tool executions';
COMMENT ON TABLE langchain.agent_sessions IS 'Stores agent session information';
COMMENT ON TABLE langchain.agent_executions IS 'Stores agent execution records';
COMMENT ON TABLE langchain.metrics IS 'Stores integration metrics and telemetry';
COMMENT ON TABLE langchain.performance_logs IS 'Stores detailed performance logs';