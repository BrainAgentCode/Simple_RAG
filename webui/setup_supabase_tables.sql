-- Create table for user answers
CREATE TABLE IF NOT EXISTS user_answers (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    task_id TEXT NOT NULL,
    answer TEXT,
    is_correct BOOLEAN,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entry_type TEXT,
    model_used TEXT,
    query TEXT
);

-- Create table for user evaluations
CREATE TABLE IF NOT EXISTS user_evaluations (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sus_responses JSONB,
    task_difficulty JSONB,
    ai_helpfulness TEXT,
    ai_relevance TEXT,
    retrieval_quality TEXT,
    traditional_comparison TEXT,
    improvement_suggestions TEXT,
    favorite_feature TEXT,
    skipped_tasks JSONB
);

-- Create table for system logs
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    logger TEXT,
    pathname TEXT,
    lineno INTEGER
);

-- Create table for RAG queries
CREATE TABLE IF NOT EXISTS rag_queries (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    username TEXT NOT NULL,
    task_id TEXT,
    question TEXT NOT NULL,
    model TEXT NOT NULL,
    processing_time FLOAT,
    num_docs_retrieved INTEGER,
    answer_length INTEGER
);

-- Enable Row Level Security (RLS)
ALTER TABLE user_answers ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_queries ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations for authenticated users
CREATE POLICY "Allow all operations for authenticated users" ON user_answers
    FOR ALL USING (true);

CREATE POLICY "Allow all operations for authenticated users" ON user_evaluations
    FOR ALL USING (true);
    
CREATE POLICY "Allow all operations for authenticated users" ON system_logs
    FOR ALL USING (true);
    
CREATE POLICY "Allow all operations for authenticated users" ON rag_queries
    FOR ALL USING (true); 