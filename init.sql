-- init.sql
-- Runs automatically on first container boot
-- Safe to re-run: uses CREATE IF NOT EXISTS throughout

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Tools ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tools (
    id             SERIAL      PRIMARY KEY,
    name           TEXT        NOT NULL UNIQUE,
    description    TEXT        NOT NULL,
    input_schema   JSONB       NOT NULL,
    embedding      vector(384),
    search_vector  tsvector
);

CREATE INDEX IF NOT EXISTS tools_embedding_idx
    ON tools USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS tools_search_vector_idx
    ON tools USING gin (search_vector);

-- ── Users ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id                  SERIAL      PRIMARY KEY,
    name                TEXT        NOT NULL,
    phone               TEXT        NOT NULL UNIQUE,
    preferences_prompt  TEXT        NOT NULL DEFAULT 'You are a helpful personal AI assistant.',
    timezone            TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone);

-- ── Reminders ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS reminders (
    id          SERIAL      PRIMARY KEY,
    user_id     INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message     TEXT        NOT NULL,
    remind_at   TIMESTAMPTZ NOT NULL,
    recurrence  TEXT        CHECK (recurrence IN ('daily', 'weekly', 'monthly')),
    fired       BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reminders_remind_at
    ON reminders (remind_at)
    WHERE fired = FALSE;

CREATE INDEX IF NOT EXISTS idx_reminders_user_id
    ON reminders (user_id);

-- ── Messages (Conversation History) ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id           SERIAL      PRIMARY KEY,
    user_id      INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role         TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    message_type TEXT        NOT NULL DEFAULT 'raw'
                             CHECK (message_type IN ('raw', 'summary', 'fact')),
    content      JSONB       NOT NULL,

    -- Time span (for raw messages both are set to created_at via trigger)
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    period_start TIMESTAMPTZ NOT NULL,
    period_end   TIMESTAMPTZ NOT NULL,

    -- Embeddings
    embedding           vector(384),
    embedding_pruned_at TIMESTAMPTZ,

    -- Keyword search
    search_vector tsvector,

    -- Retrieval scoring (feeds summarisation pruning decisions)
    retrieval_count   INT         NOT NULL DEFAULT 0,
    last_retrieved_at TIMESTAMPTZ,

    -- Summarisation lineage
    source_message_ids   INT[],
    source_message_count INT
);

-- Auto-set period_start/end to created_at for raw message inserts
CREATE OR REPLACE FUNCTION set_period_defaults()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.period_start IS NULL THEN NEW.period_start = NEW.created_at; END IF;
    IF NEW.period_end   IS NULL THEN NEW.period_end   = NEW.created_at; END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER messages_period_defaults
    BEFORE INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION set_period_defaults();

CREATE INDEX IF NOT EXISTS idx_messages_user_period
    ON messages (user_id, period_end DESC);

CREATE INDEX IF NOT EXISTS idx_messages_user_retrieval
    ON messages (user_id, retrieval_count);

CREATE INDEX IF NOT EXISTS messages_embedding_idx
    ON messages USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL AND embedding_pruned_at IS NULL;

CREATE INDEX IF NOT EXISTS messages_search_vector_idx
    ON messages USING gin (search_vector);

-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_phone_isolation ON users
    FOR ALL TO smsai
    USING (phone = current_setting('app.current_phone', TRUE));

ALTER TABLE reminders ENABLE ROW LEVEL SECURITY;

CREATE POLICY reminders_user_isolation ON reminders
    FOR ALL TO smsai
    USING (
        user_id = (
            SELECT id FROM users
            WHERE phone = current_setting('app.current_phone', TRUE)
        )
    );

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY messages_user_isolation ON messages
    FOR ALL TO smsai
    USING (
        user_id = (
            SELECT id FROM users
            WHERE phone = current_setting('app.current_phone', TRUE)
        )
    );