CREATE TABLE IF NOT EXISTS boxscore_failures (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50),
    attempt_count INT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
