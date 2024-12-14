CREATE TABLE IF NOT EXISTS game_log_failures (
    id SERIAL PRIMARY KEY,
    start_date_str VARCHAR(50),
    end_date_str VARCHAR(50),
    attempt_count INT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
