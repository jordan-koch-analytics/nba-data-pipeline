CREATE TABLE IF NOT EXISTS teams (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(255),
    nickname VARCHAR(255),
    team_abbreviation VARCHAR(50),
    city VARCHAR(255),
    state VARCHAR(255),
    year_founded INT
);