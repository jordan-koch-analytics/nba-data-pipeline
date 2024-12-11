import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams, players
import psycopg2
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Endpoint Functions
# ==================
def get_game_logs(start_date_str, end_date_str, max_retries=3, initial_wait=2):
    """
    Fetches game logs over a specified period from nba-api.
    
    Paramters:
    - start_date (str): The first date in the range
    - end_date (str): The last day in the range
    - max_retries (int): Maximum number of retry attempts.
    - initial_wait (int): Initial wait time (in seconds) before the first retry.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing game logs from the specified period.
    """
    
    # Initialize the NBA Game Finder
    gamefinder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable='00',            # '00' corresponds to the NBA
        season_nullable='2024-25',          # Get current season
        date_from_nullable=start_date_str,
        date_to_nullable=end_date_str
    )
    
    for attempt in range(max_retries):
        try:
            game_logs_df = gamefinder.get_data_frames()[0]
            
            return game_logs_df
        
        except Exception as e:
            logger.error(f'Error in get_game_logs: {str(e)}')
            
            # Retry until max retries
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # All attempts have failed
                raise RuntimeError(f"Failed to fetch game logs after {max_retries} attempts.")

def process_game_logs(df):
    """
    Processes raw boxscore data and returns the processed data as a Pandas dataframe.
    
    Paramters:
    - pd.DataFrame: a Pandas dataframe containing raw box score data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing processed player data.
    """
    
    try:
        # Fix column structure
        processed_df = df.drop(['TEAM_ABBREVIATION', 'TEAM_NAME', 'FG_PCT', 'FT_PCT', 'FG3_PCT'], axis=1, errors='ignore')    # Drop unneccary columns
        processed_df.columns = processed_df.columns.str.lower()    # Lowercase columns
        
        # Fix datatypes
        processed_df['season_id'] = pd.to_numeric(processed_df['season_id'], errors='coerce').fillna(0).astype('int64')
        processed_df['game_date'] = pd.to_datetime(processed_df['game_date'])
        processed_df['season_id'] = pd.to_numeric(processed_df['season_id'], errors='coerce').fillna(0).astype('int64')
        
        return processed_df
    
    except Exception as e:
        logger.error(f'Error in process_game_logs: {str(e)}')
        raise e

def get_boxscores(game_id, max_retries=3, initial_wait=2):
    """
    Fetches box score data for a specific game.
    
    Parameters:
    - game_id (str): The unique identifier for the game.
    - max_retries (int): Maximum number of retry attempts.
    - initial_wait (int): Initial wait time in seconds before the first retry.
    
    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the boxscore for the game.
                    Returns an empty DataFrame if no data or if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            boxscore_df = boxscore.get_data_frames()[0]
            logger.info(f"Fetched box scores for game_id: {game_id} on attempt {attempt+1}")
            return boxscore_df
        except Exception as e:
            logger.error(f"Error fetching box scores for game_id {game_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying get_boxscores in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch box scores for game_id {game_id} after {max_retries} attempts.")
                return pd.DataFrame()

def process_boxscores(boxscores_df):
    """
    Processes boxscore data to prepare for RDS insertion.
    
    Parameters:
    - boxscores_df (pd.DataFrame): A Pandas DataFrame containing boxscore data.
    
    Returns:
    - pd.DataFrame: A Pandas DataFrame containing processed boxscore data.
    """
    try:
        clean_boxscores_df = boxscores_df.drop(
            ['TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_NAME', 'NICKNAME', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
            axis=1, errors='ignore'
        )
        clean_boxscores_df.columns = clean_boxscores_df.columns.str.lower()

        # Convert columns to Int64
        cols_to_convert_to_int64 = [
            'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
            'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk',
            'to', 'pf', 'pts', 'plus_minus'
        ]
        for col in cols_to_convert_to_int64:
            if col in clean_boxscores_df.columns:
                clean_boxscores_df[col] = clean_boxscores_df[col].astype('Int64')

        # Convert "min" column to float if it exists
        if 'min' in clean_boxscores_df.columns:
            clean_boxscores_df['min'] = clean_boxscores_df['min'].replace('None', np.nan)
            min_split = clean_boxscores_df['min'].astype(str).str.split(':', expand=True)
            clean_boxscores_df['minutes'] = pd.to_numeric(min_split[0], errors='coerce')
            clean_boxscores_df['seconds'] = pd.to_numeric(min_split[1], errors='coerce')
            clean_boxscores_df['min_float'] = (clean_boxscores_df['minutes'] + (clean_boxscores_df['seconds'] / 60)).round(2)
            clean_boxscores_df['min'] = clean_boxscores_df['min_float']
            clean_boxscores_df.drop(['minutes', 'seconds', 'min_float'], axis=1, errors='ignore', inplace=True)
        else:
            logger.warning("No 'min' column in boxscores. Setting 'min' to null.")
            clean_boxscores_df['min'] = pd.NA

        return clean_boxscores_df
    except Exception as e:
        logger.error(f'Error in process_boxscores: {str(e)}')
        raise e



# Static Data Functions
# =====================
def get_teams():
    """
    Fetches static team data from nba_api and returns it as a DataFrame.
    
    Parameters:
    - None
    
    Returns:
    - pd.DataFrame: Pandas dataframe comtaining team data.
    """
    
    try:
        # Fetch teams data
        nba_teams = teams.get_teams()
        teams_df = pd.DataFrame(nba_teams)
        
        # Process data
        teams_df = process_teams(teams_df)
        logger.info(f"Fetched and processed {len(teams_df)} teams from nba_api")
        
        return teams_df
    
    except Exception as e:
        logger.error(f"Error in fetch_and_store_teams: {str(e)}")
        raise e

def process_teams(teams_df):
    """
    Processes static team data and returns the processed data as a dataframe.
    
    Parameters:
    - teams_df (pd.DataFrame): A Pandas dataframe containing raw team data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe contaning processed team data.
    """
    
    try:
        required_cols = {'id', 'full_name', 'nickname', 'abbreviation', 'city', 'state', 'year_founded'}
        missing = required_cols - set(teams_df.columns)
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        
        processed_df = teams_df[['id', 'full_name', 'nickname', 'abbreviation', 'city', 'state', 'year_founded']]
        processed_df.rename(columns={
            'id': 'team_id',
            'abbreviation': 'team_abbreviation',
            'full_name': 'team_name'
        }, inplace=True)
        
        # Logging
        logger.info(f'Processed team data.')
        
        return processed_df
    
    except Exception as e:
        logger.error(f'Error in process_teams: {str(e)}')
        raise e

def get_players():
    """
    Fetches static player data from nba_api and returns it as a Pandas dataframe.
    
    Parameters:
    - None
    
    Returns:
    - pd.DataFrame: a Pandass dataframe containing player data.
    """
    
    try:
        # Fetch players data
        nba_players = players.get_players()
        players_df = pd.DataFrame(nba_players)
        
        # Process player data
        processed_players_df = process_players(players_df)
        logger.info(f"Fetched {len(players_df)} players from nba_api")
        
        return processed_players_df
    
    except Exception as e:
        logger.error(f"Error in fetch_and_store_players: {str(e)}")
        raise e

def process_players(players_df):
    """
    Processes static player data and returns the processed data as a Pandas dataframe.
    
    Paramters:
    - players_df (pd.DataFrame): A Pandas dataframe containing raw player data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing processed player data.
    """
    
    try:
        required_cols = {'is_active', 'id', 'full_name'}
        missing = required_cols - set(players_df.columns)
        if missing:
            logger.warning(f"Missing expected columns: {missing}")

        df_processed = players_df[players_df['is_active'] == True]    # Get active players
        df_processed.drop('is_active', axis=1, errors='ignore', inplace=True)
        
        df_processed.rename(columns={
            'id': 'player_id',
            'full_name': 'player_name'
        }, inplace=True)
        
        # Logging
        logger.info('Processed player data.')
        
        return df_processed
    
    except Exception as e:
        logger.error(f'Error in process_players: {str(e)}')
        raise e


        
        
        
        

# Database Functions
# ==================        
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise e

def read_sql_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def store_teams_in_rds(teams_df):
    """
    Placeholder function to store teams into RDS.
    Implement connection and insertion logic here.
    """
    logger.info(f"Storing {len(teams_df)} teams to RDS...")
    # TODO: Implement DB insertion logic
    pass

def store_players_in_rds(players_df):
    """
    Placeholder function to store players into RDS.
    Implement connection and insertion logic here.
    """
    logger.info(f"Storing {len(players_df)} players to RDS...")
    # TODO: Implement DB insertion logic
    pass